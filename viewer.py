"""
Live preview: ray-knight facial rig with glowing energy-line wireframe.

The face plays back its baked FACS performance (280 frames) while a bright
cyan-blue highlight continuously travels across the wireframe tracing the
current frame's geometry. Drag with the mouse to orbit, scroll to zoom; the
camera also auto-rotates slowly when idle.

`FaceRig.set_external_frame(idx)` is the hook for driving the face from an
outside source (e.g. a viseme/phoneme stream from a TTS pipeline) instead of
playing the baked clip - call it with a frame index each tick, or None to
resume normal playback.

Run: python viewer.py
(requires model_data.npz - generate it once with build_data.py)
"""
import bisect
import os

import glm
import moderngl
import moderngl_window as mglw
import numpy as np

ENERGY_VERT = """
#version 330
uniform mat4 mvp;
in vec3 in_position;
in vec3 in_bary;
in float in_t;
out vec3 v_bary;
out float v_t;
void main() {
    v_bary = in_bary;
    v_t = in_t;
    gl_Position = mvp * vec4(in_position, 1.0);
}
"""

ENERGY_FRAG = """
#version 330
uniform float u_time;
uniform float u_speed;
uniform float u_line_width;
uniform float u_pulse_width;
uniform int u_pulse_count;
uniform vec3 u_base_color;
uniform vec3 u_pulse_color;
in vec3 v_bary;
in float v_t;
out vec4 f_color;

float edge_factor(vec3 bary, float width) {
    vec3 d = fwidth(bary);
    vec3 a3 = smoothstep(vec3(0.0), d * width, bary);
    return min(min(a3.x, a3.y), a3.z);
}

void main() {
    float line = 1.0 - edge_factor(v_bary, u_line_width);
    if (line <= 0.01) discard;

    float pulse = 0.0;
    for (int i = 0; i < u_pulse_count; i++) {
        float center = fract(u_time * u_speed + float(i) / float(u_pulse_count));
        float d = v_t - center;
        d -= floor(d + 0.5);
        pulse += exp(-pow(d / u_pulse_width, 2.0));
    }
    pulse = min(pulse, 1.0);

    vec3 color = u_base_color * 0.55 + u_pulse_color * pulse * 3.0;
    float alpha = line * (0.30 + pulse * 1.2);
    f_color = vec4(color * line, clamp(alpha, 0.0, 1.0));
}
"""

BODY_VERT = """
#version 330
uniform mat4 mvp;
in vec3 in_position;
in vec3 in_normal;
out vec3 v_normal;
out vec3 v_pos;
void main() {
    v_normal = in_normal;
    v_pos = in_position;
    gl_Position = mvp * vec4(in_position, 1.0);
}
"""

BODY_FRAG = """
#version 330
uniform vec3 u_eye;
in vec3 v_normal;
in vec3 v_pos;
out vec4 f_color;
void main() {
    vec3 n = normalize(v_normal);
    vec3 light_dir = normalize(vec3(0.4, 0.8, 0.6));
    float diffuse = max(dot(n, light_dir), 0.0);
    vec3 view_dir = normalize(u_eye - v_pos);
    float fresnel = pow(1.0 - max(dot(n, view_dir), 0.0), 2.5);
    vec3 base = vec3(0.05, 0.07, 0.11);
    vec3 color = base * (0.25 + diffuse * 0.6) + vec3(0.15, 0.22, 0.30) * fresnel;
    f_color = vec4(color, 1.0);
}
"""


class FaceRig:
    """Selects which baked FACS frame is active at a given time.

    Plays the model's authored clip by default. Call set_external_frame()
    to override with frames driven by an outside source (e.g. lip sync);
    pass None to hand control back to the clip.
    """

    def __init__(self, activation_times):
        self.activation_times = activation_times
        frame_dt = activation_times[-1] - activation_times[-2]
        self.loop_duration = float(activation_times[-1] + frame_dt)
        self._external_frame = None

    def set_external_frame(self, frame_idx):
        self._external_frame = frame_idx

    def frame_for_time(self, t):
        if self._external_frame is not None:
            return self._external_frame
        t_clip = t % self.loop_duration
        return bisect.bisect_right(self.activation_times, t_clip) - 1


class EnergyLinesViewer(mglw.WindowConfig):
    title = "Energy Lines - Ray Knight Face Rig"
    window_size = (1280, 800)
    aspect_ratio = None
    resizable = True
    resource_dir = os.path.dirname(os.path.abspath(__file__))
    gl_version = (3, 3)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx.enable(moderngl.DEPTH_TEST)

        data = np.load(os.path.join(self.resource_dir, "model_data.npz"))
        self.center = glm.vec3(*data["center"])
        self.radius = float(data["radius"])

        self.energy_frames = data["energy_frames"]  # (F, V, 7) pos3 bary3 t1
        self.body_frames = data["body_frames"]  # (F, V, 6) pos3 normal3
        self.verts_per_frame = self.energy_frames.shape[1]
        self.rig = FaceRig(data["activation_times"])
        self.current_frame = -1

        self.energy_prog = self.ctx.program(vertex_shader=ENERGY_VERT, fragment_shader=ENERGY_FRAG)
        self.body_prog = self.ctx.program(vertex_shader=BODY_VERT, fragment_shader=BODY_FRAG)

        self.energy_vbo = self.ctx.buffer(reserve=self.energy_frames[0].nbytes)
        self.energy_vao = self.ctx.vertex_array(
            self.energy_prog,
            [(self.energy_vbo, "3f 3f 1f", "in_position", "in_bary", "in_t")],
        )

        self.body_vbo = self.ctx.buffer(reserve=self.body_frames[0].nbytes)
        self.body_vao = self.ctx.vertex_array(
            self.body_prog,
            [(self.body_vbo, "3f 3f", "in_position", "in_normal")],
        )

        self._upload_frame(0)

        # orbit camera state
        self.yaw = 0.6
        self.pitch = 0.15
        self.distance = self.radius * 3.2
        self.auto_rotate = True

        self.energy_prog["u_base_color"].value = (0.85, 0.90, 1.0)
        self.energy_prog["u_pulse_color"].value = (0.10, 0.85, 1.0)
        self.energy_prog["u_speed"].value = 0.35
        self.energy_prog["u_line_width"].value = 1.4
        self.energy_prog["u_pulse_width"].value = 0.10
        self.energy_prog["u_pulse_count"].value = 3

    def _upload_frame(self, frame_idx):
        self.current_frame = frame_idx
        self.energy_vbo.write(np.ascontiguousarray(self.energy_frames[frame_idx]).tobytes())
        self.body_vbo.write(np.ascontiguousarray(self.body_frames[frame_idx]).tobytes())

    def _camera_matrices(self):
        eye = self.center + glm.vec3(
            self.distance * glm.cos(self.pitch) * glm.sin(self.yaw),
            self.distance * glm.sin(self.pitch),
            self.distance * glm.cos(self.pitch) * glm.cos(self.yaw),
        )
        view = glm.lookAt(eye, self.center, glm.vec3(0, 1, 0))
        proj = glm.perspective(glm.radians(45.0), self.wnd.aspect_ratio, 0.01, self.radius * 50)
        return eye, view, proj

    def on_render(self, time, frametime):
        if self.auto_rotate:
            self.yaw += frametime * 0.15

        frame_idx = self.rig.frame_for_time(time)
        if frame_idx != self.current_frame:
            self._upload_frame(frame_idx)

        self.ctx.clear(0.015, 0.02, 0.045)
        eye, view, proj = self._camera_matrices()
        mvp = proj * view

        self.body_prog["mvp"].write(mvp.to_bytes())
        self.body_prog["u_eye"].value = tuple(eye)
        self.body_vao.render(moderngl.TRIANGLES, vertices=self.verts_per_frame)

        self.energy_prog["mvp"].write(mvp.to_bytes())
        self.energy_prog["u_time"].value = time

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        self.ctx.depth_func = "<="
        self.energy_vao.render(moderngl.TRIANGLES, vertices=self.verts_per_frame)
        self.ctx.depth_func = "<"
        self.ctx.disable(moderngl.BLEND)

    def on_mouse_drag_event(self, x, y, dx, dy):
        self.auto_rotate = False
        self.yaw += dx * 0.005
        self.pitch = max(-1.4, min(1.4, self.pitch - dy * 0.005))

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self.distance = max(self.radius * 0.5, min(self.radius * 8.0, self.distance * (1.0 - y_offset * 0.1)))

    def on_key_event(self, key, action, modifiers):
        keys = self.wnd.keys
        if action == keys.ACTION_PRESS and key == keys.SPACE:
            self.auto_rotate = not self.auto_rotate


if __name__ == "__main__":
    mglw.run_window_config(EnergyLinesViewer)
