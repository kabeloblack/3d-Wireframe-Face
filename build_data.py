"""
Extracts the ray-knight face rig from scene.gltf into a .npz cache the
viewer loads instantly.

The glTF stores its facial performance as a baked "flipbook": 280 separate
meshes (mesh_mid_FACS_001_39_0 .. _279), each scaled from ~0 to 1 at its own
timestamp and never scaled back down. We treat the most-recently-activated
frame as the currently visible one (the only ordering that produces a single
coherent face shape per instant instead of all 280 overlapping).

For every frame we bake a triangle SOUP (each triangle gets its own 3 verts)
carrying:
  - a barycentric attribute (drives the wireframe-edge shading)
  - a 0..1 "t" coordinate along a FIXED axis (computed once from frame 0,
    reused for every frame so the energy-pulse direction doesn't jitter as
    the mouth moves)
  - a vertex normal (drives the dim solid-shading pass underneath the wires)

Run once: python build_data.py
"""
import os
import struct

import numpy as np
import pygltflib
import trimesh

MODEL_PATH = "../scene.gltf"
OUT_PATH = "model_data.npz"
FRAME_COUNT = 280
FACE_NAME_TEMPLATE = "mesh_mid_FACS_001_39_{i}"
TRIS_PER_FRAME = 9000  # constant across all 280 frames (verified)
BARY = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)


ACTIVE_SCALE = 0.01  # frame 0's (always-visible) scale; the canonical "shown" state


def world_geometry(scene, geom_name):
    """World-space verts/faces, using the *activated* (scale=1) state of each
    FACS frame node rather than its frozen pre-animation rest scale (which
    trimesh reports as-is, near 0, since it doesn't evaluate animations)."""
    for node_name in scene.graph.nodes_geometry:
        transform, g_name = scene.graph.get(node_name)
        if g_name == geom_name:
            geom = scene.geometry[geom_name]
            linear = transform[:3, :3]
            scale = abs(np.linalg.det(linear)) ** (1 / 3)
            if scale < 1e-6:
                transform = transform.copy()
                transform[:3, :3] = linear * (ACTIVE_SCALE / scale)
            verts = trimesh.transform_points(geom.vertices, transform)
            return verts, geom.faces.copy()
    raise KeyError(geom_name)


def fixed_axis_from_frame0(verts0):
    centered = verts0 - verts0.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    proj = centered @ axis
    return verts0.mean(axis=0), axis, proj.min(), proj.max()


def path_coordinate(verts, center, axis, lo, hi):
    proj = (verts - center) @ axis
    span = hi - lo
    t = (proj - lo) / span if span > 1e-9 else np.zeros_like(proj)
    return np.clip(t, 0.0, 1.0).astype(np.float32)


def read_accessor(gltf, raw_by_buffer, idx):
    acc = gltf.accessors[idx]
    bv = gltf.bufferViews[acc.bufferView]
    n_comp = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}[acc.type]
    start = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    fmt = "<" + "f" * n_comp * acc.count
    data = struct.unpack_from(fmt, raw_by_buffer[bv.buffer], start)
    return np.array(data).reshape(acc.count, n_comp)


def read_frame_activation_times():
    """Returns a (FRAME_COUNT,) array: the time at which each frame becomes active."""
    gltf = pygltflib.GLTF2().load(MODEL_PATH)
    base_dir = os.path.dirname(os.path.abspath(MODEL_PATH))
    raw_by_buffer = []
    for buf in gltf.buffers:
        with open(os.path.join(base_dir, buf.uri), "rb") as f:
            raw_by_buffer.append(f.read())

    def mesh_name_under(node_idx):
        """The channel target is a wrapper node; the mesh lives on a descendant."""
        n = gltf.nodes[node_idx]
        if n.mesh is not None:
            return gltf.meshes[n.mesh].name
        for c in n.children or []:
            found = mesh_name_under(c)
            if found:
                return found
        return None

    times_by_frame_idx = {0: 0.0}
    for ch in gltf.animations[0].channels:
        if ch.target.path != "scale":
            continue
        node_idx = ch.target.node
        name = mesh_name_under(node_idx) or ""
        if not name.startswith("mesh_mid_FACS_001_39_"):
            continue
        frame_idx = int(name.rsplit("_", 1)[1])
        if frame_idx == 0:
            continue  # frame 0 is the permanent baseline, active from t=0
        sampler = gltf.animations[0].samplers[ch.sampler]
        times = read_accessor(gltf, raw_by_buffer, sampler.input).ravel()
        scales = read_accessor(gltf, raw_by_buffer, sampler.output)
        activation_time = times[-1] if scales[-1][0] > 0.5 else times[0]
        times_by_frame_idx[frame_idx] = float(activation_time)

    return np.array([times_by_frame_idx[i] for i in range(FRAME_COUNT)], dtype=np.float32)


def main():
    scene = trimesh.load(MODEL_PATH, process=False)

    verts0, faces0 = world_geometry(scene, FACE_NAME_TEMPLATE.format(i=0))
    center, axis, lo, hi = fixed_axis_from_frame0(verts0)

    n_soup_verts = TRIS_PER_FRAME * 3
    energy_all = np.empty((FRAME_COUNT, n_soup_verts, 7), dtype=np.float32)  # pos3 bary3 t1
    body_all = np.empty((FRAME_COUNT, n_soup_verts, 6), dtype=np.float32)  # pos3 normal3

    for i in range(FRAME_COUNT):
        verts, faces = world_geometry(scene, FACE_NAME_TEMPLATE.format(i=i))
        assert faces.shape[0] == TRIS_PER_FRAME, f"frame {i} has {faces.shape[0]} tris"

        t = path_coordinate(verts, center, axis, lo, hi)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        normals = mesh.vertex_normals.astype(np.float32)

        tri_pos = verts[faces].reshape(-1, 3).astype(np.float32)
        tri_bary = np.tile(BARY, (faces.shape[0], 1))
        tri_t = t[faces].reshape(-1, 1)
        tri_normal = normals[faces].reshape(-1, 3)

        energy_all[i] = np.hstack([tri_pos, tri_bary, tri_t])
        body_all[i] = np.hstack([tri_pos, tri_normal])

        if i % 40 == 0:
            print(f"baked frame {i}/{FRAME_COUNT}")

    activation_times = read_frame_activation_times()

    all_pts = energy_all[:, :, :3].reshape(-1, 3)
    bbox_center = all_pts.mean(axis=0)
    radius = np.linalg.norm(all_pts - bbox_center, axis=1).max()

    np.savez(
        OUT_PATH,
        energy_frames=energy_all,
        body_frames=body_all,
        activation_times=activation_times,
        center=bbox_center.astype(np.float32),
        radius=np.float32(radius),
    )
    print(f"frames: {FRAME_COUNT}, tris/frame: {TRIS_PER_FRAME}")
    print(f"clip duration: {activation_times[-1]:.3f}s")
    print(f"center={bbox_center}, radius={radius:.4f}")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
