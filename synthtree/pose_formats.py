"""Pose exports: Nerfstudio transforms.json, TUM RGB-D, raw .npy.

transforms.json  : c2w in OpenGL camera convention (== Blender camera axes),
                   world = Blender world (+Z up, metres).
TUM groundtruth  : 'timestamp tx ty tz qx qy qz qw', c2w in OpenCV camera
                   convention (TUM optical frame: x right, y down, z forward),
                   world = Blender world.
poses/*.npy      : c2w_opencv (N,4,4) and c2w_blender (N,4,4), plus per-frame files.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .intrinsics import Intrinsics
from .transforms import c2w_blender_to_opencv, c2w_blender_to_opengl, matrix_to_quaternion, quaternion_to_matrix


def transforms_json(intr: Intrinsics, frames: list, extra: dict = None) -> dict:
    """frames: dicts with c2w_blender, file_path, optional depth_file_path, time, extra keys."""
    out = {
        "camera_model": "OPENCV",
        "fl_x": intr.fx, "fl_y": intr.fy, "cx": intr.cx, "cy": intr.cy,
        "w": intr.width, "h": intr.height,
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        "frames": [],
    }
    for fr in frames:
        d = {k: v for k, v in fr.items() if k != "c2w_blender"}
        d["transform_matrix"] = c2w_blender_to_opengl(fr["c2w_blender"]).tolist()
        out["frames"].append(d)
    if extra:
        out.update(extra)
    return out


def write_tum(path, times, c2w_blender):
    with open(path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw  (c2w, OpenCV camera axes, Blender world, metres)\n")
        for t, M in zip(times, c2w_blender):
            C = c2w_blender_to_opencv(M)
            q = matrix_to_quaternion(C[:3, :3])
            tx, ty, tz = C[:3, 3]
            f.write(f"{t:.6f} {tx:.9f} {ty:.9f} {tz:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")


def read_tum(path):
    times, poses = [], []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        v = [float(x) for x in line.split()]
        M = np.eye(4)
        M[:3, :3] = quaternion_to_matrix(v[4:8])
        M[:3, 3] = v[1:4]
        times.append(v[0])
        poses.append(M)
    return np.asarray(times), np.asarray(poses)  # c2w OpenCV


def write_tum_list(path, times, rel_paths, header):
    with open(path, "w") as f:
        f.write(f"# {header}\n# timestamp filename\n")
        for t, p in zip(times, rel_paths):
            f.write(f"{t:.6f} {p}\n")


def save_pose_arrays(pose_dir, c2w_blender, per_frame=True):
    os.makedirs(pose_dir, exist_ok=True)
    c2w_blender = np.asarray(c2w_blender, np.float64)
    c2w_cv = np.stack([c2w_blender_to_opencv(M) for M in c2w_blender])
    np.save(os.path.join(pose_dir, "c2w_blender.npy"), c2w_blender)
    np.save(os.path.join(pose_dir, "c2w_opencv.npy"), c2w_cv)
    if per_frame:
        d = os.path.join(pose_dir, "c2w_opencv")
        os.makedirs(d, exist_ok=True)
        for i, M in enumerate(c2w_cv):
            np.save(os.path.join(d, f"{i:05d}.npy"), M)


def intrinsics_json(intr: Intrinsics, depth_units_m=0.001):
    return {"width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
            "K": intr.K.tolist(), "distortion": [0, 0, 0, 0, 0], "pixel_convention": "opencv (pixel centre at integer coords)",
            "depth_units_m": depth_units_m, "depth_type": "z-depth along optical axis; 0 = invalid"}


def load_json(path):
    with open(path) as f:
        return json.load(f)
