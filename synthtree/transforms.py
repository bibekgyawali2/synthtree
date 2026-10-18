"""Rotations, rigid transforms and camera-axis conventions.

CONVENTIONS (the single most common source of bugs; see docs/PITFALLS.md):

  World:   Blender world, right-handed, +Z up, metres.
  Camera, Blender == OpenGL == Nerfstudio:  +X right, +Y up,   looks along -Z.
  Camera, OpenCV  == TUM == MonoGS:         +X right, +Y down, looks along +Z.

  All stored poses are camera-to-world (c2w) 4x4 matrices.
  c2w_opencv = c2w_blender @ diag(1, -1, -1, 1)  (flip the camera's Y and Z axes;
  the camera centre is unchanged). The same matrix converts back (it is its own inverse).
"""
from __future__ import annotations

import numpy as np

BLENDER_TO_OPENCV_CAM = np.diag([1.0, -1.0, -1.0, 1.0])


def normalize(v, axis=-1, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def rodrigues(axis, angle):
    """Rotation matrices about unit `axis` (...,3) by `angle` (...) radians.

    Written element-wise so that angle == 0 gives *exactly* the identity
    (sin 0 = 0, 1 - cos 0 = 0): the calm sequence must be bit-exact rest pose.
    """
    axis = np.asarray(axis, dtype=float)
    angle = np.asarray(angle, dtype=float)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    s, c = np.sin(angle), np.cos(angle)
    C = 1.0 - c
    shape = np.broadcast(x, angle).shape
    R = np.empty(shape + (3, 3))
    R[..., 0, 0] = c + x * x * C
    R[..., 0, 1] = x * y * C - z * s
    R[..., 0, 2] = x * z * C + y * s
    R[..., 1, 0] = y * x * C + z * s
    R[..., 1, 1] = c + y * y * C
    R[..., 1, 2] = y * z * C - x * s
    R[..., 2, 0] = z * x * C - y * s
    R[..., 2, 1] = z * y * C + x * s
    R[..., 2, 2] = c + z * z * C
    return R


def perpendicular_basis(t):
    """Two unit vectors (e1, e2) with e1 x e2 = t for unit t (...,3)."""
    t = normalize(t)
    ref = np.where(np.abs(t[..., 2:3]) < 0.9, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    e1 = normalize(np.cross(ref, t))
    e2 = np.cross(t, e1)
    return e1, e2


def make_rigid(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_rigid(T):
    T = np.asarray(T, dtype=float)
    Ti = np.zeros_like(T)
    R = T[..., :3, :3]
    Rt = np.swapaxes(R, -1, -2)
    Ti[..., :3, :3] = Rt
    Ti[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, T[..., :3, 3])
    Ti[..., 3, 3] = 1.0
    return Ti


def c2w_blender_to_opencv(c2w):
    return np.asarray(c2w, dtype=float) @ BLENDER_TO_OPENCV_CAM


def c2w_opencv_to_blender(c2w):
    return np.asarray(c2w, dtype=float) @ BLENDER_TO_OPENCV_CAM


def c2w_blender_to_opengl(c2w):
    """Blender's camera axes already are the OpenGL/Nerfstudio camera axes."""
    return np.array(c2w, dtype=float)


def look_at_blender(eye, target, up=(0.0, 0.0, 1.0)):
    """Blender-convention c2w for a camera at `eye` looking at `target`."""
    eye = np.asarray(eye, dtype=float)
    fwd = normalize(np.asarray(target, dtype=float) - eye)
    right = normalize(np.cross(fwd, up))
    cam_up = np.cross(right, fwd)
    R = np.stack([right, cam_up, -fwd], axis=1)
    return make_rigid(R, eye)


def matrix_to_quaternion(R):
    """3x3 rotation -> unit quaternion (x, y, z, w), w >= 0 (Shepperd's method)."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.asarray(q)
    q = q / np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def quaternion_to_matrix(q):
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def apply_rigid(T, X):
    """Apply (...,4,4) or (...,3,4) rigid transforms to points X (...,3)."""
    T = np.asarray(T)
    return np.einsum("...ij,...j->...i", T[..., :3, :3], X) + T[..., :3, 3]
