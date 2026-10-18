"""Camera conventions, intrinsics <-> Blender mapping, pose file round trips."""
import numpy as np

from synthtree.intrinsics import Intrinsics, backproject, from_blender, project, to_blender
from synthtree.pose_formats import read_tum, transforms_json, write_tum
from synthtree.transforms import (c2w_blender_to_opencv, c2w_opencv_to_blender, look_at_blender,
                                  matrix_to_quaternion, quaternion_to_matrix, rodrigues)


def test_rodrigues_exact_identity():
    R = rodrigues(np.array([[0.3, 0.4, np.sqrt(0.75)]]), np.array([0.0]))
    assert np.array_equal(R[0], np.eye(3))


def test_quaternion_roundtrip(rng):
    for _ in range(200):
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        R = rodrigues(ax, rng.uniform(-np.pi, np.pi))
        assert np.allclose(quaternion_to_matrix(matrix_to_quaternion(R)), R, atol=1e-12)


def test_blender_opencv_axes():
    # a camera at (2,0,1) looking at the origin-ish target
    c2w_b = look_at_blender([2, 0, 1], [0, 0, 1])
    c2w_cv = c2w_blender_to_opencv(c2w_b)
    assert np.allclose(c2w_opencv_to_blender(c2w_cv), c2w_b)
    # OpenCV +Z must point from the camera towards the target, +Y down (world -Z)
    assert np.allclose(c2w_cv[:3, 2], [-1, 0, 0])
    assert np.allclose(c2w_cv[:3, 1], [0, 0, -1])
    # Blender camera looks along -Z, +Y up
    assert np.allclose(-c2w_b[:3, 2], [-1, 0, 0]) and np.allclose(c2w_b[:3, 1], [0, 0, 1])
    intr = Intrinsics(640, 480, 500, 500, 319.5, 239.5)
    u, v, z = project(intr, c2w_cv, np.array([0.0, 0.0, 1.0]))
    assert np.isclose(u, 319.5) and np.isclose(v, 239.5) and np.isclose(z, 2.0)
    u, v, _ = project(intr, c2w_cv, np.array([0.0, 0.0, 1.5]))  # above target => upper half
    assert v < 239.5
    # looking along world -X with +Z up, camera right = fwd x up = world +Y
    u, v, _ = project(intr, c2w_cv, np.array([0.0, 0.5, 1.0]))
    assert u > 319.5


def test_intrinsics_blender_roundtrip():
    for intr in [Intrinsics(848, 480, 603, 603, 423.5, 239.5), Intrinsics(640, 480, 610.2, 606.9, 322.1, 236.4),
                 Intrinsics(1280, 720, 900, 910, 650.0, 355.0)]:
        back = from_blender(to_blender(intr))
        for k in ("fx", "fy", "cx", "cy"):
            assert abs(getattr(back, k) - getattr(intr, k)) < 1e-9, k


def test_backproject_project_inverse(rng):
    intr = Intrinsics(64, 48, 50, 52, 31.0, 24.5)
    c2w_cv = c2w_blender_to_opencv(look_at_blender([1, 1, 1], [0, 0, 0.3]))
    depth = rng.uniform(0.5, 2.0, (48, 64))
    X = backproject(intr, c2w_cv, depth)
    u, v, z = project(intr, c2w_cv, X)
    uu, vv = np.meshgrid(np.arange(64), np.arange(48))
    assert np.allclose(u, uu) and np.allclose(v, vv) and np.allclose(z, depth)


def test_scaled_intrinsics_centre():
    intr = Intrinsics(848, 480, 603, 603, 423.5, 239.5).scaled(0.5)
    assert intr.width == 424 and np.isclose(intr.cx, 211.5) and np.isclose(intr.fx, 301.5)


def test_tum_and_transforms_roundtrip(tmp_path):
    poses = [look_at_blender([np.cos(a), np.sin(a), 0.8], [0, 0, 0.6]) for a in np.linspace(0, 6, 10)]
    times = np.arange(10) / 7.5
    write_tum(tmp_path / "gt.txt", times, poses)
    t2, cv = read_tum(tmp_path / "gt.txt")
    assert np.allclose(t2, times)
    for P, C in zip(poses, cv):
        assert np.allclose(c2w_blender_to_opencv(P), C, atol=1e-8)
    tj = transforms_json(Intrinsics(64, 48, 50, 50, 31.5, 23.5), [{"c2w_blender": P, "file_path": "x.png"} for P in poses])
    assert np.allclose(tj["frames"][3]["transform_matrix"], poses[3])
