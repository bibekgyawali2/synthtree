"""Noise simulator, calibration recovery, encodings, PLY I/O, geometry metrics, decomposition."""
import os
import sys

import numpy as np
import pytest

from synthtree.imageio import decode_flow_kitti, decode_normals16, encode_flow_kitti, encode_normals16
from synthtree.noise import DEFAULT_PROFILE, deep_merge, depth_sigma, simulate_depth, simulate_rgb
from synthtree.ply import ply_xyz, read_ply, write_ply

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "eval"))

PLAIN = deep_merge(DEFAULT_PROFILE, {"depth": {"effective_scale": 1.0, "corr_px": 0.0, "holes": {"random": 0.0},
                                                "min_m": 0.05, "max_m": 10.0, "units_m": 1e-5}})


@pytest.mark.parametrize("z0", [0.5, 1.5, 3.0])
def test_depth_sigma_matches_profile(z0):
    z = np.full((200, 200), z0)
    zn = simulate_depth(z, PLAIN, np.random.default_rng(1))
    d = (zn - z)[zn > 0]
    assert abs(d.std() / depth_sigma(z0, PLAIN) - 1) < 0.05
    assert abs(d.mean()) < 3 * depth_sigma(z0, PLAIN) / 200


def test_disparity_model_and_quantisation():
    prof = deep_merge(PLAIN, {"depth": {"model": "disparity", "units_m": 0.001}})
    z = np.full((150, 150), 2.0)
    zn = simulate_depth(z, prof, np.random.default_rng(0))
    assert np.allclose(zn * 1000, np.round(zn * 1000))
    assert 0.5 < (zn - z)[zn > 0].std() / depth_sigma(2.0, prof) < 1.6  # sub-pixel quantisation adds error


def test_noise_deterministic_and_range():
    z = np.full((64, 64), 1.0)
    z[:, :20] = 6.0  # beyond max range
    z[:5] = 0.0
    a = simulate_depth(z, DEFAULT_PROFILE, np.random.default_rng(3))
    b = simulate_depth(z, DEFAULT_PROFILE, np.random.default_rng(3))
    assert np.array_equal(a, b)
    assert (a[:, :18] == 0).all() and (a[:5] == 0).all()


def test_flying_pixels_between_layers():
    z = np.full((100, 100), 2.5)
    z[:, :50] = 0.8
    prof = deep_merge(PLAIN, {"depth": {"a_m": 0.0, "b_per_m": 0.0, "flying": {"rate": 1.0}, "holes": {"edge": 0.0}}})
    zn, dbg = simulate_depth(z, prof, np.random.default_rng(0), return_debug=True)
    between = (zn > 0.85) & (zn < 2.45)
    assert between.any() and between[:, 40:60].sum() == between.sum()


def test_rgb_noise_runs():
    img = np.full((32, 32, 3), 128, np.uint8)
    out = simulate_rgb(img, DEFAULT_PROFILE, np.random.default_rng(0), ev=0.2)
    assert out.shape == img.shape and out.dtype == np.uint8 and out.mean() > 128


def test_calibration_recovers_known_profile(tmp_path):
    import calibrate_noise as cn
    from synthtree.intrinsics import Intrinsics
    intr = Intrinsics(160, 120, 150, 150, 79.5, 59.5)
    truth = deep_merge(DEFAULT_PROFILE, {"depth": {"a_m": 0.001, "b_per_m": 0.006, "corr_px": 1.0, "effective_scale": 1.0,
                                                   "min_m": 0.1, "max_m": 6.0, "flying": {"rate": 0.4},
                                                   "holes": {"random": 0.01, "edge": 0.3, "thin": 0.5}}})
    cn.make_demo(str(tmp_path), intr, truth)
    prof, _ = cn.calibrate(str(tmp_path / "wall"), str(tmp_path / "leaves"), intr)
    d = prof["depth"]
    assert abs(d["a_m"] - 0.001) < 0.0005 and abs(d["b_per_m"] - 0.006) / 0.006 < 0.1
    assert abs(d["corr_px"] - 1.0) < 0.2
    assert abs(d["flying"]["rate"] - 0.4) < 0.1 and abs(d["holes"]["edge"] - 0.3) < 0.1 and abs(d["holes"]["thin"] - 0.5) < 0.15


def test_encodings_roundtrip(rng):
    n = rng.normal(size=(20, 30, 3))
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    n[0, 0] = 0
    d = decode_normals16(encode_normals16(n))
    assert np.abs(d - n).max() < 5e-5 and (d[0, 0] == 0).all()
    f = rng.uniform(-100, 100, (20, 30, 2))
    v = rng.random((20, 30)) > 0.3
    f2, v2 = decode_flow_kitti(encode_flow_kitti(f, v))
    assert np.array_equal(v, v2) and np.abs(f2[v] - f[v]).max() <= 1 / 128 + 1e-6


def test_ply_roundtrip(tmp_path, rng):
    V = rng.random((10, 3)).astype(np.float32)
    F = np.array([[0, 1, 2], [2, 3, 4], [5, 6, 7]])
    write_ply(tmp_path / "m.ply", V, F, face_props={"leaf_id": np.array([1, 2, 3])})
    p = read_ply(tmp_path / "m.ply")
    assert np.allclose(ply_xyz(p["vertex"]), V) and np.array_equal(p["face"]["v"], F)
    assert list(p["face"]["leaf_id"]) == [1, 2, 3]


def test_geometry_metrics(small_tree, tmp_path):
    from eval_geometry import evaluate, load_recon, sample_mesh
    rng = np.random.default_rng(0)
    P, N, _, _ = sample_mesh(small_tree["leaf_verts"], small_tree["leaf_faces"], 150_000, rng)
    perfect = evaluate(P, N, small_tree, tau=0.003, n_gt=100_000)
    assert perfect["accuracy_mm"] < 1.0 and perfect["fscore"] > 0.95  # floor = GT sample spacing
    assert perfect["leaf_angle_w1_deg"] < 2 and perfect["centroid_err_mm_median"] < 1.0
    assert perfect["leaf_area_rel_error"] < 0.05
    shifted = evaluate(P + [0.004, 0, 0], N, small_tree, tau=0.003, n_gt=100_000)
    assert shifted["accuracy_mm"] > perfect["accuracy_mm"] + 1.0 and shifted["centroid_err_mm_median"] > 3.0
    # a 3DGS-style PLY: flat gaussians whose smallest axis is the surface normal
    n = 5000
    idx = rng.choice(len(P), n, replace=False)
    rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("opacity", "<f4")] +
                   [(f"scale_{i}", "<f4") for i in range(3)] + [(f"rot_{i}", "<f4") for i in range(4)])
    rec["x"], rec["y"], rec["z"] = P[idx].T
    rec["opacity"] = 3.0
    rec["scale_0"], rec["scale_1"], rec["scale_2"] = -5, -5, -9  # z axis smallest
    # rotation taking +z to the GT normal (w, x, y, z)
    nz = N[idx]
    axis = np.cross([0, 0, 1.0], nz)
    s = np.linalg.norm(axis, axis=1, keepdims=True)
    ang = np.arctan2(s[:, 0], nz[:, 2])
    axis = axis / np.maximum(s, 1e-12)
    rec["rot_0"] = np.cos(ang / 2)
    rec["rot_1"], rec["rot_2"], rec["rot_3"] = (axis * np.sin(ang / 2)[:, None]).T
    with open(tmp_path / "g.ply", "wb") as f:
        hdr = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"] + [f"property float {k}" for k in rec.dtype.names] + ["end_header"]
        f.write(("\n".join(hdr) + "\n").encode())
        f.write(rec.tobytes())
    Pg, Ng, kind = load_recon(str(tmp_path / "g.ply"), "auto", 0.5, rng)
    assert kind == "gaussians" and np.allclose(np.abs(np.einsum("ij,ij->i", Ng, nz)), 1, atol=1e-4)


def test_decomposition_arithmetic():
    from decompose_errors import decompose
    rows = []
    for tree in ("a", "b"):
        base = 2.0 if tree == "a" else 4.0
        rows += [{"method": "m", "tree": tree, "wind_U": 0.0, "sensor": "clean", "e": base},
                 {"method": "m", "tree": tree, "wind_U": 0.0, "sensor": "noisy", "e": base + 1},
                 {"method": "m", "tree": tree, "wind_U": 5.0, "sensor": "noisy", "e": base + 4},
                 {"method": "m", "tree": tree, "wind_U": 5.0, "sensor": "clean", "e": base + 2.5}]
    s = decompose(rows, "e")["m"]
    assert s["method"]["0"]["mean"] == 3.0 and s["sensor"]["0"]["mean"] == 1.0
    assert s["wind"]["5"]["mean"] == 3.0 and s["interaction"]["5"]["mean"] == 0.5
