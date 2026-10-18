"""Tree generator, leaf meshes, wind model and pairing invariants (no Blender needed)."""
import numpy as np
import pytest

from synthtree.camera_path import CameraPath, tree_bbox
from synthtree.intrinsics import Intrinsics
from synthtree.leaves import blade, face_normals_areas, leaf_template, needle
from synthtree.meshing import branch_tube, pot_meshes
from synthtree.tree import generate_tree
from synthtree.wind import WindModel

WIND = {"gust": {"intensity": 0.4}}


def test_tree_deterministic(species):
    a = generate_tree(species["small_dense"], 7)
    b = generate_tree(species["small_dense"], 7)
    for k in a:
        if k != "meta_json":
            assert a[k].tobytes() == b[k].tobytes(), k
    c = generate_tree(species["small_dense"], 8)
    assert not np.array_equal(a["leaf_attach"][:5], c["leaf_attach"][:5])


@pytest.mark.parametrize("name", ["broad_medium", "small_dense", "compound_medium", "needle_dense", "broad_medium_autumn"])
def test_tree_invariants(species, name):
    t = generate_tree(species[name], 3)
    L = len(t["leaf_id"])
    assert np.array_equal(t["leaf_id"], np.arange(1, L + 1))
    assert set(np.unique(t["leaf_face_leaf"])) == set(range(L))
    # every leaf attaches to the bark of the segment starting at its node
    p, r = t["nodes_p0"], t["nodes_radius"]
    for i in range(0, L, max(1, L // 50)):
        n = t["leaf_node"][i]
        nxt = n + 1  # next node on the same branch (nodes of a branch are contiguous)
        assert t["nodes_branch"][nxt] == t["leaf_branch"][i]
        d = p[nxt] - p[n]
        s = np.clip(np.dot(t["leaf_attach"][i] - p[n], d) / np.dot(d, d), 0, 1)
        dist = np.linalg.norm(t["leaf_attach"][i] - (p[n] + s * d))
        assert abs(dist - ((1 - s) * r[n] + s * r[nxt])) < 1e-6
    # leaf faces are non-degenerate, and the rest frame is orthonormal
    _, area = face_normals_areas(t["leaf_verts"], t["leaf_faces"])
    assert area.min() > 0
    F = t["leaf_frame"]
    assert np.allclose(np.einsum("nji,njk->nik", F, F), np.eye(3), atol=1e-9)
    assert (t["leaf_area"] > 0).all()
    # parents precede children (required by forward kinematics)
    par = t["nodes_parent"]
    assert (par[1:] < np.arange(1, len(par))).all() and par[0] == -1


def test_leaf_winding_up():
    v, f = blade(1.0, 0.5, 6, "ovate", curl=0.0, droop=0.0)
    n, _ = face_normals_areas(v, f)
    assert (n[:, 2] > 0.99).all()
    for kind in ("broad", "compound"):
        v, f = leaf_template({"type": kind})
        n, _ = face_normals_areas(v, f)
        assert (n[:, 2] > 0).all()


def test_closed_meshes_outward():
    v, f = needle(1.0, 0.05, 3)
    n, a = face_normals_areas(v, f)
    c = v[f].mean(1)
    radial = c - np.array([0.4, 0, 0]) * [1, 0, 0]
    radial[:, 0] = 0
    side = np.linalg.norm(radial, axis=1) > 1e-9
    assert (np.einsum("ij,ij->i", n[side], radial[side]) > 0).all()
    P = np.array([[0, 0, 0], [0, 0, 0.5], [0, 0.1, 1.0]], float)
    v, f, _ = branch_tube(P, np.array([0.05, 0.04, 0.02]), 8)
    n, _ = face_normals_areas(v, f)
    c = v[f].mean(1)
    radial = c.copy()
    radial[:, 2] = 0
    radial[:, 1] -= 0.1 * np.clip(c[:, 2] - 0.5, 0, None) / 0.5
    assert np.mean(np.einsum("ij,ij->i", n, radial) > 0) > 0.95
    (pv, pf), _ = pot_meshes(0.12, 0.09, 0.18, 0.025)
    n, _ = face_normals_areas(pv, pf)
    assert n.shape[0] == pf.shape[0]


def test_wind_calm_is_exact_rest(small_tree):
    m = WindModel(small_tree, WIND, 0.0, seed=1)
    for t in (0.0, 0.37, 5.0):
        st = m.evaluate(t)
        assert np.array_equal(st.wood_verts, small_tree["wood_verts"])
        assert np.array_equal(st.leaf_verts, small_tree["leaf_verts"])
        assert np.array_equal(st.leaf_A, np.broadcast_to(np.eye(3), st.leaf_A.shape))
        assert np.array_equal(st.leaf_t, np.zeros_like(st.leaf_t))


@pytest.mark.parametrize("U", [1.5, 5.0])
def test_wind_leaves_rigid_and_consistent(small_tree, needle_tree, U):
    for tree in (small_tree, needle_tree):
        m = WindModel(tree, WIND, U, seed=1)
        st = m.evaluate(1.234)
        A = st.leaf_A
        assert np.allclose(np.einsum("nji,njk->nik", A, A), np.eye(3), atol=1e-12)
        assert np.allclose(np.linalg.det(A), 1.0)
        # vertices equal the exported rigid transform applied to the canonical leaf
        vl = tree["leaf_vert_leaf"]
        pred = np.einsum("nij,nj->ni", A[vl], tree["leaf_verts"]) + st.leaf_t[vl]
        assert np.abs(pred - st.leaf_verts).max() < 1e-12
        # attachment point stays on the moving segment axis offset (follows its node frame)
        n = tree["leaf_node"]
        a_pred = st.node_disp[n] + tree["nodes_p0"][n] + np.einsum("nij,nj->ni", st.node_A[n], tree["leaf_attach"] - tree["nodes_p0"][n])
        assert np.abs(a_pred - st.leaf_attach).max() < 1e-12
        # it actually moves, and more for stronger wind
        assert np.abs(st.leaf_verts - tree["leaf_verts"]).max() > 1e-3


def test_wind_amplitude_grows_with_height_and_U(small_tree):
    disp = {}
    for U in (1.5, 3.0, 5.0):
        m = WindModel(small_tree, {"gust": {"intensity": 0.0}}, U, seed=1)
        ts = np.linspace(0, 4, 40)
        disp[U] = np.mean([np.linalg.norm(m.evaluate(t).node_disp, axis=1) for t in ts], axis=0)
    assert disp[1.5].max() < disp[3.0].max() < disp[5.0].max()
    z = small_tree["nodes_p0"][:, 2]
    trunk = small_tree["nodes_branch"] == 0
    assert np.corrcoef(z[trunk], disp[5.0][trunk])[0, 1] > 0.9


def test_wind_phases_independent_of_U(small_tree):
    a = WindModel(small_tree, WIND, 1.5, seed=5)
    b = WindModel(small_tree, WIND, 5.0, seed=5)
    assert np.array_equal(a.ph_branch, b.ph_branch) and np.array_equal(a.f_leaf, b.f_leaf)


def test_camera_pairing_and_smoothness(small_tree):
    intr = Intrinsics(848, 480, 603, 603, 423.5, 239.5)
    bb = tree_bbox(small_tree)
    p1 = CameraPath({}, intr, bb, seed=11)
    p2 = CameraPath({}, intr, bb, seed=11)
    ts = p1.frame_times()
    assert len(ts) == 150
    A = np.stack([p1.c2w(t) for t in ts])
    B = np.stack([p2.c2w(t) for t in ts])
    assert A.tobytes() == B.tobytes()
    R = A[:, :3, :3]
    assert np.allclose(np.einsum("nji,njk->nik", R, R), np.eye(3), atol=1e-12)
    step = np.linalg.norm(np.diff(A[:, :3, 3], axis=0), axis=1)
    assert step.max() < 3 * np.median(step)
    # camera looks roughly at the tree: forward (-Z) points at the look-at target
    fwd = -A[:, :3, 2]
    to_c = np.array([p1.center_xy[0], p1.center_xy[1], p1.look_z]) - A[:, :3, 3]
    cosang = np.einsum("ij,ij->i", fwd, to_c) / np.linalg.norm(to_c, axis=1)
    assert cosang.min() > np.cos(np.radians(5))
