"""Procedural sapling generator (pure NumPy).

Why not Sapling Tree Gen / a GN tree: we need the branch hierarchy, per-node
frames, leaf attachment points and parent-branch ids as first-class data for exact
wind and ground truth. Sapling does not expose them (and since Blender 4.2 it is a
separately installed extension); a GN generator loses that bookkeeping on
realisation unless every attribute is carefully re-stored. A deterministic NumPy
generator also runs in pytest without Blender.

Model: a skeleton of *nodes* (positions, radii, parent node) grouped into
*branches* (level 0 = trunk). Child branches start exactly at a node of their
parent, so forward kinematics in `wind.py` keeps everything connected. Leaves
attach at nodes, offset to the bark surface.
"""
from __future__ import annotations

import copy
import json

import numpy as np

from . import __version__
from .leaves import face_normals_areas, leaf_template
from .meshing import branch_tube, ground_mesh, pot_meshes
from .rng import substream
from .transforms import normalize, perpendicular_basis, rodrigues

Z = np.array([0.0, 0.0, 1.0])

DEFAULT_PRESET = {
    "height_m": [1.0, 1.2],
    "pot": {"radius_top": 0.12, "radius_bottom": 0.09, "height": 0.18, "soil_depth": 0.025},
    "trunk": {"radius_base": 0.010, "radius_tip": 0.0015, "segments": 14, "wander_deg": 5.0,
              "lean_deg": 3.0, "upturn": 0.3, "sides": 10, "taper_exp": 1.0},
    "levels": [],
    "leaves": {"type": "broad", "shape": "ovate", "length_m": [0.05, 0.07], "width_ratio": 0.55,
               "levels": [1, 2], "start_frac": 0.25, "spacing_m": 0.02, "per_site": 1.0,
               "phyllotaxis_deg": 137.5, "petiole_frac": 0.2,
               "petiole_angle_deg": [40, 70], "up_bias": 0.5, "roll_deg": 20.0, "curl": 0.06,
               "droop": 0.04, "segments": 6, "max_leaves": 3000, "size_jitter": 0.12},
    "color": {"front_rgb": [0.06, 0.20, 0.03], "back_tint": [1.4, 1.4, 1.25],
              "hsv_jitter": [0.025, 0.10, 0.15], "autumn": False},
    "bark_rgb": [0.16, 0.10, 0.06],
}

LEVEL_DEFAULTS = {"count": 0, "count_per_m": 0.0, "start_frac": 0.3, "end_frac": 0.95,
                  "down_angle_deg": [45, 65], "phyllotaxis_deg": 137.5, "whorl": 1,
                  "length_ratio": 0.4, "length_shape": "spherical", "radius_ratio": 0.6,
                  "radius_tip": 0.001, "seg_len_m": 0.04, "wander_deg": 8.0, "upturn": 0.1,
                  "sides": 6, "taper_exp": 1.0}


def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _length_shape(name, f):
    if name == "conical":
        return 1.0 - 0.85 * f
    if name == "spherical":
        return 0.25 + 0.75 * np.sin(np.pi * np.clip(f, 0, 1))
    if name == "flame":
        return 0.4 + 0.6 * np.sin(np.pi * np.clip(f, 0, 1) ** 0.6)
    return 1.0


def _perturb(d, sigma, rng):
    if sigma <= 0:
        return d
    e1, e2 = perpendicular_basis(d)
    phi = rng.uniform(0, 2 * np.pi)
    axis = np.cos(phi) * e1 + np.sin(phi) * e2
    return normalize(rodrigues(axis, rng.normal(0, sigma)) @ d)


class _Skeleton:
    def __init__(self, soil_z):
        self.p, self.parent, self.branch, self.radius = [], [], [], []
        self.branches = []
        self.soil_z = soil_z

    def grow(self, start, direction, length, r0, r1, n_seg, level, parent_branch, parent_node,
             wander_deg, upturn, sides, taper_exp, rng):
        bid = len(self.branches)
        n_seg = max(2, int(n_seg))
        seg = length / n_seg
        d, pos, ids = normalize(direction), np.asarray(start, float), []
        for k in range(n_seg + 1):
            frac = k / n_seg
            r = r1 + (r0 - r1) * (1.0 - frac) ** taper_exp
            ids.append(len(self.p))
            self.p.append(pos.copy())
            self.parent.append(parent_node if k == 0 else ids[-2])
            self.branch.append(bid)
            self.radius.append(r)
            if k < n_seg:
                d = _perturb(d, np.radians(wander_deg), rng)
                w = upturn / n_seg
                d = normalize((1 - w) * d + w * Z)
                if pos[2] + d[2] * seg < self.soil_z + 0.03 and d[2] < 0:
                    d = normalize(d * [1, 1, 0] + [0, 0, 0.05])
                pos = pos + d * seg
        self.branches.append({"id": bid, "level": level, "parent_branch": parent_branch,
                              "parent_node": parent_node, "nodes": ids, "length": length, "sides": sides})
        return bid

    def tangent(self, nid):
        b = self.branches[self.branch[nid]]
        ids = b["nodes"]
        k = ids.index(nid)
        a, c = ids[max(k - 1, 0)], ids[min(k + 1, len(ids) - 1)]
        return normalize(np.asarray(self.p[c]) - np.asarray(self.p[a]))


def generate_tree(preset: dict, seed: int) -> dict:
    """Generate a canonical (rest-pose) tree. Returns a dict of NumPy arrays + 'meta'."""
    p = deep_merge(DEFAULT_PRESET, preset)
    rng = substream(seed, "tree", "skeleton")
    pot = p["pot"]
    soil_z = pot["height"] - pot["soil_depth"]
    sk = _Skeleton(soil_z)

    tr = p["trunk"]
    H = rng.uniform(*p["height_m"])
    lean, az = np.radians(tr["lean_deg"]) * rng.uniform(0.3, 1.0), rng.uniform(0, 2 * np.pi)
    d0 = np.array([np.sin(lean) * np.cos(az), np.sin(lean) * np.sin(az), np.cos(lean)])
    sk.grow([0, 0, soil_z - 0.01], d0, H + 0.01, tr["radius_base"], tr["radius_tip"], tr["segments"], 0, -1, -1,
            tr["wander_deg"], tr["upturn"], tr["sides"], tr["taper_exp"], rng)

    for li, lv in enumerate(p["levels"], start=1):
        lv = deep_merge(LEVEL_DEFAULTS, lv)
        parents = [b for b in sk.branches if b["level"] == li - 1]
        for pb in parents:
            ids = pb["nodes"]
            nseg = len(ids) - 1
            if li == 1:
                n = int(lv["count"])
            else:
                n = int(round(lv["count_per_m"] * pb["length"] * rng.uniform(0.8, 1.2)))
            if n <= 0:
                continue
            whorl = max(1, int(lv["whorl"]))
            n_groups = int(np.ceil(n / whorl))
            fr = np.linspace(lv["start_frac"], lv["end_frac"], n_groups)
            if n_groups > 1:
                fr = fr + rng.uniform(-0.3, 0.3, n_groups) * (fr[1] - fr[0])
            fr = np.clip(fr, 0.0, 1.0)
            az0 = rng.uniform(0, 2 * np.pi)
            made = 0
            for g, f in enumerate(fr):
                k = int(np.clip(round(f * nseg), 1, nseg - 1))
                nid = ids[k]
                T = sk.tangent(nid)
                e1, e2 = perpendicular_basis(T)
                for w in range(whorl):
                    if made >= n:
                        break
                    if whorl > 1:
                        phi = az0 + 2 * np.pi * w / whorl + g * np.pi / whorl + rng.normal(0, 0.15)
                    else:
                        phi = az0 + g * np.radians(lv["phyllotaxis_deg"]) + rng.normal(0, 0.1)
                    down = np.radians(rng.uniform(*lv["down_angle_deg"]))
                    r = np.cos(phi) * e1 + np.sin(phi) * e2
                    d = normalize(np.cos(down) * T + np.sin(down) * r)
                    frac_par = k / nseg
                    span = max(lv["end_frac"] - lv["start_frac"], 1e-6)
                    shape = _length_shape(lv["length_shape"], (frac_par - lv["start_frac"]) / span)
                    L = pb["length"] * lv["length_ratio"] * shape * np.exp(rng.normal(0, 0.1))
                    L = max(L, 0.03)
                    r_par = sk.radius[nid]
                    r0 = max(min(r_par * lv["radius_ratio"], 0.9 * r_par), lv["radius_tip"] * 1.2)
                    sk.grow(sk.p[nid], d, L, r0, lv["radius_tip"], round(L / lv["seg_len_m"]), li, pb["id"], nid,
                            lv["wander_deg"], lv["upturn"], lv["sides"], lv["taper_exp"], rng)
                    made += 1

    nodes_p = np.asarray(sk.p, float)
    nodes_parent = np.asarray(sk.parent, np.int64)
    nodes_branch = np.asarray(sk.branch, np.int64)
    nodes_radius = np.asarray(sk.radius, float)

    # ---- wood mesh -------------------------------------------------------
    wv, wf, wnode, wfb = [], [], [], []
    off = 0
    for b in sk.branches:
        ids = np.asarray(b["nodes"])
        v, f, vn = branch_tube(nodes_p[ids], nodes_radius[ids], b["sides"])
        wv.append(v)
        wf.append(f + off)
        wnode.append(ids[vn])
        wfb.append(np.full(len(f), b["id"], np.int64))
        off += len(v)
    wood_verts, wood_faces = np.concatenate(wv), np.concatenate(wf)
    wood_vert_node, wood_face_branch = np.concatenate(wnode), np.concatenate(wfb)

    # ---- leaves ------------------------------------------------------------
    leaves = _place_leaves(p, sk, nodes_p, nodes_radius, seed)

    # ---- pot / soil / ground ---------------------------------------------------
    (pot_v, pot_f), (soil_v, soil_f) = pot_meshes(pot["radius_top"], pot["radius_bottom"], pot["height"], pot["soil_depth"])
    ground_v, ground_f = ground_mesh()

    meta = {
        "generator": "synthtree.tree", "version": __version__, "seed": int(seed),
        "preset": p, "height_m": float(H), "n_nodes": int(len(nodes_p)), "n_branches": len(sk.branches),
        "n_leaves": int(len(leaves["leaf_attach"])), "n_wood_tris": int(len(wood_faces)),
        "n_leaf_tris": int(len(leaves["leaf_faces"])), "leaf_type": p["leaves"]["type"],
    }
    out = {
        "nodes_p0": nodes_p, "nodes_parent": nodes_parent, "nodes_branch": nodes_branch, "nodes_radius": nodes_radius,
        "branch_level": np.asarray([b["level"] for b in sk.branches], np.int64),
        "branch_parent": np.asarray([b["parent_branch"] for b in sk.branches], np.int64),
        "branch_parent_node": np.asarray([b["parent_node"] for b in sk.branches], np.int64),
        "wood_verts": wood_verts, "wood_faces": wood_faces, "wood_vert_node": wood_vert_node,
        "wood_face_branch": wood_face_branch,
        "pot_verts": pot_v, "pot_faces": pot_f, "soil_verts": soil_v, "soil_faces": soil_f,
        "ground_verts": ground_v, "ground_faces": ground_f,
        "bark_rgb": np.asarray(p["bark_rgb"], float),
        "meta_json": np.array(json.dumps(meta)),
    }
    out.update(leaves)
    return out


def _leaf_colors(cfg, n, rng):
    base = np.asarray(cfg["front_rgb"], float)
    if cfg.get("autumn"):
        palette = np.array([[0.55, 0.35, 0.02], [0.60, 0.16, 0.01], [0.40, 0.04, 0.02], [0.22, 0.12, 0.04], base])
        w = np.array(cfg.get("autumn_weights", [0.3, 0.3, 0.2, 0.1, 0.1]), float)
        rgb = palette[rng.choice(len(palette), n, p=w / w.sum())]
    else:
        rgb = np.tile(base, (n, 1))
    jh, js, jv = cfg["hsv_jitter"]
    # jitter in a simple HSV-like way on linear RGB
    mx = rgb.max(1, keepdims=True)
    mn = rgb.min(1, keepdims=True)
    grey = rgb.mean(1, keepdims=True)
    sat = 1 + rng.normal(0, js, (n, 1))
    rgb = grey + (rgb - grey) * sat
    rgb = rgb * np.exp(rng.normal(0, jv, (n, 1)))
    shift = rng.normal(0, jh, (n, 1))  # hue-ish: trade red against blue
    rgb = rgb + shift * (mx - mn) * np.array([1.0, 0.0, -1.0])
    front = np.clip(rgb, 0.002, 0.9)
    back = np.clip(front * np.asarray(cfg["back_tint"], float), 0.002, 0.95)
    return front, back


def _place_leaves(p, sk, nodes_p, nodes_radius, seed):
    lc = p["leaves"]
    rng = substream(seed, "tree", "leaves")
    # Leaf sites are spaced along each segment (node k-1 -> k) and attach rigidly to
    # the segment's start node k-1; under the FK in wind.py the whole segment moves
    # rigidly with that node's frame, so the attachment stays exactly on the axis.
    cand = []  # (node id, branch id, azimuth, segment fraction)
    site_count = 0
    for b in sk.branches:
        if b["level"] not in lc["levels"]:
            continue
        ids = b["nodes"]
        for k in range(1, len(ids)):
            if k / (len(ids) - 1) < lc["start_frac"]:
                continue
            seglen = np.linalg.norm(nodes_p[ids[k]] - nodes_p[ids[k - 1]])
            n_sites = max(1, int(round(seglen / lc["spacing_m"])))
            for si in range(n_sites):
                site_count += 1
                per = lc["per_site"]
                n = int(np.floor(per)) + int(rng.random() < per - np.floor(per))
                for j in range(n):
                    phi = site_count * np.radians(lc["phyllotaxis_deg"]) + 2 * np.pi * j / max(n, 1) + rng.normal(0, 0.2)
                    cand.append((ids[k - 1], b["id"], phi, (si + rng.uniform(0.25, 0.75)) / n_sites))
    if len(cand) > lc["max_leaves"]:
        keep = np.sort(rng.choice(len(cand), lc["max_leaves"], replace=False))
        cand = [cand[i] for i in keep]

    tmpl_v, tmpl_f = leaf_template(lc)
    L = len(cand)
    attach = np.zeros((L, 3))
    frames = np.zeros((L, 3, 3))
    lnode = np.zeros(L, np.int64)
    lbranch = np.zeros(L, np.int64)
    V, F, vleaf, fleaf = [], [], [], []
    for i, (nid, bid, phi, s) in enumerate(cand):
        nxt = sk.branches[bid]["nodes"][sk.branches[bid]["nodes"].index(nid) + 1]
        T = normalize(nodes_p[nxt] - nodes_p[nid])
        axis_pt = nodes_p[nid] + s * (nodes_p[nxt] - nodes_p[nid])
        rad = (1 - s) * nodes_radius[nid] + s * nodes_radius[nxt]
        e1, e2 = perpendicular_basis(T)
        r = np.cos(phi) * e1 + np.sin(phi) * e2
        alpha = np.radians(rng.uniform(*lc["petiole_angle_deg"]))
        d = normalize(np.cos(alpha) * T + np.sin(alpha) * r)
        d = normalize(d + lc["up_bias"] * Z)
        zl = Z - np.dot(Z, d) * d
        zl = normalize(zl) if np.linalg.norm(zl) > 1e-3 else r
        zl = rodrigues(d, np.radians(rng.normal(0, lc["roll_deg"]))) @ zl
        yl = np.cross(zl, d)
        Fm = np.stack([d, yl, zl], axis=1)  # columns: local X, Y, Z in world
        rp = r - np.dot(r, T) * T
        a0 = axis_pt + rad * normalize(rp)
        size = np.exp(np.log(rng.uniform(*lc["length_m"])) + rng.normal(0, lc["size_jitter"]))
        vw = a0 + (tmpl_v * size) @ Fm.T
        attach[i], frames[i], lnode[i], lbranch[i] = a0, Fm, nid, bid
        F.append(tmpl_f + len(tmpl_v) * i)
        V.append(vw)
        vleaf.append(np.full(len(tmpl_v), i, np.int64))
        fleaf.append(np.full(len(tmpl_f), i, np.int64))
    if L == 0:
        raise ValueError("preset produced no leaves")
    V, F = np.concatenate(V), np.concatenate(F)
    vleaf, fleaf = np.concatenate(vleaf), np.concatenate(fleaf)

    fn, fa = face_normals_areas(V, F)
    fc = V[F].mean(1)
    area = np.bincount(fleaf, fa, L)
    centroid = np.stack([np.bincount(fleaf, fa * fc[:, k], L) for k in range(3)], 1) / area[:, None]
    if lc["type"] == "needle":
        normal = frames[:, :, 2].copy()
        area = area / 2.0  # hemi-surface area, the usual conifer leaf-area convention
    else:
        normal = normalize(np.stack([np.bincount(fleaf, fa * fn[:, k], L) for k in range(3)], 1))
    front, back = _leaf_colors(p["color"], L, substream(seed, "tree", "colors"))
    return {
        "leaf_verts": V, "leaf_faces": F, "leaf_vert_leaf": vleaf, "leaf_face_leaf": fleaf,
        "leaf_id": np.arange(1, L + 1, dtype=np.int64),  # 0 is reserved for "not a leaf"
        "leaf_attach": attach, "leaf_frame": frames, "leaf_node": lnode, "leaf_branch": lbranch,
        "leaf_area": area, "leaf_centroid": centroid, "leaf_normal": normal,
        "leaf_color_front": front, "leaf_color_back": back,
    }


def load_tree(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta_json"]))
    return d
