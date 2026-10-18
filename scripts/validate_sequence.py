#!/usr/bin/env python
"""Validation checks on *exported* products (what a user of the dataset sees).

    python scripts/validate_sequence.py --seq data/output/tree_01/wind_high
    python scripts/validate_sequence.py --master configs/master.yaml --pairs   # calm/windy pairing

Checks (all written to <seq>/validation.json):
  self_reprojection   world-position AOV projects onto its own pixel (from qc.json), < 0.01 px
  cross_reprojection  depth of frame i back-projected and re-projected into frame j lands on the
                      same surface point (sub-pixel solve on j's back-projected depth); static
                      geometry only for windy sequences; median < 0.05 px (EXR) / 0.3 px (mm PNG)
  ground_plane        z-depth back-projected on the ground (world z = 0) stays on the plane,
                      also in the image corners: proves z-depth (ray distance would be off by cm)
  depth_png_vs_exr    mm PNG / 1000 == float EXR within 0.5 mm
  leaf_ids            every leaf pixel id is in the leaf table; leaf_id>0 <=> semantic==leaf
  rgb_alignment       RGB gradients peak exactly on leaf-id boundaries (shift search +-2 px)
  noise_stats         noisy-clean depth std per distance bin within [0.6, 1.6] x profile sigma
Pairing (per tree): identical camera poses across wind levels, shared canonical tree,
leaf transforms exactly identity at U = 0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from synthtree.config import write_json  # noqa: E402
from synthtree.imageio import read_exr, read_png  # noqa: E402
from synthtree.intrinsics import Intrinsics, backproject, project  # noqa: E402


def _intr(seq):
    ij = json.load(open(os.path.join(seq, "intrinsics.json")))
    return Intrinsics(**{k: ij[k] for k in ("width", "height", "fx", "fy", "cx", "cy")})


def _depth(seq, i, prefer_exr=True):
    e = os.path.join(seq, "clean", "depth_exr", f"{i:05d}.exr")
    if prefer_exr and os.path.exists(e):
        return read_exr(e)["Z"].astype(np.float64), True
    return read_png(os.path.join(seq, "clean", "depth", f"{i:05d}.png")).astype(np.float64) * 0.001, False


def _interior(mask):
    m = mask.copy()
    m[1:-1, 1:-1] = mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    m[0], m[-1], m[:, 0], m[:, -1] = False, False, False, False
    return m


def cross_reprojection(seq, i, j, intr, poses, static_only, n_samples=4000, rng=None):
    rng = rng or np.random.default_rng(0)
    zi, exr_i = _depth(seq, i)
    zj, exr_j = _depth(seq, j)
    si = read_png(os.path.join(seq, "clean", "semantic", f"{i:05d}.png"))
    sj = read_png(os.path.join(seq, "clean", "semantic", f"{j:05d}.png"))
    allowed = np.isin(si, [3, 4] if static_only else [1, 2, 3, 4])
    Xi = backproject(intr, poses[i], zi)
    Xj = backproject(intr, poses[j], zj)
    cand = np.argwhere(_interior((zi > 0) & allowed) & _interior(si == si))
    if len(cand) == 0:
        return None
    cand = cand[rng.choice(len(cand), min(n_samples, len(cand)), replace=False)]
    H, W = zi.shape
    okj = _interior(zj > 0)
    errs = []
    for (v, u) in cand:
        X = Xi[v, u]
        uq, vq, zq = project(intr, poses[j], X)
        if not (np.isfinite(uq) and zq > 0):
            continue
        u0, v0 = int(round(uq)), int(round(vq))
        if not (1 <= u0 < W - 1 and 1 <= v0 < H - 1) or not okj[v0, u0] or sj[v0, u0] != si[v, u]:
            continue
        if abs(zj[v0, u0] - zq) > 0.02 * zq:  # occluded in j
            continue
        P0 = Xj[v0, u0]
        J = np.stack([Xj[v0, u0 + 1] - P0, Xj[v0 + 1, u0] - P0], 1)
        d, *_ = np.linalg.lstsq(J, X - P0, rcond=None)
        errs.append(np.hypot(u0 + d[0] - uq, v0 + d[1] - vq))
    if not errs:
        return None
    e = np.asarray(errs)
    return {"pair": [i, j], "n": int(len(e)), "median_px": float(np.median(e)), "p90_px": float(np.percentile(e, 90)),
            "float_depth": bool(exr_i and exr_j)}


def ground_plane(seq, i, intr, poses):
    z, is_exr = _depth(seq, i)
    sem = read_png(os.path.join(seq, "clean", "semantic", f"{i:05d}.png"))
    g = (sem == 4) & (z > 0)
    if g.sum() < 50:
        return None
    X = backproject(intr, poses[i], z)
    H, W = z.shape
    corner = np.zeros_like(g)
    c = max(4, W // 8)
    corner[-c:, :c] = corner[-c:, -c:] = True  # ground lives in the lower corners
    res = np.abs(X[..., 2])
    out = {"frame": i, "float_depth": is_exr, "median_m": float(np.median(res[g])), "max_m": float(res[g].max())}
    if (g & corner).sum() > 10:
        out["corner_median_m"] = float(np.median(res[g & corner]))
        r = np.linalg.norm(X - poses[i][:3, 3], axis=-1)  # what ray-distance would have implied
        Xr = backproject(intr, poses[i], np.where(g, z * z / np.maximum(r, 1e-9), 0))
        out["corner_median_if_ray_distance_m"] = float(np.median(np.abs(Xr[..., 2])[g & corner]))
    return out


def rgb_alignment(seq, i):
    rgb = read_png(os.path.join(seq, "clean", "rgb", f"{i:05d}.png"))[..., :3].astype(np.float64)
    lid = read_png(os.path.join(seq, "clean", "leaf_id", f"{i:05d}.png")).astype(np.int64)
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    gx = np.zeros_like(lum)
    gx[:, 1:] = np.abs(lum[:, 1:] - lum[:, :-1])
    gy = np.zeros_like(lum)
    gy[1:] = np.abs(lum[1:] - lum[:-1])
    ex = np.zeros(lid.shape, bool)
    ex[:, 1:] = (lid[:, 1:] > 0) != (lid[:, :-1] > 0)
    ey = np.zeros(lid.shape, bool)
    ey[1:] = (lid[1:] > 0) != (lid[:-1] > 0)
    if ex.sum() < 20:
        return None
    score = {}
    for s in range(-2, 3):
        score[s] = float(np.mean(np.roll(gx, -s, 1)[ex]) + np.mean(np.roll(gy, -s, 0)[ey]))
    if max(score.values()) < 1e-6:  # no visible leaf edges with contrast
        return None
    best = max(score, key=score.get)
    return {"frame": i, "best_shift_px": best, "scores": score}


def flow_and_leaf_tracks(seq, i, intr, poses, tracks):
    """Flow (from rendered per-vertex positions) vs prediction from depth + poses (+ leaf transforms).

    Static pixels: X_{i+1} = X_i. Leaf pixels: X_{i+1} = T_{i+1} T_i^{-1} X_i using the exported
    per-leaf 3x4 transforms -> verifies that gt/leaf_tracks.npz describes the rendered leaves.
    """
    from synthtree.imageio import decode_flow_kitti
    fp = os.path.join(seq, "clean", "flow", f"{i:05d}.png")
    if not os.path.exists(fp) or i + 1 >= len(poses):
        return None
    flow, valid = decode_flow_kitti(read_png(fp))
    z, _ = _depth(seq, i)
    sem = read_png(os.path.join(seq, "clean", "semantic", f"{i:05d}.png"))
    lid = read_png(os.path.join(seq, "clean", "leaf_id", f"{i:05d}.png")).astype(np.int64)
    X = backproject(intr, poses[i], z)
    T = tracks["transform_3x4"]
    out = {"frame": i}
    for name, m in (("static", _interior(np.isin(sem, [3, 4]) & valid)), ("leaves", _interior((lid > 0) & valid) & _interior(lid == lid))):
        if m.sum() < 20:
            continue
        P = X[m]
        if name == "leaves":
            k = lid[m] - 1
            Ri, ti = T[i, k, :, :3].astype(np.float64), T[i, k, :, 3].astype(np.float64)
            Rn, tn = T[i + 1, k, :, :3].astype(np.float64), T[i + 1, k, :, 3].astype(np.float64)
            P0 = np.einsum("nji,nj->ni", Ri, P - ti)
            P = np.einsum("nij,nj->ni", Rn, P0) + tn
        u, v, _ = project(intr, poses[i + 1], P)
        vv, uu = np.nonzero(m)
        e = np.hypot(u - (uu + flow[m][:, 0]), v - (vv + flow[m][:, 1]))
        out[name] = {"n": int(m.sum()), "median_px": float(np.median(e)), "p90_px": float(np.percentile(e, 90))}
    return out


def noise_stats(seq):
    meta_p = os.path.join(seq, "noisy", "noise_meta.json")
    if not os.path.exists(meta_p):
        return None
    prof = json.load(open(meta_p))["profile"]
    from synthtree.noise import depth_sigma, discontinuities
    names = sorted(os.listdir(os.path.join(seq, "noisy", "depth")))[:10]
    zc_all, d_all = [], []
    for f in names:
        zc = read_png(os.path.join(seq, "clean", "depth", f)).astype(np.float64) * 0.001
        zn = read_png(os.path.join(seq, "noisy", "depth", f)).astype(np.float64) * 0.001
        sem = read_png(os.path.join(seq, "clean", "semantic", f))
        smooth = _interior(_interior((zc > 0) & (sem == 4))) & ~discontinuities(zc, 0.02)
        m = smooth & (zn > 0)
        zc_all.append(zc[m])
        d_all.append(zn[m] - zc[m])
    zc, d = np.concatenate(zc_all), np.concatenate(d_all)
    if zc.size < 1000:
        return None
    bins = np.quantile(zc, [0, 0.33, 0.66, 1.0])
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (zc >= lo) & (zc < hi)
        if m.sum() < 200:
            continue
        zm = float(np.median(zc[m]))
        mad = 1.4826 * np.median(np.abs(d[m] - np.median(d[m])))
        expected = float(np.sqrt(depth_sigma(zm, prof) ** 2 + (prof["depth"]["units_m"] ** 2) / 12))
        rows.append({"z_m": zm, "measured_sigma_m": float(mad), "profile_sigma_m": expected, "ratio": float(mad / expected)})
    return {"bins": rows}


def validate(seq):
    intr = _intr(seq)
    poses = np.load(os.path.join(seq, "poses", "c2w_opencv.npy"))
    meta = json.load(open(os.path.join(seq, "metadata.json")))
    qc = json.load(open(os.path.join(seq, "qc.json")))["summary"]
    n = len(poses)
    calm = meta["wind_U_mps"] == 0
    checks = {}
    checks["self_reprojection"] = {"ok": qc["reproj_max_px"] < 0.01, "max_px": qc["reproj_max_px"],
                                   "blender_depth_pass": qc.get("blender_depth_pass_matches")}
    exr_frames = sorted(int(f[:5]) for f in os.listdir(os.path.join(seq, "clean", "depth_exr"))) if os.path.isdir(os.path.join(seq, "clean", "depth_exr")) else []
    pairs = list(zip(exr_frames[:-1], exr_frames[1:]))[:3] or [(i, i + 1) for i in range(min(n - 1, 3))]
    cr = [r for r in (cross_reprojection(seq, i, j, intr, poses, static_only=not calm) for i, j in pairs) if r]
    if cr:
        lim = 0.05 if all(r["float_depth"] for r in cr) else 0.3
        checks["cross_reprojection"] = {"ok": all(r["median_px"] < lim for r in cr), "limit_px": lim, "pairs": cr,
                                        "geometry": "all" if calm else "static (pot, ground)"}
    gp = [g for g in (ground_plane(seq, i, intr, poses) for i in (exr_frames[:2] or [0])) if g]
    if gp:
        checks["ground_plane"] = {"ok": all(g["median_m"] < (0.0005 if g["float_depth"] else 0.002) for g in gp), "frames": gp}
    if exr_frames:
        i = exr_frames[0]
        ze, _ = _depth(seq, i, True)
        zp, _ = _depth(seq, i, False)
        diff = np.abs(ze - zp)
        checks["depth_png_vs_exr"] = {"ok": bool(diff.max() <= 0.0005 + 1e-6 and np.array_equal(ze > 0, zp > 0)), "max_diff_m": float(diff.max())}
    tracks = np.load(os.path.join(seq, "gt", "leaf_tracks.npz"))
    L = len(tracks["leaf_id"])
    fl = [f for f in (flow_and_leaf_tracks(seq, i, intr, poses, tracks) for i in range(0, max(n - 1, 0), max(1, n // 3))) if f]
    if fl:
        meds = [f[k]["median_px"] for f in fl for k in ("static", "leaves") if k in f]
        checks["flow_and_leaf_tracks"] = {"ok": max(meds) < 0.1, "limit_median_px": 0.1, "frames": fl}
    checks["leaf_ids"] = {"ok": qc["leaf_ids_unknown_px"] == 0 and qc["leafid_semantic_mismatch_px"] == 0,
                          "unknown_px": qc["leaf_ids_unknown_px"], "mismatch_px": qc["leafid_semantic_mismatch_px"], "n_leaves": L}
    rnd = meta["render"]
    temporal = (rnd.get("motion_blur", {}).get("enabled") or rnd.get("rolling_shutter", {}).get("enabled")) and meta["kind"] == "orbit"
    al = [] if temporal else [a for a in (rgb_alignment(seq, i) for i in range(0, n, max(1, n // 3))) if a]
    if temporal:  # RGB is intentionally blurred / row-shifted w.r.t. the frame-centre GT
        checks["rgb_alignment"] = {"ok": True, "skipped": "motion blur / rolling shutter enabled"}
    elif al:
        checks["rgb_alignment"] = {"ok": all(a["best_shift_px"] == 0 for a in al), "frames": al}
    ns = noise_stats(seq)
    if ns and ns["bins"]:
        checks["noise_stats"] = {"ok": all(0.6 < b["ratio"] < 1.6 for b in ns["bins"]), **ns}
    res = {"ok": all(c["ok"] for c in checks.values()), "checks": checks}
    write_json(os.path.join(seq, "validation.json"), res)
    return res


def validate_pairs(master, seqs, log=print):
    """Calm/windy pairing per tree."""
    by_tree = {}
    for s in seqs:
        if s["kind"] == "orbit" and os.path.exists(os.path.join(s["out_dir"], "poses", "c2w_blender.npy")):
            by_tree.setdefault(s["tree"]["id"], []).append(s)
    results = {}
    for tid, ss in by_tree.items():
        P = [np.load(os.path.join(s["out_dir"], "poses", "c2w_blender.npy")) for s in ss]
        n = min(len(p) for p in P)
        same_pose = all(np.array_equal(P[0][:n], p[:n]) for p in P[1:])
        same_tree = len({s["tree_dir"] for s in ss}) == 1
        ident = True
        for s in ss:
            if s["wind_U"] == 0:
                T = np.load(os.path.join(s["out_dir"], "gt", "leaf_tracks.npz"))["transform_3x4"]
                eye = np.zeros_like(T)
                eye[..., 0, 0] = eye[..., 1, 1] = eye[..., 2, 2] = 1
                ident &= bool(np.array_equal(T, eye))
        ok = same_pose and same_tree and ident
        results[tid] = {"ok": ok, "identical_poses": same_pose, "shared_canonical_tree": same_tree,
                        "calm_leaf_transforms_identity": ident, "sequences": [s["sequence_id"] for s in ss]}
        log("pairing", tree=tid, **{k: v for k, v in results[tid].items() if k != "sequences"})
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq")
    ap.add_argument("--master")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pairs", action="store_true")
    a = ap.parse_args()
    if a.seq:
        r = validate(a.seq)
        print(json.dumps(r, indent=1))
        sys.exit(0 if r["ok"] else 1)
    if a.master and a.pairs:
        from synthtree.config import expand_master, load_master
        m = load_master(a.master, smoke=a.smoke)
        r = validate_pairs(m, expand_master(m), lambda msg, **kw: print(msg, json.dumps(kw)))
        sys.exit(0 if all(v["ok"] for v in r.values()) else 1)


if __name__ == "__main__":
    main()
