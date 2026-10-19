#!/usr/bin/env python
"""Leaf-geometry metrics of a reconstruction against the canonical (rest-pose) ground truth.

    python scripts/eval/eval_geometry.py --recon point_cloud.ply --tree-dir trees/tree_01 \
        [--kind auto|gaussians|mesh|points] [--opacity-min 0.5] [--tau 0.005] \
        [--dataparser-transforms dataparser_transforms.json] [--out metrics.json]

Reconstruction inputs:
  gaussians  3DGS PLY (x,y,z, opacity logit, scale_0..2 log, rot_0..3 wxyz): centres, filtered by opacity;
             normal = axis of the smallest scale
  mesh       triangle PLY: area-weighted surface samples with face normals
  points     any PLY with x,y,z (normals via PCA of 16 nearest neighbours)
If the method normalised the scene (e.g. nerfstudio without --orientation-method none ...),
pass its dataparser_transforms.json to map back into metric world coordinates.

Metrics. Every reconstruction point is attributed to its nearest GT surface - leaf, wood, pot/soil
or FLOOR (z = 0) - within --crop of the tree; points farther than --gate (3 cm) from every surface
are floaters (floater_frac, reported separately). Gaussians are evaluated by default as samples
spread over each Gaussian's footprint (--gaussian-mode samples), not just their centres.
Means are capped at 5 cm (robust to floaters); medians are reported too.
  chamfer_mm         mean of accuracy (recon->GT leaves, incl. floaters nearest a leaf) and completeness
                     (GT leaves->leaf-attributed recon), capped means in mm
  accuracy_mm / completeness_mm, precision / recall / fscore at tau
  leaf_area_error    recovered leaf area (GT leaf area whose samples lie within tau of the recon)
                     vs total GT leaf area; per leaf and overall (relative)
  leaf_angle_w1_deg  1-D Wasserstein distance between the leaf inclination-angle distributions
                     (angle between normal and vertical, folded to [0,90] deg) - the leaf angle
                     distribution (LAD) used in plant phenotyping; also KL of 9-bin histograms
  centroid_err_mm    per-leaf centroid of attributed recon points vs GT centroid (leaves with
                     >= min_pts points), median / p90; leaf_coverage = fraction of leaves recovered
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from synthtree.ply import ply_xyz, read_ply  # noqa: E402
from synthtree.tree import load_tree  # noqa: E402


def sample_mesh(V, F, n, rng, face_attr=None):
    a = V[F[:, 1]] - V[F[:, 0]]
    b = V[F[:, 2]] - V[F[:, 0]]
    cr = np.cross(a, b)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    fi = rng.choice(len(F), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    P = V[F[fi, 0]] * (1 - s)[:, None] + V[F[fi, 1]] * (s * (1 - r2))[:, None] + V[F[fi, 2]] * (s * r2)[:, None]
    N = cr[fi] / np.maximum(np.linalg.norm(cr[fi], axis=1, keepdims=True), 1e-30)
    return P, N, fi, area.sum()


def load_recon(path, kind, opacity_min, rng, n_mesh=400_000, gaussian_mode="samples", n_samples=400_000, bbox=None):
    """bbox: optional (lo, hi) world box; Gaussians centred outside it are dropped BEFORE footprint
    sampling (otherwise huge background splats - floor, walls - absorb the sample budget)."""
    ply = read_ply(path)
    v = ply["vertex"]
    names = v.dtype.names
    if kind == "auto":
        kind = "gaussians" if "opacity" in names and "scale_0" in names else ("mesh" if "face" in ply else "points")
    P = ply_xyz(v)
    N = None
    if kind == "gaussians":
        op = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
        keep = op >= opacity_min
        if bbox is not None:
            keep &= np.all((P > bbox[0]) & (P < bbox[1]), axis=1)
        sc = np.stack([v[f"scale_{i}"] for i in range(3)], 1)[keep]
        q = np.stack([v[f"rot_{i}"] for i in range(4)], 1)[keep].astype(np.float64)  # w, x, y, z
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        w, x, y, z = q.T
        R = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                      np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                      np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)
        N = R[np.arange(len(R)), :, np.argmin(sc, 1)]
        P = P[keep]
        if gaussian_mode == "samples":
            P, N = sample_gaussians(P, R, np.exp(sc.astype(np.float64)), op[keep], n_samples, rng)
    elif kind == "mesh":
        F = ply["face"]["v"].astype(np.int64)
        P, N, _, _ = sample_mesh(P, F, n_mesh, rng)
    return P, N, kind


def sample_gaussians(mu, R, S, opacity, n, rng, clip=2.0, max_sigma=0.015):
    """Surface samples spread over each Gaussian's footprint (flattened onto its two largest axes).

    Centres alone under-cover the surface: a Gaussian is a blob, and ~1 cm centre spacing
    leaves most leaf surface > 5 mm from any centre even when the blobs fully cover it.
    Samples per Gaussian ~ opacity x footprint area; offsets ~ N(0, s) clipped to +-2 s.
    Normals = the Gaussian's smallest axis. In-plane sigmas are capped at max_sigma (1.5 cm,
    i.e. footprints <= +-3 cm) so a few oversized blurry blobs cannot spray points over the canopy.
    """
    S = np.minimum(S, max_sigma)
    order = np.argsort(S, axis=1)  # smallest axis first
    idx = np.arange(len(S))
    a1, a2, a0 = order[:, 2], order[:, 1], order[:, 0]
    w = opacity * S[idx, a1] * S[idx, a2]
    k = rng.multinomial(n, w / w.sum())
    g = np.repeat(idx, k)
    z = np.clip(rng.standard_normal((len(g), 2)), -clip, clip)
    off = (R[g, :, a1[g]] * (S[g, a1[g]] * z[:, 0])[:, None] + R[g, :, a2[g]] * (S[g, a2[g]] * z[:, 1])[:, None])
    return mu[g] + off, R[g, :, a0[g]]


def pca_normals(P, k=16):
    from scipy.spatial import cKDTree
    _, idx = cKDTree(P).query(P, k=min(k, len(P)))
    Q = P[idx] - P[idx].mean(1, keepdims=True)
    C = np.einsum("nki,nkj->nij", Q, Q)
    _, vec = np.linalg.eigh(C)
    return vec[:, :, 0]


def apply_dataparser_inverse(P, N, path):
    """Undo nerfstudio's applied transform + scale: x_ns = s * (T @ [x;1])."""
    d = json.load(open(path))
    T = np.asarray(d["transform"], float)
    s = float(d["scale"])
    R, t = T[:3, :3], T[:3, 3]
    Pw = np.einsum("ji,nj->ni", R, P / s - t)
    Nw = None if N is None else np.einsum("ji,nj->ni", R, N)
    return Pw, Nw


def inclination(N):
    return np.degrees(np.arccos(np.clip(np.abs(N[:, 2]), 0, 1)))


def wasserstein1(a, b):
    a, b = np.sort(a), np.sort(b)
    q = np.linspace(0, 1, 512)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def kl_hist(a, b, bins=9):
    ha, _ = np.histogram(a, bins, (0, 90))
    hb, _ = np.histogram(b, bins, (0, 90))
    pa, pb = (ha + 1) / (ha + 1).sum(), (hb + 1) / (hb + 1).sum()
    return float(np.sum(pa * np.log(pa / pb)))


def evaluate(recon_P, recon_N, tree, tau=0.005, n_gt=300_000, crop=0.3, min_pts=10, seed=0, leaf_T=None,
             gate=0.03, cap=0.05):
    """leaf_T: optional (L,3,4) to evaluate against a posed (non-canonical) leaf state.
    gate: points farther than this from every GT surface (leaf/wood/pot/floor) are floaters.
    cap: distances are capped here in the means (medians are reported uncapped)."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    LV, LF = tree["leaf_verts"].copy(), tree["leaf_faces"]
    if leaf_T is not None:
        vl = tree["leaf_vert_leaf"]
        LV = np.einsum("nij,nj->ni", leaf_T[vl, :, :3], LV) + leaf_T[vl, :, 3]
    face_leaf = tree["leaf_face_leaf"]
    Pg, Ng, fi, total_area = sample_mesh(LV, LF, n_gt, rng)
    g_leaf = face_leaf[fi]
    Pw, _, _, _ = sample_mesh(tree["wood_verts"], tree["wood_faces"], n_gt // 4, rng)
    Pp, _, _, _ = sample_mesh(np.concatenate([tree["pot_verts"], tree["soil_verts"]]),
                              np.concatenate([tree["pot_faces"], tree["soil_faces"] + len(tree["pot_verts"])]), n_gt // 8, rng)
    lo, hi = np.concatenate([Pg, Pw, Pp]).min(0) - crop, np.concatenate([Pg, Pw, Pp]).max(0) + crop
    # the floor must be part of the attribution, otherwise floor Gaussians under the canopy
    # are matched to the leaves above them
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0], 0.003), np.arange(lo[1], hi[1], 0.003))
    Pf = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], 1)
    allP = np.concatenate([Pg, Pw, Pp, Pf])
    owner = np.concatenate([np.zeros(len(Pg), int), np.ones(len(Pw), int), np.full(len(Pp), 2), np.full(len(Pf), 3)])
    inbox = np.all((recon_P > lo) & (recon_P < hi), axis=1)
    R = recon_P[inbox]
    RN = None if recon_N is None else recon_N[inbox]
    if len(R) == 0:
        return {"error": "no reconstruction points near the tree"}
    t_all = cKDTree(allP)
    d_all, j = t_all.query(R)
    nearest_leaf = owner[j] == 0
    floater = d_all >= gate
    is_leaf = nearest_leaf & ~floater  # attributed to leaves
    RL = R[is_leaf]
    t_leaf = cKDTree(Pg)
    acc = d_all[nearest_leaf]  # floaters closest to leaves count against accuracy / precision
    comp, _ = cKDTree(RL).query(Pg) if len(RL) else (np.full(len(Pg), np.inf), None)
    prec = float(np.mean(acc < tau)) if len(acc) else 0.0
    rec = float(np.mean(comp < tau))
    out = {
        "n_recon_points_near_tree": int(len(R)), "n_leaf_attributed": int(len(RL)), "tau_m": tau, "gate_m": gate,
        "floater_frac": float(floater.mean()),
        "floor_frac": float(np.mean(owner[j] == 3)),
        "accuracy_mm_median": float(np.median(acc) * 1000) if len(acc) else None,
        "accuracy_mm": float(np.mean(np.minimum(acc, cap)) * 1000) if len(acc) else None,
        "completeness_mm_median": float(np.median(np.minimum(comp, cap)) * 1000),
        "completeness_mm": float(np.mean(np.minimum(comp, cap)) * 1000),
        "precision": prec, "recall": rec, "fscore": 2 * prec * rec / max(prec + rec, 1e-12),
    }
    out["chamfer_mm"] = 0.5 * (out["accuracy_mm"] + out["completeness_mm"]) if out["accuracy_mm"] is not None else None
    # leaf area
    L = len(tree["leaf_id"])
    covered = comp < tau
    per_leaf_area = tree["leaf_area"] if leaf_T is None else tree["leaf_area"]
    frac = np.bincount(g_leaf, covered, L) / np.maximum(np.bincount(g_leaf, minlength=L), 1)
    rec_area = frac * per_leaf_area
    out["leaf_area_gt_m2"] = float(per_leaf_area.sum())
    out["leaf_area_recovered_m2"] = float(rec_area.sum())
    out["leaf_area_rel_error"] = float(abs(rec_area.sum() - per_leaf_area.sum()) / per_leaf_area.sum())
    out["leaf_area_per_leaf_mae_rel"] = float(np.mean(np.abs(rec_area - per_leaf_area) / per_leaf_area))
    # leaf angle distribution
    if RN is None:
        RN = pca_normals(R) if len(R) >= 16 else None
    if RN is not None and is_leaf.sum() > 16:
        gt_ang = inclination(Ng)
        rc_ang = inclination(RN[is_leaf])
        out["leaf_angle_w1_deg"] = wasserstein1(rc_ang, gt_ang)
        out["leaf_angle_kl"] = kl_hist(rc_ang, gt_ang)
        out["leaf_angle_mean_gt_deg"], out["leaf_angle_mean_recon_deg"] = float(gt_ang.mean()), float(rc_ang.mean())
    # per-leaf centroid error (attribution = leaf of nearest GT leaf sample)
    if len(RL):
        _, jl = t_leaf.query(RL)
        lid = g_leaf[jl]
        cnt = np.bincount(lid, minlength=L)
        cen = np.stack([np.bincount(lid, RL[:, k], L) for k in range(3)], 1) / np.maximum(cnt, 1)[:, None]
        gtc = tree["leaf_centroid"] if leaf_T is None else np.einsum("nij,nj->ni", leaf_T[:, :, :3], tree["leaf_centroid"]) + leaf_T[:, :, 3]
        ok = cnt >= min_pts
        e = np.linalg.norm(cen[ok] - gtc[ok], axis=1) * 1000
        out["leaf_coverage"] = float(ok.mean())
        out["centroid_err_mm_median"] = float(np.median(e)) if e.size else None
        out["centroid_err_mm_p90"] = float(np.percentile(e, 90)) if e.size else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon", required=True)
    ap.add_argument("--tree-dir", required=True, help="trees/tree_XX (contains canonical/)")
    ap.add_argument("--kind", default="auto", choices=["auto", "gaussians", "mesh", "points"])
    ap.add_argument("--opacity-min", type=float, default=0.1)
    ap.add_argument("--gaussian-mode", default="samples", choices=["samples", "centers"],
                    help="samples: points spread over each Gaussian footprint (default); centers: Gaussian means only")
    ap.add_argument("--samples", type=int, default=600_000)
    ap.add_argument("--gate", type=float, default=0.03, help="floater distance gate (m)")
    ap.add_argument("--crop", type=float, default=0.3, help="margin around the tree bbox (m)")
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--dataparser-transforms")
    ap.add_argument("--out")
    a = ap.parse_args()
    tree = load_tree(os.path.join(a.tree_dir, "canonical", "canonical_tree.npz"))
    rng = np.random.default_rng(0)
    pts = np.concatenate([tree["leaf_verts"], tree["wood_verts"], tree["pot_verts"]])
    bbox = (pts.min(0) - a.crop, pts.max(0) + a.crop)
    if a.dataparser_transforms:  # bbox must be in the recon's frame; crop after mapping instead
        bbox = None
    P, N, kind = load_recon(a.recon, a.kind, a.opacity_min, rng, gaussian_mode=a.gaussian_mode, n_samples=a.samples, bbox=bbox)
    if a.dataparser_transforms:
        P, N = apply_dataparser_inverse(P, N, a.dataparser_transforms)
    m = evaluate(P, N, tree, tau=a.tau, gate=a.gate, crop=a.crop)
    m["recon_kind"] = kind + (f" ({a.gaussian_mode})" if kind == "gaussians" else "")
    if kind == "gaussians":  # diagnostics: how many splats are near the tree and how big they are
        v = read_ply(a.recon)["vertex"]
        c = ply_xyz(v)
        op = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
        smax = np.exp(np.max(np.stack([v[f"scale_{i}"] for i in range(3)], 1), axis=1))
        near = np.all((c > bbox[0]) & (c < bbox[1]), axis=1) if bbox is not None else np.ones(len(c), bool)
        m["gaussians_total"], m["gaussians_near_tree"] = int(len(c)), int(near.sum())
        m["gaussians_near_tree_opacity_ge_min"] = int((near & (op >= a.opacity_min)).sum())
        m["near_tree_max_scale_mm_p50_p90"] = [float(np.percentile(smax[near], 50) * 1000), float(np.percentile(smax[near], 90) * 1000)] if near.any() else None
    print(json.dumps(m, indent=1))
    if a.out:
        json.dump(m, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
