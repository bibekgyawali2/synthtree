#!/usr/bin/env python
"""How complex is the GT leaf motion? Rank, global-rigid share and spectrum of per-leaf trajectories.

    python scripts/paper2/motion_rank.py --master configs/master.yaml --level high --out results/paper2/motion_rank.json

Straight from the closed-form wind model (no rendering). Per tree, over the full orbit at `--fps`:
  r90_centroid / r90_normal  number of singular values holding 90 % of the trajectory energy of the
                             (3L x F) matrix of leaf centroid / normal displacements. Shared-basis motion
                             models (a few SE(3) bases, smooth control points) need this to be small.
  eta_global                 share of centroid motion explained by ONE rigid transform per frame
                             (Kabsch). A deformation that collapsed to a global shift has eta ~ 1.
  frac_above_<f>Hz           share of normal-trajectory energy above f, e.g. the 3.75 Hz Nyquist limit
                             of the 7.5 fps paper-1 frames and the 15 Hz limit of 30 fps.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from synthtree.config import load_master, tree_spec, write_json  # noqa: E402
from synthtree.tree import generate_tree  # noqa: E402
from synthtree.wind import WindModel  # noqa: E402


def r_energy(X, q=0.90):
    s = np.linalg.svd(X - X.mean(1, keepdims=True), compute_uv=False)
    e = np.cumsum(s ** 2) / max((s ** 2).sum(), 1e-30)
    return int(np.searchsorted(e, q)) + 1


def kabsch(P, Q):
    """R, t minimising |R P + t - Q| for (n,3) point sets."""
    mp, mq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, mq - R @ mp


def analyse(tree, wind_cfg, U, seed, fps, duration, cut_hz):
    wm = WindModel(tree, wind_cfg, U, seed)
    L = len(tree["leaf_attach"])
    c0 = np.zeros((L, 3))
    np.add.at(c0, tree["leaf_vert_leaf"], tree["leaf_verts"])
    c0 /= np.bincount(tree["leaf_vert_leaf"], minlength=L)[:, None]
    n0 = tree["leaf_normal"]
    ts = np.arange(int(round(duration * fps))) / fps
    C = np.empty((len(ts), L, 3))
    N = np.empty((len(ts), L, 3))
    for i, t in enumerate(ts):
        st = wm.evaluate(float(t))
        C[i] = np.einsum("lij,lj->li", st.leaf_A, c0) + st.leaf_t
        N[i] = np.einsum("lij,lj->li", st.leaf_A, n0)

    Cm = C.mean(0)
    eta = []
    for i in range(len(ts)):
        R, tr = kabsch(Cm, C[i])
        res = C[i] - (Cm @ R.T + tr)
        eta.append(1.0 - (res ** 2).sum() / max(((C[i] - Cm) ** 2).sum(), 1e-30))

    spec = np.abs(np.fft.rfft(N - N.mean(0), axis=0)) ** 2
    f = np.fft.rfftfreq(len(ts), 1.0 / fps)
    out = {"n_leaves": L, "frames": len(ts), "fps": fps,
           "r90_centroid": r_energy(C.transpose(1, 2, 0).reshape(-1, len(ts))),
           "r90_normal": r_energy(N.transpose(1, 2, 0).reshape(-1, len(ts))),
           "eta_global": float(np.mean(eta)),
           "centroid_rms_mm": float(1e3 * np.sqrt(((C - Cm) ** 2).sum(-1).mean())),
           "normal_rms_deg": float(np.degrees(np.sqrt((np.arccos(np.clip((N * N.mean(0) / np.linalg.norm(N.mean(0), axis=-1, keepdims=True)).sum(-1), -1, 1)) ** 2).mean())))}
    for c in cut_hz:
        out[f"frac_above_{c:g}Hz"] = float(spec[f > c].sum() / max(spec.sum(), 1e-30))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="configs/master.yaml")
    ap.add_argument("--level", default="high")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--trees", type=int, nargs="*", default=None)
    ap.add_argument("--cut-hz", type=float, nargs="*", default=[3.75, 15.0])
    ap.add_argument("--out", default="results/paper2/motion_rank.json")
    a = ap.parse_args()
    m = load_master(a.master)
    U = float(m["wind"]["levels"][a.level])
    res = {}
    for t in m["trees"]:
        if a.trees is not None and int(t["id"]) not in a.trees:
            continue
        s = tree_spec(m, t)
        tree = generate_tree(s["preset"], s["seed"])
        r = analyse(tree, m["wind"], U, s["seed"], a.fps, m["capture"]["duration_s"], a.cut_hz)
        r["preset"] = s["preset_name"]
        res[f"tree_{s['id']:02d}"] = r
        print(f"tree_{s['id']:02d} {s['preset_name']:<20s} " + " ".join(
            f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items() if k != "preset"), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    write_json(a.out, {"level": a.level, "U": U, "trees": res})


if __name__ == "__main__":
    main()
