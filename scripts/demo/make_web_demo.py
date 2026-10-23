#!/usr/bin/env python
"""Convert trained splatfacto models into the browser demo in web/.

    python scripts/demo/make_web_demo.py --models data/models_v1/models \
        --results data/models_v1/results.csv results/benchmark_v1/results.csv

Each models/<run>/splat.ply becomes web/splats/<run>.splat (full scene) and <run>.tree.splat (cropped
to the tree and its pot), in the 32-byte-per-Gaussian .splat layout read by web viewers
(position, linear scale, RGBA from the DC colour term and opacity, quantised quaternion). The
view-dependent SH terms are dropped. web/manifest.json lists the runs with their metrics, taken from
the first results file that has the run.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

SH_C0 = 0.28209479177387814
LEVELS = {"calm": 0.0, "low": 1.5, "medium": 3.0, "high": 5.0}
LEAF = {1: "broad", 3: "small", 5: "compound", 7: "needle"}
METRICS = ["fscore", "precision", "recall", "chamfer_mm", "centroid_err_mm_median", "leaf_area_rel_error",
           "floater_frac", "psnr", "gaussians_total"]


def read_ply(path):
    b = open(path, "rb").read()
    i = b.index(b"end_header\n") + len(b"end_header\n")
    props = [ln.split()[-1] for ln in b[:i].decode().splitlines() if ln.startswith("property")]
    a = np.frombuffer(b[i:], dtype="<f4").reshape(-1, len(props))
    return {k: a[:, j] for j, k in enumerate(props)}


def to_splat(d, mask):
    pos = np.stack([d["x"], d["y"], d["z"]], 1)[mask]
    scale = np.exp(np.stack([d["scale_0"], d["scale_1"], d["scale_2"]], 1)[mask])
    rgb = 0.5 + SH_C0 * np.stack([d["f_dc_0"], d["f_dc_1"], d["f_dc_2"]], 1)[mask]
    alpha = 1.0 / (1.0 + np.exp(-d["opacity"][mask]))
    q = np.stack([d["rot_0"], d["rot_1"], d["rot_2"], d["rot_3"]], 1)[mask]  # w, x, y, z
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    order = np.argsort(-scale.prod(1) * alpha)  # large, opaque splats first (progressive loading)
    out = np.zeros(len(order), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    out["p"] = pos[order]
    out["s"] = scale[order]
    out["c"][:, :3] = np.clip(rgb[order] * 255, 0, 255).astype(np.uint8)
    out["c"][:, 3] = np.clip(alpha[order] * 255, 0, 255).astype(np.uint8)
    out["r"] = np.clip(q[order] * 128 + 128, 0, 255).astype(np.uint8)
    return out.tobytes(), len(order)


def load_results(paths):
    rows = {}
    for p in paths:
        for r in csv.DictReader(open(p)):
            key = (r["method"], int(r["tree"]), float(r["wind_U"]), r["sensor"], int(r["seed"]))
            rows.setdefault(key, dict(r, _source=p))  # earlier files take precedence
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True)
    ap.add_argument("--results", nargs="+", default=[])
    ap.add_argument("--out", default="web")
    ap.add_argument("--crop-radius", type=float, default=0.9, help="tree crop: horizontal radius [m]")
    a = ap.parse_args()

    results = load_results([p for p in a.results if os.path.exists(p)])
    os.makedirs(os.path.join(a.out, "splats"), exist_ok=True)
    runs = []
    for name in sorted(os.listdir(a.models)):
        ply = os.path.join(a.models, name, "splat.ply")
        if not os.path.exists(ply):
            continue
        method, tree, level, sensor, seed = name.split("__")
        tree, seed = int(tree.split("_")[1]), int(seed[1:])
        try:
            d = read_ply(ply)
        except ValueError:  # partially downloaded file
            print(f"skip {name} (incomplete)")
            continue
        full, n_full = to_splat(d, np.ones(len(d["x"]), bool))
        r_xy = np.hypot(d["x"], d["y"])
        # tree crop: everything above the floor within the radius, plus the floor just around the pot
        crop = (r_xy < a.crop_radius) & (d["z"] < 2.3) & ((d["z"] > 0.02) | ((r_xy < 0.25) & (d["z"] > -0.03)))
        tree_b, n_tree = to_splat(d, crop)
        open(os.path.join(a.out, "splats", name + ".splat"), "wb").write(full)
        open(os.path.join(a.out, "splats", name + ".tree.splat"), "wb").write(tree_b)
        r = results.get((method, tree, LEVELS[level], sensor, seed), {})
        runs.append({
            "name": name, "method": method, "tree": tree, "leaf": LEAF.get(tree, "?"), "level": level,
            "U": LEVELS[level], "sensor": sensor, "seed": seed, "gaussians": n_full, "gaussians_tree": n_tree,
            "params": n_full * 59,
            "metrics": {k: (float(r[k]) if r.get(k) not in (None, "") else None) for k in METRICS},
            "metrics_source": r.get("_source"),
        })
        print(f"{name}: {n_full} Gaussians ({n_tree} on the tree)")
    json.dump({"runs": runs, "crop_radius_m": a.crop_radius}, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(f"{len(runs)} runs -> {a.out}/manifest.json")


if __name__ == "__main__":
    main()
