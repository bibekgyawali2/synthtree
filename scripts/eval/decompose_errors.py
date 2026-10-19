#!/usr/bin/env python
"""Three-way error decomposition + plots.

    python scripts/eval/decompose_errors.py --results results.csv --metric chamfer_mm --out decomposition/

results.csv: one row per run with columns
    method, tree, wind_U, sensor (clean|noisy), <metric columns...>
(e.g. collect the JSON outputs of eval_geometry.py / eval_images.py into a table).

Protocol (per method and tree; then mean +- std over trees):
    E_method(occlusion)  = E(calm,  clean)
    E_sensor             = E(calm,  noisy) - E(calm, clean)
    E_wind(U)            = E(U,     noisy) - E(calm, noisy)
If windy+clean runs are present, the interaction term is reported too:
    E_interaction(U)     = [E(U,noisy) - E(U,clean)] - E_sensor
A large interaction means sensor and wind errors are not additive (e.g. noise hurts
more when leaves also move) - report it rather than hide it in E_wind.
For "higher is better" metrics (PSNR, SSIM, F-score) pass --higher-better: errors are then
computed as drops (E = -metric differences), so all components read as "how much worse".
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


def load(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["wind_U"] = float(r["wind_U"])
    return rows


def decompose(rows, metric, higher_better=False):
    sign = -1.0 if higher_better else 1.0
    raw = defaultdict(list)  # (method, tree, U, sensor) -> values over seeds
    for r in rows:
        if r.get(metric) in (None, ""):
            continue
        raw[(r["method"], r["tree"], r["wind_U"], r["sensor"])].append(sign * float(r[metric]))
    E = defaultdict(dict)  # (method, tree) -> {(U, sensor): seed-mean value}
    seed_sd = defaultdict(list)  # method -> per-configuration std over seeds (training noise)
    for (m, t, U, s), v in raw.items():
        E[(m, t)][(U, s)] = float(np.mean(v))
        if len(v) > 1:
            seed_sd[m].append(float(np.std(v, ddof=1)))
    per = defaultdict(lambda: defaultdict(list))  # method -> component -> [(U, value)]
    for (method, tree), e in E.items():
        if (0.0, "clean") not in e or (0.0, "noisy") not in e:
            continue
        cc, cn = e[(0.0, "clean")], e[(0.0, "noisy")]
        per[method]["method"].append((0.0, cc))
        per[method]["sensor"].append((0.0, cn - cc))
        for (U, s), v in e.items():
            if U > 0 and s == "noisy":
                per[method]["wind"].append((U, v - cn))
                if (U, "clean") in e:
                    per[method]["interaction"].append((U, (v - e[(U, "clean")]) - (cn - cc)))
                    per[method]["wind_clean_only"].append((U, e[(U, "clean")] - cc))
    summary = {}
    for method, comps in per.items():
        summary[method] = {"_seed_noise_std": float(np.mean(seed_sd[method])) if seed_sd[method] else None}
        for comp, vals in comps.items():
            byU = defaultdict(list)
            for U, v in vals:
                byU[U].append(v)
            summary[method][comp] = {f"{U:g}": {"mean": float(np.mean(v)), "std": float(np.std(v)), "n_trees": len(v)}
                                     for U, v in sorted(byU.items())}
    return summary


def plot(summary, metric, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = sorted(summary)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    # left: grouped bars at the strongest wind (grouped, not stacked: sensor error can be negative)
    ax = axs[0]
    labels = ["occlusion/method", "sensor", "wind (strongest U)", "interaction"]
    colors = ["#4c72b0", "#dd8452", "#55a868", "#8172b3"]
    width = 0.8 / len(labels)
    for k, m in enumerate(methods):
        s = summary[m]
        Umax = max(s.get("wind", {"0": 0}), key=float)
        vals = [s["method"]["0"], s["sensor"]["0"], s.get("wind", {}).get(Umax), s.get("interaction", {}).get(Umax)]
        for i, (v, lab, col) in enumerate(zip(vals, labels, colors)):
            if v is None:
                continue
            ax.bar(k + (i - 1.5) * width, v["mean"], width, yerr=v["std"] if v["n_trees"] > 1 else None,
                   color=col, capsize=2, label=lab if k == 0 else None)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(methods)), methods, rotation=20)
    ax.set_ylabel(metric)
    ax.set_title("error components (mean +- std over trees)")
    ax.legend(fontsize=8)
    # right: wind component vs U per method
    ax = axs[1]
    for m in methods:
        w = summary[m].get("wind", {})
        if not isinstance(w, dict):
            continue
        if not w:
            continue
        U = [0.0] + [float(u) for u in w]
        mu = [0.0] + [w[u]["mean"] for u in w]
        sd = [0.0] + [w[u]["std"] for u in w]
        ax.errorbar(U, mu, yerr=sd, marker="o", capsize=3, label=m)
    ax.set_xlabel("wind speed U at canopy [m/s]")
    ax.set_ylabel(f"wind error in {metric}")
    ax.set_title("wind error vs U (mean +- std over trees)")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend()
    fig.tight_layout()
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, f"decomposition_{metric}.png")
    fig.savefig(p, dpi=130)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--metric", required=True)
    ap.add_argument("--higher-better", action="store_true")
    ap.add_argument("--out", default="decomposition")
    a = ap.parse_args()
    s = decompose(load(a.results), a.metric, a.higher_better)
    os.makedirs(a.out, exist_ok=True)
    json.dump(s, open(os.path.join(a.out, f"decomposition_{a.metric}.json"), "w"), indent=1)
    print(json.dumps(s, indent=1))
    print("plot:", plot(s, a.metric, a.out))


if __name__ == "__main__":
    main()
