#!/usr/bin/env python
"""Collect per-run metric JSONs into the results.csv used by decompose_errors.py.

    python scripts/eval/collect_results.py --dir results/ --out results.csv

Expects files named   <method>__tree_<id>__<wind level or U>__<clean|noisy>[__s<seed>][__<anything>].json
e.g. splatfacto__tree_01__high__noisy__s1__geom.json (no seed part = seed 0). Wind levels are mapped to U via --master
(default configs/master.yaml) or parsed as a number. Keys of 'mean' (eval_images) or the
top level (eval_geometry) become columns; several JSONs of the same run are merged.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--master", default=os.path.join(os.path.dirname(__file__), "..", "..", "configs", "master.yaml"))
    a = ap.parse_args()
    from synthtree.config import load_master
    levels = load_master(a.master)["wind"]["levels"]
    runs = {}
    for f in sorted(glob.glob(os.path.join(a.dir, "*.json"))):
        parts = os.path.basename(f)[:-5].split("__")
        if len(parts) < 4:
            continue
        method, tree, wind, sensor = parts[:4]
        seed = next((int(x[1:]) for x in parts[4:] if x[:1] == "s" and x[1:].isdigit()), 0)
        U = float(levels[wind]) if wind in levels else float(wind)
        d = json.load(open(f))
        d = d.get("mean") or d.get("results") or d  # eval_images: mean; ns-eval: results
        key = (method, tree.replace("tree_", ""), U, sensor, seed)
        runs.setdefault(key, {}).update({k: v for k, v in d.items() if isinstance(v, (int, float))})
    cols = sorted({k for v in runs.values() for k in v})
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "tree", "wind_U", "sensor", "seed"] + cols)
        for (m, t, U, s, sd), v in sorted(runs.items()):
            w.writerow([m, t, U, s, sd] + [v.get(c, "") for c in cols])
    print(f"wrote {a.out}: {len(runs)} runs, {len(cols)} metrics")


if __name__ == "__main__":
    main()
