#!/usr/bin/env python
"""Render 10 frames per beauty engine, measure time + disk, extrapolate to the full config.

    python scripts/benchmark.py --master configs/master.yaml [--tree 2] [--frames 10] [--engines eevee,cycles]

Uses the densest tree by default (worst case), full resolution, the master's render
settings, a windy level. Writes data/benchmark/report.json and prints a table.
The first frame of each Blender launch includes shader / GPU-kernel compilation and is
excluded from the steady-state average (the Cycles Metal kernels compile once per
Blender version, ~2 min, then they are cached).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import numpy as np  # noqa: E402

import noise_sim  # noqa: E402
import postprocess  # noqa: E402
from make_tree import export_tree  # noqa: E402
from run_batch import find_blender, run_blender  # noqa: E402
from synthtree.config import expand_master, load_master, write_json  # noqa: E402
from synthtree.tree import generate_tree  # noqa: E402


def dir_bytes(d):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", required=True)
    ap.add_argument("--tree", type=int, default=None, help="default: tree with most leaves")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--engines", default="eevee,cycles")
    ap.add_argument("--blender")
    a = ap.parse_args()
    m = load_master(a.master)
    all_seqs = expand_master(m)
    root = os.path.join(os.path.dirname(m["output_root"]), "benchmark")
    if a.tree is None:
        counts = {}
        for t in m["trees"]:
            spec = next(s["tree"] for s in all_seqs if s["tree"]["id"] == t["id"])
            counts[t["id"]] = generate_tree(spec["preset"], spec["seed"])["leaf_id"].size
        a.tree = max(counts, key=counts.get)
    base = next(s for s in all_seqs if s["tree"]["id"] == a.tree and s["wind_U"] > 0 and s["kind"] == "orbit")
    blender = find_blender(a.blender)
    tree_dir = os.path.join(root, f"tree_{a.tree:02d}")
    if not os.path.exists(os.path.join(tree_dir, "canonical", "canonical_tree.npz")):
        export_tree(generate_tree(base["tree"]["preset"], base["tree"]["seed"]), os.path.join(tree_dir, "canonical"))
    report = {"tree": a.tree, "frames": a.frames, "engines": {}}
    for eng in a.engines.split(","):
        c = copy.deepcopy(base)
        c["tree_dir"] = tree_dir
        c["out_dir"] = os.path.join(root, eng)
        c["render"]["beauty_engine"] = eng
        shutil.rmtree(c["out_dir"], ignore_errors=True)
        write_json(os.path.join(c["out_dir"], "config.json"), c)
        rc, dt, fr = run_blender(blender, os.path.join(c["out_dir"], "config.json"), os.path.join(c["out_dir"], "blender.log"),
                                 ["--max-frames", str(a.frames)])
        if rc:
            sys.exit(f"Blender failed; see {c['out_dir']}/blender.log")
        import time
        t0 = time.time()
        postprocess.process(c["out_dir"], keep_raw=False)
        t_post = (time.time() - t0) / a.frames
        t0 = time.time()
        noise_sim.process(c["out_dir"])
        t_noise = (time.time() - t0) / a.frames
        sizes = {k: dir_bytes(os.path.join(c["out_dir"], *k.split("/"))) / a.frames
                 for k in ("clean/rgb", "clean/depth", "clean/depth_exr", "clean/normal", "clean/leaf_id",
                           "clean/semantic", "clean/flow", "noisy/rgb", "noisy/depth")
                 if os.path.isdir(os.path.join(c["out_dir"], *k.split("/")))}
        steady = {k: float(np.mean(v[1:])) if len(v) > 1 else None for k, v in fr.items()}
        report["engines"][eng] = {
            "sec_per_frame": {"data_pass": steady["data"], "beauty": steady["beauty"], "postprocess": t_post, "noise": t_noise,
                              "first_frame_overhead_s": float(sum(v[0] for v in fr.values() if v))},
            "bytes_per_frame": sizes, "bytes_per_frame_total": float(sum(sizes.values())),
            "gt_bytes_per_frame": dir_bytes(os.path.join(c["out_dir"], "gt")) / a.frames,
        }
    # extrapolate to the full master config
    n_seq = len(all_seqs)
    frames_total = 0
    for s in all_seqs:
        cap = s["capture"]
        frames_total += (2 * cap.get("ring", {}).get("views_per_ring", 24)) if s["kind"] == "ring" else \
            int(cap["duration_s"] * cap["fps"] / cap["frame_stride"])
    report["full_config"] = {"sequences": n_seq, "frames": frames_total}
    print(f"\nBenchmark: tree {a.tree}, {a.frames} frames, resolution x{base['render']['resolution_scale']}")
    print(f"Full config: {n_seq} sequences, {frames_total} frames\n")
    print(f"{'engine':8s} {'data s/f':>9s} {'beauty s/f':>11s} {'post+noise':>11s} {'MB/frame':>9s} {'total h':>8s} {'total GB':>9s}")
    for eng, r in report["engines"].items():
        s = r["sec_per_frame"]
        per = s["data_pass"] + s["beauty"] + s["postprocess"] + s["noise"]
        mb = (r["bytes_per_frame_total"] + r["gt_bytes_per_frame"]) / 1e6
        r["extrapolated"] = {"hours": per * frames_total / 3600 + n_seq * 5 / 3600, "GB": mb * frames_total / 1e3}
        print(f"{eng:8s} {s['data_pass']:9.2f} {s['beauty']:11.2f} {s['postprocess'] + s['noise']:11.2f} {mb:9.2f} "
              f"{r['extrapolated']['hours']:8.1f} {r['extrapolated']['GB']:9.1f}")
    write_json(os.path.join(root, "report.json"), report)
    print("\nper-product MB/frame:", {k: round(v / 1e6, 3) for k, v in next(iter(report["engines"].values()))["bytes_per_frame"].items()})
    print("report:", os.path.join(root, "report.json"))


if __name__ == "__main__":
    main()
