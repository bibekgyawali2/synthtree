#!/usr/bin/env python
"""Run the decomposition benchmark matrix on Kaggle: convert -> train -> eval -> export -> geometry.

    python run_benchmark.py --trees 1 7                       # default matrix per tree (6 runs)
    python run_benchmark.py --trees 1 --runs calm:clean high:clean --iters 7000
    python run_benchmark.py --trees 1 7 --dry-run             # print the commands only
    python run_benchmark.py --trees 1 3 5 7 --seeds 0 1 --methods splatfacto depth-nerfacto

Default matrix per tree (what decompose_errors.py needs):
    calm:clean   occlusion/method error
    calm:noisy   + sensor error
    low:noisy  medium:noisy  high:noisy   + wind error per U
    high:clean   interaction term (sensor x wind)
Methods: splatfacto (3DGS; Gaussians exported and evaluated in hybrid mode) and depth-nerfacto
(NeRF with depth supervision; a 1M-point cloud is exported and evaluated with PCA normals).
Results -> /kaggle/working/results/<method>__tree_XX__<level>__<variant>__s<seed>__{geom,ns}.json
(read by scripts/eval/collect_results.py). Resumable: finished runs are skipped, so after a
session reset just re-run the same command. Each run's model/export is deleted after evaluation
to stay inside /kaggle/working's ~20 GB unless --keep is given.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

DEFAULT_RUNS = ["calm:clean", "calm:noisy", "low:noisy", "medium:noisy", "high:noisy", "high:clean"]

# per method: training iterations, export command, exported file
METHODS = {
    "splatfacto": {"iters": 7000, "export": "gaussian-splat", "file": "splat.ply", "extra": ""},
    "depth-nerfacto": {"iters": 15000, "export": "pointcloud --num-points 1000000 --remove-outliers True "
                       "--normal-method open3d", "file": "point_cloud.ply", "extra": ""},
}


def sh(cmd, log, dry):
    print("  $", cmd if len(cmd) < 220 else cmd[:220] + " ...", flush=True)
    if dry:
        return 0
    with open(log, "a") as f:
        f.write(f"\n$ {cmd}\n")
        f.flush()
        r = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode


def tail(path, n=25):
    try:
        return "".join(open(path).readlines()[-n:])
    except OSError:
        return ""


def one_run(a, M, method, iters, tid, level, variant, seed, name, root, res_dir, geom_json):
    t0 = time.time()
    print(f"[run ] {name}", flush=True)
    seq = f"{root}/tree_{tid:02d}/wind_{level}"
    data = os.path.join(a.work, "ns", name)
    runs = os.path.join(a.work, "runs", name)
    export = os.path.join(a.work, "export", name)
    log = os.path.join(res_dir, name + ".log")
    init = variant if a.init_ply == "auto" else a.init_ply
    steps = [
        f"python {HERE}/convert.py nerfstudio --seq {seq} --variant {variant} --out {data} --init-ply {init}",
        f"ns-train {method} --data {data} --output-dir {runs} --max-num-iterations {iters} --machine.seed {seed} "
        f"--steps-per-save 1000 --viewer.quit-on-train-completion True {M['extra']} "
        f"nerfstudio-data --orientation-method none --center-method none --auto-scale-poses False --eval-mode filename",
    ]
    ok = all(sh(cmd, log, a.dry_run) == 0 for cmd in steps)
    cfgs = sorted(glob.glob(os.path.join(runs, "*", method, "*", "nerfstudio_models", "..", "config.yml")))
    if ok and not a.dry_run and not cfgs:
        ok = False
        print("  training produced no checkpoint")
    if ok:
        cfg = cfgs[-1] if cfgs else "<config.yml>"
        post = [
            f"python {HERE}/ns_safe.py eval --load-config {cfg} --output-path {res_dir}/{name}__ns.json",
            f"python {HERE}/ns_safe.py export {M['export']} --load-config {cfg} --output-dir {export}",
            f"python {REPO}/scripts/eval/eval_geometry.py --recon {export}/{M['file']} "
            f"--tree-dir {root}/trees/tree_{tid:02d} --out {geom_json}",
        ]
        ok = all(sh(cmd, log, a.dry_run) == 0 for cmd in post)
    if not ok:
        print(f"  FAILED - last lines of {log}:\n{tail(log)}")
        if "Error building extension" in tail(log, 400):  # environment problem: every later run would fail too
            sys.exit("CUDA extension build failed - stopping this worker instead of failing every remaining run")
        for d in (runs, export, data):  # allow a clean retry on the next invocation
            shutil.rmtree(d, ignore_errors=True)
        return None
    if a.dry_run:
        return None
    g = json.load(open(geom_json))
    print(f"  done in {(time.time() - t0) / 60:.1f} min: F={g['fscore']:.3f} chamfer={g['chamfer_mm']:.1f} mm "
          f"floaters={g['floater_frac']:.2f}", flush=True)
    if a.keep:  # converted images are re-creatable from the dataset; keep only the model and export
        shutil.rmtree(data, ignore_errors=True)
    else:
        for d in (runs, export, data):
            shutil.rmtree(d, ignore_errors=True)
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trees", type=int, nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="level:variant pairs")
    ap.add_argument("--dataset", default="synthtree-v1")
    ap.add_argument("--work", default="/kaggle/working")
    ap.add_argument("--methods", nargs="+", default=["splatfacto"], choices=list(METHODS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--iters", type=int, default=None, help="override the per-method default iterations")
    ap.add_argument("--init-ply", default="auto", help="auto = same depth variant as the images; or clean/noisy/none")
    ap.add_argument("--keep", action="store_true", help="keep trained models (checkpoint + config) and exports")
    ap.add_argument("--shard", type=int, default=0, help="this worker's index (use one worker per GPU)")
    ap.add_argument("--num-shards", type=int, default=1, help="number of parallel workers")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    # gsplat JIT-compiles its CUDA kernels on first use; ninja's default parallelism (-j 10)
    # runs Kaggle out of RAM ("Killed", exit 137). Cap it; later runs reuse the built kernels.
    os.environ.setdefault("MAX_JOBS", "2")
    import synthtree_dataset
    root = synthtree_dataset.find_dataset_root(a.dataset) if not a.dry_run else f"/kaggle/input/{a.dataset}"
    res_dir = os.path.join(a.work, "results")
    os.makedirs(res_dir, exist_ok=True)
    summary = []
    jobs = [(method, tid, run, seed) for method in a.methods for tid in a.trees for run in a.runs for seed in a.seeds]
    jobs = jobs[a.shard::a.num_shards]  # round-robin split across workers / GPUs
    print(f"worker {a.shard}/{a.num_shards}: {len(jobs)} runs", flush=True)
    for method, tid, run, seed in jobs:
        M = METHODS[method]
        iters = a.iters or M["iters"]
        level, variant = run.split(":")
        name = f"{method}__tree_{tid:02d}__{level}__{variant}__s{seed}"
        geom_json = os.path.join(res_dir, name + "__geom.json")
        legacy = os.path.join(res_dir, f"{method}__tree_{tid:02d}__{level}__{variant}__geom.json")
        done = geom_json if os.path.exists(geom_json) else (legacy if seed == 0 and os.path.exists(legacy) else None)
        if done:
            print(f"[skip] {name} (done)")
            summary.append((name, json.load(open(done))))
            continue
        r = one_run(a, M, method, iters, tid, level, variant, seed, name, root, res_dir, geom_json)
        if r is not None:
            summary.append((name, r))
    if summary:
        print(f"\n{'run':52s} {'F@5mm':>6s} {'chamfer':>8s} {'area_err':>8s} {'cover':>6s} {'floaters':>8s}")
        for name, g in summary:
            f = lambda k, w, p: f"{g[k]:{w}.{p}f}" if g.get(k) is not None else f"{'-':>{w}s}"  # noqa: E731
            print(f"{name:52s} {f('fscore', 6, 3)} {f('chamfer_mm', 8, 1)} {f('leaf_area_rel_error', 8, 2)} "
                  f"{f('leaf_coverage', 6, 2)} {f('floater_frac', 8, 2)}")


if __name__ == "__main__":
    main()
