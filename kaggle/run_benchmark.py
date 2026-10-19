#!/usr/bin/env python
"""Run the decomposition benchmark matrix on Kaggle: convert -> train -> eval -> export -> geometry.

    python run_benchmark.py --trees 1 7                       # default matrix per tree (6 runs)
    python run_benchmark.py --trees 1 --runs calm:clean high:clean --iters 7000
    python run_benchmark.py --trees 1 7 --dry-run             # print the commands only

Default matrix per tree (what decompose_errors.py needs):
    calm:clean   occlusion/method error
    calm:noisy   + sensor error
    low:noisy  medium:noisy  high:noisy   + wind error per U
    high:clean   interaction term (sensor x wind)
Results -> /kaggle/working/results/splatfacto__tree_XX__<level>__<variant>__{geom,ns}.json
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trees", type=int, nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="level:variant pairs")
    ap.add_argument("--dataset", default="synthtree-v1")
    ap.add_argument("--work", default="/kaggle/working")
    ap.add_argument("--iters", type=int, default=7000)
    ap.add_argument("--init-ply", default="auto", help="auto = same depth variant as the images; or clean/noisy/none")
    ap.add_argument("--keep", action="store_true", help="keep trained models and exports")
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
    for tid in a.trees:
        for run in a.runs:
            level, variant = run.split(":")
            name = f"splatfacto__tree_{tid:02d}__{level}__{variant}"
            geom_json = os.path.join(res_dir, name + "__geom.json")
            if os.path.exists(geom_json):
                print(f"[skip] {name} (done)")
                summary.append((name, json.load(open(geom_json))))
                continue
            t0 = time.time()
            print(f"[run ] {name}", flush=True)
            seq = f"{root}/tree_{tid:02d}/wind_{level}"
            data = os.path.join(a.work, "ns", name)
            runs = os.path.join(a.work, "runs", name)
            log = os.path.join(res_dir, name + ".log")
            init = variant if a.init_ply == "auto" else a.init_ply
            steps = [
                f"python {HERE}/convert.py nerfstudio --seq {seq} --variant {variant} --out {data} --init-ply {init}",
                f"ns-train splatfacto --data {data} --output-dir {runs} --max-num-iterations {a.iters} "
                f"--steps-per-save 1000 --viewer.quit-on-train-completion True "
                f"nerfstudio-data --orientation-method none --center-method none --auto-scale-poses False --eval-mode filename",
            ]
            ok = True
            for cmd in steps:
                if sh(cmd, log, a.dry_run) != 0:
                    ok = False
                    break
            cfgs = sorted(glob.glob(os.path.join(runs, "*", "splatfacto", "*", "nerfstudio_models", "..", "config.yml")))
            if ok and not a.dry_run and not cfgs:
                ok = False
                print("  training produced no checkpoint")
            if ok:
                cfg = cfgs[-1] if cfgs else "<config.yml>"
                export = os.path.join(a.work, "export", name)
                for cmd in [
                    f"python {HERE}/ns_safe.py eval --load-config {cfg} --output-path {res_dir}/{name}__ns.json",
                    f"python {HERE}/ns_safe.py export gaussian-splat --load-config {cfg} --output-dir {export}",
                    f"python {REPO}/scripts/eval/eval_geometry.py --recon {export}/splat.ply "
                    f"--tree-dir {root}/trees/tree_{tid:02d} --out {geom_json}",
                ]:
                    if sh(cmd, log, a.dry_run) != 0:
                        ok = False
                        break
            if not ok:
                print(f"  FAILED - last lines of {log}:\n{tail(log)}")
                for d in (runs, os.path.join(a.work, "export", name), data):  # allow a clean retry
                    shutil.rmtree(d, ignore_errors=True)
                continue
            if not a.dry_run:
                g = json.load(open(geom_json))
                summary.append((name, g))
                print(f"  done in {(time.time() - t0) / 60:.1f} min: F={g['fscore']:.3f} chamfer={g['chamfer_mm']:.1f} mm "
                      f"area_err={g['leaf_area_rel_error']:.2f} floaters={g['floater_frac']:.2f}", flush=True)
                if not a.keep:
                    for d in (runs, export, data):
                        shutil.rmtree(d, ignore_errors=True)
    if summary:
        print(f"\n{'run':45s} {'F@5mm':>6s} {'chamfer':>8s} {'area_err':>8s} {'cover':>6s} {'floaters':>8s}")
        for name, g in summary:
            print(f"{name:45s} {g['fscore']:6.3f} {g['chamfer_mm']:8.1f} {g['leaf_area_rel_error']:8.2f} "
                  f"{g['leaf_coverage']:6.2f} {g['floater_frac']:8.2f}")


if __name__ == "__main__":
    main()
