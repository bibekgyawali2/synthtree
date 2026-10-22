#!/usr/bin/env python
"""Run the static-vs-deformable 3DGS comparison (dyngs.py) over trees / wind levels, resumable.

    python run_dyn.py --trees 1 3 5 7 --runs calm:noisy low:noisy medium:noisy high:noisy \
        --modes static deform [--iters 15000] [--shard 0 --num-shards 2]
Results: /kaggle/working/results/{static3dgs|dyn3dgs}__tree_XX__<level>__<variant>__s<seed>__geom.json
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
NAMES = {"static": "static3dgs", "deform": "dyn3dgs", "robust": "robust3dgs"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trees", type=int, nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", default=["calm:noisy", "low:noisy", "medium:noisy", "high:noisy"])
    ap.add_argument("--modes", nargs="+", default=["static", "deform"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--warmup", type=int, default=3000)
    ap.add_argument("--eval-frames", type=int, default=4)
    ap.add_argument("--dataset", default="synthtree-v1")
    ap.add_argument("--work", default="/kaggle/working")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()
    import synthtree_dataset
    root = synthtree_dataset.find_dataset_root(a.dataset)
    res = os.path.join(a.work, "results")
    os.makedirs(res, exist_ok=True)
    jobs = [(m, t, r, s) for t in a.trees for r in a.runs for m in a.modes for s in a.seeds][a.shard::a.num_shards]
    print(f"worker {a.shard}/{a.num_shards}: {len(jobs)} runs", flush=True)
    for mode, tid, run, seed in jobs:
        level, variant = run.split(":")
        name = f"{NAMES[mode]}__tree_{tid:02d}__{level}__{variant}__s{seed}"
        out = os.path.join(res, name + "__geom.json")
        if os.path.exists(out):
            print(f"[skip] {name}", flush=True)
            continue
        log = os.path.join(res, name + ".log")
        cmd = (f"python {HERE}/dyngs.py --seq {root}/tree_{tid:02d}/wind_{level} --tree-dir {root}/trees/tree_{tid:02d} "
               f"--variant {variant} --mode {mode} --iters {a.iters} --warmup {a.warmup} --seed {seed} "
               f"--eval-frames {a.eval_frames} --out {out}")
        print(f"[run ] {name}", flush=True)
        t0 = time.time()
        with open(log, "w") as f:
            rc = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT).returncode
        last = open(log).read().strip().splitlines()[-1:] if os.path.exists(log) else [""]
        if rc != 0 or not os.path.exists(out):
            txt = open(log).read()
            print(f"  FAILED ({(time.time() - t0) / 60:.1f} min):\n" + "\n".join(txt.strip().splitlines()[-25:]), flush=True)
            if "Error building extension" in txt:
                sys.exit("CUDA extension build failed - stopping this worker")
            continue
        print(f"  done in {(time.time() - t0) / 60:.1f} min: {last[0][:300]}", flush=True)


if __name__ == "__main__":
    main()
