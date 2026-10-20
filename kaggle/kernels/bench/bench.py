"""Kaggle script kernel: full SynthTree benchmark on a 2x T4 machine (one worker per GPU).

Pushed with `kaggle kernels push -p kaggle/kernels/bench`. Stage 1: splatfacto on trees 1, 3, 5, 7
(broad, small, compound, needle) x 6 conditions x 2 seeds. Stage 2: depth-nerfacto on trees 1, 7.
"""
import os
import subprocess
import sys
import threading
import time

REPO_URL = "https://github.com/bibekgyawali2/synthtree.git"
WORK = "/kaggle/working"
CODE = f"{WORK}/synthtree"
STAGES = [
    "--methods splatfacto --trees 1 3 5 7 --seeds 0 1",
    "--methods depth-nerfacto --trees 1 7 --seeds 0",
]


def sh(cmd):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True).returncode


def clone():
    for attempt in range(5):
        if sh(f"git clone -q {REPO_URL} {CODE}") == 0:
            return
        print(f"clone failed (attempt {attempt + 1}/5); the repository must be PUBLIC", flush=True)
        time.sleep(20 * (attempt + 1))
    sys.exit("could not clone the code repository - is github.com/bibekgyawali2/synthtree public?")


def run_workers(args, n_gpus):
    """One run_benchmark worker per GPU, output prefixed with the GPU index."""
    procs = []
    for k in range(n_gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k))
        cmd = f"python kaggle/run_benchmark.py {args} --shard {k} --num-shards {n_gpus}"
        print(f"[gpu{k}] $ {cmd}", flush=True)
        p = subprocess.Popen(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        def pump(p=p, k=k):
            for line in p.stdout:
                print(f"[gpu{k}] {line}", end="", flush=True)
        threading.Thread(target=pump, daemon=True).start()
        procs.append(p)
        time.sleep(60)  # let worker 0 start compiling the CUDA kernels first (torch holds a build lock)
    return [p.wait() for p in procs]


clone()
sh("pip install -q nerfstudio 2>&1 | tail -n 2")
os.chdir(CODE)
sh("git log --oneline -1")
n = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 1)
print(f"GPUs: {n}", flush=True)
for stage in STAGES:
    print(f"\n===== stage: {stage}", flush=True)
    print("exit codes:", run_workers(stage, max(1, n)), flush=True)
    sh(f"python scripts/eval/collect_results.py --dir {WORK}/results --out {WORK}/results.csv")
for metric, hb in [("fscore", "--higher-better"), ("chamfer_mm", ""), ("centroid_err_mm_median", ""),
                   ("leaf_area_rel_error", ""), ("leaf_angle_w1_deg", "")]:
    sh(f"python scripts/eval/decompose_errors.py --results {WORK}/results.csv --metric {metric} {hb} "
       f"--out {WORK}/decomposition > /dev/null")
sh(f"rm -rf {WORK}/ns {WORK}/runs {WORK}/export")
