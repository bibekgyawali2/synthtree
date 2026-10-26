"""Kaggle script kernel: per-condition reconstructions for the qualitative figure (2x T4).

The per-cell rigid-motion model on tree 1 of the paper-1 dataset (bibekgyawali2/synthtree-v1: 150 frames per
orbit, i.e. every 4th frame of 30 fps) for every condition of the paper-1 design: calm, 1.5, 3, 5 m/s, each with
the clean and the noisy sensor. Saves figure data (run_dyn.py --fig) for scripts/paper/make_condition_figure.py.
Code comes from the `saurav` branch. Pushed with `kaggle kernels push -p kaggle/kernels/stage1_conditions`.
"""
import os
import subprocess
import sys
import threading
import time

os.environ.setdefault("DYN_SMOKE", "0")  # "1" = 2-run smoke test
REPO_URL = "https://github.com/bibekgyawali2/synthtree.git"
BRANCH = "saurav"
WORK = "/kaggle/working"
CODE = f"{WORK}/synthtree"
RUNS = "calm:clean low:clean medium:clean high:clean calm:noisy low:noisy medium:noisy high:noisy"
# 8 runs, ~40-50 min on 2x T4
STAGES = ([f"--dataset synthtree-v1 --trees 1 --runs high:clean --modes cluster --iters 600 --warmup 200 --eval-frames 2 --fig"]
          if os.environ.get("DYN_SMOKE") == "1" else
          [f"--dataset synthtree-v1 --trees 1 --runs {RUNS} --modes cluster --iters 10000 --warmup 2000 --fig"])


T0 = time.time()


def sh(cmd):
    print(f"[{(time.time() - T0) / 60:5.1f} min] $ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True).returncode


def clone():
    for attempt in range(5):
        if sh(f"git clone -q -b {BRANCH} {REPO_URL} {CODE}") == 0:
            return
        print(f"clone failed (attempt {attempt + 1}/5); the repository must be PUBLIC", flush=True)
        time.sleep(20 * (attempt + 1))
    sys.exit("could not clone the code repository - is github.com/bibekgyawali2/synthtree public?")


WARMUP = r"""
import torch, gsplat
d = "cuda"
means = torch.zeros(1, 3, device=d)
quats = torch.tensor([[1.0, 0, 0, 0]], device=d)
scales = torch.full((1, 3), 0.01, device=d)
viewmat = torch.eye(4, device=d)[None]
viewmat[0, 2, 3] = 2.0
K = torch.tensor([[[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]]], device=d)
img, _, _ = gsplat.rasterization(means, quats, scales, torch.ones(1, device=d), torch.ones(1, 3, device=d), viewmat, K, 100, 100)
img.sum().backward() if img.requires_grad else None
print("gsplat CUDA kernels ready", tuple(img.shape))
"""


def warmup_gsplat():
    """Compile gsplat's CUDA extension ONCE before the workers start. Two processes JIT-compiling
    into the same torch_extensions folder delete each other's build files, failing every run."""
    with open(f"{WORK}/warmup_gsplat.py", "w") as f:
        f.write(WARMUP)
    env = dict(os.environ, MAX_JOBS="4", CUDA_VISIBLE_DEVICES="0")
    for attempt in range(2):
        print(f"[{(time.time() - T0) / 60:5.1f} min] $ compiling gsplat (attempt {attempt + 1}, ~5-15 min)", flush=True)
        if subprocess.run(f"python {WORK}/warmup_gsplat.py", shell=True, env=env).returncode == 0:
            return
    sys.exit("gsplat CUDA extension failed to compile - aborting before wasting GPU time")


def run_workers(args, n_gpus):
    """One run_benchmark worker per GPU, output prefixed with the GPU index."""
    procs = []
    for k in range(n_gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k))
        cmd = f"python kaggle/run_dyn.py {args} --shard {k} --num-shards {n_gpus}"
        print(f"[gpu{k}] $ {cmd}", flush=True)
        p = subprocess.Popen(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        def pump(p=p, k=k):
            for line in p.stdout:
                print(f"[gpu{k}] {line}", end="", flush=True)
        threading.Thread(target=pump, daemon=True).start()
        procs.append(p)
        time.sleep(5)
    return [p.wait() for p in procs]


clone()
sh("pip install --progress-bar off gsplat==1.5.3 ninja jaxtyping rich")  # same gsplat as the local tests
os.chdir(CODE)
sh("git log --oneline -1")
n = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 1)
warmup_gsplat()
print(f"[{(time.time() - T0) / 60:5.1f} min] GPUs: {n}", flush=True)
for stage in STAGES:
    print(f"\n[{(time.time() - T0) / 60:5.1f} min] ===== stage: {stage}", flush=True)
    print("exit codes:", run_workers(stage, max(1, n)), flush=True)
    sh(f"python scripts/eval/collect_results.py --dir {WORK}/results --out {WORK}/results.csv")
sh(f"rm -rf {WORK}/ns {WORK}/runs {WORK}/export {CODE}")  # keep only results in the output
