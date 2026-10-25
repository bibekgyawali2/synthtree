"""Kaggle script kernel, paper 2 stage 1: is the paper-1 deformation collapse an aliasing artefact?

per-cluster rigid motion (the fix) vs the paper-1 deformation field, trained at 30 fps (every frame) and
7.5 fps (every 4th frame, the paper-1 rate) on the SAME stride-1 renders (synthtree-v2-stage1), trees 1/3/5/7
(the paper-1 trees), calm + high, noisy variant (2x T4).
Code comes from the `saurav` branch. Pushed with `kaggle kernels push -p kaggle/kernels/paper2_stage1`.
Set DYN_SMOKE=1 (top of file) for a 2-run smoke test.
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
COMMON = "--dataset synthtree-v2-stage1 --runs calm:noisy high:noisy --iters 10000 --warmup 2000"  # paper-1 schedule
# Most important first, in case the session runs short (36 runs, ~5-5.5 h on 2x T4; cluster/oracle ~12 min,
# deform ~26 min per run):
#   A   the fix vs the paper-1 model at 30 fps                          16 runs
#   A2  oracle grouping (one cell per GT leaf) at 30 fps: upper bound    8 runs
#   B   the fix at 7.5 fps -> compared with paper 1's published numbers   8 runs
#   C   paper-1 model at 7.5 fps on the new renders (reproduction check)  4 runs
STAGES = ([f"{COMMON} --trees 1 --runs high:noisy --modes cluster deform oracle --train-strides 1 --iters 600 --warmup 200 --eval-frames 2"]
          if os.environ.get("DYN_SMOKE") == "1" else
          [f"{COMMON} --trees 1 3 5 7 --modes cluster deform --train-strides 1",
           f"{COMMON} --trees 1 3 5 7 --modes oracle --train-strides 1",
           f"{COMMON} --trees 1 3 5 7 --modes cluster --train-strides 4",
           f"{COMMON} --trees 1 5 --modes deform --train-strides 4"])


def sh(cmd):
    print("$", cmd, flush=True)
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
        print(f"$ compiling gsplat (attempt {attempt + 1})", flush=True)
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
sh("pip install -q nerfstudio 2>&1 | tail -n 2")
os.chdir(CODE)
sh("git log --oneline -1")
n = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 1)
warmup_gsplat()
print(f"GPUs: {n}", flush=True)
for stage in STAGES:
    print(f"\n===== stage: {stage}", flush=True)
    print("exit codes:", run_workers(stage, max(1, n)), flush=True)
    sh(f"python scripts/eval/collect_results.py --dir {WORK}/results --out {WORK}/results.csv")
sh(f"rm -rf {WORK}/ns {WORK}/runs {WORK}/export")
