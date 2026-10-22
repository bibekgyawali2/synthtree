"""Kaggle script kernel: retrain the 48 benchmark splatfacto models and KEEP them (2x T4).

Pushed with `kaggle kernels push -p kaggle/kernels/models`.
Same matrix and settings as the benchmark (trees 1, 3, 5, 7 x 6 conditions x 2 seeds); outputs are
reorganised into /kaggle/working/models/<run>/ with splat.ply, config.yml and the final checkpoint.
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
    "--methods splatfacto --trees 1 3 5 7 --seeds 0 1 --keep",
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
        cmd = f"python kaggle/run_benchmark.py {args} --shard {k} --num-shards {n_gpus}"
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
sh(f"rm -rf {WORK}/ns")
# one folder per run: splat.ply (open in any Gaussian-splat viewer), config.yml + checkpoint (ns-viewer / ns-render)
os.makedirs(f"{WORK}/models", exist_ok=True)
for name in sorted(os.listdir(f"{WORK}/export")):
    dst = f"{WORK}/models/{name}"
    os.makedirs(dst, exist_ok=True)
    sh(f"mv {WORK}/export/{name}/splat.ply {dst}/")
    run = subprocess.run(f"ls -d {WORK}/runs/{name}/*/splatfacto/*", shell=True, capture_output=True, text=True).stdout.split()
    if run:
        sh(f"mv {run[-1]}/config.yml {run[-1]}/nerfstudio_models {dst}/")
sh(f"rm -rf {WORK}/runs {WORK}/export {WORK}/warmup_gsplat.py {CODE}")
sh(f"du -sh {WORK}/models; ls {WORK}/models | wc -l")
