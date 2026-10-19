"""Kaggle script kernel: SynthTree benchmark (splatfacto). Pushed with `kaggle kernels push -p kaggle/kernels/bench_splatfacto`."""
import os
import subprocess


def sh(cmd):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True).returncode


sh("git clone -q https://github.com/bibekgyawali2/synthtree.git /kaggle/working/synthtree")
sh("pip install -q nerfstudio 2>&1 | tail -n 3")
os.chdir("/kaggle/working/synthtree")
sh("git log --oneline -1 && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
sh("python kaggle/run_benchmark.py --trees 1 3 5 7 --seeds 0 1 --methods splatfacto")
sh("python scripts/eval/collect_results.py --dir /kaggle/working/results --out /kaggle/working/results.csv")
for metric, hb in [("fscore", "--higher-better"), ("chamfer_mm", ""), ("centroid_err_mm_median", ""),
                   ("leaf_area_rel_error", ""), ("leaf_angle_w1_deg", "")]:
    sh(f"python scripts/eval/decompose_errors.py --results /kaggle/working/results.csv --metric {metric} {hb} "
       f"--out /kaggle/working/decomposition > /dev/null")
sh("rm -rf /kaggle/working/ns /kaggle/working/runs /kaggle/working/export")
