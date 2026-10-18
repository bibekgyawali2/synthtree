#!/usr/bin/env python
"""Convert a SynthTree sequence (read-only input) into a training-pipeline layout (writable dir).

    # Nerfstudio: splatfacto / depth-nerfacto (transforms.json with absolute paths, no image copies)
    python convert.py nerfstudio --seq /kaggle/input/synthtree-v1/tree_01/wind_calm --variant clean \
        --out /kaggle/working/ns_tree01_calm [--init-ply noisy|clean|none]
    # MonoGS (TUM RGB-D layout + config yaml); also usable by other TUM-format RGB-D SLAM (e.g. Co-SLAM's TUM loader)
    python convert.py monogs --seq /kaggle/input/synthtree-v1/tree_01/wind_calm --variant noisy --out /kaggle/working/monogs_t01

--init-ply writes a point cloud back-projected from sensor depth (subsampled) and sets
"ply_file_path" so splatfacto starts from it instead of random points. Using the NOISY depth
is the fair choice (a real RGB-D capture provides it); clean depth gives an upper bound.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthtree_dataset import SynthTreeSequence, ensure_extracted  # noqa: E402


def write_ply_xyzrgb(path, xyz, rgb):
    n = len(xyz)
    rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        f.write(rec.tobytes())


def init_point_cloud(seq_dir, variant, max_points=200_000, frame_step=4, max_depth=3.5):
    s = SynthTreeSequence(seq_dir, variant=variant, split="train", load=("rgb", "depth"), with_leaf_gt=False)
    K = s.K
    pts, cols = [], []
    for k in range(0, len(s), frame_step):
        it = s[k]
        d = it["depth"]
        m = (d > 0) & (d < max_depth)
        v, u = np.nonzero(m)
        z = d[m]
        Xc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
        c2w = it["c2w_opencv"]
        pts.append(np.einsum("ij,nj->ni", c2w[:3, :3], Xc) + c2w[:3, 3])
        cols.append((it["rgb"][m] * 255).astype(np.uint8))
    P, C = np.concatenate(pts), np.concatenate(cols)
    if len(P) > max_points:
        idx = np.random.default_rng(0).choice(len(P), max_points, replace=False)
        P, C = P[idx], C[idx]
    return P, C


def to_nerfstudio(seq, variant, out, init_ply="noisy"):
    seq = ensure_extracted(seq)
    os.makedirs(out, exist_ok=True)
    tj = json.load(open(os.path.join(seq, "transforms.json" if variant == "clean" else "transforms_noisy.json")))
    for fr in tj["frames"]:
        fr["file_path"] = os.path.join(seq, fr["file_path"])
        fr["depth_file_path"] = os.path.join(seq, fr["depth_file_path"])
    for key in ("train_filenames", "val_filenames", "test_filenames"):
        if key in tj:
            tj[key] = [os.path.join(seq, p) for p in tj[key]]
    if init_ply != "none":
        P, C = init_point_cloud(seq, init_ply)
        write_ply_xyzrgb(os.path.join(out, "init_points.ply"), P, C)
        tj["ply_file_path"] = "init_points.ply"
    with open(os.path.join(out, "transforms.json"), "w") as f:
        json.dump(tj, f, indent=1)
    print(f"nerfstudio data -> {out}\n"
          f"  ns-train splatfacto --data {out} nerfstudio-data --orientation-method none --center-method none "
          f"--auto-scale-poses False --eval-mode filename\n"
          f"  ns-train depth-nerfacto --data {out} nerfstudio-data --orientation-method none --center-method none "
          f"--auto-scale-poses False --eval-mode filename")


def to_monogs(seq, variant, out):
    seq = ensure_extracted(seq)
    os.makedirs(out, exist_ok=True)
    s = SynthTreeSequence(seq, variant=variant, split="all", load=(), with_leaf_gt=False)
    for sub in ("rgb", "depth"):
        link = os.path.join(out, sub)
        if not os.path.exists(link):
            os.symlink(os.path.join(seq, variant, sub), link)
    names = [f"{i:05d}.png" for i in range(len(s.c2w_opencv))]
    with open(os.path.join(out, "rgb.txt"), "w") as f:
        f.write("# timestamp filename\n" + "".join(f"{t:.6f} rgb/{n}\n" for t, n in zip(s.times, names)))
    with open(os.path.join(out, "depth.txt"), "w") as f:
        f.write("# timestamp filename\n" + "".join(f"{t:.6f} depth/{n}\n" for t, n in zip(s.times, names)))
    with open(os.path.join(seq, "poses", "groundtruth_tum.txt")) as src, open(os.path.join(out, "groundtruth.txt"), "w") as dst:
        dst.write(src.read())
    K = s.K
    cfg = f"""inherit_from: "configs/rgbd/tum/base_config.yaml"
Dataset:
  type: "tum"
  dataset_path: "{out}"
  sensor_type: "depth"
  Calibration:
    fx: {K[0, 0]:.6f}
    fy: {K[1, 1]:.6f}
    cx: {K[0, 2]:.6f}
    cy: {K[1, 2]:.6f}
    k1: 0.0
    k2: 0.0
    p1: 0.0
    p2: 0.0
    k3: 0.0
    width: {s.width}
    height: {s.height}
    depth_scale: {1.0 / s.depth_units:.1f}
    distorted: False
"""
    with open(os.path.join(out, "monogs_config.yaml"), "w") as f:
        f.write(cfg)
    print(f"MonoGS data -> {out}\n  copy monogs_config.yaml into MonoGS/configs/rgbd/ and run: "
          f"python slam.py --config configs/rgbd/monogs_config.yaml  (TUM timestamps are seconds from 0; "
          f"the TUM depth scale is {1.0 / s.depth_units:.0f}, not TUM's 5000)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("format", choices=["nerfstudio", "monogs"])
    ap.add_argument("--seq", required=True)
    ap.add_argument("--variant", default="clean", choices=["clean", "noisy"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-ply", default="noisy", choices=["noisy", "clean", "none"])
    a = ap.parse_args()
    if a.format == "nerfstudio":
        to_nerfstudio(a.seq, a.variant, a.out, a.init_ply)
    else:
        to_monogs(a.seq, a.variant, a.out)


if __name__ == "__main__":
    main()
