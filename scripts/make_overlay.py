#!/usr/bin/env python
"""Visual QC montage for one frame: RGB | depth | normals | leaf ids | semantic | RGB+id-edges overlay.

    python scripts/make_overlay.py --seq data/output/tree_01/wind_high --frame 0 --out overlay.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from synthtree.imageio import read_png, write_png  # noqa: E402


def colorize_ids(ids):
    rng = np.random.default_rng(0)
    lut = rng.integers(40, 255, (int(ids.max()) + 1, 3)).astype(np.uint8)
    lut[0] = 0
    return lut[ids]


def montage(seq, i, variant="clean"):
    f = f"{i:05d}.png"
    rgb = read_png(os.path.join(seq, variant, "rgb", f))[..., :3]
    dep = read_png(os.path.join(seq, variant, "depth", f)).astype(np.float32)
    v = dep[dep > 0]
    dn = np.zeros_like(dep) if v.size == 0 else np.clip((dep - v.min()) / (np.percentile(v, 99) - v.min() + 1e-6), 0, 1)
    dvis = np.where(dep[..., None] > 0, (np.stack([dn, 1 - np.abs(dn - 0.5) * 2, 1 - dn], -1) * 255), 0).astype(np.uint8)
    nrm = read_png(os.path.join(seq, "clean", "normal", f))
    nvis = (nrm.astype(np.float32) / 256).astype(np.uint8)
    lid = read_png(os.path.join(seq, "clean", "leaf_id", f)).astype(np.int64)
    sem = read_png(os.path.join(seq, "clean", "semantic", f))
    pal = np.array([[0, 0, 0], [40, 200, 40], [140, 80, 30], [220, 120, 60], [120, 120, 120], [200, 200, 230]], np.uint8)
    edge = np.zeros(lid.shape, bool)
    edge[:, 1:] |= lid[:, 1:] != lid[:, :-1]
    edge[1:, :] |= lid[1:, :] != lid[:-1, :]
    ov = rgb.copy()
    ov[edge] = [255, 0, 255]
    top = np.concatenate([rgb, dvis, nvis], 1)
    bot = np.concatenate([colorize_ids(lid), pal[np.minimum(sem, 5)], ov], 1)
    return np.concatenate([top, bot], 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--variant", default="clean")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    write_png(a.out, montage(a.seq, a.frame, a.variant))
