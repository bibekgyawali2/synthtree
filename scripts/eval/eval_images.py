#!/usr/bin/env python
"""Held-out-view image and depth metrics, masked to leaves (and full-image for reference).

    python scripts/eval/eval_images.py --seq data/output/tree_01/wind_calm --renders renders_dir \
        [--depth-renders depth_dir] [--depth-scale 0.001] [--split test] [--lpips] [--out metrics.json]

renders_dir holds the method's RGB renders of the held-out views named like the GT
(NNNNN.png). Optional depth renders: 16-bit PNG (x depth-scale -> metres) or .npy (metres).
The GT is always the CLEAN rgb/depth (we evaluate what the method recovers, not the noise).
Masked PSNR: MSE over leaf pixels only. Masked SSIM: SSIM map (11x11 Gaussian, sigma 1.5)
averaged over leaf pixels. Masked LPIPS (optional, needs `pip install lpips torch`): spatial
LPIPS map averaged over the leaf mask. Depth: MAE / RMSE / AbsRel / delta<1.05 over leaf pixels
where both depths are valid.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402

from synthtree.imageio import read_png  # noqa: E402


def psnr_masked(a, b, m):
    if m.sum() == 0:
        return None
    mse = np.mean((a[m] - b[m]) ** 2)
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def ssim_map(a, b):
    import cv2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    out = []
    for c in range(a.shape[2]):
        x, y = a[..., c].astype(np.float64), b[..., c].astype(np.float64)
        g = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)  # noqa: E731
        mx, my = g(x), g(y)
        sxx, syy, sxy = g(x * x) - mx ** 2, g(y * y) - my ** 2, g(x * y) - mx * my
        out.append(((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2)))
    return np.mean(out, 0)


class LPIPS:
    def __init__(self):
        import lpips
        import torch
        self.torch = torch
        self.net = lpips.LPIPS(net="alex", spatial=True).eval()

    def map(self, a, b):
        t = lambda x: self.torch.from_numpy(x).permute(2, 0, 1)[None].float() * 2 - 1  # noqa: E731
        with self.torch.no_grad():
            return self.net(t(a), t(b))[0, 0].numpy()


def depth_metrics(pred, gt, m):
    m = m & (gt > 0) & (pred > 0)
    if m.sum() == 0:
        return {}
    e = pred[m] - gt[m]
    r = np.maximum(pred[m] / gt[m], gt[m] / pred[m])
    return {"depth_mae_mm": float(np.mean(np.abs(e)) * 1000), "depth_rmse_mm": float(np.sqrt(np.mean(e ** 2)) * 1000),
            "depth_absrel": float(np.mean(np.abs(e) / gt[m])), "depth_delta1.05": float(np.mean(r < 1.05)),
            "depth_valid_px": float(m.sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--renders", required=True)
    ap.add_argument("--depth-renders")
    ap.add_argument("--depth-scale", type=float, default=0.001)
    ap.add_argument("--split", default="test")
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    frames = json.load(open(os.path.join(a.seq, "splits.json")))[a.split]
    lp = LPIPS() if a.lpips else None
    rows = []
    for i in frames:
        name = f"{i:05d}.png"
        rp = os.path.join(a.renders, name)
        if not os.path.exists(rp):
            continue
        gt = read_png(os.path.join(a.seq, "clean", "rgb", name))[..., :3].astype(np.float64) / 255
        pr = read_png(rp)[..., :3].astype(np.float64) / 255
        leaf = read_png(os.path.join(a.seq, "clean", "leaf_id", name)) > 0
        full = np.ones_like(leaf)
        s = ssim_map(pr, gt)
        r = {"frame": i, "psnr_leaf": psnr_masked(pr, gt, leaf), "psnr_full": psnr_masked(pr, gt, full),
             "ssim_leaf": float(s[leaf].mean()) if leaf.any() else None, "ssim_full": float(s.mean())}
        if lp:
            lm = lp.map(pr, gt)
            r["lpips_leaf"] = float(lm[leaf].mean()) if leaf.any() else None
            r["lpips_full"] = float(lm.mean())
        if a.depth_renders:
            dp = os.path.join(a.depth_renders, name)
            dn = os.path.join(a.depth_renders, f"{i:05d}.npy")
            pred = np.load(dn) if os.path.exists(dn) else (read_png(dp).astype(np.float64) * a.depth_scale if os.path.exists(dp) else None)
            if pred is not None:
                gtd = read_png(os.path.join(a.seq, "clean", "depth", name)).astype(np.float64) * 0.001
                r.update(depth_metrics(pred, gtd, leaf))
        rows.append(r)
    if not rows:
        sys.exit("no renders matched the split frames")
    keys = [k for k in rows[0] if k != "frame"]
    summary = {k: float(np.mean([r[k] for r in rows if r.get(k) is not None])) for k in keys
               if any(r.get(k) is not None for r in rows)}
    res = {"n_views": len(rows), "mean": summary, "per_view": rows}
    print(json.dumps(summary, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
