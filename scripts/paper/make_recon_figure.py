#!/usr/bin/env python
"""Qualitative figures: reconstructions vs ground truth at one held-out instant, calm and high wind.

    python scripts/paper/make_recon_figure.py --runs data/fig_runs --out paper_synthetic/generated

Reads the --fig-out .npz files written by kaggle/dyngs.py (tree 1, 30 fps, one per method and wind level) and
writes two figures, both seen from the held-out camera:
  fig_recon_images.pdf    ground-truth image and the rendered reconstruction of each method
  fig_recon_geometry.pdf  the true leaf surface posed at that instant, coloured by whether the reconstruction
                          covers it within 5 mm (the recall side of the F@5mm in the tables)
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METHODS = [("static", "Static 3DGS"), ("deform", "Deformable 3DGS"), ("cluster", "Per-cell rigid motion"),
           ("oracle", "Oracle groups")]
LEVELS = [("calm", "Calm"), ("high", "5 m/s")]
COVERED, MISSED = "#2a78d6", "#eb6834"  # categorical slots 1-2 of the reference palette (validated, CVD-safe)
INK, MUTED = "#1f1f1e", "#6b6a64"
TAU = 0.005

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "text.color": INK,
                     "axes.labelcolor": INK, "axes.edgecolor": "#c9c8c2"})


def project(P, viewmat, K):
    pc = P @ viewmat[:3, :3].T + viewmat[:3, 3]
    z = pc[:, 2]
    uv = pc[:, :2] / z[:, None] * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    return uv, z


def crop_box(z, H, W, pad=24):
    uv, _ = project(z["P_gt"].astype(np.float64), z["viewmat"], z["K"])
    x0, y0 = np.maximum(uv.min(0) - pad, 0).astype(int)
    x1, y1 = np.minimum(uv.max(0) + pad, [W, H]).astype(int)
    return x0, x1, y0, y1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join(REPO, "data", "fig_runs"))
    ap.add_argument("--tree", default="tree_01")
    ap.add_argument("--out", default=os.path.join(REPO, "paper_synthetic", "generated"))
    a = ap.parse_args()
    Z = {}
    for m, _ in METHODS:
        for lv, _ in LEVELS:
            f = os.path.join(a.runs, f"{m}__{a.tree}__{lv}.npz")
            if os.path.exists(f):
                Z[(m, lv)] = np.load(f)
    if not Z:
        raise SystemExit(f"no figure data in {a.runs}")
    H, W = next(iter(Z.values()))["img_gt"].shape[:2]
    ncol = 1 + len(METHODS)

    # ---- rendered images
    fig, axes = plt.subplots(len(LEVELS), ncol, figsize=(7.2, 3.6), squeeze=False)
    for r, (lv, lvname) in enumerate(LEVELS):
        ref = next((Z[(m, lv)] for m, _ in METHODS if (m, lv) in Z), None)
        box = crop_box(ref, H, W) if ref is not None else (0, W, 0, H)
        for c in range(ncol):
            ax = axes[r, c]
            ax.set_xticks([]), ax.set_yticks([])
            if c == 0:
                img, title = (ref["img_gt"] if ref is not None else None), "Ground truth"
            else:
                m, title = METHODS[c - 1]
                img = Z[(m, lv)]["img_render"] if (m, lv) in Z else None
            if img is not None:
                ax.imshow(img[box[2]:box[3], box[0]:box[1]])
            else:
                ax.text(0.5, 0.5, "not run", ha="center", va="center", color=MUTED, transform=ax.transAxes)
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(lvname)
    fig.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.3)
    fig.savefig(os.path.join(a.out, "fig_recon_images.pdf"), dpi=300)
    plt.close(fig)

    # ---- geometry: true leaf surface at that instant, covered (within 5 mm) or missed
    fig, axes = plt.subplots(len(LEVELS), ncol, figsize=(7.2, 3.9), squeeze=False)
    for r, (lv, lvname) in enumerate(LEVELS):
        ref = next((Z[(m, lv)] for m, _ in METHODS if (m, lv) in Z), None)
        if ref is not None:
            P = ref["P_gt"].astype(np.float64)
            uv, depth = project(P, ref["viewmat"], ref["K"])
            order = np.argsort(-depth)  # far points first, near points drawn on top
            box = crop_box(ref, H, W)
        for c in range(ncol):
            ax = axes[r, c]
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_facecolor("#fcfcfb")
            if ref is None:
                continue
            if c == 0:
                ax.scatter(uv[order, 0], uv[order, 1], s=0.15, c=INK, linewidths=0, rasterized=True)
                title = "Ground-truth leaves"
            else:
                m, title = METHODS[c - 1]
                if (m, lv) in Z:
                    cov = Z[(m, lv)]["d_gt"][order] < TAU
                    ax.scatter(uv[order, 0], uv[order, 1], s=0.15, c=np.where(cov, COVERED, MISSED),
                               linewidths=0, rasterized=True)
                    ax.text(0.03, 0.03, f"covered {100 * cov.mean():.0f}%", transform=ax.transAxes, color=INK,
                            fontsize=7, va="bottom")
                else:
                    ax.text(0.5, 0.5, "not run", ha="center", va="center", color=MUTED, transform=ax.transAxes)
            ax.set_xlim(box[0], box[1]), ax.set_ylim(box[3], box[2]), ax.set_aspect("equal")
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(lvname)
    handles = [Line2D([], [], marker="o", ls="", color=COVERED, label="true leaf surface within 5 mm of the reconstruction"),
               Line2D([], [], marker="o", ls="", color=MISSED, label="missed")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1), pad=0.4, w_pad=0.3, h_pad=0.3)
    fig.savefig(os.path.join(a.out, "fig_recon_geometry.pdf"), dpi=300)
    plt.close(fig)
    print(f"{len(Z)} runs -> fig_recon_images.pdf, fig_recon_geometry.pdf in {a.out}")


if __name__ == "__main__":
    main()
