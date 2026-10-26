#!/usr/bin/env python
"""Qualitative figure: one reconstruction per condition (4 wind speeds x clean / noisy sensor).

    python scripts/paper/make_condition_figure.py --runs results/paper2_stage1_conditions --out paper_synthetic/generated

Reads the figure .npz files of the Kaggle kernel stage1_conditions (per-cell rigid motion, tree 1 of synthtree-v1,
run_dyn.py --fig). Columns are the wind speeds; for each sensor variant one row shows the rendered held-out view
and one the true leaf surface at that instant, coloured by whether the reconstruction covers it within 5 mm.
Calm and windy sequences of a tree share the camera path, so every column shows the same view and instant.
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
WINDS = [("calm", "Calm"), ("low", "1.5 m/s"), ("medium", "3 m/s"), ("high", "5 m/s")]
VARIANTS = [("clean", "clean sensor"), ("noisy", "noisy sensor")]
COVERED, MISSED = "#2a78d6", "#eb6834"  # categorical slots 1-2 of the reference palette (validated, CVD-safe)
INK, MUTED = "#1f1f1e", "#6b6a64"
TAU = 0.005

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "text.color": INK,
                     "axes.labelcolor": INK, "axes.edgecolor": "#c9c8c2"})


def project(P, viewmat, K):
    pc = P @ viewmat[:3, :3].T + viewmat[:3, 3]
    uv = pc[:, :2] / pc[:, 2:3] * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    return uv, pc[:, 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join(REPO, "results", "paper2_stage1_conditions"))
    ap.add_argument("--method", default="cluster3dgs")
    ap.add_argument("--tree", default="tree_01")
    ap.add_argument("--out", default=os.path.join(REPO, "paper_synthetic", "generated"))
    a = ap.parse_args()
    Z = {}
    for v, _ in VARIANTS:
        for w, _ in WINDS:
            f = os.path.join(a.runs, f"{a.method}__{a.tree}__{w}__{v}__s0__fig.npz")
            if os.path.exists(f):
                Z[(v, w)] = np.load(f)
    if not Z:
        raise SystemExit(f"no figure data in {a.runs}")
    any_z = next(iter(Z.values()))
    H, W = any_z["img_gt"].shape[:2]
    # one crop for all panels: the union of the projected leaf surfaces over all conditions
    uv_all = np.concatenate([project(z["P_gt"].astype(np.float64), z["viewmat"], z["K"])[0] for z in Z.values()])
    x0, y0 = np.maximum(uv_all.min(0) - 20, 0).astype(int)
    x1, y1 = np.minimum(uv_all.max(0) + 20, [W, H]).astype(int)

    rows = [(v, vname, kind) for v, vname in VARIANTS for kind in ("render", "geometry")]
    fig, axes = plt.subplots(len(rows), len(WINDS), figsize=(7.2, 7.6), squeeze=False)
    for r, (v, vname, kind) in enumerate(rows):
        for c, (w, wname) in enumerate(WINDS):
            ax = axes[r, c]
            ax.set_xticks([]), ax.set_yticks([])
            z = Z.get((v, w))
            if z is None:
                ax.text(0.5, 0.5, "not run", ha="center", va="center", color=MUTED, transform=ax.transAxes)
            elif kind == "render":
                ax.imshow(z["img_render"][y0:y1, x0:x1])
            else:
                ax.set_facecolor("#fcfcfb")
                uv, depth = project(z["P_gt"].astype(np.float64), z["viewmat"], z["K"])
                o = np.argsort(-depth)
                cov = z["d_gt"][o] < TAU
                ax.scatter(uv[o, 0], uv[o, 1], s=0.12, c=np.where(cov, COVERED, MISSED), linewidths=0, rasterized=True)
                ax.set_xlim(x0, x1), ax.set_ylim(y1, y0), ax.set_aspect("equal")
                ax.text(0.03, 0.03, f"covered {100 * cov.mean():.0f}%", transform=ax.transAxes, color=INK,
                        fontsize=7, va="bottom")
            if r == 0:
                ax.set_title(wname)
            if c == 0:
                ax.set_ylabel(f"{vname}\n{'rendered view' if kind == 'render' else 'leaf surface'}")
    handles = [Line2D([], [], marker="o", ls="", color=COVERED, label="true leaf surface within 5 mm of the reconstruction"),
               Line2D([], [], marker="o", ls="", color=MISSED, label="missed")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1), pad=0.4, w_pad=0.3, h_pad=0.4)
    os.makedirs(a.out, exist_ok=True)
    fig.savefig(os.path.join(a.out, "fig_conditions.pdf"), dpi=300)
    fig.savefig(os.path.join(a.out, "fig_conditions.png"), dpi=200)
    plt.close(fig)
    # covered fractions for the caption (every number in the paper is generated)
    names = {"calm": "Calm", "low": "Low", "medium": "Medium", "high": "High"}
    with open(os.path.join(a.out, "numbers_conditions.tex"), "w") as f:
        f.write("% generated by scripts/paper/make_condition_figure.py - do not edit\n")
        for (v, w), z in sorted(Z.items()):
            f.write(f"\\newcommand{{\\Cov{v.capitalize()}{names[w]}}}{{{100 * (z['d_gt'] < TAU).mean():.0f}}}\n")
    print(f"{len(Z)} conditions -> fig_conditions.pdf / .png, numbers_conditions.tex in {a.out}")


if __name__ == "__main__":
    main()
