#!/usr/bin/env python
"""Fit a depth-noise profile from REAL sensor recordings; plot real vs simulated.

Recordings (16-bit depth PNG in mm, aligned to colour; intrinsics JSON with fx, fy, cx, cy):

  wall/                      flat, matte, textured wall, camera roughly fronto-parallel
    0.5/depth/*.png          one folder per nominal distance in metres (>= 3 distances,
    1.0/depth/*.png          e.g. 0.4 ... 3.0 m), 30+ frames each, static camera
    ...
  leaves/                    real leaves with the background >= 1 m behind them
    depth/*.png
    rgb/*.png                same names; used for a vegetation (ExG) silhouette
    mask/*.png  (optional)   your own binary leaf mask (overrides ExG)

    python scripts/calibrate_noise.py --wall rec/wall --leaves rec/leaves --intrinsics rec/intrinsics.json \
        --out configs/noise_profile_d435i.json --plots calib_plots/
    python scripts/calibrate_noise.py --demo --out /tmp/demo_profile.json --plots /tmp/demo_plots

Fitted: sigma(z) = a + b z^2 (weighted LSQ on per-distance plane residual std),
spatial correlation (lag-1 autocorrelation of residuals), random hole rate (wall),
flying-pixel rate, edge-band hole rate, thin-structure hole rate (leaves).
`--demo` synthesises recordings with a known profile, then checks it is recovered.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from synthtree.imageio import read_png, write_png  # noqa: E402
from synthtree.intrinsics import Intrinsics, backproject  # noqa: E402
from synthtree.noise import DEFAULT_PROFILE, deep_merge, discontinuities, simulate_depth, thin_mask  # noqa: E402


# ----------------------------------------------------------------------------- wall
def plane_residuals(z, intr, roi=0.5):
    """Fit a plane to the central ROI of a depth map; return residual image (m) along z and mask."""
    H, W = z.shape
    h0, h1 = int(H * (1 - roi) / 2), int(H * (1 + roi) / 2)
    w0, w1 = int(W * (1 - roi) / 2), int(W * (1 + roi) / 2)
    sub = np.zeros_like(z)
    sub[h0:h1, w0:w1] = z[h0:h1, w0:w1]
    X = backproject(intr, np.eye(4), sub)
    m = sub > 0
    P = X[m]
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2]
    dist = (X - c) @ n
    res = np.where(m, dist / max(abs(n[2]), 1e-6), 0.0)  # convert to depth-direction residual
    return res, m, (h0, h1, w0, w1)


def lag1_corr(res, m):
    a, b = res[:, :-1], res[:, 1:]
    mm = m[:, :-1] & m[:, 1:]
    if mm.sum() < 100:
        return 0.0
    return float(np.corrcoef(a[mm], b[mm])[0, 1])


def corr_px_from_rho(rho):
    """Gaussian-filtered white noise with sigma s has lag-1 correlation exp(-1/(4 s^2))."""
    rho = min(max(rho, 1e-4), 0.999)
    return float(math.sqrt(-1.0 / (4.0 * math.log(rho))))


def analyse_wall(wall_dir, intr, max_frames=60):
    rows = []
    for dd in sorted(glob.glob(os.path.join(wall_dir, "*"))):
        files = sorted(glob.glob(os.path.join(dd, "depth", "*.png")))[:max_frames]
        if not files:
            continue
        stds, holes, rhos, zs, resid = [], [], [], [], []
        for f in files:
            z = read_png(f).astype(np.float64) * 0.001
            res, m, (h0, h1, w0, w1) = plane_residuals(z, intr)
            roi_valid = z[h0:h1, w0:w1] > 0
            holes.append(1 - roi_valid.mean())
            r = res[m]
            r = r[np.abs(r - np.median(r)) < 6 * 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9]
            stds.append(r.std())
            rhos.append(lag1_corr(res, m))
            zs.append(np.median(z[h0:h1, w0:w1][roi_valid]))
            resid.append(r)
        rows.append({"dir": dd, "z_m": float(np.median(zs)), "sigma_m": float(np.median(stds)),
                     "hole_rate": float(np.median(holes)), "rho": float(np.median(rhos)),
                     "residuals": np.concatenate(resid)})
    return rows


def fit_sigma(rows):
    z = np.array([r["z_m"] for r in rows])
    s = np.array([r["sigma_m"] for r in rows])
    A = np.stack([np.ones_like(z), z ** 2], 1)
    w = 1.0 / np.maximum(s, 1e-5)  # relative error weighting
    coef, *_ = np.linalg.lstsq(A * w[:, None], s * w, rcond=None)
    a, b = coef
    if a < 0:  # constrain to non-negative
        a, b = 0.0, float(np.sum(w ** 2 * s * z ** 2) / np.sum(w ** 2 * z ** 4))
    return float(max(a, 0)), float(max(b, 0))


# ----------------------------------------------------------------------------- leaves
def exg_mask(rgb):
    import cv2
    f = rgb.astype(np.float32) / 255.0
    exg = 2 * f[..., 1] - f[..., 0] - f[..., 2]
    e8 = np.clip((exg + 1) * 127.5, 0, 255).astype(np.uint8)
    _, m = cv2.threshold(e8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m.astype(bool)


def leaf_stats(z, mask, edge_band_px=2, thin_px=3, margin_m=0.05):
    import cv2
    valid = z > 0
    k = np.ones((3, 3), np.uint8)
    boundary = mask ^ cv2.erode(mask.astype(np.uint8), k).astype(bool)
    band = cv2.dilate(boundary.astype(np.uint8), np.ones((2 * edge_band_px + 1,) * 2, np.uint8)).astype(bool)
    interior = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    bg = ~cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    thin = thin_mask(mask, thin_px)
    zfg = np.percentile(z[interior & valid], 95) if (interior & valid).sum() > 50 else np.nan
    zbg = np.percentile(z[bg & valid], 5) if (bg & valid).sum() > 50 else np.nan
    between = valid & (z > zfg + margin_m) & (z < zbg - margin_m)
    return {
        "flying_rate": float(between[band].sum() / max((band & valid).sum(), 1)),
        "edge_hole_rate": float(1 - valid[band].mean()) if band.any() else 0.0,
        "thin_hole_rate": float(1 - valid[thin].mean()) if thin.any() else 0.0,
        "interior_hole_rate": float(1 - valid[interior].mean()) if interior.any() else 0.0,
        "z_fg": float(zfg), "z_bg": float(zbg),
    }


def analyse_leaves(leaf_dir, max_frames=60):
    out = []
    for f in sorted(glob.glob(os.path.join(leaf_dir, "depth", "*.png")))[:max_frames]:
        name = os.path.basename(f)
        z = read_png(f).astype(np.float64) * 0.001
        mp = os.path.join(leaf_dir, "mask", name)
        if os.path.exists(mp):
            mask = read_png(mp) > 0
        else:
            mask = exg_mask(read_png(os.path.join(leaf_dir, "rgb", name))[..., :3])
        out.append((leaf_stats(z, mask), mask, z))
    return out


def clean_proxy(mask, z_fg, z_bg):
    """Two-level 'clean' depth from a silhouette, used to simulate the leaf scene."""
    return np.where(mask, z_fg, z_bg).astype(np.float64)


def invert_rates(real, sim_fn, prof, iters=3):
    """Adjust flying / edge / thin rates so simulated statistics match the real ones (fixed-point)."""
    p = json.loads(json.dumps(prof))
    for _ in range(iters):
        sim = sim_fn(p)
        ho, fl = p["depth"]["holes"], p["depth"]["flying"]
        fl["rate"] = float(np.clip(fl["rate"] * real["flying_rate"] / max(sim["flying_rate"], 1e-4), 0, 1))
        rnd = ho["random"]
        ho["edge"] = float(np.clip(ho["edge"] + (real["edge_hole_rate"] - sim["edge_hole_rate"]), 0, 1 - rnd))
        ho["thin"] = float(np.clip(ho["thin"] + (real["thin_hole_rate"] - sim["thin_hole_rate"]), 0, 1 - rnd))
    return p


# ----------------------------------------------------------------------------- demo data
def make_demo(root, intr, prof_true, seed=0):
    import cv2
    rng = np.random.default_rng(seed)
    H, W = intr.height, intr.width
    for d in (0.4, 0.8, 1.2, 1.8, 2.5, 3.2):
        os.makedirs(os.path.join(root, "wall", f"{d}", "depth"), exist_ok=True)
        z = np.full((H, W), d)
        for k in range(12):
            zn = simulate_depth(z, prof_true, np.random.default_rng(rng.integers(1 << 31)))
            write_png(os.path.join(root, "wall", f"{d}", "depth", f"{k:04d}.png"), np.round(zn * 1000).astype(np.uint16))
    for sub in ("depth", "rgb", "mask"):
        os.makedirs(os.path.join(root, "leaves", sub), exist_ok=True)
    for k in range(8):
        mask = np.zeros((H, W), np.uint8)
        for _ in range(40):
            c = (int(rng.integers(0, W)), int(rng.integers(0, H)))
            cv2.ellipse(mask, c, (int(rng.integers(6, 25)), int(rng.integers(3, 12))), float(rng.uniform(0, 180)), 0, 360, 1, -1)
        for _ in range(15):  # thin twigs
            p0 = (int(rng.integers(0, W)), int(rng.integers(0, H)))
            p1 = (p0[0] + int(rng.integers(-80, 80)), p0[1] + int(rng.integers(-80, 80)))
            cv2.line(mask, p0, p1, 1, 1)
        m = mask.astype(bool)
        z = clean_proxy(m, 0.8, 2.5)
        zn = simulate_depth(z, prof_true, np.random.default_rng(rng.integers(1 << 31)), semantic=m.astype(np.uint8))
        write_png(os.path.join(root, "leaves", "depth", f"{k:04d}.png"), np.round(zn * 1000).astype(np.uint16))
        write_png(os.path.join(root, "leaves", "mask", f"{k:04d}.png"), (m * 255).astype(np.uint8))
        write_png(os.path.join(root, "leaves", "rgb", f"{k:04d}.png"), np.zeros((H, W, 3), np.uint8))


# ----------------------------------------------------------------------------- main
def calibrate(wall_dir, leaf_dir, intr, base_profile=None, plots=None):
    prof = deep_merge(DEFAULT_PROFILE, base_profile or {})
    prof["depth"]["model"] = "quadratic"
    prof["depth"]["effective_scale"] = 1.0  # real recordings already contain the sensor's sampling
    report = {}
    rows = analyse_wall(wall_dir, intr) if wall_dir else []
    if rows:
        a, b = fit_sigma(rows)
        rho = float(np.median([r["rho"] for r in rows]))
        prof["depth"]["a_m"], prof["depth"]["b_per_m"] = a, b
        prof["depth"]["corr_px"] = corr_px_from_rho(rho) if rho > 0.05 else 0.0
        prof["depth"]["holes"]["random"] = float(np.min([r["hole_rate"] for r in rows]))
        report["wall"] = [{k: v for k, v in r.items() if k != "residuals"} for r in rows]
        report["fit"] = {"a_m": a, "b_per_m": b, "lag1_rho": rho, "corr_px": prof["depth"]["corr_px"]}
    leaves = analyse_leaves(leaf_dir) if leaf_dir else []
    if leaves:
        real = {k: float(np.mean([s[k] for s, _, _ in leaves])) for k in leaves[0][0]}

        def sim_fn(p):
            st = []
            for s, mask, _ in leaves[:6]:
                z = clean_proxy(mask, s["z_fg"], s["z_bg"])
                zn = simulate_depth(z, p, np.random.default_rng(1), semantic=mask.astype(np.uint8))
                st.append(leaf_stats(zn, mask))
            return {k: float(np.mean([x[k] for x in st])) for k in st[0]}
        prof = invert_rates(real, sim_fn, prof)
        report["leaves_real"] = real
        report["leaves_sim_after_fit"] = sim_fn(prof)
    prof["name"] = "calibrated from real recordings"
    if plots:
        make_plots(plots, rows, leaves, prof, report)
    return prof, report


def make_plots(out, rows, leaves, prof, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    if rows:
        z = np.array([r["z_m"] for r in rows])
        s = np.array([r["sigma_m"] for r in rows])
        zz = np.linspace(0, z.max() * 1.1, 100)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(z, s * 1000, "o", label="real (plane residual std)")
        ax.plot(zz, (prof["depth"]["a_m"] + prof["depth"]["b_per_m"] * zz ** 2) * 1000, "-", label="fit a + b z^2")
        ax.set_xlabel("distance z [m]")
        ax.set_ylabel("sigma [mm]")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, "sigma_vs_z.png"), dpi=120)
        fig, axs = plt.subplots(1, len(rows), figsize=(3.2 * len(rows), 3), squeeze=False)
        for ax, r in zip(axs[0], rows):
            zc = np.full((240, 320), r["z_m"])
            sim = simulate_depth(zc, deep_merge(prof, {"depth": {"holes": {"random": 0.0}}}), np.random.default_rng(0))
            sres = (sim - zc)[sim > 0]
            lim = 5 * max(r["sigma_m"], 1e-4)
            bins = np.linspace(-lim, lim, 60) * 1000
            ax.hist(r["residuals"] * 1000, bins, density=True, alpha=0.5, label="real")
            ax.hist(sres * 1000, bins, density=True, alpha=0.5, label="sim")
            ax.set_title(f"z = {r['z_m']:.2f} m")
            ax.set_xlabel("residual [mm]")
        axs[0][0].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, "residual_hist_real_vs_sim.png"), dpi=120)
    if leaves and "leaves_real" in report:
        keys = ["flying_rate", "edge_hole_rate", "thin_hole_rate", "interior_hole_rate"]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        x = np.arange(len(keys))
        ax.bar(x - 0.2, [report["leaves_real"][k] for k in keys], 0.4, label="real")
        ax.bar(x + 0.2, [report["leaves_sim_after_fit"][k] for k in keys], 0.4, label="sim")
        ax.set_xticks(x, keys, rotation=20)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, "leaf_rates_real_vs_sim.png"), dpi=120)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wall")
    ap.add_argument("--leaves")
    ap.add_argument("--intrinsics", help="JSON with width, height, fx, fy, cx, cy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plots")
    ap.add_argument("--demo", action="store_true", help="synthesise recordings with a known profile and recover it")
    a = ap.parse_args()
    if a.demo:
        import tempfile
        intr = Intrinsics(320, 240, 300, 300, 159.5, 119.5)
        truth = deep_merge(DEFAULT_PROFILE, {"depth": {"a_m": 0.001, "b_per_m": 0.006, "corr_px": 1.0, "effective_scale": 1.0,
                                                       "min_m": 0.1, "max_m": 6.0,
                                                       "flying": {"rate": 0.4}, "holes": {"random": 0.01, "edge": 0.3, "thin": 0.5}}})
        root = tempfile.mkdtemp(prefix="calib_demo_")
        make_demo(root, intr, truth)
        prof, rep = calibrate(os.path.join(root, "wall"), os.path.join(root, "leaves"), intr, plots=a.plots)
        d, t = prof["depth"], truth["depth"]
        print(json.dumps({"true": {"a": t["a_m"], "b": t["b_per_m"], "corr": t["corr_px"], "random": t["holes"]["random"],
                                   "fly": t["flying"]["rate"], "edge": t["holes"]["edge"], "thin": t["holes"]["thin"]},
                          "fit": {"a": d["a_m"], "b": d["b_per_m"], "corr": d["corr_px"], "random": d["holes"]["random"],
                                  "fly": d["flying"]["rate"], "edge": d["holes"]["edge"], "thin": d["holes"]["thin"]}}, indent=1))
    else:
        ij = json.load(open(a.intrinsics))
        intr = Intrinsics(**{k: ij[k] for k in ("width", "height", "fx", "fy", "cx", "cy")})
        prof, rep = calibrate(a.wall, a.leaves, intr, plots=a.plots)
    with open(a.out, "w") as f:
        json.dump(prof, f, indent=2)
    with open(os.path.splitext(a.out)[0] + "_report.json", "w") as f:
        json.dump(rep, f, indent=2, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
