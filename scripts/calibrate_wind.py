#!/usr/bin/env python
"""Calibrate wind amplitudes against real static-camera fan clips.

    python scripts/calibrate_wind.py --clips rec/clips --master configs/master.yaml --tree 1 \
        [--seconds 4] [--scale 0.5] [--out calib_wind/]
    python scripts/calibrate_wind.py --self-test --master configs/master.yaml --tree 1   # recover a known factor

Real clips: rec/clips/fan_<level>/{rgb,depth}/NNNNN.png (+ meta.json from rs_capture.py, or
--intrinsics), one per wind level name in master.yaml (calm, low, medium, high). Camera static,
the whole sapling in view, recorded at the same fps as the dataset.

Statistic (identical for real and synthetic, so estimator bias cancels): dense Farneback optical
flow between consecutive frames on vegetation pixels (ExG > Otsu threshold, valid depth), converted
to metric image-plane leaf speed  v = |flow| * z / f * fps  [m/s]; we report median and p90 over
all leaf pixels and frames. Synthetic clips are rendered with the camera frozen at the real clip's
median leaf distance and the master's wind settings.

Recommendation: amplitude factor k = geometric mean over windy levels of real/synthetic median
speed (multiply branch static/osc/cross_deg and leaf pitch/twist/mean_deg by k), and an exponent
correction dp = p_real - p_synth from log-log fits of speed vs U (add dp to both law exponents).
Speed is not exactly linear in the amplitudes (leaf-law saturation, flow underestimating fast
motion), so the synthetic clips are re-rendered with the corrected wind until k is within --tol
of 1 (--iterations, default 3); the printed overrides are the converged values.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import numpy as np  # noqa: E402

from synthtree.config import expand_master, load_master, write_json  # noqa: E402
from synthtree.imageio import read_png  # noqa: E402
from synthtree.tree import deep_merge  # noqa: E402


def exg_mask(rgb):
    import cv2
    f = rgb.astype(np.float32) / 255.0
    exg = 2 * f[..., 1] - f[..., 0] - f[..., 2]
    e8 = np.clip((exg + 1) * 127.5, 0, 255).astype(np.uint8)
    _, m = cv2.threshold(e8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m.astype(bool)


def leaf_speed(clip_dir, fx, fps, max_frames=300):
    """Median / p90 metric leaf speed (m/s) and median leaf depth of a static-camera clip."""
    import cv2
    names = sorted(os.listdir(os.path.join(clip_dir, "rgb")))[:max_frames]
    speeds, depths = [], []
    prev = None
    for n in names:
        rgb = read_png(os.path.join(clip_dir, "rgb", n))[..., :3]
        z = read_png(os.path.join(clip_dir, "depth", n)).astype(np.float32) * 0.001
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev[0], gray, None, 0.5, 4, 15, 3, 5, 1.2, 0)
            m = prev[1] & (prev[2] > 0)
            if m.sum() > 50:
                mag = np.linalg.norm(flow[m], axis=1)
                speeds.append(mag * prev[2][m] / fx * fps)
                depths.append(prev[2][m])
        prev = (gray, exg_mask(rgb), z)
    if not speeds:
        raise RuntimeError(f"no vegetation pixels found in {clip_dir}")
    s, d = np.concatenate(speeds), np.concatenate(depths)
    return {"median_mps": float(np.median(s)), "p90_mps": float(np.percentile(s, 90)),
            "leaf_depth_m": float(np.median(d)), "frames": len(names)}


def render_synthetic(master, tree_id, levels, distance, seconds, scale, root, wind_override=None, blender=None):
    from run_batch import find_blender, run_blender
    import postprocess
    m = copy.deepcopy(master)
    m["only_trees"] = [tree_id]
    blender = find_blender(blender)
    out = {}
    for s in [s for s in expand_master(m) if s["kind"] == "orbit" and s["wind_level"] in levels]:
        c = copy.deepcopy(s)
        c["out_dir"] = os.path.join(root, s["wind_level"])
        c["capture"].update({"frame_stride": 1, "duration_s": seconds, "freeze_camera_at_s": 0.0})
        c["capture"]["jitter"]["enabled"] = False
        c["capture"]["orbit"]["radius_m"] = float(distance)
        c["render"].update({"resolution_scale": scale, "beauty_engine": "eevee", "eevee": {"samples": 8}})
        c["outputs"] = {"depth_exr_every": 0, "normals_every": 0, "flow": False, "png_compression": 1}
        if wind_override:
            from synthtree.tree import deep_merge
            c["wind"] = deep_merge(c["wind"], wind_override)
        if not os.path.exists(os.path.join(c["out_dir"], "metadata.json")):
            write_json(os.path.join(c["out_dir"], "config.json"), c)
            rc, _, _ = run_blender(blender, os.path.join(c["out_dir"], "config.json"), os.path.join(c["out_dir"], "blender.log"))
            if rc:
                sys.exit(f"Blender failed: {c['out_dir']}/blender.log")
            postprocess.process(c["out_dir"])
        ij = json.load(open(os.path.join(c["out_dir"], "intrinsics.json")))
        out[s["wind_level"]] = (os.path.join(c["out_dir"], "clean"), ij["fx"], s["wind_U"])
    return out


def fit_power(U, v):
    U, v = np.asarray(U), np.asarray(v)
    ok = (U > 0) & (v > 0)
    if ok.sum() < 2:
        return None
    p, _ = np.polyfit(np.log(U[ok]), np.log(v[ok]), 1)
    return float(p)


def calibrate(real, synth, levels_U, floor=None):
    rows, ratios = [], []
    for lvl, U in levels_U.items():
        if lvl not in real or lvl not in synth:
            continue
        r, s = real[lvl]["median_mps"], synth[lvl]["median_mps"]
        if floor is not None:  # remove the static noise floor measured on the calm clip
            r = max(r - floor["real"], 1e-6)
            s = max(s - floor["synth"], 1e-6)
        rows.append({"level": lvl, "U": U, "real_mps": r, "synth_mps": s, "ratio": r / s if U > 0 else None})
        if U > 0:
            ratios.append(r / s)
    k = float(np.exp(np.mean(np.log(ratios)))) if ratios else None
    pr = fit_power([x["U"] for x in rows], [x["real_mps"] for x in rows])
    ps = fit_power([x["U"] for x in rows], [x["synth_mps"] for x in rows])
    return {"levels": rows, "amplitude_factor": k, "exponent_real": pr, "exponent_synth": ps,
            "exponent_correction": (pr - ps) if pr is not None and ps is not None else None}


def suggestion(wind, res):
    k = res["amplitude_factor"] or 1.0
    dp = res["exponent_correction"] or 0.0
    b, lf = wind["branch"], wind["leaf"]
    return {"wind": {
        "branch": {"static_deg": round(b.get("static_deg", 6.0) * k, 3), "osc_deg": round(b.get("osc_deg", 3.0) * k, 3),
                   "cross_deg": round(b.get("cross_deg", 1.5) * k, 3),
                   "law": {**b.get("law", {}), "exponent": round(b.get("law", {}).get("exponent", 2.0) + dp, 3)}},
        "leaf": {"pitch_deg": round(lf.get("pitch_deg", 25.0) * k, 3), "twist_deg": round(lf.get("twist_deg", 18.0) * k, 3),
                 "mean_deg": round(lf.get("mean_deg", 15.0) * k, 3),
                 "law": {**lf.get("law", {}), "exponent": round(lf.get("law", {}).get("exponent", 1.0) + dp, 3)}}}}


def plot(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = res["levels"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([r["U"] for r in rows], [r["real_mps"] * 1000 for r in rows], "o-", label="real")
    ax.plot([r["U"] for r in rows], [r["synth_mps"] * 1000 for r in rows], "s--", label="synthetic (current)")
    ax.set_xlabel("wind speed U [m/s]")
    ax.set_ylabel("median leaf speed [mm/s]")
    ax.set_title(f"amplitude factor k = {res['amplitude_factor'] or float('nan'):.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "wind_calibration.png"), dpi=120)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default=os.path.join(REPO, "rec", "clips"))
    ap.add_argument("--master", required=True)
    ap.add_argument("--tree", type=int, required=True, help="synthetic tree that best matches the filmed sapling")
    ap.add_argument("--intrinsics", help="JSON with fx (default: meta.json of each clip)")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(REPO, "data", "calib_wind"))
    ap.add_argument("--self-test", action="store_true", help="fake 'real' clips = synthetic with amplitudes x1.6")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--blender")
    a = ap.parse_args()
    m = load_master(a.master)
    levels = m["wind"]["levels"]
    os.makedirs(a.out, exist_ok=True)
    fps = float(m["capture"]["fps"])

    real = {}
    if a.self_test:
        w = m["wind"]
        fake = {"branch": {k: w["branch"][k] * 1.6 for k in ("static_deg", "osc_deg", "cross_deg")},
                "leaf": {k: w["leaf"][k] * 1.6 for k in ("pitch_deg", "twist_deg", "mean_deg")}}
        clips = render_synthetic(m, a.tree, list(levels), 1.5, a.seconds, a.scale, os.path.join(a.out, "selftest_real"), fake, a.blender)
        for lvl, (d, fx, _) in clips.items():
            real[lvl] = leaf_speed(d, fx, fps)
    else:
        for lvl in levels:
            d = os.path.join(a.clips, f"fan_{lvl}")
            if not os.path.isdir(d):
                print(f"missing clip for level {lvl}: {d}")
                continue
            if a.intrinsics:
                fx = json.load(open(a.intrinsics))["fx"]
            else:
                fx = json.load(open(os.path.join(d, "meta.json")))["intrinsics"]["color"]["fx"]
            real[lvl] = leaf_speed(d, fx, fps)
    if not real:
        sys.exit("no real clips found")
    dist = float(np.median([r["leaf_depth_m"] for r in real.values()]))
    print("real:", json.dumps(real, indent=1), f"\nmedian leaf distance {dist:.2f} m -> synthetic camera radius")
    wind = copy.deepcopy(m["wind"])
    history = []
    for it in range(max(1, a.iterations)):
        override = {"branch": {k: wind["branch"][k] for k in ("static_deg", "osc_deg", "cross_deg", "law")},
                    "leaf": {k: wind["leaf"][k] for k in ("pitch_deg", "twist_deg", "mean_deg", "law")}}
        syn_clips = render_synthetic(m, a.tree, list(real), dist, a.seconds, a.scale,
                                     os.path.join(a.out, f"synthetic_it{it}"), override, a.blender)
        synth = {lvl: leaf_speed(d, fx, fps) for lvl, (d, fx, _) in syn_clips.items()}
        floor = None
        calm = [lvl for lvl, U in levels.items() if U == 0 and lvl in real and lvl in synth]
        if calm:
            floor = {"real": real[calm[0]]["median_mps"], "synth": synth[calm[0]]["median_mps"]}
        res = calibrate(real, synth, levels, floor)
        res["noise_floor_mps"] = floor
        history.append({"iteration": it, "amplitude_factor": res["amplitude_factor"],
                        "exponent_correction": res["exponent_correction"], "levels": res["levels"]})
        print(f"iteration {it}: k = {res['amplitude_factor']:.3f}, dp = {res['exponent_correction']:+.3f}")
        converged = res["amplitude_factor"] is None or (abs(res["amplitude_factor"] - 1) < a.tol and abs(res["exponent_correction"] or 0) < 0.05)
        if converged:
            break
        wind = deep_merge(wind, suggestion(wind, res)["wind"])
    res["iterations"] = history
    res["converged"] = bool(converged)
    res["suggested_master_overrides"] = {"wind": {
        "branch": {k: wind["branch"][k] for k in ("static_deg", "osc_deg", "cross_deg", "law")},
        "leaf": {k: wind["leaf"][k] for k in ("pitch_deg", "twist_deg", "mean_deg", "law")}}}
    res["cumulative_amplitude_factor_vs_master"] = wind["leaf"]["pitch_deg"] / m["wind"]["leaf"]["pitch_deg"]
    write_json(os.path.join(a.out, "wind_calibration.json"), res)
    plot(res, a.out)
    print(json.dumps({k: v for k, v in res.items() if k not in ("levels", "iterations")}, indent=1))
    for r in res["levels"]:
        print(f"  {r['level']:7s} U={r['U']:.1f}  real {r['real_mps'] * 1000:7.1f} mm/s  synth {r['synth_mps'] * 1000:7.1f} mm/s"
              + (f"  ratio {r['ratio']:.2f}" if r["ratio"] else ""))


if __name__ == "__main__":
    main()
