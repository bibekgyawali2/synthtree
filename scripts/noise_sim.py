#!/usr/bin/env python
"""Write the NOISY version of a clean sequence (clean data is never modified).

    python scripts/noise_sim.py --seq data/output/tree_01/wind_high [--profile noise_profile.json]

Reads clean/{rgb,depth,normal,semantic}, writes noisy/{rgb,depth} and
noisy/noise_meta.json (full profile + seed). Noise draws are keyed by
(tree seed + seed_offset, frame index) and therefore identical across the
wind levels of a tree.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from synthtree.config import write_json  # noqa: E402
from synthtree.imageio import decode_normals16, read_png, write_png  # noqa: E402
from synthtree.intrinsics import Intrinsics  # noqa: E402
from synthtree.noise import DEFAULT_PROFILE, deep_merge, exposure_track, frame_rng, simulate_depth, simulate_rgb  # noqa: E402
from synthtree.rng import substream  # noqa: E402


def load_profile(path, sensor_depth=None):
    prof = DEFAULT_PROFILE
    if sensor_depth:  # sensor config provides range / units / sampling unless the profile overrides them
        prof = deep_merge(prof, {"depth": {k: sensor_depth[k] for k in ("min_m", "max_m", "units_m", "effective_scale") if k in sensor_depth}})
    if path:
        prof = deep_merge(prof, json.load(open(path)))
    return prof


def process(seq, profile_path=None):
    cfg = json.load(open(os.path.join(seq, "config.json")))
    ncfg = cfg.get("noise", {})
    prof = load_profile(profile_path or ncfg.get("profile"), cfg["sensor"].get("depth"))
    seed = int(cfg["tree"]["seed"]) + int(ncfg.get("seed_offset", 7))
    ij = json.load(open(os.path.join(seq, "intrinsics.json")))
    intr = Intrinsics(**{k: ij[k] for k in ("width", "height", "fx", "fy", "cx", "cy")})
    names = sorted(f for f in os.listdir(os.path.join(seq, "clean", "rgb")) if f.endswith(".png"))
    od = {k: os.path.join(seq, "noisy", k) for k in ("rgb", "depth")}
    for p in od.values():
        os.makedirs(p, exist_ok=True)
    ev = exposure_track(len(names), prof, substream(seed, "noise", "exposure"))
    units = prof["depth"]["units_m"]
    comp = cfg.get("outputs", {}).get("png_compression", 3)
    for i, f in enumerate(names):
        z = read_png(os.path.join(seq, "clean", "depth", f)).astype(np.float64) * 0.001
        nrm_p = os.path.join(seq, "clean", "normal", f)
        nrm = decode_normals16(read_png(nrm_p)) if os.path.exists(nrm_p) else None
        sem = read_png(os.path.join(seq, "clean", "semantic", f))
        zn = simulate_depth(z, prof, frame_rng(seed, i, "depth"), nrm, intr.K, sem)
        write_png(os.path.join(od["depth"], f), np.clip(np.round(zn / units), 0, 65535).astype(np.uint16), comp)
        rgb = read_png(os.path.join(seq, "clean", "rgb", f))[..., :3]
        write_png(os.path.join(od["rgb"], f), simulate_rgb(rgb, prof, frame_rng(seed, i, "rgb"), ev[i]), comp)
    write_json(os.path.join(seq, "noisy", "noise_meta.json"),
               {"profile": prof, "seed": seed, "exposure_ev": ev.tolist(), "depth_units_m": units,
                "note": "depth PNG value * depth_units_m = metres; 0 = invalid"})
    return {"frames": len(names), "seed": seed, "profile": prof.get("name")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--profile")
    a = ap.parse_args()
    print(json.dumps(process(a.seq, a.profile)))


if __name__ == "__main__":
    main()
