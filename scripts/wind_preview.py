#!/usr/bin/env python
"""Render a short low-res clip of one tree at every wind level, side by side.

    python scripts/wind_preview.py --master configs/master.yaml --tree 1 [--seconds 3] [--scale 0.4]

Writes data/previews/tree_XX/wind_preview.mp4 (+ a contact-sheet PNG). Uses EEVEE
beauty only, real-time wind (stride 1), a camera frozen at t = 0 so only the
tree moves, and the same tree seed as the dataset.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

import numpy as np  # noqa: E402

from run_batch import find_blender, run_blender  # noqa: E402
from synthtree.colour import linear_to_srgb, to_uint8  # noqa: E402
from synthtree.config import expand_master, load_master, write_json  # noqa: E402
from synthtree.imageio import exr_passes, read_exr, write_png  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", required=True)
    ap.add_argument("--tree", type=int, required=True)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--scale", type=float, default=0.4)
    ap.add_argument("--blender")
    a = ap.parse_args()
    m = load_master(a.master)
    m["only_trees"] = [a.tree]
    seqs = [s for s in expand_master(m) if s["kind"] == "orbit"]
    root = os.path.join(os.path.dirname(m["output_root"]), "previews", f"tree_{a.tree:02d}")
    blender = find_blender(a.blender)
    clips = []
    for s in seqs:
        c = copy.deepcopy(s)
        c["out_dir"] = os.path.join(root, s["wind_level"])
        c["capture"]["frame_stride"] = 1
        c["capture"]["jitter"]["enabled"] = False
        c["capture"]["freeze_camera_at_s"] = 0.0
        c["capture"]["duration_s"] = a.seconds
        c["render"]["resolution_scale"] = a.scale
        c["render"]["beauty_engine"] = "eevee"
        c["render"]["eevee"] = {"samples": 8}
        n = int(a.seconds * c["capture"]["fps"])
        cfgp = os.path.join(c["out_dir"], "config.json")
        write_json(cfgp, c)
        if not os.path.exists(c["tree_dir"]):
            sys.exit("generate the tree first (scripts/make_tree.py or run_batch --stages tree)")
        print(f"rendering {s['wind_level']} (U={s['wind_U']} m/s, {n} frames)")
        rc, dt, _ = run_blender(blender, cfgp, os.path.join(c["out_dir"], "blender.log"), ["--passes", "beauty", "--max-frames", str(n)])
        if rc:
            sys.exit(f"Blender failed, see {c['out_dir']}/blender.log")
        frames = []
        for i in range(n):
            ch = exr_passes(read_exr(os.path.join(c["out_dir"], "raw", "beauty", f"{i:05d}_b00_s00.exr")))
            rgb = next(v for v in ch.values() if v.ndim == 3)[..., :3]
            frames.append(to_uint8(linear_to_srgb(rgb)))
        clips.append((f"{s['wind_level']} {s['wind_U']:.1f} m/s", frames))

    import cv2
    H, W = clips[0][1][0].shape[:2]
    out = os.path.join(root, "wind_preview.mp4")
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), m["capture"]["fps"], (W * len(clips), H))
    for i in range(len(clips[0][1])):
        row = []
        for label, fr in clips:
            im = np.ascontiguousarray(fr[i][..., ::-1])
            cv2.putText(im, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            row.append(im)
        vw.write(np.concatenate(row, 1))
    vw.release()
    # contact sheet: |frame_t - frame_0| per level shows where/how much the tree moves
    sheet = []
    for label, fr in clips:
        f0 = fr[0].astype(np.int16)
        diff = np.max([np.abs(f.astype(np.int16) - f0).max(-1) for f in fr], axis=0)
        sheet.append(np.concatenate([fr[len(fr) // 2], np.repeat(np.clip(diff * 3, 0, 255).astype(np.uint8)[..., None], 3, -1)], 0))
    write_png(os.path.join(root, "wind_preview_sheet.png"), np.concatenate(sheet, 1))
    print("wrote", out)


if __name__ == "__main__":
    main()
