#!/usr/bin/env python
"""Convert Blender raw outputs of one sequence into dataset products + ground truth.

    python scripts/postprocess.py --seq data/output/tree_01/wind_high

Per frame (OpenCV camera convention everywhere):
  clean/rgb/NNNNN.png        8-bit sRGB (Standard transform of linear render; sub-frames averaged)
  clean/depth/NNNNN.png      16-bit z-depth in mm, 0 = no geometry
  clean/depth_exr/NNNNN.exr  float32 z-depth in metres (every k-th frame)
  clean/normal/NNNNN.png     16-bit camera-space normals, facing the camera, 0 = invalid
  clean/leaf_id/NNNNN.png    16-bit leaf id, 0 = not a leaf
  clean/semantic/NNNNN.png   8-bit: 0 bg, 1 leaf, 2 wood, 3 pot/soil, 4 ground, 5 room
  clean/flow/NNNNN.png       forward optical flow to the next rendered frame, KITTI 16-bit PNG encoding
Sequence level: poses/ (npy, TUM), transforms.json, intrinsics.json, splits.json,
gt/leaf_tracks.npz (per-leaf per-frame transform/centroid/normal/visible px), metadata.json, qc.json.

Depth is computed as z = ((P - C) . optical_axis) from the exact world-position
AOV, so it is z-depth by construction regardless of how Blender defines its own
Depth pass (which is only used as a cross-check, reported in qc.json).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from synthtree import __version__  # noqa: E402
from synthtree.colour import linear_to_srgb, to_uint8  # noqa: E402
from synthtree.config import write_json  # noqa: E402
from synthtree.imageio import encode_flow_kitti, encode_normals16, exr_passes, read_exr, write_exr, write_png  # noqa: E402
from synthtree.intrinsics import Intrinsics, pixel_grid  # noqa: E402
from synthtree.pose_formats import (intrinsics_json, save_pose_arrays, transforms_json, write_tum,  # noqa: E402
                                    write_tum_list)
from synthtree.transforms import c2w_blender_to_opencv  # noqa: E402
from synthtree.tree import load_tree  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git_info():
    try:
        h = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(["git", "-C", REPO, "status", "--porcelain"], stderr=subprocess.DEVNULL).strip())
        return {"commit": h, "dirty": dirty}
    except Exception:
        return {"commit": "unknown (no git commit)", "dirty": None}


def seq_intrinsics(cfg):
    intr = Intrinsics.from_dict(cfg["sensor"]["rgb"])
    s = float(cfg["render"].get("resolution_scale", 1.0))
    return intr if s == 1.0 else intr.scaled(s)


def assemble_rgb(raw, i, H, W):
    done = json.load(open(os.path.join(raw, "beauty", f"{i:05d}.done")))
    img = np.zeros((H, W, 3), np.float64)
    nsub = len(done["subframe_offsets_s"])
    for b, (r0, r1) in enumerate(done["bands"]):
        acc = np.zeros((H, W, 3), np.float64)
        for s in range(nsub):
            ch = exr_passes(read_exr(os.path.join(raw, "beauty", f"{i:05d}_b{b:02d}_s{s:02d}.exr")))
            rgb = next(v for v in ch.values() if v.ndim == 3)[..., :3]
            acc += rgb
        img[r0:r1] = acc[r0:r1] / nsub
    return img, done


def process(seq, keep_raw=None):
    cfg = json.load(open(os.path.join(seq, "config.json")))
    raw = os.path.join(seq, "raw")
    outs = cfg.get("outputs", {})
    keep_raw = outs.get("keep_raw", False) if keep_raw is None else keep_raw
    comp = outs.get("png_compression", 3)
    intr = seq_intrinsics(cfg)
    H, W = intr.height, intr.width
    frames_meta = json.load(open(os.path.join(raw, "frames.json")))
    n = len([f for f in os.listdir(os.path.join(raw, "gt")) if f.endswith(".npz") and ".part" not in f])
    times = np.asarray(frames_meta["times"][:n])
    tree = load_tree(os.path.join(cfg["tree_dir"], "canonical", "canonical_tree.npz"))
    L = len(tree["leaf_id"])
    c0, n0 = tree["leaf_centroid"], tree["leaf_normal"]

    d = {k: os.path.join(seq, "clean", k) for k in ("rgb", "depth", "depth_exr", "normal", "leaf_id", "semantic", "flow", "canonical_xyz")}
    for k in ("rgb", "depth", "normal", "leaf_id", "semantic"):
        os.makedirs(d[k], exist_ok=True)
    exr_every = int(outs.get("depth_exr_every", 10))
    nrm_every = int(outs.get("normals_every", 1))
    xyz_every = int(outs.get("canonical_xyz_every", 0))
    do_flow = bool(outs.get("flow", True))
    for k, on in (("depth_exr", exr_every > 0), ("flow", do_flow), ("canonical_xyz", xyz_every > 0)):
        if on:
            os.makedirs(d[k], exist_ok=True)

    uu, vv = pixel_grid(W, H)
    c2w_all = np.zeros((n, 4, 4))
    T_all = np.zeros((n, L, 3, 4), np.float32)
    vis_all = np.zeros((n, L), np.int32)
    qc = {"reproj_max_px": [], "depth_vs_blender_planar_med_m": [], "depth_vs_blender_ray_med_m": [],
          "leaf_pixels": [], "leaf_ids_unknown": 0, "leafid_semantic_mismatch_px": 0}

    for i in range(n):
        gt = np.load(os.path.join(raw, "gt", f"{i:05d}.npz"))
        c2w_b = gt["c2w_blender"]
        c2w_all[i] = c2w_b
        T_all[i] = gt["leaf_T"]
        cv = c2w_blender_to_opencv(c2w_b)
        R, C = cv[:3, :3], cv[:3, 3]
        p = exr_passes(read_exr(os.path.join(raw, "data", f"{i:05d}.exr")))
        sem = np.round(p["semantic"]).astype(np.uint8)
        valid = sem > 0
        P = p["P"][..., :3].astype(np.float64)
        Xc = np.einsum("ji,hwj->hwi", R, P - C)  # R^T (P - C)
        z = np.where(valid, Xc[..., 2], 0.0)

        # --- QC: world position must project onto its own pixel centre
        with np.errstate(divide="ignore", invalid="ignore"):
            u = intr.fx * Xc[..., 0] / Xc[..., 2] + intr.cx
            v = intr.fy * Xc[..., 1] / Xc[..., 2] + intr.cy
        err = np.hypot(u - uu, v - vv)[valid]
        qc["reproj_max_px"].append(float(err.max()) if err.size else 0.0)
        if "Depth" in p:
            bz = p["Depth"].astype(np.float64)[valid]
            ray = np.linalg.norm(P - C, axis=-1)[valid]
            qc["depth_vs_blender_planar_med_m"].append(float(np.median(np.abs(bz - z[valid]))))
            qc["depth_vs_blender_ray_med_m"].append(float(np.median(np.abs(bz - ray))))

        # --- depth
        mm = np.clip(np.round(z * 1000.0), 0, 65535).astype(np.uint16)
        write_png(os.path.join(d["depth"], f"{i:05d}.png"), mm, comp)
        if exr_every > 0 and i % exr_every == 0:
            write_exr(os.path.join(d["depth_exr"], f"{i:05d}.exr"), {"Z": z.astype(np.float32)})

        # --- normals (camera space, oriented towards the camera)
        if nrm_every > 0 and i % nrm_every == 0:
            Nw = p["N"][..., :3].astype(np.float64)
            Nc = np.einsum("ji,hwj->hwi", R, Nw)
            flip = np.einsum("hwi,hwi->hw", Nc, Xc) > 0
            Nc[flip] *= -1
            Nc /= np.maximum(np.linalg.norm(Nc, axis=-1, keepdims=True), 1e-12)
            Nc[~valid] = 0
            write_png(os.path.join(d["normal"], f"{i:05d}.png"), encode_normals16(Nc), comp)

        # --- ids
        lid = np.round(p["leaf_id"]).astype(np.int64)
        assert lid.max() < 65536, "leaf ids exceed uint16"
        write_png(os.path.join(d["leaf_id"], f"{i:05d}.png"), lid.astype(np.uint16), comp)
        write_png(os.path.join(d["semantic"], f"{i:05d}.png"), sem, comp)
        qc["leaf_ids_unknown"] += int(np.sum(lid > L))
        qc["leafid_semantic_mismatch_px"] += int(np.sum((lid > 0) != (sem == 1)))
        qc["leaf_pixels"].append(int(np.sum(lid > 0)))
        vis_all[i] = np.bincount(lid.ravel(), minlength=L + 1)[1:L + 1]

        # --- forward flow to the next rendered frame (exact, from per-vertex next positions)
        if do_flow:
            cvn = c2w_blender_to_opencv(gt["c2w_next_blender"])
            Pn = p["P_next"][..., :3].astype(np.float64)
            Xn = np.einsum("ji,hwj->hwi", cvn[:3, :3], Pn - cvn[:3, 3])
            with np.errstate(divide="ignore", invalid="ignore"):
                un = intr.fx * Xn[..., 0] / Xn[..., 2] + intr.cx
                vn = intr.fy * Xn[..., 1] / Xn[..., 2] + intr.cy
            fvalid = valid & (Xn[..., 2] > 0)
            flow = np.stack([np.where(fvalid, un - uu, 0), np.where(fvalid, vn - vv, 0)], -1)
            write_png(os.path.join(d["flow"], f"{i:05d}.png"), encode_flow_kitti(flow, fvalid), comp)
        if xyz_every > 0 and i % xyz_every == 0:
            Pr = np.where(valid[..., None], p["P_rest"][..., :3], 0)
            write_exr(os.path.join(d["canonical_xyz"], f"{i:05d}.exr"), {"X": Pr[..., 0], "Y": Pr[..., 1], "Z": Pr[..., 2]})

        # --- rgb
        lin, _ = assemble_rgb(raw, i, H, W)
        write_png(os.path.join(d["rgb"], f"{i:05d}.png"), to_uint8(linear_to_srgb(lin)), comp)
        if i % 25 == 0:
            print(f"[post] {cfg['sequence_id']} frame {i}/{n}", flush=True)

    # ------------------------------------------------------------- sequence level
    os.makedirs(os.path.join(seq, "gt"), exist_ok=True)
    A = T_all[..., :3].astype(np.float64)
    cent = (np.einsum("flij,lj->fli", A, c0) + T_all[..., 3]).astype(np.float32)
    nrm = np.einsum("flij,lj->fli", A, n0).astype(np.float32)
    np.savez_compressed(os.path.join(seq, "gt", "leaf_tracks.npz"), times=times, leaf_id=tree["leaf_id"],
                        transform_3x4=T_all, centroid=cent, normal=nrm, visible_px=vis_all)

    save_pose_arrays(os.path.join(seq, "poses"), c2w_all)
    write_tum(os.path.join(seq, "poses", "groundtruth_tum.txt"), times, c2w_all)
    names = [f"{i:05d}.png" for i in range(n)]
    for variant in ("clean", "noisy"):
        write_tum_list(os.path.join(seq, "poses", f"rgb_{variant}.txt"), times, [f"{variant}/rgb/{x}" for x in names], "color images")
        write_tum_list(os.path.join(seq, "poses", f"depth_{variant}.txt"), times, [f"{variant}/depth/{x}" for x in names], "depth images (mm)")

    k = int(cfg.get("splits", {}).get("holdout_every", 8))
    test = [i for i in range(n) if k > 0 and i % k == 0]
    train = [i for i in range(n) if i not in set(test)]
    write_json(os.path.join(seq, "splits.json"), {"train": train, "test": test, "holdout_every": k,
                                                  "heldout_tree": cfg["tree"]["id"] in cfg.get("splits", {}).get("heldout_trees", [])})
    write_json(os.path.join(seq, "intrinsics.json"), intrinsics_json(intr))
    for variant in ("clean", "noisy"):
        frames = [{"file_path": f"{variant}/rgb/{names[i]}", "depth_file_path": f"{variant}/depth/{names[i]}",
                   "time": float(times[i]), "c2w_blender": c2w_all[i]} for i in range(n)]
        tj = transforms_json(intr, frames, {
            "train_filenames": [f"{variant}/rgb/{names[i]}" for i in train],
            "val_filenames": [f"{variant}/rgb/{names[i]}" for i in test],
            "test_filenames": [f"{variant}/rgb/{names[i]}" for i in test],
            "synthtree_note": "transform_matrix = c2w, OpenGL/Blender camera axes, Blender world (+Z up), metres. "
                              "Train nerfstudio with --orientation-method none --center-method none --auto-scale-poses False "
                              "to keep metric world coordinates.",
        })
        write_json(os.path.join(seq, "transforms.json" if variant == "clean" else "transforms_noisy.json"), tj)

    binfo = json.load(open(os.path.join(raw, "blender_info.json")))
    qc_sum = {
        "frames": n,
        "reproj_max_px": float(max(qc["reproj_max_px"])) if n else None,
        "blender_depth_pass_matches": None,
        "leaf_ids_unknown_px": qc["leaf_ids_unknown"],
        "leafid_semantic_mismatch_px": qc["leafid_semantic_mismatch_px"],
        "mean_leaf_px": float(np.mean(qc["leaf_pixels"])) if n else 0,
        "leaves_ever_visible": int(np.sum(vis_all.max(0) > 0)), "n_leaves": L,
    }
    if qc["depth_vs_blender_planar_med_m"]:
        pl, ry = np.median(qc["depth_vs_blender_planar_med_m"]), np.median(qc["depth_vs_blender_ray_med_m"])
        qc_sum["blender_depth_pass_matches"] = "planar z-depth" if pl < ry else "ray distance"
        qc_sum["blender_depth_vs_ours_median_m"] = {"planar": float(pl), "ray": float(ry)}
    write_json(os.path.join(seq, "qc.json"), {"summary": qc_sum, "per_frame": qc})

    meta = {
        "dataset": cfg["dataset_name"], "sequence_id": cfg["sequence_id"], "kind": cfg["kind"],
        "tree": cfg["tree"], "canonical_tree_dir": os.path.relpath(os.path.join(cfg["tree_dir"], "canonical"), seq),
        "wind_level": cfg["wind_level"], "wind_U_mps": cfg["wind_U"], "wind": cfg["wind"],
        "capture": cfg["capture"], "render": cfg["render"], "sensor": cfg["sensor"], "outputs": outs,
        "intrinsics": intrinsics_json(intr), "n_frames": n, "times_s": times.tolist(),
        "blender": binfo, "synthtree_version": __version__, "git": git_info(), "config_hash": cfg["config_hash"],
        "generated_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "conventions": {
            "world": "Blender world, right-handed, +Z up, metres",
            "poses": "camera-to-world. transforms.json: OpenGL camera axes; poses/c2w_opencv & TUM: OpenCV axes",
            "depth": "z-depth along optical axis (not ray distance); 0 = invalid/no geometry",
            "normals": "camera space (OpenCV axes), oriented towards the camera; png16: n = (v-1)/65534*2-1",
            "leaf_transform": "x_t = R x_canonical + t, gt/leaf_tracks.npz transform_3x4 = [R | t]",
            "flow": "forward flow (du, dv) in pixels from frame i to the next rendered frame, KITTI PNG "
                    "(R,G = flow*64 + 32768, B = valid); no occlusion test (valid = surface exists in frame i)",
        },
        "qc": qc_sum,
    }
    write_json(os.path.join(seq, "metadata.json"), meta)
    if not keep_raw:
        for sub in ("data", "beauty"):
            shutil.rmtree(os.path.join(raw, sub), ignore_errors=True)
    return qc_sum


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--keep-raw", action="store_true", default=None)
    a = ap.parse_args()
    print(json.dumps(process(a.seq, a.keep_raw), indent=2))


if __name__ == "__main__":
    main()
