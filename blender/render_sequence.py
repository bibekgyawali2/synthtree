"""Render one sequence headless (resumable, frame-level).

    blender -b --factory-startup -P blender/render_sequence.py -- --config <out_dir>/config.json
            [--passes data,beauty] [--max-frames N] [--save-blend]

Pass 1 "data" (Cycles, 1 spp, no AA) -> raw/data/NNNNN.exr (32-bit multilayer) with
AOVs P, N, P_next, P_rest (world space), leaf_id, semantic, plus Blender's Depth.
Pass 2 "beauty" (EEVEE or Cycles) -> raw/beauty/NNNNN_bBB_sSS.exr (linear RGB),
one file per rolling-shutter band b and motion-blur sub-frame s.
Per frame ground truth (computed by the *same* call that moved the vertices):
raw/gt/NNNNN.npz with time, camera c2w read back from Blender, per-leaf 3x4 transforms.

Time handling: wind and camera are evaluated at an explicit time t (seconds);
Blender's frame/sub-frame machinery is not used at all, so there are no frame
offset or fps-base surprises. Frame i of the sequence is at t = i * stride / fps.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

import bl_common as bc  # noqa: E402
from synthtree.camera_path import CameraPath, tree_bbox  # noqa: E402
from synthtree.intrinsics import Intrinsics  # noqa: E402
from synthtree.tree import load_tree  # noqa: E402
from synthtree.wind import WindModel  # noqa: E402


def sequence_setup(cfg, max_frames=None):
    tree = load_tree(os.path.join(cfg["tree_dir"], "canonical", "canonical_tree.npz"))
    intr_full = Intrinsics.from_dict(cfg["sensor"]["rgb"])
    s = float(cfg["render"].get("resolution_scale", 1.0))
    intr = intr_full if s == 1.0 else intr_full.scaled(s)
    seed = cfg["tree"]["seed"]
    path = CameraPath(cfg["capture"], intr_full, tree_bbox(tree), seed)
    wind = WindModel(tree, cfg["wind"], cfg["wind_U"], seed)
    if cfg["kind"] == "ring":
        poses = path.ring_poses()
        times = np.full(len(poses), path.ring_time())
    else:
        times = path.frame_times()
        if max_frames is not None:
            times = times[:max_frames]
        poses = [path.c2w(t) for t in times]
    return tree, intr, path, wind, times, poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--passes", default="data,beauty")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--save-blend", action="store_true")
    a = ap.parse_args(bc.script_args())
    cfg = json.load(open(a.config))
    out = cfg["out_dir"]
    raw = os.path.join(out, "raw")
    for d in ("data", "beauty", "gt"):
        os.makedirs(os.path.join(raw, d), exist_ok=True)

    t0 = time.time()
    tree, intr, path, wind, times, poses = sequence_setup(cfg, a.max_frames)
    n = len(times) if a.max_frames is None else min(len(times), a.max_frames)
    h = bc.build_scene(tree, cfg, room_half_size=path.radius + 1.5)
    scene = h.scene
    bl_cam = bc.set_intrinsics(scene, h.cam, intr)
    bc.log("setup", seconds=round(time.time() - t0, 2), frames=n, leaves=int(len(tree["leaf_id"])),
           blender=bpy.app.version_string)
    rcfg = cfg["render"]
    is_ring = cfg["kind"] == "ring"
    dt = path.dt_frame

    def apply_state(t_wind, c2w, t_next=None):
        st = wind.evaluate(t_wind)
        bc.set_verts(h.wood.data, st.wood_verts)
        bc.set_verts(h.leaves.data, st.leaf_verts)
        if t_next is not None:
            sn = st if t_next == t_wind else wind.evaluate(t_next)
            bc.set_attr(h.wood.data, "P_next", "FLOAT_VECTOR", "POINT", sn.wood_verts)
            bc.set_attr(h.leaves.data, "P_next", "FLOAT_VECTOR", "POINT", sn.leaf_verts)
        h.wood.data.update()
        h.leaves.data.update()
        bc.set_camera_pose(h.cam, c2w)
        return st

    with open(os.path.join(raw, "frames.json"), "w") as f:
        json.dump({"times": [float(t) for t in times[:n]], "dt_frame": dt, "kind": cfg["kind"],
                   "planned_c2w_blender": [np.asarray(p).tolist() for p in poses[:n]]}, f)

    info = {"blender_version": bpy.app.version_string, "blender_build_hash": bpy.app.build_hash.decode() if isinstance(bpy.app.build_hash, bytes) else str(bpy.app.build_hash),
            "python": sys.version.split()[0], "numpy": np.__version__, "camera_blender_params": bl_cam,
            "eevee_engine_id": bc.compat_eevee_id(), "passes": {}}
    passes = a.passes.split(",")

    # ---------------------------------------------------------------- data pass
    if "data" in passes and rcfg.get("data_pass", {}).get("enabled", True):
        dev = bc.configure_data_pass(scene, rcfg.get("device", "auto"))
        tp = time.time()
        rendered = 0
        for i in range(n):
            exr = os.path.join(raw, "data", f"{i:05d}.exr")
            gtp = os.path.join(raw, "gt", f"{i:05d}.npz")
            if os.path.exists(exr) and os.path.exists(gtp):
                continue
            tf = time.time()
            t = float(times[i])
            t_next = t if is_ring else t + dt
            st = apply_state(t, poses[i], t_next)
            c2w = bc.read_camera_pose(h.cam)
            bc.render_to(scene, exr)
            c2w_next = poses[i] if is_ring else path.c2w(t_next)
            np.savez_compressed(gtp + ".part.npz", t=t, t_next=t_next, c2w_blender=c2w, c2w_planned=np.asarray(poses[i]),
                                c2w_next_blender=np.asarray(c2w_next), leaf_T=st.leaf_T34(np.float32),
                                U_eff=st.U_eff, U=cfg["wind_U"])
            os.replace(gtp + ".part.npz", gtp)
            rendered += 1
            bc.log("frame", stage="data", i=i, seconds=round(time.time() - tf, 3))
        info["passes"]["data"] = {"engine": "CYCLES", "device": dev, "samples": 1, "filter": "BOX 0.01px",
                                  "frames_rendered": rendered, "seconds": round(time.time() - tp, 2)}

    # ---------------------------------------------------------------- beauty pass
    if "beauty" in passes:
        dev = bc.configure_beauty_pass(scene, rcfg)
        mb, rs = rcfg.get("motion_blur", {}), rcfg.get("rolling_shutter", {})
        fps = path.fps
        if mb.get("enabled") and not is_ring:
            ns = int(mb.get("subframes", 5))
            shutter = mb.get("shutter_frames", 0.5) / fps
            sub = [((k + 0.5) / ns - 0.5) * shutter for k in range(ns)]
        else:
            sub = [0.0]
        H = intr.height
        if rs.get("enabled") and not is_ring:
            nb = int(rs.get("bands", 8))
            edges = np.linspace(0, H, nb + 1).round().astype(int)
            bands = [(int(edges[b]), int(edges[b + 1])) for b in range(nb)]
            band_off = [((r0 + r1) / 2 / H - 0.5) * rs.get("readout_s", 0.02) for r0, r1 in bands]
        else:
            bands, band_off = [(0, H)], [0.0]
        tp = time.time()
        rendered = 0
        for i in range(n):
            done = os.path.join(raw, "beauty", f"{i:05d}.done")
            if os.path.exists(done):
                continue
            tf = time.time()
            for b, (r0, r1) in enumerate(bands):
                bc.set_border_rows(scene, r0, r1, H)
                for s, so in enumerate(sub):
                    p = os.path.join(raw, "beauty", f"{i:05d}_b{b:02d}_s{s:02d}.exr")
                    if os.path.exists(p):
                        continue
                    t = float(times[i]) + band_off[b] + so
                    c2w = poses[i] if is_ring else path.c2w(t)
                    apply_state(float(times[i]) if is_ring else t, c2w)
                    bc.render_to(scene, p)
            bc.set_border_rows(scene, 0, H, H)
            with open(done, "w") as f:
                json.dump({"bands": bands, "band_time_offsets_s": band_off, "subframe_offsets_s": sub}, f)
            rendered += 1
            bc.log("frame", stage="beauty", i=i, seconds=round(time.time() - tf, 3))
        info["passes"]["beauty"] = {"engine": scene.render.engine, "device": dev, "motion_blur": mb, "rolling_shutter": rs,
                                    "frames_rendered": rendered, "seconds": round(time.time() - tp, 2),
                                    "view_transform": scene.view_settings.view_transform}

    if a.save_blend or rcfg.get("save_blend"):
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(raw, "scene.blend"))
    info_path = os.path.join(raw, "blender_info.json")
    if os.path.exists(info_path):  # merge with earlier (resumed) runs
        old = json.load(open(info_path))
        old.setdefault("passes", {}).update(info["passes"])
        info["passes"] = old["passes"]
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    bc.log("done", seconds=round(time.time() - t0, 2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
