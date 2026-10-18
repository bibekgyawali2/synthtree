"""Build a sequence's scene and save it as a .blend for visual inspection (optional).

    blender -b --factory-startup -P blender/build_scene.py -- --config <seq>/config.json --out tree.blend [--time 1.5]

The saved file shows the tree at time --time (default 0) with the first camera
pose; open it in the Blender UI to look around. Nothing in the pipeline reads it.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402

import bl_common as bc  # noqa: E402
from render_sequence import sequence_setup  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time", type=float, default=0.0)
    a = ap.parse_args(bc.script_args())
    cfg = json.load(open(a.config))
    tree, intr, path, wind, times, poses = sequence_setup(cfg, 1)
    h = bc.build_scene(tree, cfg, room_half_size=path.radius + 1.5)
    bc.set_intrinsics(h.scene, h.cam, intr)
    st = wind.evaluate(a.time)
    bc.set_verts(h.wood.data, st.wood_verts)
    bc.set_verts(h.leaves.data, st.leaf_verts)
    h.wood.data.update()
    h.leaves.data.update()
    bc.set_camera_pose(h.cam, path.c2w(a.time))
    bc.configure_beauty_pass(h.scene, cfg["render"])
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out))
    bc.log("saved", path=os.path.abspath(a.out), U=cfg["wind_U"], t=a.time)


if __name__ == "__main__":
    main()
