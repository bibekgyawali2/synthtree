"""Blender-side check: our numpy sRGB conversion == Blender's 'Standard' view transform.

    blender -b --factory-startup -P blender/check_render_setup.py -- --config <seq>/config.json --out /tmp/cm_check

Renders frame 0 of a sequence twice with the beauty settings: once to an 8-bit PNG
(Blender applies the Standard view transform) and once to a linear EXR. Then run
    python scripts/check_colour.py /tmp/cm_check
which converts the EXR with synthtree.colour.linear_to_srgb and compares (max |diff| <= 1 LSB).
Also prints the engine ids / colour-management settings actually in effect.
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
    a = ap.parse_args(bc.script_args())
    cfg = json.load(open(a.config))
    os.makedirs(a.out, exist_ok=True)
    tree, intr, path, wind, times, poses = sequence_setup(cfg, 1)
    h = bc.build_scene(tree, cfg, room_half_size=path.radius + 1.5)
    bc.set_intrinsics(h.scene, h.cam, intr)
    st = wind.evaluate(float(times[0]))
    bc.set_verts(h.wood.data, st.wood_verts)
    bc.set_verts(h.leaves.data, st.leaf_verts)
    bc.set_camera_pose(h.cam, poses[0])
    bc.configure_beauty_pass(h.scene, cfg["render"])
    bc.render_to(h.scene, os.path.join(a.out, "linear.exr"))
    ims = h.scene.render.image_settings
    if hasattr(ims, "media_type"):
        ims.media_type = "IMAGE"
    ims.file_format = "PNG"
    ims.color_mode = "RGB"
    ims.color_depth = "8"
    bc.render_to(h.scene, os.path.join(a.out, "blender_standard.png"))
    s = h.scene
    info = {"blender": bpy.app.version_string, "engine": s.render.engine, "view_transform": s.view_settings.view_transform,
            "look": s.view_settings.look, "exposure": s.view_settings.exposure, "gamma": s.view_settings.gamma,
            "display_device": s.display_settings.display_device, "dither": s.render.dither_intensity}
    json.dump(info, open(os.path.join(a.out, "settings.json"), "w"), indent=2)
    bc.log("check_render_setup", **info)


if __name__ == "__main__":
    main()
