#!/usr/bin/env python
"""Compare Blender's Standard-view PNG with our linear->sRGB conversion of the same render.

    python scripts/check_colour.py /tmp/cm_check     (after blender/check_render_setup.py)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from synthtree.colour import linear_to_srgb, to_uint8  # noqa: E402
from synthtree.imageio import exr_passes, read_exr, read_png  # noqa: E402

d = sys.argv[1]
ref = read_png(os.path.join(d, "blender_standard.png"))[..., :3].astype(int)
lin = next(v for v in exr_passes(read_exr(os.path.join(d, "linear.exr"))).values() if v.ndim == 3)[..., :3]
ours = to_uint8(linear_to_srgb(lin)).astype(int)
diff = np.abs(ours - ref)
res = {"max_abs_diff_8bit": int(diff.max()), "mean_abs_diff": float(diff.mean()), "frac_pixels_gt1": float((diff > 1).mean()),
       "ok": bool((diff > 1).mean() < 1e-3)}
res.update(json.load(open(os.path.join(d, "settings.json"))))
print(json.dumps(res, indent=1))
sys.exit(0 if res["ok"] else 1)
