"""Colour transforms.

RGB products are written as 8-bit sRGB = the IEC 61966-2-1 piecewise transfer
function applied to clipped linear Rec.709 scene values. This equals Blender's
"Standard" view transform with look None, exposure 0, gamma 1 on an sRGB display
(verified by blender/check_render_setup.py). Filmic/AgX are NOT used because they
tone-map, which a real sensor's ISP does differently anyway; if you prefer AgX,
render beauty to PNG in Blender instead of converting here.
"""
from __future__ import annotations

import numpy as np


def linear_to_srgb(x):
    x = np.clip(np.asarray(x, np.float64), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def srgb_to_linear(x):
    x = np.clip(np.asarray(x, np.float64), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def to_uint8(x01):
    return np.clip(np.round(np.asarray(x01) * 255.0), 0, 255).astype(np.uint8)
