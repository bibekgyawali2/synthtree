"""Pinhole intrinsics and their mapping onto a Blender camera.

Pixel convention: OpenCV, i.e. the centre of pixel (u, v) is at integer
coordinates and the image centre is ((W-1)/2, (H-1)/2).

Derivation of the Blender mapping (BKE_camera_params_compute_viewplane, sensor
fit HORIZONTAL, pixel aspect xasp/yasp, ycor = yasp/xasp):
    fx = lens * W * xasp / sensor_width
    fy = fx / ycor
    cx = W/2 - 0.5 - shift_x * W * xasp
    cy = H/2 - 0.5 + shift_y * W * xasp / ycor
The inverse is implemented below and checked by the reprojection validator,
which projects the rendered world-position pass with K and must hit pixel
centres to ~1e-3 px.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_dict(cls, d: dict) -> "Intrinsics":
        if "fx" in d:
            return cls(int(d["width"]), int(d["height"]), float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"]))
        return cls.from_fov(int(d["width"]), int(d["height"]), float(d["hfov_deg"]), d.get("vfov_deg"))

    @classmethod
    def from_fov(cls, width, height, hfov_deg, vfov_deg=None):
        fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        fy = fx if vfov_deg is None else (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
        return cls(width, height, fx, fy, (width - 1) / 2.0, (height - 1) / 2.0)

    def to_dict(self):
        return asdict(self)

    def scaled(self, s: float) -> "Intrinsics":
        """Intrinsics of the same camera rendered at a resolution scale s (pixel-corner model)."""
        W, H = max(1, round(self.width * s)), max(1, round(self.height * s))
        sx, sy = W / self.width, H / self.height
        return Intrinsics(W, H, self.fx * sx, self.fy * sy, (self.cx + 0.5) * sx - 0.5, (self.cy + 0.5) * sy - 0.5)

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @property
    def hfov_deg(self):
        return math.degrees(2 * math.atan(self.width / (2 * self.fx)))

    @property
    def vfov_deg(self):
        return math.degrees(2 * math.atan(self.height / (2 * self.fy)))


def to_blender(intr: Intrinsics, sensor_width_mm: float = 36.0) -> dict:
    ratio = intr.fx / intr.fy  # = ycor
    if ratio >= 1.0:
        xasp, yasp = 1.0, ratio
    else:
        xasp, yasp = 1.0 / ratio, 1.0
    ycor = yasp / xasp
    W, H = intr.width, intr.height
    return {
        "lens": intr.fx * sensor_width_mm / (W * xasp),
        "sensor_width": sensor_width_mm,
        "sensor_fit": "HORIZONTAL",
        "shift_x": -(intr.cx - (W - 1) / 2.0) / (W * xasp),
        "shift_y": (intr.cy - (H - 1) / 2.0) * ycor / (W * xasp),
        "pixel_aspect_x": xasp,
        "pixel_aspect_y": yasp,
        "resolution_x": W,
        "resolution_y": H,
    }


def from_blender(p: dict) -> Intrinsics:
    W, H = p["resolution_x"], p["resolution_y"]
    xasp, yasp = p["pixel_aspect_x"], p["pixel_aspect_y"]
    ycor = yasp / xasp
    fx = p["lens"] * W * xasp / p["sensor_width"]
    return Intrinsics(W, H, fx, fx / ycor,
                      W / 2 - 0.5 - p["shift_x"] * W * xasp,
                      H / 2 - 0.5 + p["shift_y"] * W * xasp / ycor)


def pixel_grid(W, H):
    u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    return u, v


def project(intr: Intrinsics, c2w_cv, X):
    """World points X (...,3) -> (u, v, z) with OpenCV-convention c2w."""
    w2c = np.linalg.inv(c2w_cv)
    Xc = np.einsum("ij,...j->...i", w2c[:3, :3], X) + w2c[:3, 3]
    z = Xc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intr.fx * Xc[..., 0] / z + intr.cx
        v = intr.fy * Xc[..., 1] / z + intr.cy
    return u, v, z


def backproject(intr: Intrinsics, c2w_cv, depth):
    """z-depth image (H,W) metres -> world points (H,W,3)."""
    u, v = pixel_grid(intr.width, intr.height)
    Xc = np.stack([(u - intr.cx) / intr.fx * depth, (v - intr.cy) / intr.fy * depth, depth], axis=-1)
    return np.einsum("ij,...j->...i", c2w_cv[:3, :3], Xc) + c2w_cv[:3, 3]


def ray_dirs_cam(intr: Intrinsics):
    """Unnormalised OpenCV camera-space ray directions with z = 1 per pixel."""
    u, v = pixel_grid(intr.width, intr.height)
    return np.stack([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, np.ones_like(u)], axis=-1)
