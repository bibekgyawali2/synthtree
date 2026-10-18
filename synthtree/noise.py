"""RGB-D sensor noise model (host side; uses OpenCV for filtering/resizing/JPEG).

Depth pipeline (applied to clean z-depth in metres, 0 = invalid):
  1. resolution: nearest-downsample to the sensor's effective depth sampling, back up
  2. depth noise: sigma(z) = a + b z^2 (stereo-like), spatially correlated, OR
     disparity-space noise d = fB/z + N(0, sigma_d), quantised to sub-pixel steps
  3. flying pixels at discontinuities: z = alpha z_fg + (1-alpha) z_bg
  4. holes: random + edge band + thin structures + grazing angle (blob-shaped)
  5. range limits, RGB-depth misregistration, quantisation to sensor units
RGB pipeline: per-frame exposure (AR(1) in EV), blur, shot + read noise in
linear space, JPEG round trip.

All randomness comes from `substream(seed, frame, ...)`: noise is keyed by
(tree seed, frame index), so calm and windy sequences get the same noise draws.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .colour import linear_to_srgb, srgb_to_linear, to_uint8
from .rng import substream

DEFAULT_PROFILE = {
    "name": "d435-like defaults (uncalibrated)",
    "depth": {
        "model": "quadratic",              # quadratic | disparity
        "a_m": 0.0005, "b_per_m": 0.0038,  # sigma(z) = a + b z^2
        "disparity": {"fx_px": 447.0, "baseline_m": 0.050, "sigma_px": 0.08, "subpixel": 0.03125},
        "corr_px": 1.5,                    # spatial correlation (Gaussian sigma, px) of depth noise
        "flying": {"rate": 0.3, "grad_rel": 0.10, "band_px": 1},
        "holes": {"random": 0.002, "edge": 0.20, "edge_band_px": 2, "thin": 0.35, "thin_px": 3,
                  "grazing": 0.6, "grazing_deg": 83.0, "grazing_width_deg": 2.5, "blob_px": 1.2},
        "min_m": 0.2, "max_m": 4.0, "effective_scale": 0.74,
        "misregistration_px": [0.0, 0.0],
        "units_m": 0.001,
    },
    "rgb": {"shot": 0.004, "read": 0.002, "exposure_sigma_ev": 0.08, "exposure_rho": 0.9,
            "blur_sigma_px": 0.5, "jpeg_quality": 92},
}


def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _cv2():
    import cv2
    return cv2


def correlated_normal(shape, sigma_px, rng):
    """Unit-variance Gaussian field with Gaussian spatial correlation."""
    n = rng.standard_normal(shape).astype(np.float32)
    if sigma_px <= 0:
        return n
    cv2 = _cv2()
    k = int(2 * math.ceil(3 * sigma_px) + 1)
    f = cv2.GaussianBlur(n, (k, k), sigma_px, borderType=cv2.BORDER_REFLECT)
    # variance of white noise after a normalised Gaussian filter = 1 / (4 pi sigma^2)
    return f * np.float32(math.sqrt(4 * math.pi) * sigma_px)


def correlated_uniform(shape, sigma_px, rng):
    """U(0,1) marginals with Gaussian spatial correlation (blob-shaped thresholds)."""
    from scipy.special import ndtr  # host only
    g = correlated_normal(shape, sigma_px, rng).astype(np.float64)
    g /= max(g.std(), 1e-9)
    return ndtr(g)


def depth_sigma(z, prof):
    d = prof["depth"]
    if d["model"] == "disparity":
        p = d["disparity"]
        return z ** 2 / (p["fx_px"] * p["baseline_m"]) * p["sigma_px"]
    return d["a_m"] + d["b_per_m"] * z ** 2


def discontinuities(z, grad_rel):
    cv2 = _cv2()
    valid = z > 0
    big = np.where(valid, z, 0).astype(np.float32)
    small = np.where(valid, z, 1e6).astype(np.float32)
    k = np.ones((3, 3), np.uint8)
    zmax = cv2.dilate(big, k)
    zmin = cv2.erode(small, k)
    zmin = np.where(zmin > 1e5, 0, zmin)
    jump = (zmax - zmin) > grad_rel * np.maximum(zmin, 1e-3)
    holes_next = cv2.dilate((~valid).astype(np.uint8), k).astype(bool) & valid
    return (jump & valid) | holes_next


def thin_mask(mask, thin_px):
    """Pixels of `mask` belonging to structures thinner than ~thin_px."""
    if thin_px <= 1:
        return np.zeros_like(mask)
    cv2 = _cv2()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thin_px, thin_px))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
    return mask & ~opened


def grazing_cos(normals_cam, intr_K):
    """|cos| of angle between surface normal and viewing ray per pixel."""
    H, W = normals_cam.shape[:2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    r = np.stack([(u - intr_K[0, 2]) / intr_K[0, 0], (v - intr_K[1, 2]) / intr_K[1, 1], np.ones((H, W))], -1)
    r /= np.linalg.norm(r, axis=-1, keepdims=True)
    return np.abs(np.einsum("hwi,hwi->hw", normals_cam, r))


def simulate_depth(z, prof, rng, normals_cam=None, K=None, semantic=None, return_debug=False):
    """z: (H,W) clean z-depth in metres. Returns noisy depth in metres (0 = invalid)."""
    cv2 = _cv2()
    d = prof["depth"]
    H, W = z.shape
    z = z.astype(np.float64)
    dbg = {}

    # 1. resolution
    s = float(d.get("effective_scale", 1.0))
    if s < 1.0:
        h2, w2 = max(1, round(H * s)), max(1, round(W * s))
        zl = cv2.resize(z.astype(np.float32), (w2, h2), interpolation=cv2.INTER_NEAREST)
        z = cv2.resize(zl, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float64)
    valid = z > 0

    # 2. depth noise
    n = correlated_normal((H, W), d.get("corr_px", 0.0), rng).astype(np.float64)
    if d["model"] == "disparity":
        p = d["disparity"]
        fB = p["fx_px"] * p["baseline_m"]
        with np.errstate(divide="ignore"):
            disp = np.where(valid, fB / np.maximum(z, 1e-6), 0)
        disp = disp + p["sigma_px"] * n
        if p.get("subpixel"):
            disp = np.round(disp / p["subpixel"]) * p["subpixel"]
        zn = np.where(valid & (disp > 0), fB / np.maximum(disp, 1e-9), 0)
    else:
        zn = np.where(valid, z + depth_sigma(z, prof) * n, 0)

    # 3. flying pixels
    fl = d["flying"]
    edges = discontinuities(z, fl["grad_rel"])
    if fl.get("band_px", 1) > 1:
        kb = np.ones((fl["band_px"], fl["band_px"]), np.uint8)
        edges = cv2.dilate(edges.astype(np.uint8), kb).astype(bool) & valid
    k5 = np.ones((5, 5), np.uint8)
    zfg = cv2.erode(np.where(valid, z, 1e6).astype(np.float32), k5).astype(np.float64)
    zbg = cv2.dilate(np.where(valid, z, 0).astype(np.float32), k5).astype(np.float64)
    fly = edges & (rng.random((H, W)) < fl["rate"]) & (zfg < 1e5)
    alpha = rng.random((H, W))
    zn = np.where(fly, alpha * zfg + (1 - alpha) * zbg, zn)
    dbg["edges"], dbg["flying"] = edges, fly

    # 4. holes (blob-shaped via a correlated uniform field, exact marginal probability)
    ho = d["holes"]
    p = np.full((H, W), ho["random"], np.float64)
    band = cv2.dilate(edges.astype(np.uint8), np.ones((2 * ho["edge_band_px"] + 1,) * 2, np.uint8)).astype(bool)
    p += ho["edge"] * band
    if semantic is not None:
        thin = thin_mask((semantic == 1) | (semantic == 2), int(ho["thin_px"]))
        p += ho["thin"] * thin
        dbg["thin"] = thin
    if normals_cam is not None and K is not None:
        c = grazing_cos(normals_cam, K)
        ang = np.degrees(np.arccos(np.clip(c, 0, 1)))
        p += ho["grazing"] / (1 + np.exp(-(ang - ho["grazing_deg"]) / max(ho["grazing_width_deg"], 1e-3)))
    u = correlated_uniform((H, W), ho.get("blob_px", 0.0), rng)
    holes = u < np.clip(p, 0, 1)
    zn = np.where(holes, 0, zn)
    dbg["holes"] = holes

    # 5. range, misregistration, quantisation
    zn = np.where((zn < d["min_m"]) | (zn > d["max_m"]), 0, zn)
    dx, dy = d.get("misregistration_px", [0, 0])
    if dx or dy:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        zn = cv2.warpAffine(zn.astype(np.float32), M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0).astype(np.float64)
    q = d["units_m"]
    zn = np.round(zn / q) * q
    return (zn, dbg) if return_debug else zn


def exposure_track(n_frames, prof, rng):
    r = prof["rgb"]
    ev = np.zeros(n_frames)
    rho, s = r["exposure_rho"], r["exposure_sigma_ev"]
    for i in range(n_frames):
        ev[i] = (rho * ev[i - 1] if i else 0.0) + math.sqrt(1 - rho ** 2) * s * rng.standard_normal()
    return ev


def simulate_rgb(rgb8, prof, rng, ev=0.0):
    cv2 = _cv2()
    r = prof["rgb"]
    lin = srgb_to_linear(rgb8.astype(np.float64) / 255.0) * (2.0 ** ev)
    if r["blur_sigma_px"] > 0:
        lin = cv2.GaussianBlur(lin, (0, 0), r["blur_sigma_px"])
    var = r["shot"] * np.clip(lin, 0, None) + r["read"] ** 2
    lin = lin + np.sqrt(var) * rng.standard_normal(lin.shape)
    out = to_uint8(linear_to_srgb(lin))
    q = int(r.get("jpeg_quality", 0))
    if 0 < q < 100:
        ok, buf = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
    return np.ascontiguousarray(out)


def frame_rng(seed, frame, what):
    return substream(seed, "noise", what, int(frame))
