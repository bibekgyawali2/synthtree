"""Host-side image I/O (EXR via the `OpenEXR` package, PNG via OpenCV).

Not importable inside Blender (no OpenEXR/cv2 there) - and never needed there.
"""
from __future__ import annotations

import numpy as np


def read_exr(path) -> dict:
    """All channels of a (multilayer) EXR as float32 arrays keyed by full channel name.

    Blender <= 4.5 writes multilayer EXR as ONE part with 'Layer.Pass.C' channels;
    Blender 5.x writes a MULTI-PART file (one part per pass). Readers that only
    look at part 0 silently see just 'Combined' - so iterate over all parts.
    """
    import OpenEXR
    out = {}
    with OpenEXR.File(str(path), separate_channels=True) as f:
        for part in f.parts:
            for name, ch in part.channels.items():
                out[name] = np.asarray(ch.pixels, dtype=np.float32)
    return out


_ORDER = {c: i for i, c in enumerate("RGBAXYZWV")}


def exr_passes(channels: dict, layer_prefix: str = "") -> dict:
    """Group 'ViewLayer.Pass.C' channels into {'Pass': (H,W[,C]) array}.

    Blender names multilayer channels '<view layer>.<pass>.<channel>'; AOV value
    passes have a single channel. Channel order follows R,G,B,A / X,Y,Z,W.
    """
    groups = {}
    for name, arr in channels.items():
        parts = name.split(".")
        if len(parts) >= 3:
            pas, ch = parts[-2], parts[-1]
        elif len(parts) == 2:
            pas, ch = parts
        else:
            pas, ch = "", parts[0]
        groups.setdefault(pas, []).append((ch, arr))
    out = {}
    for pas, chans in groups.items():
        chans.sort(key=lambda c: _ORDER.get(c[0], 99))
        out[pas] = chans[0][1] if len(chans) == 1 else np.stack([c[1] for c in chans], axis=-1)
    return out


def write_exr(path, channels: dict, half=False):
    """Write single-part scanline EXR, ZIP compressed. channels: name -> (H,W) array."""
    import OpenEXR
    dt = np.float16 if half else np.float32
    chans = {k: np.ascontiguousarray(v.astype(dt)) for k, v in channels.items()}
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    with OpenEXR.File(header, chans) as f:
        f.write(str(path))


def read_png(path):
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 3:
        im = im[..., ::-1]  # BGR(A) -> RGB(A)
    return np.ascontiguousarray(im)


def write_png(path, img, compression=3):
    """uint8 or uint16, (H,W) or (H,W,3) RGB."""
    import cv2
    img = np.asarray(img)
    assert img.dtype in (np.uint8, np.uint16), img.dtype
    if img.ndim == 3:
        img = img[..., ::-1]
    ok = cv2.imwrite(str(path), np.ascontiguousarray(img), [cv2.IMWRITE_PNG_COMPRESSION, int(compression)])
    if not ok:
        raise IOError(f"failed to write {path}")


def read_depth_png_m(path, units_m=0.001):
    return read_png(path).astype(np.float32) * units_m


def encode_normals16(n):
    """Unit normals in [-1, 1] -> uint16 RGB; background (0,0,0) -> 0."""
    valid = np.any(n != 0, axis=-1, keepdims=True)
    return np.where(valid, np.round((np.clip(n, -1, 1) + 1) * 0.5 * 65534) + 1, 0).astype(np.uint16)


def decode_normals16(im):
    valid = np.any(im != 0, axis=-1, keepdims=True)
    return np.where(valid, (im.astype(np.float32) - 1) / 65534 * 2 - 1, 0.0)


def encode_flow_kitti(flow, valid):
    """KITTI 16-bit flow PNG: R,G = flow * 64 + 2^15, B = valid (1/64 px precision, +-512 px)."""
    f = np.clip(np.round(flow * 64.0 + 32768.0), 0, 65535)
    out = np.zeros(flow.shape[:2] + (3,), np.uint16)
    out[..., 0] = np.where(valid, f[..., 0], 0)
    out[..., 1] = np.where(valid, f[..., 1], 0)
    out[..., 2] = valid.astype(np.uint16)
    return out


def decode_flow_kitti(im):
    valid = im[..., 2] > 0
    flow = (im[..., :2].astype(np.float32) - 32768.0) / 64.0
    flow[~valid] = 0
    return flow, valid
