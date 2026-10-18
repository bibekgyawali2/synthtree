"""Orbit camera trajectory with smooth seeded hand-held jitter, plus a static ring.

All poses are analytic functions of time t (seconds), so sub-frame samples for
motion blur / rolling shutter are consistent with the frame poses. Poses depend
only on the capture config and the *tree* seed, never on wind, so calm and windy
sequences of a tree have bit-identical cameras.
"""
from __future__ import annotations

import math

import numpy as np

from .intrinsics import Intrinsics
from .rng import substream
from .transforms import look_at_blender, rodrigues

DEFAULT_CAPTURE = {
    "fps": 30.0,
    "duration_s": 20.0,
    "frame_stride": 4,  # render every k-th video frame; wind time stays real-time
    "orbit": {"radius_m": "auto", "heights_m": "auto", "loops": 1, "start_angle_deg": 0.0,
              "look_at_z": "auto", "fit_margin": 1.15, "min_radius_m": 0.6, "height_transition_frac": 0.15},
    "jitter": {"enabled": True, "translation_m": 0.01, "rotation_deg": 0.6, "freq_hz": [0.1, 1.2], "n_components": 4},
    "ring": {"enabled": False, "views_per_ring": 24, "heights_m": "auto", "time_s": "mid"},
}


def _smooth_noise(rng, n_axes, f_range, n_comp):
    f = rng.uniform(*f_range, (n_axes, n_comp))
    ph = rng.uniform(0, 2 * np.pi, (n_axes, n_comp))
    a = rng.normal(0, 1, (n_axes, n_comp)) / np.sqrt(f)
    norm = np.sqrt(0.5 * np.sum(a ** 2, axis=1, keepdims=True))  # unit RMS per axis

    def fn(t):
        return np.sum(a * np.sin(2 * np.pi * f * t + ph), axis=1) / norm[:, 0]
    return fn


class CameraPath:
    def __init__(self, capture: dict, intr: Intrinsics, tree_bbox: np.ndarray, seed: int):
        from .tree import deep_merge
        self.cfg = deep_merge(DEFAULT_CAPTURE, capture or {})
        self.intr = intr
        o = self.cfg["orbit"]
        lo, hi = np.asarray(tree_bbox[0], float), np.asarray(tree_bbox[1], float)
        self.center_xy = 0.5 * (lo[:2] + hi[:2])
        r_tree = 0.5 * float(np.max(hi[:2] - lo[:2]))
        h_tree = float(hi[2] - lo[2])
        self.look_z = 0.5 * (lo[2] + hi[2]) if o["look_at_z"] == "auto" else float(o["look_at_z"])
        if o["radius_m"] == "auto":
            tv, th = math.tan(math.radians(intr.vfov_deg) / 2), math.tan(math.radians(intr.hfov_deg) / 2)
            d = o["fit_margin"] * max(0.5 * h_tree / tv, r_tree / th) + r_tree
            self.radius = max(d, o["min_radius_m"])
        else:
            self.radius = float(o["radius_m"])
        if o["heights_m"] == "auto":
            self.heights = [self.look_z + 0.10 * h_tree] if o["loops"] <= 1 else \
                list(np.linspace(self.look_z - 0.15 * h_tree, self.look_z + 0.30 * h_tree, int(o["loops"])))
        else:
            self.heights = [float(h) for h in o["heights_m"]]
        j = self.cfg["jitter"]
        rng = substream(seed, "camera", "jitter")
        self._jt = _smooth_noise(rng, 3, j["freq_hz"], j["n_components"])
        self._jr = _smooth_noise(rng, 3, j["freq_hz"], j["n_components"])

    # -- timing -------------------------------------------------------------
    @property
    def fps(self):
        return float(self.cfg["fps"])

    @property
    def dt_frame(self):
        """Time between consecutive *rendered* frames."""
        return self.cfg["frame_stride"] / self.fps

    def frame_times(self, n_frames=None):
        c = self.cfg
        n = int(math.floor(c["duration_s"] * c["fps"] / c["frame_stride"])) if n_frames is None else int(n_frames)
        return np.arange(n) * self.dt_frame

    # -- poses ----------------------------------------------------------------
    def _height(self, t):
        o = self.cfg["orbit"]
        loops = max(1, int(o["loops"]))
        T = self.cfg["duration_s"]
        x = np.clip(t / T, 0, 1) * loops
        k = min(int(x), loops - 1)
        h = self.heights[k % len(self.heights)]
        frac = x - k
        tf = o["height_transition_frac"]
        if k + 1 < loops and frac > 1 - tf:
            s = (frac - (1 - tf)) / tf
            s = s * s * (3 - 2 * s)
            h = (1 - s) * h + s * self.heights[(k + 1) % len(self.heights)]
        return h

    def c2w(self, t: float, jitter: bool = True) -> np.ndarray:
        """Blender-convention camera-to-world at time t."""
        if self.cfg.get("freeze_camera_at_s") is not None:  # previews: only the tree moves
            t = float(self.cfg["freeze_camera_at_s"])
        o = self.cfg["orbit"]
        ang = math.radians(o["start_angle_deg"]) + 2 * math.pi * max(1, int(o["loops"])) * t / self.cfg["duration_s"]
        eye = np.array([self.center_xy[0] + self.radius * math.cos(ang),
                        self.center_xy[1] + self.radius * math.sin(ang), self._height(t)])
        target = np.array([self.center_xy[0], self.center_xy[1], self.look_z])
        j = self.cfg["jitter"]
        if jitter and j["enabled"]:
            eye = eye + j["translation_m"] * self._jt(t)
        M = look_at_blender(eye, target)
        if jitter and j["enabled"]:
            r = np.radians(j["rotation_deg"]) * self._jr(t)
            Rj = rodrigues(np.array([1.0, 0, 0]), r[0]) @ rodrigues(np.array([0, 1.0, 0]), r[1]) @ rodrigues(np.array([0, 0, 1.0]), r[2])
            M[:3, :3] = M[:3, :3] @ Rj  # perturbation in the camera frame
        return M

    def ring_poses(self):
        rc = self.cfg["ring"]
        hs = self.heights if rc["heights_m"] == "auto" else [float(h) for h in rc["heights_m"]]
        if rc["heights_m"] == "auto" and len(hs) == 1:
            hs = [self.look_z - 0.1 * self.radius, self.look_z + 0.35 * self.radius]
        target = np.array([self.center_xy[0], self.center_xy[1], self.look_z])
        poses = []
        for ri, h in enumerate(hs):
            for k in range(int(rc["views_per_ring"])):
                ang = 2 * math.pi * (k + 0.5 * ri) / rc["views_per_ring"]
                eye = [self.center_xy[0] + self.radius * math.cos(ang), self.center_xy[1] + self.radius * math.sin(ang), h]
                poses.append(look_at_blender(eye, target))
        return poses

    def ring_time(self):
        t = self.cfg["ring"]["time_s"]
        return 0.5 * self.cfg["duration_s"] if t == "mid" else float(t)


def tree_bbox(tree: dict, include_pot=True):
    pts = [tree["wood_verts"], tree["leaf_verts"]]
    if include_pot:
        pts.append(tree["pot_verts"])
    P = np.concatenate(pts)
    return np.stack([P.min(0), P.max(0)])
