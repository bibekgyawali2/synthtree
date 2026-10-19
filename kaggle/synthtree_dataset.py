"""SynthTree reader + PyTorch Dataset (self-contained: copy this file into a Kaggle notebook).

Dependencies: numpy, opencv-python (preinstalled on Kaggle); torch only for the Dataset wrapper.

    from synthtree_dataset import SynthTreeSequence, SynthTreeTorch, list_sequences
    seqs = list_sequences("/kaggle/input/synthtree-v1")              # ['tree_00/wind_calm', ...]
    ds = SynthTreeTorch("/kaggle/input/synthtree-v1/tree_01/wind_high", variant="noisy", split="train")
    sample = ds[0]   # dict of tensors: rgb (3,H,W) [0,1], depth (1,H,W) metres, valid, leaf_id, semantic,
                     # normal (3,H,W), K (3,3), c2w_opencv (4,4), c2w_opengl (4,4), time, leaf_T (L,3,4), leaf_visible (L,)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)

Conventions: world +Z up metres; c2w = camera-to-world; OpenCV camera axes for c2w_opencv,
OpenGL/Blender axes for c2w_opengl; depth = z-depth, 0 = invalid.
"""
from __future__ import annotations

import json
import os
import zipfile

import numpy as np

SEMANTIC = {0: "background", 1: "leaf", 2: "wood", 3: "pot", 4: "ground", 5: "room"}


def _imread(path):
    """RGB(A) or single-channel PNG, keeping 16-bit depth. OpenCV if importable, else Pillow
    (e.g. after `pip install nerfstudio` downgrades numpy below what Kaggle's OpenCV needs)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        import cv2
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise FileNotFoundError(path)
        return im[..., ::-1].copy() if im.ndim == 3 else im
    except ImportError:
        with open(path, "rb") as f:
            head = f.read(26)
        bit_depth, color_type = head[24], head[25]
        if bit_depth == 16 and color_type in (2, 6):  # Pillow would silently drop 16-bit RGB to 8 bit
            raise ImportError(f"{path}: 16-bit RGB PNG (normals/flow) needs OpenCV; "
                              "`pip install opencv-python-headless` matching your numpy")
        from PIL import Image
        with Image.open(path) as im:
            a = np.asarray(im)
        return (a.astype(np.uint16) if bit_depth == 16 else a).copy()


def find_dataset_root(name="synthtree-v1", base="/kaggle/input"):
    """Locate the mounted dataset (Kaggle mounts at /kaggle/input/<name> or /kaggle/input/datasets/<user>/<name>)."""
    for r, dirs, _ in os.walk(base):
        if os.path.basename(r) == name or ("trees" in dirs and any(d.startswith("tree_") for d in dirs)):
            return r
        if r.count(os.sep) - base.count(os.sep) >= 3:
            dirs[:] = []
    raise FileNotFoundError(f"dataset '{name}' not found under {base}; is it added as an Input? Found: {os.listdir(base)}")


def list_sequences(root, kind="wind"):
    out = []
    for t in sorted(os.listdir(root)):
        if not t.startswith("tree_") or not os.path.isdir(os.path.join(root, t)):
            continue
        for s in sorted(os.listdir(os.path.join(root, t))):
            if s.startswith(kind + "_"):
                out.append(f"{t}/{s.replace('.zip', '')}")
    return out


def ensure_extracted(path, cache="/kaggle/working/synthtree_cache"):
    """Accept a sequence dir or a per-sequence .zip (extracted once into `cache`)."""
    if os.path.isdir(path):
        return path
    zp = path if path.endswith(".zip") else path + ".zip"
    if not os.path.exists(zp):
        raise FileNotFoundError(f"sequence not found: {path} (nor {zp}). Locate the dataset with find_dataset_root() "
                                f"or `!find /kaggle/input -maxdepth 4 -name tree_01`")
    tree = os.path.basename(os.path.dirname(zp))
    dst = os.path.join(cache, tree, os.path.basename(zp)[:-4])
    if not os.path.isdir(dst):
        with zipfile.ZipFile(zp) as z:
            z.extractall(os.path.join(cache, tree))
    return dst


class SynthTreeSequence:
    """NumPy reader for one sequence. Items are dicts of numpy arrays."""

    def __init__(self, seq_dir, variant="clean", split="all", load=("rgb", "depth", "leaf_id", "semantic", "normal"),
                 with_leaf_gt=True):
        assert variant in ("clean", "noisy")
        self.dir = ensure_extracted(seq_dir)
        self.variant = variant
        self.meta = json.load(open(os.path.join(self.dir, "metadata.json")))
        ij = json.load(open(os.path.join(self.dir, "intrinsics.json")))
        self.K = np.asarray(ij["K"], np.float32)
        self.width, self.height = ij["width"], ij["height"]
        self.depth_units = ij.get("depth_units_m", 0.001)
        self.c2w_opencv = np.load(os.path.join(self.dir, "poses", "c2w_opencv.npy")).astype(np.float32)
        self.times = np.asarray(self.meta["times_s"], np.float64)
        n = len(self.c2w_opencv)
        sp = json.load(open(os.path.join(self.dir, "splits.json")))
        self.frames = list(range(n)) if split == "all" else sp[split]
        stride = self.meta.get("packaged_frame_stride", 1)
        self.frames = [i for i in self.frames if i % stride == 0]
        self.load = set(load)
        self.leaf = None
        if with_leaf_gt:
            z = np.load(os.path.join(self.dir, "gt", "leaf_tracks.npz"))
            self.leaf = {"T": z["transform_3x4"], "visible": z["visible_px"], "leaf_id": z["leaf_id"]}

    def __len__(self):
        return len(self.frames)

    def path(self, product, i):
        var = self.variant if product in ("rgb", "depth") else "clean"
        return os.path.join(self.dir, var, product, f"{i:05d}.png")

    def __getitem__(self, k):
        i = self.frames[k]
        c2w = self.c2w_opencv[i]
        out = {"frame_index": i, "time": float(self.times[i]), "K": self.K, "c2w_opencv": c2w,
               "c2w_opengl": c2w @ np.diag([1, -1, -1, 1]).astype(np.float32)}
        if "rgb" in self.load:
            out["rgb"] = _imread(self.path("rgb", i))[..., :3].astype(np.float32) / 255.0
        if "depth" in self.load:
            d = _imread(self.path("depth", i)).astype(np.float32) * self.depth_units
            out["depth"], out["valid"] = d, d > 0
        if "leaf_id" in self.load:
            out["leaf_id"] = _imread(self.path("leaf_id", i)).astype(np.int64)
        if "semantic" in self.load:
            out["semantic"] = _imread(self.path("semantic", i)).astype(np.int64)
        if "normal" in self.load and os.path.exists(self.path("normal", i)):
            im = _imread(self.path("normal", i)).astype(np.float32)
            v = np.any(im != 0, -1, keepdims=True)
            out["normal"] = np.where(v, (im - 1) / 65534 * 2 - 1, 0).astype(np.float32)
        if "flow" in self.load and os.path.exists(self.path("flow", i)):
            im = _imread(self.path("flow", i))
            out["flow_valid"] = im[..., 2] > 0
            out["flow"] = np.where(out["flow_valid"][..., None], (im[..., :2].astype(np.float32) - 32768) / 64, 0)
        if self.leaf is not None:
            out["leaf_T"] = self.leaf["T"][i]
            out["leaf_visible"] = self.leaf["visible"][i]
        return out

    def canonical_dir(self):
        return os.path.normpath(os.path.join(self.dir, self.meta["canonical_tree_dir"]))

    def leaf_table(self):
        z = np.load(os.path.join(self.canonical_dir(), "leaf_table.npz"))
        return {k: z[k] for k in z.files}


class SynthTreeTorch:
    """torch.utils.data.Dataset wrapper (channels-first tensors)."""

    def __init__(self, *args, **kw):
        import torch  # noqa: F401
        self.seq = SynthTreeSequence(*args, **kw)

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, k):
        import torch
        s = self.seq[k]
        out = {}
        for key, v in s.items():
            if isinstance(v, np.ndarray):
                t = torch.from_numpy(np.ascontiguousarray(v))
                if key in ("rgb", "normal"):
                    t = t.permute(2, 0, 1)
                elif key in ("depth", "valid"):
                    t = t[None]
                out[key] = t
            else:
                out[key] = v
        return out
