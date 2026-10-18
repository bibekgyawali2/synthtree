"""Leaf meshes as real geometry (no alpha cards).

Leaf-local frame: origin = petiole base (attachment point), +X along the midrib,
+Y lateral, +Z = upper-side normal. Templates are built for unit leaf length and
scaled uniformly per leaf. Triangles are wound counter-clockwise seen from +Z so
face normals point to the upper side (needles: outward).
"""
from __future__ import annotations

import numpy as np

from .transforms import rodrigues


def _shape_fn(name):
    fns = {
        "ovate": lambda s: np.sin(np.pi * s ** 0.7) ** 0.9,
        "elliptic": lambda s: np.sin(np.pi * s),
        "lanceolate": lambda s: np.sin(np.pi * s ** 0.5) * (1 - s) ** 0.3,
        "obovate": lambda s: np.sin(np.pi * s ** 1.4) ** 0.9,
    }
    f = fns[name]
    peak = f(np.linspace(1e-4, 1 - 1e-4, 2001)).max()
    return lambda s: f(s) / peak


def blade(length, width, segments=6, shape="ovate", curl=0.0, droop=0.0, x0=0.0):
    """One blade along +X from x0 to x0+length; 3 vertices across (edge, midrib, edge).

    curl: lateral cupping, edge height = curl * half-width (V/U section).
    droop: tip drops by droop * length (quadratic along the midrib).
    """
    f = _shape_fn(shape)
    n = max(3, int(segments))
    s = np.linspace(0.0, 1.0, n + 1)
    x = x0 + s * length
    hw = 0.5 * width * f(s)
    zd = -droop * length * s ** 2
    verts = [[x[0], 0.0, zd[0]]]  # 0: blade base
    for i in range(1, n):  # L, M, R per interior station
        ze = zd[i] + curl * hw[i]
        verts += [[x[i], hw[i], ze], [x[i], 0.0, zd[i]], [x[i], -hw[i], ze]]
    verts.append([x[n], 0.0, zd[n]])  # blade tip
    end = len(verts) - 1
    L = lambda i: 1 + 3 * (i - 1)
    M = lambda i: 2 + 3 * (i - 1)
    R = lambda i: 3 + 3 * (i - 1)
    faces = [[0, R(1), M(1)], [0, M(1), L(1)]]
    for i in range(1, n - 1):
        faces += [[M(i), R(i), R(i + 1)], [M(i), R(i + 1), M(i + 1)],
                  [L(i), M(i), M(i + 1)], [L(i), M(i + 1), L(i + 1)]]
    faces += [[M(n - 1), R(n - 1), end], [L(n - 1), M(n - 1), end]]
    return np.asarray(verts, float), np.asarray(faces, np.int64)


def strip(x0, x1, width):
    """Flat quad along X (petiole / rachis)."""
    h = width / 2
    v = np.array([[x0, -h, 0], [x1, -h, 0], [x1, h, 0], [x0, h, 0]], float)
    return v, np.array([[0, 1, 2], [0, 2, 3]], np.int64)


def needle(length, width, segments=3):
    """Triangular prism tapering to a point, closed at the base. Outward normals."""
    n = max(2, int(segments))
    th = 2 * np.pi * np.arange(3) / 3
    verts, faces = [], []
    for k in range(n):
        x = length * k / n
        r = 0.5 * width * (1 - 0.6 * k / n)
        verts += [[x, r * np.cos(t), r * np.sin(t)] for t in th]
    verts.append([length, 0.0, 0.0])
    tip = len(verts) - 1
    for k in range(n - 1):
        for j in range(3):
            a, b = 3 * k + j, 3 * k + (j + 1) % 3
            c, d = a + 3, b + 3
            faces += [[a, d, c], [a, b, d]]
    last = 3 * (n - 1)
    for j in range(3):
        faces.append([last + j, last + (j + 1) % 3, tip])
    faces.append([0, 2, 1])  # base cap, facing -X
    return np.asarray(verts, float), np.asarray(faces, np.int64)


def _merge(parts):
    V, F, off = [], [], 0
    for v, f in parts:
        V.append(v)
        F.append(f + off)
        off += len(v)
    return np.concatenate(V), np.concatenate(F)


def leaf_template(cfg: dict):
    """Unit-length leaf template (verts, faces) for the leaves config of a preset."""
    kind = cfg["type"]
    wr = cfg.get("width_ratio", 0.5)
    seg = cfg.get("segments", 6)
    curl, droop = cfg.get("curl", 0.0), cfg.get("droop", 0.0)
    pf = cfg.get("petiole_frac", 0.2)
    pw = cfg.get("petiole_width_frac", 0.025)
    if kind in ("broad", "small"):
        parts = [strip(0.0, pf, pw), blade(1.0 - pf, (1.0 - pf) * wr, seg, cfg.get("shape", "ovate"), curl, droop, x0=pf)]
        return _merge(parts)
    if kind == "compound":
        npairs = int(cfg.get("leaflet_pairs", 3))
        ll = cfg.get("leaflet_frac", 0.3)  # leaflet length relative to whole leaf
        parts = [strip(0.0, 1.0 - 0.5 * ll, pw)]
        stations = np.linspace(pf + 0.1, 1.0 - 0.75 * ll, npairs)
        for xs in stations:
            for side in (1, -1):
                v, f = blade(ll, ll * wr, max(3, seg - 2), cfg.get("shape", "elliptic"), curl, droop)
                R = rodrigues(np.array([0, 0, 1.0]), side * np.radians(cfg.get("leaflet_angle_deg", 60)))
                parts.append((v @ R.T + [xs, 0, 0], f))
        v, f = blade(ll, ll * wr, max(3, seg - 2), cfg.get("shape", "elliptic"), curl, droop, x0=1.0 - ll)
        parts.append((v, f))
        return _merge(parts)
    if kind == "needle":
        return needle(1.0, cfg.get("width_ratio", 0.04), cfg.get("segments", 3))
    raise ValueError(f"unknown leaf type {kind!r}")


def face_normals_areas(V, F):
    cr = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    a2 = np.linalg.norm(cr, axis=1)
    return cr / np.maximum(a2[:, None], 1e-30), 0.5 * a2
