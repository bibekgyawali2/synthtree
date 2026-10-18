"""Wood tubes, pot, ground and room meshes (triangles, outward normals)."""
from __future__ import annotations

import numpy as np

from .transforms import normalize, perpendicular_basis


def branch_tube(P, radii, sides):
    """Tube through node positions P (n,3) with radii (n,), closed with a tip vertex.

    Returns verts, faces, vert_node_local (index into P for every vertex).
    Rings use parallel-transported frames so the tube does not twist.
    """
    n = len(P)
    seg = np.diff(P, axis=0)
    tang = np.empty_like(P)
    tang[0], tang[-1] = seg[0], seg[-1]
    if n > 2:
        tang[1:-1] = normalize(seg[:-1]) + normalize(seg[1:])
    tang = normalize(tang)
    e1, _ = perpendicular_basis(tang[0])
    th = 2 * np.pi * np.arange(sides) / sides
    verts, vnode = [], []
    for k in range(n):
        t = tang[k]
        e1 = normalize(e1 - np.dot(e1, t) * t)
        e2 = np.cross(t, e1)
        ring = P[k] + radii[k] * (np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2)
        verts.append(ring)
        vnode += [k] * sides
    verts.append((P[-1] + tang[-1] * radii[-1])[None])
    vnode.append(n - 1)
    verts = np.concatenate(verts)
    faces = []
    for k in range(n - 1):
        for j in range(sides):
            a, b = k * sides + j, k * sides + (j + 1) % sides
            c, d = a + sides, b + sides
            faces += [[a, b, d], [a, d, c]]
    tip = len(verts) - 1
    last = (n - 1) * sides
    for j in range(sides):
        faces.append([last + j, last + (j + 1) % sides, tip])
    return verts, np.asarray(faces, np.int64), np.asarray(vnode, np.int64)


def _revolve(profile_rz, sides, flip=False):
    """Surface of revolution about +Z from a (r, z) profile polyline."""
    th = 2 * np.pi * np.arange(sides) / sides
    rings = [np.stack([r * np.cos(th), r * np.sin(th), np.full(sides, z)], 1) for r, z in profile_rz]
    V = np.concatenate(rings)
    F = []
    for k in range(len(profile_rz) - 1):
        for j in range(sides):
            a, b = k * sides + j, k * sides + (j + 1) % sides
            c, d = a + sides, b + sides
            F += [[a, b, d], [a, d, c]] if not flip else [[a, d, b], [a, c, d]]
    return V, np.asarray(F, np.int64)


def _disk(radius, z, sides, up=True):
    th = 2 * np.pi * np.arange(sides) / sides
    V = np.concatenate([[[0, 0, z]], np.stack([radius * np.cos(th), radius * np.sin(th), np.full(sides, z)], 1)])
    F = [[0, 1 + j, 1 + (j + 1) % sides] if up else [0, 1 + (j + 1) % sides, 1 + j] for j in range(sides)]
    return V, np.asarray(F, np.int64)


def _merge(parts):
    V, F, off = [], [], 0
    for v, f in parts:
        V.append(v)
        F.append(f + off)
        off += len(v)
    return np.concatenate(V), np.concatenate(F)


def pot_meshes(r_top, r_bottom, height, soil_depth, wall=0.008, sides=48):
    """Returns (pot_verts, pot_faces), (soil_verts, soil_faces). Pot rests on z = 0."""
    outer = _revolve([(r_bottom, 0.0), (r_top, height)], sides)  # normals outward
    rim = _revolve([(r_top, height), (r_top - wall, height)], sides)  # annulus facing up
    inner = _revolve([(r_top - wall, height), (r_top - wall - 0.004, height - soil_depth - 0.005)], sides)
    bottom = _disk(r_bottom, 0.0, sides, up=False)
    soil = _disk(r_top - wall - 0.002, height - soil_depth, sides, up=True)
    return _merge([outer, rim, inner, bottom]), soil


def ground_mesh(half_size=15.0):
    V = np.array([[-half_size, -half_size, 0], [half_size, -half_size, 0], [half_size, half_size, 0], [-half_size, half_size, 0]], float)
    return V, np.array([[0, 1, 2], [0, 2, 3]], np.int64)


def room_mesh(half_size=3.0, height=3.0):
    """Four walls and a ceiling with normals facing inwards (floor = ground mesh)."""
    s, h = half_size, height
    c = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0], [-s, -s, h], [s, -s, h], [s, s, h], [-s, s, h]], float)
    quads = [[0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0], [4, 7, 6, 5]]
    F = []
    for a, b, cc, d in quads:
        F += [[a, b, cc], [a, cc, d]]
    return c, np.asarray(F, np.int64)
