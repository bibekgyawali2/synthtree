"""Minimal binary PLY writer/reader with per-face integer properties.

Blender's PLY exporter cannot write per-face attributes such as leaf_id, so the
canonical meshes are written here directly from the generator arrays.
"""
from __future__ import annotations

import numpy as np


def write_ply(path, verts, faces=None, vert_props=None, face_props=None):
    verts = np.asarray(verts, np.float32)
    vert_props = vert_props or {}
    face_props = face_props or {}
    vdt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")] + [(k, _dt(v)) for k, v in vert_props.items()]
    vrec = np.empty(len(verts), dtype=vdt)
    vrec["x"], vrec["y"], vrec["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
    for k, v in vert_props.items():
        vrec[k] = v
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(verts)}",
              "property float x", "property float y", "property float z"]
    header += [f"property {_ply_type(vrec.dtype[k])} {k}" for k in vert_props]
    frec = None
    if faces is not None:
        faces = np.asarray(faces)
        fdt = [("n", "u1"), ("v", "<i4", (3,))] + [(k, _dt(v)) for k, v in face_props.items()]
        frec = np.empty(len(faces), dtype=fdt)
        frec["n"] = 3
        frec["v"] = faces
        for k, v in face_props.items():
            frec[k] = v
        header += [f"element face {len(faces)}", "property list uchar int vertex_indices"]
        header += [f"property {_ply_type(frec.dtype[k])} {k}" for k in face_props]
    header.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(vrec.tobytes())
        if frec is not None:
            f.write(frec.tobytes())


def _dt(v):
    v = np.asarray(v)
    if v.dtype.kind in "iu":
        return "<i4"
    if v.dtype.kind == "f":
        return "<f4"
    raise TypeError(v.dtype)


def _ply_type(dt):
    return {"i": "int", "u": "uint", "f": "float"}[dt.kind] if dt.itemsize == 4 else {"u1": "uchar"}[dt.str[1:]]


_PLY2NP = {"char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2", "int": "<i4", "uint": "<u4",
           "float": "<f4", "double": "<f8", "int8": "i1", "uint8": "u1", "int16": "<i2", "uint16": "<u2",
           "int32": "<i4", "uint32": "<u4", "float32": "<f4", "float64": "<f8"}


def read_ply(path):
    """Reads binary-little-endian PLY (vertex props + triangle faces with scalar props).

    Returns dict: 'vertex' -> structured array, 'face' -> (indices (F,3), structured props) or None.
    Handles 3DGS PLYs (vertex-only, many float properties).
    """
    with open(path, "rb") as f:
        elems, cur = [], None
        line = f.readline().decode().strip()
        assert line == "ply", "not a PLY file"
        fmt = None
        while True:
            line = f.readline().decode().strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element"):
                _, name, n = line.split()
                cur = {"name": name, "n": int(n), "props": []}
                elems.append(cur)
            elif line.startswith("property"):
                parts = line.split()
                if parts[1] == "list":
                    cur["props"].append(("__list__", parts[2], parts[3], parts[4]))
                else:
                    cur["props"].append((parts[2], parts[1]))
            elif line == "end_header":
                break
        assert fmt == "binary_little_endian", f"only binary_little_endian PLY supported, got {fmt}"
        out = {}
        for e in elems:
            if any(p[0] == "__list__" for p in e["props"]):
                # assume triangle lists: fixed record size
                dt = []
                for p in e["props"]:
                    if p[0] == "__list__":
                        dt += [("n", _PLY2NP[p[1]]), ("v", _PLY2NP[p[2]], (3,))]
                    else:
                        dt.append((p[0], _PLY2NP[p[1]]))
                rec = np.frombuffer(f.read(np.dtype(dt).itemsize * e["n"]), dtype=dt)
                assert (rec["n"] == 3).all(), "only triangle faces supported"
                out[e["name"]] = rec
            else:
                dt = [(p[0], _PLY2NP[p[1]]) for p in e["props"]]
                out[e["name"]] = np.frombuffer(f.read(np.dtype(dt).itemsize * e["n"]), dtype=dt)
    return out


def ply_xyz(rec):
    return np.stack([rec["x"], rec["y"], rec["z"]], 1).astype(np.float64)
