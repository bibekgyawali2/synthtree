"""Config loading, master-config expansion and hashing.

Host side reads YAML (master + presets) and writes one fully *resolved* JSON per
sequence; Blender only ever reads that JSON (Blender's Python has no PyYAML).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os

from .tree import deep_merge


def load_any(path):
    with open(path) as f:
        if str(path).endswith((".yaml", ".yml")):
            import yaml  # host only
            return yaml.safe_load(f)
        return json.load(f)


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def config_hash(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()[:16]


def _rel(base, p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(os.path.dirname(base), p))


def _load_with_extends(path):
    m = load_any(path)
    for k in ("sensor", "species", "output_root"):  # paths are relative to the file that defines them
        if isinstance(m.get(k), str):
            m[k] = os.path.abspath(_rel(path, m[k]))
    if isinstance(m.get("smoke", {}).get("output_root"), str):
        m["smoke"]["output_root"] = os.path.abspath(_rel(path, m["smoke"]["output_root"]))
    if m.get("extends"):
        base = _load_with_extends(_rel(path, m.pop("extends")))
        m = deep_merge(base, m)
    return m


def load_master(path, smoke=False):
    m = _load_with_extends(path)
    if isinstance(m.get("sensor"), str):
        m["sensor"] = load_any(_rel(path, m["sensor"]))
    if isinstance(m.get("species"), str):
        m["species"] = load_any(_rel(path, m["species"]))
    if smoke and m.get("smoke"):
        m = deep_merge(m, m["smoke"])
    m["output_root"] = os.path.abspath(_rel(path, m.get("output_root", "../data/output")))
    m["_master_path"] = os.path.abspath(path)
    return m


def tree_dir(master, tid):
    return os.path.join(master["output_root"], f"tree_{int(tid):02d}")


def tree_spec(master, t):
    preset = deep_merge(master["species"][t["preset"]], t.get("preset_overrides", {}))
    return {"id": int(t["id"]), "preset_name": t["preset"], "seed": int(t["seed"]), "preset": preset}


def expand_master(master) -> list:
    """One resolved config per (tree, wind level[, ring]) sequence."""
    seqs = []
    levels = master["wind"]["levels"]
    trees = master["trees"]
    only = master.get("only_trees")
    for t in trees:
        if only is not None and int(t["id"]) not in only:
            continue
        spec = tree_spec(master, t)
        for level_name, U in levels.items():
            if master.get("only_levels") and level_name not in master["only_levels"]:
                continue
            kinds = ["orbit"] + (["ring"] if master["capture"].get("ring", {}).get("enabled") else [])
            for kind in kinds:
                prefix = "wind" if kind == "orbit" else "ring"
                seq_id = f"tree_{spec['id']:02d}/{prefix}_{level_name}"
                cfg = {
                    "dataset_name": master.get("dataset_name", "synthtree"),
                    "sequence_id": seq_id, "kind": kind,
                    "tree": spec,
                    "tree_dir": tree_dir(master, spec["id"]),
                    "out_dir": os.path.join(master["output_root"], seq_id),
                    "wind": deep_merge({k: v for k, v in master["wind"].items() if k != "levels"},
                                       t.get("wind_overrides", {})),
                    "wind_level": level_name, "wind_U": float(U),
                    "sensor": master["sensor"],
                    "capture": deep_merge(master["capture"], t.get("capture_overrides", {})),
                    "render": master["render"],
                    "outputs": master.get("outputs", {}),
                    "noise": master.get("noise", {}),
                    "splits": master.get("splits", {}),
                }
                cfg["config_hash"] = config_hash(cfg)
                seqs.append(cfg)
    return seqs


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    import numpy as np
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


def copy_cfg(c):
    return copy.deepcopy(c)
