#!/usr/bin/env python
"""Generate one canonical (rest-pose) tree and export its ground truth.

    python scripts/make_tree.py --master configs/master.yaml --tree 1
    python scripts/make_tree.py --preset broad_medium --seed 1001 --out data/trees/tree_01

Outputs in <tree_dir>/canonical/:
    canonical_tree.npz   all generator arrays (the single source of truth, read by Blender)
    tree_full.ply        wood + leaves + pot/soil, face props: semantic, leaf_id, branch_id
    leaves.ply           leaves only, face prop leaf_id
    wood.ply             wood only, face prop branch_id
    leaf_table.csv/.npz  id, attachment, centroid, normal, area, parent branch, parent node
    tree_meta.json
Runs with host Python or Blender's Python (no extra deps besides NumPy, PyYAML for --master).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from synthtree.config import load_any, load_master, tree_dir, tree_spec  # noqa: E402
from synthtree.ply import write_ply  # noqa: E402
from synthtree.tree import generate_tree  # noqa: E402

SEM = {"background": 0, "leaf": 1, "wood": 2, "pot": 3, "ground": 4, "room": 5}


def export_tree(tree: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "canonical_tree.npz"), **tree)
    lv, lf, fl = tree["leaf_verts"], tree["leaf_faces"], tree["leaf_face_leaf"]
    wv, wf = tree["wood_verts"], tree["wood_faces"]
    leaf_id_face = tree["leaf_id"][fl]
    write_ply(os.path.join(out_dir, "leaves.ply"), lv, lf, face_props={"leaf_id": leaf_id_face})
    write_ply(os.path.join(out_dir, "wood.ply"), wv, wf, face_props={"branch_id": tree["wood_face_branch"]})
    parts = [(lv, lf, SEM["leaf"], leaf_id_face, tree["leaf_branch"][fl]),
             (wv, wf, SEM["wood"], np.zeros(len(wf), int), tree["wood_face_branch"]),
             (tree["pot_verts"], tree["pot_faces"], SEM["pot"], np.zeros(len(tree["pot_faces"]), int), np.full(len(tree["pot_faces"]), -1)),
             (tree["soil_verts"], tree["soil_faces"], SEM["pot"], np.zeros(len(tree["soil_faces"]), int), np.full(len(tree["soil_faces"]), -1))]
    V, F, sem, lid, bid, off = [], [], [], [], [], 0
    for v, f, s, l, b in parts:
        V.append(v)
        F.append(f + off)
        sem.append(np.full(len(f), s))
        lid.append(l)
        bid.append(b)
        off += len(v)
    write_ply(os.path.join(out_dir, "tree_full.ply"), np.concatenate(V), np.concatenate(F),
              face_props={"semantic": np.concatenate(sem), "leaf_id": np.concatenate(lid), "branch_id": np.concatenate(bid)})
    table = {
        "leaf_id": tree["leaf_id"], "attach": tree["leaf_attach"], "centroid": tree["leaf_centroid"],
        "normal": tree["leaf_normal"], "area_m2": tree["leaf_area"], "parent_branch": tree["leaf_branch"],
        "parent_node": tree["leaf_node"], "frame": tree["leaf_frame"],
    }
    np.savez_compressed(os.path.join(out_dir, "leaf_table.npz"), **table)
    with open(os.path.join(out_dir, "leaf_table.csv"), "w") as f:
        f.write("leaf_id,attach_x,attach_y,attach_z,centroid_x,centroid_y,centroid_z,normal_x,normal_y,normal_z,area_m2,parent_branch,parent_node\n")
        for i in range(len(tree["leaf_id"])):
            a, c, n = tree["leaf_attach"][i], tree["leaf_centroid"][i], tree["leaf_normal"][i]
            f.write(f"{tree['leaf_id'][i]},{a[0]:.6f},{a[1]:.6f},{a[2]:.6f},{c[0]:.6f},{c[1]:.6f},{c[2]:.6f},"
                    f"{n[0]:.6f},{n[1]:.6f},{n[2]:.6f},{tree['leaf_area'][i]:.8f},{tree['leaf_branch'][i]},{tree['leaf_node'][i]}\n")
    meta = json.loads(str(tree["meta_json"]))
    with open(os.path.join(out_dir, "tree_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master")
    ap.add_argument("--tree", type=int, help="tree id in the master config")
    ap.add_argument("--species", default=os.path.join(os.path.dirname(__file__), "..", "configs", "species.yaml"))
    ap.add_argument("--preset")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", help="tree directory (canonical/ is created inside)")
    a = ap.parse_args()
    if a.master:
        m = load_master(a.master)
        t = next(t for t in m["trees"] if int(t["id"]) == a.tree)
        spec = tree_spec(m, t)
        out = a.out or tree_dir(m, a.tree)
    else:
        spec = {"preset": load_any(a.species)[a.preset], "seed": a.seed}
        out = a.out
    tree = generate_tree(spec["preset"], spec["seed"])
    meta = export_tree(tree, os.path.join(out, "canonical"))
    print(json.dumps({k: meta[k] for k in ("seed", "height_m", "n_branches", "n_leaves", "n_leaf_tris", "n_wood_tris")}))


if __name__ == "__main__":
    main()
