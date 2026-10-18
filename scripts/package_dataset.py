#!/usr/bin/env python
"""Build the Kaggle dataset folder (+ dataset card, licence, splits, size report).

    python scripts/package_dataset.py --master configs/master.yaml --out data/kaggle/synthtree-v1 \
        [--variants clean,noisy] [--no-flow] [--no-normals] [--frame-stride 1] [--zip] \
        [--kaggle-id your-username/synthtree-v1]
    python scripts/package_dataset.py --master configs/master.yaml --estimate     # sizes only

Layout produced:
    <out>/README.md  LICENSE  dataset-metadata.json  splits.json  size_report.json
    <out>/trees/tree_XX/canonical/{canonical_tree.npz, tree_full.ply, leaves.ply, wood.ply, leaf_table.*}
    <out>/tree_XX/wind_<level>/{clean,noisy}/...  poses/ gt/ metadata.json transforms*.json intrinsics.json
      (or <out>/tree_XX/wind_<level>.zip with --zip)
Upload:  kaggle datasets create -p <out> --dir-mode zip     (first time)
         kaggle datasets version -p <out> -m "msg" --dir-mode zip
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from synthtree.config import expand_master, load_master, write_json  # noqa: E402

SEQ_FILES = ["metadata.json", "intrinsics.json", "splits.json", "qc.json", "validation.json", "config.json"]


def dir_bytes(d):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs) if os.path.isdir(d) else 0


def copy_seq(src, dst, variants, flow, normals, stride):
    os.makedirs(dst, exist_ok=True)
    meta = json.load(open(os.path.join(src, "metadata.json")))
    n = meta["n_frames"]
    keep = set(range(0, n, stride))
    for f in SEQ_FILES:
        if os.path.exists(os.path.join(src, f)):
            shutil.copy2(os.path.join(src, f), dst)
    for v in variants:
        tj = json.load(open(os.path.join(src, "transforms.json" if v == "clean" else "transforms_noisy.json")))
        tj["frames"] = [fr for k, fr in enumerate(tj["frames"]) if k in keep]
        for key in ("train_filenames", "val_filenames", "test_filenames"):
            if key in tj:
                tj[key] = [p for p in tj[key] if int(os.path.basename(p)[:5]) in keep]
        write_json(os.path.join(dst, "transforms.json" if v == "clean" else "transforms_noisy.json"), tj)
    subs = [("clean", "rgb"), ("clean", "depth"), ("clean", "leaf_id"), ("clean", "semantic"), ("clean", "depth_exr")]
    if normals:
        subs.append(("clean", "normal"))
    if flow:
        subs.append(("clean", "flow"))
    if "noisy" in variants:
        subs += [("noisy", "rgb"), ("noisy", "depth")]
    for a, b in subs:
        s = os.path.join(src, a, b)
        if not os.path.isdir(s):
            continue
        d = os.path.join(dst, a, b)
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(s):
            if int(f[:5]) in keep:
                shutil.copy2(os.path.join(s, f), d)
    if os.path.exists(os.path.join(src, "noisy", "noise_meta.json")) and "noisy" in variants:
        shutil.copy2(os.path.join(src, "noisy", "noise_meta.json"), os.path.join(dst, "noisy"))
    shutil.copytree(os.path.join(src, "poses"), os.path.join(dst, "poses"), dirs_exist_ok=True)
    shutil.copytree(os.path.join(src, "gt"), os.path.join(dst, "gt"), dirs_exist_ok=True)
    # the packaged canonical tree lives in <out>/trees/; fix the relative pointer
    meta["canonical_tree_dir"] = f"../../trees/tree_{meta['tree']['id']:02d}/canonical"
    meta["packaged_frame_stride"] = stride
    write_json(os.path.join(dst, "metadata.json"), meta)


def zip_dir(d):
    zp = d.rstrip("/") + ".zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:  # PNGs are already compressed
        for r, _, fs in os.walk(d):
            for f in fs:
                p = os.path.join(r, f)
                z.write(p, os.path.relpath(p, os.path.dirname(d)))
    shutil.rmtree(d)
    return zp


def render_card(template, ctx):
    s = open(template).read()
    for k, v in ctx.items():
        s = s.replace("{{" + k + "}}", str(v))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", required=True)
    ap.add_argument("--out")
    ap.add_argument("--variants", default="clean,noisy")
    ap.add_argument("--no-flow", action="store_true")
    ap.add_argument("--no-normals", action="store_true")
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--zip", action="store_true", help="one zip per sequence")
    ap.add_argument("--kaggle-id", default="your-kaggle-username/synthtree-v1")
    ap.add_argument("--title", default="SynthTree: paired calm/windy synthetic RGB-D foliage")
    ap.add_argument("--estimate", action="store_true")
    a = ap.parse_args()
    m = load_master(a.master)
    seqs = [s for s in expand_master(m) if os.path.exists(os.path.join(s["out_dir"], "metadata.json"))]
    variants = a.variants.split(",")
    if not seqs:
        sys.exit("no finished sequences found under " + m["output_root"])

    sizes = {}
    for s in seqs:
        sd = s["out_dir"]
        parts = {k: dir_bytes(os.path.join(sd, *k.split("/"))) for k in
                 ("clean/rgb", "clean/depth", "clean/depth_exr", "clean/normal", "clean/leaf_id", "clean/semantic",
                  "clean/flow", "noisy/rgb", "noisy/depth", "gt", "poses")}
        if a.no_flow:
            parts["clean/flow"] = 0
        if a.no_normals:
            parts["clean/normal"] = 0
        if "noisy" not in variants:
            parts["noisy/rgb"] = parts["noisy/depth"] = 0
        sizes[s["sequence_id"]] = {k: v / a.frame_stride if k not in ("gt", "poses") else v for k, v in parts.items()}
    per_seq = {k: sum(v.values()) / 1e9 for k, v in sizes.items()}
    trees_bytes = sum(dir_bytes(os.path.join(m["output_root"], f"tree_{t:02d}", "canonical"))
                      for t in {s["tree"]["id"] for s in seqs})
    report = {"sequences": len(seqs), "GB_per_sequence": per_seq, "GB_trees": trees_bytes / 1e9,
              "GB_total": sum(per_seq.values()) + trees_bytes / 1e9,
              "GB_by_product": {k: sum(v[k] for v in sizes.values()) / 1e9 for k in next(iter(sizes.values()))}}
    print(json.dumps({k: v for k, v in report.items() if k != "GB_per_sequence"}, indent=1))
    if a.estimate:
        return
    out = a.out or os.path.join(os.path.dirname(m["output_root"]), "kaggle", "synthtree")
    os.makedirs(out, exist_ok=True)

    for tid in sorted({s["tree"]["id"] for s in seqs}):
        shutil.copytree(os.path.join(m["output_root"], f"tree_{tid:02d}", "canonical"),
                        os.path.join(out, "trees", f"tree_{tid:02d}", "canonical"), dirs_exist_ok=True)
    for s in seqs:
        dst = os.path.join(out, s["sequence_id"])
        print("packaging", s["sequence_id"])
        copy_seq(s["out_dir"], dst, variants, not a.no_flow, not a.no_normals, a.frame_stride)
        if a.zip:
            zip_dir(dst)

    heldout = set(m.get("splits", {}).get("heldout_trees", []))
    trees = sorted({s["tree"]["id"] for s in seqs})
    splits = {
        "view_split": "within each sequence: frames with index % holdout_every == 0 are test views (see <seq>/splits.json)",
        "holdout_every": m.get("splits", {}).get("holdout_every", 8),
        "train_trees": [t for t in trees if t not in heldout],
        "heldout_trees": [t for t in trees if t in heldout],
        "sequences": [s["sequence_id"] for s in seqs],
    }
    write_json(os.path.join(out, "splits.json"), splits)
    write_json(os.path.join(out, "size_report.json"), report)
    shutil.copy2(os.path.join(REPO, "dataset_card", "LICENSE_CC_BY_4.0.txt"), os.path.join(out, "LICENSE"))
    s0 = json.load(open(os.path.join(seqs[0]["out_dir"], "metadata.json")))
    rows = "\n".join(f"| {t['id']:02d} | {t['preset']} | {m['species'][t['preset']].get('description', '')} | "
                     f"{'held-out' if t['id'] in heldout else 'train'} |" for t in m["trees"] if t["id"] in trees)
    ctx = {"TITLE": a.title, "N_TREES": len(trees), "N_SEQ": len(seqs), "SIZE_GB": f"{report['GB_total']:.1f}",
           "WIND_LEVELS": ", ".join(f"{k} = {v} m/s" for k, v in m["wind"]["levels"].items()),
           "W": s0["intrinsics"]["width"], "H": s0["intrinsics"]["height"], "FX": f"{s0['intrinsics']['fx']:.2f}",
           "FY": f"{s0['intrinsics']['fy']:.2f}", "CX": f"{s0['intrinsics']['cx']:.2f}", "CY": f"{s0['intrinsics']['cy']:.2f}",
           "SENSOR": m["sensor"].get("name", ""), "FRAMES": s0["n_frames"], "FPS": m["capture"]["fps"],
           "STRIDE": m["capture"]["frame_stride"], "TREE_TABLE": rows, "DATE": datetime.date.today().isoformat(),
           "BLENDER": s0["blender"]["blender_version"], "GIT": s0["git"]["commit"], "ZIPPED": "yes" if a.zip else "no"}
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(render_card(os.path.join(REPO, "dataset_card", "README_TEMPLATE.md"), ctx))
    write_json(os.path.join(out, "dataset-metadata.json"), {"title": a.title[:50], "id": a.kaggle_id,
                                                           "licenses": [{"name": "CC-BY-4.0"}]})
    print("wrote", out, f"({dir_bytes(out) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
