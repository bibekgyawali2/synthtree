#!/usr/bin/env python
"""Batch runner: trees -> Blender render -> post-process -> noise -> validation.

    python scripts/run_batch.py --master configs/master.yaml            # full dataset
    python scripts/run_batch.py --master configs/master.yaml --smoke    # tiny end-to-end test
    python scripts/run_batch.py --master configs/debug_2x2.yaml         # start-small plan
    options: --only tree_01/wind_high  --stages tree,render,post,noise,validate  --dry-run
             --blender /path/to/Blender  --cooldown-s 30  --max-frames 10  --force

Resumable at three levels:
  * a stage is skipped when <seq>/.stage_<name> holds the current config hash;
  * the render stage resumes at the first missing frame (Blender writes frames atomically);
  * changing a sequence's config changes its hash and re-runs its stages.
Logs: logs/batch_<timestamp>.log (+ one Blender log per sequence in <seq>/logs/).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from make_tree import export_tree  # noqa: E402
from synthtree.config import expand_master, load_master, write_json  # noqa: E402

DEFAULT_BLENDER = {
    "darwin": "/Applications/Blender.app/Contents/MacOS/Blender",
    "linux": "blender",
    "win32": r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
}


def find_blender(arg):
    cand = arg or os.environ.get("BLENDER") or DEFAULT_BLENDER.get(sys.platform, "blender")
    path = shutil.which(cand) or (cand if os.path.exists(cand) else None)
    if not path:
        sys.exit(f"Blender not found ({cand}). Pass --blender or set BLENDER=/path/to/blender")
    return path


class Log:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a")

    def __call__(self, msg, **kw):
        line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}" + (f" {json.dumps(kw)}" if kw else "")
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def stage_done(seq_dir, stage, h):
    p = os.path.join(seq_dir, f".stage_{stage}")
    return os.path.exists(p) and open(p).read().strip() == h


def mark(seq_dir, stage, h):
    with open(os.path.join(seq_dir, f".stage_{stage}"), "w") as f:
        f.write(h)


def run_blender(blender, cfg_path, log_path, extra=()):
    cmd = [blender, "-b", "--factory-startup", "-noaudio", "-P", os.path.join(REPO, "blender", "render_sequence.py"),
           "--", "--config", cfg_path, *extra]
    t0 = time.time()
    frames = {"data": [], "beauty": []}
    with open(log_path, "a") as lf:
        lf.write(f"\n# {' '.join(cmd)}\n")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            lf.write(line)
            if line.startswith("[synthtree] "):
                ev = json.loads(line[len("[synthtree] "):])
                if ev["event"] == "frame":
                    frames[ev["stage"]].append(ev["seconds"])
                    if ev["i"] % 10 == 0:
                        print(f"    {ev['stage']} frame {ev['i']} ({ev['seconds']:.2f}s)", flush=True)
        p.wait()
    return p.returncode, time.time() - t0, frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*", help="sequence ids (tree_01/wind_high) or tree dirs (tree_01)")
    ap.add_argument("--stages", default="tree,render,post,noise,validate")
    ap.add_argument("--blender")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--cooldown-s", type=float, default=0.0, help="pause between sequences (thermal headroom)")
    ap.add_argument("--force", action="store_true", help="ignore stage markers")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    master = load_master(a.master, smoke=a.smoke)
    if a.smoke:
        for k in ("only_trees", "only_levels"):
            if k in master.get("smoke", {}):
                master[k] = master["smoke"][k]
    seqs = expand_master(master)
    if a.only:
        seqs = [s for s in seqs if any(s["sequence_id"] == o or s["sequence_id"].startswith(o.rstrip("/") + "/") for o in a.only)]
    stages = a.stages.split(",")
    log = Log(os.path.join(REPO, "logs", f"batch_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"))
    log("plan", sequences=[s["sequence_id"] for s in seqs], stages=stages, output_root=master["output_root"])
    if a.dry_run:
        return
    blender = find_blender(a.blender) if "render" in stages else None
    t_all = time.time()
    failures = []

    for s in seqs:
        sd = s["out_dir"]
        os.makedirs(sd, exist_ok=True)
        cfg_path = os.path.join(sd, "config.json")
        write_json(cfg_path, s)
        h = s["config_hash"]
        try:
            if "tree" in stages:  # once per tree, shared by all its wind levels
                npz = os.path.join(s["tree_dir"], "canonical", "canonical_tree.npz")
                tree_hash = json.dumps(s["tree"], sort_keys=True)
                marker = os.path.join(s["tree_dir"], ".tree_spec")
                if a.force or not os.path.exists(npz) or not os.path.exists(marker) or open(marker).read() != tree_hash:
                    from synthtree.tree import generate_tree
                    t0 = time.time()
                    meta = export_tree(generate_tree(s["tree"]["preset"], s["tree"]["seed"]), os.path.join(s["tree_dir"], "canonical"))
                    with open(marker, "w") as f:
                        f.write(tree_hash)
                    log("tree", tree=s["tree"]["id"], leaves=meta["n_leaves"], seconds=round(time.time() - t0, 2))
            if "render" in stages and (a.force or not stage_done(sd, "render", h)):
                log("render start", seq=s["sequence_id"])
                os.makedirs(os.path.join(sd, "logs"), exist_ok=True)
                extra = ["--max-frames", str(a.max_frames)] if a.max_frames else []
                rc, dt, fr = run_blender(blender, cfg_path, os.path.join(sd, "logs", "blender.log"), extra)
                steady = {k: (round(sum(v[1:]) / max(len(v) - 1, 1), 3) if v else None) for k, v in fr.items()}
                log("render end", seq=s["sequence_id"], rc=rc, seconds=round(dt, 1), sec_per_frame_steady=steady)
                if rc != 0:
                    raise RuntimeError(f"Blender exited with {rc}; see {sd}/logs/blender.log")
                if not a.max_frames:
                    mark(sd, "render", h)
            if "post" in stages and (a.force or not stage_done(sd, "post", h)):
                import postprocess
                t0 = time.time()
                qc = postprocess.process(sd)
                log("post", seq=s["sequence_id"], seconds=round(time.time() - t0, 1), reproj_max_px=qc["reproj_max_px"])
                mark(sd, "post", h)
            if "noise" in stages and (a.force or not stage_done(sd, "noise", h)):
                import noise_sim
                t0 = time.time()
                noise_sim.process(sd)
                log("noise", seq=s["sequence_id"], seconds=round(time.time() - t0, 1))
                mark(sd, "noise", h)
            if "validate" in stages:
                import validate_sequence
                res = validate_sequence.validate(sd)
                log("validate", seq=s["sequence_id"], ok=res["ok"], failed=[k for k, v in res["checks"].items() if not v["ok"]])
                if not res["ok"]:
                    failures.append(s["sequence_id"])
        except Exception as e:  # keep going with the next sequence; rerun resumes
            log("ERROR", seq=s["sequence_id"], error=repr(e))
            failures.append(s["sequence_id"])
        if a.cooldown_s:
            time.sleep(a.cooldown_s)
    if "validate" in stages:
        import validate_sequence
        validate_sequence.validate_pairs(master, seqs, log)
    log("finished", seconds=round(time.time() - t_all, 1), failures=failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
