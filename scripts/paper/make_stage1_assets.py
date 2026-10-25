#!/usr/bin/env python
"""Numbers and table for the 30 fps follow-up (stage 1) section of the paper.

    python scripts/paper/make_stage1_assets.py --out paper_synthetic/generated

Reads the per-run JSONs of the stage-1 kernels (results/paper2_stage1, plus results/paper2_stage1_seeds if
present; seeds are averaged per tree before averaging over trees), the paper-1 deformable-study runs
(results/dynamic_v2, for the reproduction check) and the motion-rank analysis (results/paper2). Writes
numbers_stage1.tex (macros prefixed \\So) and tab_stage1.tex. make_assets.py is left untouched because it
needs the rendered synthtree-v1 dataset.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TREES = ["tree_01", "tree_03", "tree_05", "tree_07"]
LABEL = {"static3dgs": "static 3DGS", "dyn3dgs": "deformable 3DGS", "cluster3dgs": "per-cell rigid motion",
         "oracle3dgs": "per-leaf rigid motion (oracle groups)"}


def load(dirs):
    """(method, tree, level) -> list of per-seed result dicts."""
    runs = defaultdict(list)
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*__geom.json"))):
            method, tree, level, _variant, _seed = os.path.basename(f)[:-len("__geom.json")].split("__")
            runs[(method, tree, level)].append(json.load(open(f)))
    return runs


def per_tree(runs, method, level, key):
    """Seed-averaged value per tree (None where missing)."""
    out = []
    for t in TREES:
        v = [r[key] for r in runs.get((method, t, level), []) if r.get(key) is not None]
        out.append(float(np.mean(v)) if v else None)
    return out


def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else float("nan")


def wind_part(runs, method, key, sign):
    """Calm-to-high change per tree, oriented so that larger is worse."""
    c, h = per_tree(runs, method, "calm", key), per_tree(runs, method, "high", key)
    return [None if a is None or b is None else sign * (b - a) for a, b in zip(c, h)]


def fmt(x, p=3):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{p}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "paper_synthetic", "generated"))
    a = ap.parse_args()
    res = os.path.join(REPO, "results")
    runs = load([os.path.join(res, "paper2_stage1"), os.path.join(res, "paper2_stage1_seeds")])
    p1 = load([os.path.join(res, "dynamic_v2")])
    rank = json.load(open(os.path.join(res, "paper2", "motion_rank_high.json")))["trees"]
    n_runs = sum(len(v) for v in runs.values())
    n_seeds_hi = {m: max(len(runs.get((m, t, "high"), [])) for t in TREES) for m in LABEL}

    M = {}
    M["SoNRuns"] = str(n_runs)
    M["SoNSeedsHi"] = str(min(n_seeds_hi[m] for m in ("static3dgs", "cluster3dgs", "oracle3dgs")))
    F = {m: mean(per_tree(runs, m, "high", "posed_fscore")) for m in list(LABEL) + ["cluster3dgs_st4"]}
    CD = {m: mean(per_tree(runs, m, "high", "posed_chamfer_mm")) for m in list(LABEL) + ["cluster3dgs_st4"]}
    WCD = {m: mean(wind_part(runs, m, "posed_chamfer_mm", +1)) for m in LABEL}
    PS = {m: mean(per_tree(runs, m, "high", "posed_psnr")) for m in LABEL}
    for m, tag in (("static3dgs", "St"), ("dyn3dgs", "Dy"), ("cluster3dgs", "Cl"), ("oracle3dgs", "Or")):
        M[f"So{tag}FHi"] = fmt(F[m])
        M[f"So{tag}CDHi"] = fmt(CD[m], 1)
        M[f"So{tag}WindCD"] = fmt(WCD[m], 1)
        M[f"So{tag}PSNRHi"] = fmt(PS[m], 2)
    M["SoClWindCDRedPct"] = fmt(100 * (1 - WCD["cluster3dgs"] / WCD["static3dgs"]), 0)
    M["SoOrWindCDRedPct"] = fmt(100 * (1 - WCD["oracle3dgs"] / WCD["static3dgs"]), 0)
    M["SoClPSNRGain"] = fmt(PS["cluster3dgs"] - PS["static3dgs"], 1)

    # deformation offsets: a uniform shift has p95 / median = 1
    for m, tag in (("dyn3dgs", "Dy"), ("cluster3dgs", "Cl")):
        med = per_tree(runs, m, "high", "deform_offset_mm_median")
        p95 = per_tree(runs, m, "high", "deform_offset_mm_p95")
        M[f"So{tag}OffMed"] = fmt(mean(med), 1)
        M[f"So{tag}OffPninetyfive"] = fmt(mean(p95), 1)
        r = [b / a for a, b in zip(med, p95) if a and b]
        M[f"So{tag}OffRatioMin"], M[f"So{tag}OffRatioMax"] = fmt(min(r), 1), fmt(max(r), 1)

    # per-tree wins of the per-cell model over static at high wind
    fs, fc = per_tree(runs, "static3dgs", "high", "posed_fscore"), per_tree(runs, "cluster3dgs", "high", "posed_fscore")
    cs, cc = per_tree(runs, "static3dgs", "high", "posed_chamfer_mm"), per_tree(runs, "cluster3dgs", "high", "posed_chamfer_mm")
    M["SoClBeatsStF"] = str(sum(b > a for a, b in zip(fs, fc)))
    M["SoClBeatsStCD"] = str(sum(b < a for a, b in zip(cs, cc)))
    M["SoNeedleStFHi"], M["SoNeedleClFHi"] = fmt(fs[3]), fmt(fc[3])
    calm_s = per_tree(runs, "static3dgs", "calm", "posed_fscore")
    calm_c = per_tree(runs, "cluster3dgs", "calm", "posed_fscore")
    M["SoNeedleStFCalm"], M["SoNeedleClFCalm"] = fmt(calm_s[3]), fmt(calm_c[3])
    M["SoStFCalm"], M["SoClFCalm"] = fmt(mean(calm_s)), fmt(mean(calm_c))
    M["SoOrFCalm"] = fmt(mean(per_tree(runs, "oracle3dgs", "calm", "posed_fscore")))

    # temporal sampling: the same per-cell model at 7.5 fps (every 4th frame of the same renders)
    M["SoClFHiSlow"], M["SoClCDHiSlow"] = fmt(F["cluster3dgs_st4"]), fmt(CD["cluster3dgs_st4"], 1)

    # reproduction of the paper-1 condition on the new renders (deformable 3DGS at 7.5 fps)
    for lv, tag in (("calm", "Calm"), ("high", "Hi")):
        new = [r["posed_fscore"] for r in runs.get(("dyn3dgs_st4", "tree_01", lv), [])]
        old = [r["posed_fscore"] for r in p1.get(("dyn3dgs", "tree_01", lv), [])]
        M[f"SoReproNew{tag}"], M[f"SoReproOld{tag}"] = fmt(mean(new)), fmt(mean(old))

    # motion complexity of the ground truth (closed-form wind model, 30 fps, high wind, the four trees)
    rk = [rank[t] for t in TREES]
    M["SoRankNormalMin"] = str(min(r["r90_normal"] for r in rk))
    M["SoRankNormalMax"] = str(max(r["r90_normal"] for r in rk))
    M["SoRankCentroidMin"] = str(min(r["r90_centroid"] for r in rk))
    M["SoRankCentroidMax"] = str(max(r["r90_centroid"] for r in rk))
    M["SoAliasPctMin"] = fmt(100 * min(r["frac_above_3.75Hz"] for r in rk), 0)
    M["SoAliasPctMax"] = fmt(100 * max(r["frac_above_3.75Hz"] for r in rk), 0)
    M["SoEtaMin"] = fmt(min(r["eta_global"] for r in rk), 2)
    M["SoEtaMax"] = fmt(max(r["eta_global"] for r in rk), 2)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "numbers_stage1.tex"), "w") as f:
        f.write("% generated by scripts/paper/make_stage1_assets.py - do not edit\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")

    rows = []
    for m, rate in (("static3dgs", "30"), ("dyn3dgs", "30"), ("cluster3dgs", "30"),  # dyn @7.5 fps (2 trees): text only
                    ("cluster3dgs_st4", "7.5"), ("oracle3dgs", "30")):
        base = m.replace("_st4", "")
        n = sum(1 for t in TREES if runs.get((m, t, "high")))
        cells = [mean(per_tree(runs, m, "high", k)) for k in
                 ("posed_fscore", "posed_chamfer_mm", "posed_centroid_err_mm_median", "posed_psnr")]
        wcd = mean(wind_part(runs, m, "posed_chamfer_mm", +1))
        med = mean(per_tree(runs, m, "high", "deform_offset_mm_median"))
        p95 = mean(per_tree(runs, m, "high", "deform_offset_mm_p95"))
        off = "--" if not np.isfinite(med) else f"{med:.1f} / {p95:.1f}"
        rows.append(f"{LABEL[base]} & {rate} & {n} & {fmt(cells[0])} & {fmt(cells[1], 1)} & {fmt(wcd, 1)} & "
                    f"{fmt(cells[2], 1)} & {fmt(cells[3], 2)} & {off} \\\\")
    with open(os.path.join(a.out, "tab_stage1.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
                "Method & fps & Trees & F (posed) & CD posed [mm] & Wind part of CD [mm] & Centroid posed [mm] & "
                "PSNR [dB] & Offset med / p95 [mm] \\\\\n\\midrule\n")
        f.write("\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    print(f"{n_runs} stage-1 runs; wrote numbers_stage1.tex ({len(M)} macros) and tab_stage1.tex to {a.out}")


if __name__ == "__main__":
    main()
