#!/usr/bin/env python
"""Static vs deformable 3D Gaussian Splatting on a SynthTree sequence (gsplat, CUDA).

    python dyngs.py --seq <dataset>/tree_01/wind_high --tree-dir <dataset>/trees/tree_01 \
        --variant noisy --mode deform --out results/dyn3dgs__tree_01__high__noisy__s0__geom.json

--mode static : plain 3DGS (same code path, no deformation) - the controlled baseline
--mode deform : canonical Gaussians + time-conditioned MLP predicting per-Gaussian offsets of
                position, rotation and scale (in the spirit of Deformable 3D Gaussians, Yang et
                al., CVPR 2024); deformation is switched on after a static warm-up.
Both modes share initialisation (back-projected depth of the same variant), losses
(0.8 L1 + 0.2 D-SSIM), densification (gsplat DefaultStrategy), background colour and iterations.

Evaluation written to --out (one JSON, keys read by collect_results.py):
  rest metrics      canonical (no deformation) Gaussians vs the canonical leaf surface
  posed_* metrics   Gaussians deformed to each of K held-out frame times vs the leaf surface
                    posed with the exported per-leaf transforms at that time (mean over frames)
  psnr / posed_psnr held-out views without / with deformation
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts", "eval"))
sys.path.insert(0, os.path.dirname(HERE))


# ----------------------------------------------------------------------------- helpers
def quat_to_rotmat_np(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)


def ssim_torch(a, b, torch, F):
    """a, b: (1,3,H,W) in [0,1]; 11x11 Gaussian window, sigma 1.5."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    g = torch.exp(-((torch.arange(11, device=a.device) - 5.0) ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    w = (g[:, None] @ g[None, :]).expand(3, 1, 11, 11).contiguous()
    f = lambda x: F.conv2d(x, w, padding=5, groups=3)  # noqa: E731
    mu_a, mu_b = f(a), f(b)
    s_a, s_b, s_ab = f(a * a) - mu_a ** 2, f(b * b) - mu_b ** 2, f(a * b) - mu_a * mu_b
    return (((2 * mu_a * mu_b + C1) * (2 * s_ab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (s_a + s_b + C2))).mean()


def positional_encoding(x, n_freq, torch):
    out = [x]
    for k in range(n_freq):
        out += [torch.sin((2.0 ** k) * math.pi * x), torch.cos((2.0 ** k) * math.pi * x)]
    return torch.cat(out, -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--tree-dir", required=True)
    ap.add_argument("--variant", default="noisy", choices=["clean", "noisy"])
    ap.add_argument("--mode", default="deform", choices=["static", "deform"])
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--warmup", type=int, default=3000, help="static iterations before deformation (deform mode)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-frames", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy

    from convert import init_point_cloud
    from eval_geometry import evaluate, hybrid_metrics, sample_gaussians
    from synthtree_dataset import SynthTreeSequence
    from synthtree.tree import load_tree

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda"
    t_start = time.time()

    # ---------------------------------------------------------------- data
    tr = SynthTreeSequence(a.seq, variant=a.variant, split="train", load=("rgb",), with_leaf_gt=False)
    te = SynthTreeSequence(a.seq, variant=a.variant, split="test", load=("rgb",), with_leaf_gt=True)
    H, W = tr.height, tr.width
    K = torch.tensor(tr.K, device=dev, dtype=torch.float32)
    tmax = max(float(tr.times.max()), 1e-6)

    def load_split(ds):
        imgs, vms, ts, idx = [], [], [], []
        for k in range(len(ds)):
            it = ds[k]
            imgs.append(torch.from_numpy(it["rgb"]))
            vms.append(torch.from_numpy(np.linalg.inv(it["c2w_opencv"]).astype(np.float32)))
            ts.append(it["time"] / tmax)
            idx.append(it["frame_index"])
        return torch.stack(imgs).to(dev), torch.stack(vms).to(dev), torch.tensor(ts, device=dev), idx
    tr_img, tr_vm, tr_t, _ = load_split(tr)
    te_img, te_vm, te_t, te_idx = load_split(te)
    top = tr_img[:, : max(1, H // 10)].reshape(-1, 3)
    bg = top.median(0).values  # sky / far background colour from the top rows of the training views
    print(f"train {len(tr_img)}, test {len(te_img)} views, {W}x{H}, bg {bg.tolist()}", flush=True)

    # ---------------------------------------------------------------- Gaussians
    P, C = init_point_cloud(a.seq, a.variant, max_points=120_000)
    pts = torch.tensor(P, device=dev, dtype=torch.float32)
    cols = torch.tensor(C, device=dev, dtype=torch.float32) / 255.0
    with torch.no_grad():
        d2 = torch.cdist(pts[torch.randperm(len(pts), device=dev)[:1000]], pts).topk(4, largest=False).values[:, 1:].pow(2).mean(1)
    init_scale = float(torch.sqrt(d2.median().clamp_min(1e-8)))
    n = len(pts)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(pts),
        "scales": torch.nn.Parameter(torch.full((n, 3), math.log(init_scale), device=dev)),
        "quats": torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0], device=dev).repeat(n, 1)),
        "opacities": torch.nn.Parameter(torch.full((n,), math.log(0.1 / 0.9), device=dev)),
        "colors": torch.nn.Parameter(torch.logit(cols.clamp(0.02, 0.98))),
    })
    scene_scale = 1.0
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "colors": 2.5e-3}
    optimizers = {k: torch.optim.Adam([params[k]], lr=lr, eps=1e-15) for k, lr in lrs.items()}
    strategy = DefaultStrategy(verbose=False, refine_start_iter=500, refine_stop_iter=int(a.iters * 0.6),
                               reset_every=3000, refine_every=100)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    # ---------------------------------------------------------------- deformation field
    class Deform(torch.nn.Module):
        def __init__(self, fx=10, ft=6, width=256, depth=8):
            super().__init__()
            self.fx, self.ft = fx, ft
            d_in = 3 * (1 + 2 * fx) + 1 * (1 + 2 * ft)
            layers, d = [], d_in
            for _ in range(depth):
                layers += [torch.nn.Linear(d, width), torch.nn.ReLU(inplace=True)]
                d = width
            self.body = torch.nn.Sequential(*layers)
            self.head = torch.nn.Linear(width, 10)
            torch.nn.init.zeros_(self.head.weight)
            torch.nn.init.zeros_(self.head.bias)

        def forward(self, x, t):
            h = torch.cat([positional_encoding(x, self.fx, torch), positional_encoding(t.expand(len(x), 1), self.ft, torch)], -1)
            o = self.head(self.body(h))
            return o[:, :3], o[:, 3:7], o[:, 7:10]

    deform = Deform().to(dev) if a.mode == "deform" else None
    opt_def = torch.optim.Adam(deform.parameters(), lr=8e-4) if deform is not None else None

    def gaussians_at(t, use_deform):
        m, q, s = params["means"], params["quats"], params["scales"]
        if use_deform and deform is not None:
            dx, dq, ds = deform(m.detach(), t.view(1, 1))
            m = m + dx
            q = q + dq
            s = s + ds
        return m, F.normalize(q, dim=-1), torch.exp(s), torch.sigmoid(params["opacities"]), torch.sigmoid(params["colors"])

    def render(t, vm, use_deform):
        m, q, s, o, c = gaussians_at(t, use_deform)
        img, alpha, info = rasterization(m, q, s, o, c, vm[None], K[None], W, H, near_plane=0.01, far_plane=100.0,
                                         backgrounds=bg[None], packed=False)  # must match strategy packed=False
        return img[0].clamp(0, 1), info

    # ---------------------------------------------------------------- training
    for step in range(a.iters):
        k = int(torch.randint(len(tr_img), (1,)))
        use_def = deform is not None and step >= a.warmup
        img, info = render(tr_t[k], tr_vm[k], use_def)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        gt = tr_img[k]
        l1 = (img - gt).abs().mean()
        ss = ssim_torch(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], torch, F)
        loss = 0.8 * l1 + 0.2 * (1 - ss)
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if use_def:
            for g in opt_def.param_groups:  # exponential decay 8e-4 -> 1.6e-5 over the deformation phase
                g["lr"] = 8e-4 * (0.02 ** ((step - a.warmup) / max(a.iters - a.warmup, 1)))
            opt_def.step()
            opt_def.zero_grad(set_to_none=True)
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        if step % 1000 == 0 or step == a.iters - 1:
            print(f"step {step:6d}  loss {loss.item():.4f}  gaussians {len(params['means'])}  "
                  f"deform {'on' if use_def else 'off'}  {time.time() - t_start:.0f}s", flush=True)

    # ---------------------------------------------------------------- evaluation
    tree = load_tree(os.path.join(a.tree_dir, "canonical", "canonical_tree.npz"))
    rng = np.random.default_rng(0)

    def geometry(t, leaf_T):
        with torch.no_grad():
            m, q, s, o, _ = gaussians_at(t, leaf_T is not None and deform is not None)
        mu, Q, S, op = (x.detach().cpu().numpy().astype(np.float64) for x in (m, q, s, o))
        keep = op >= 0.1
        mu, Q, S, op = mu[keep], Q[keep], S[keep], op[keep]
        R = quat_to_rotmat_np(Q)
        Nrm = R[np.arange(len(R)), :, np.argmin(S, 1)]
        cen = evaluate(mu, Nrm, tree, leaf_T=leaf_T)
        Ps, Ns = sample_gaussians(mu, R, S, op, 600_000, rng)
        smp = evaluate(Ps, Ns, tree, leaf_T=leaf_T)
        h = hybrid_metrics(cen, smp)
        h.pop("raw", None)
        return h

    def psnr(use_def):
        vals = []
        with torch.no_grad():
            for k in range(len(te_img)):
                img, _ = render(te_t[k], te_vm[k], use_def)
                vals.append(float(-10 * torch.log10(((img - te_img[k]) ** 2).mean())))
        return float(np.mean(vals))

    out = {k: v for k, v in geometry(torch.tensor(0.0, device=dev), None).items() if isinstance(v, (int, float))}
    out["psnr"] = psnr(False)
    sel = np.linspace(0, len(te_idx) - 1, min(a.eval_frames, len(te_idx))).round().astype(int)
    posed = []
    for j in sel:
        T = te.leaf["T"][te_idx[j]].astype(np.float64)
        posed.append(geometry(te_t[j], T))
    for key in posed[0]:
        if isinstance(posed[0][key], (int, float)) and all(p.get(key) is not None for p in posed):
            out["posed_" + key] = float(np.mean([p[key] for p in posed]))
    out["posed_psnr"] = psnr(deform is not None)
    out.update({"mode": a.mode, "variant": a.variant, "iters": a.iters, "n_gaussians": int(len(params["means"])),
                "seconds": round(time.time() - t_start, 1)})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("fscore", "chamfer_mm", "posed_fscore", "posed_chamfer_mm", "psnr", "posed_psnr",
                                          "n_gaussians", "seconds") if k in out}), flush=True)


if __name__ == "__main__":
    main()
