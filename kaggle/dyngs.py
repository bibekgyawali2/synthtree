#!/usr/bin/env python
"""Static vs deformable 3D Gaussian Splatting on a SynthTree sequence (gsplat, CUDA).

    python dyngs.py --seq <dataset>/tree_01/wind_high --tree-dir <dataset>/trees/tree_01 \
        --variant noisy --mode deform --out results/dyn3dgs__tree_01__high__noisy__s0__geom.json

--mode static : plain 3DGS (same code path, no deformation) - the controlled baseline
--mode robust : static 3DGS with a trimmed L1 loss (top 20 % residual pixels ignored) - mitigation
--mode deform : canonical Gaussians + time-conditioned MLP predicting per-Gaussian offsets of
                position, rotation and scale (in the spirit of Deformable 3D Gaussians, Yang et
                al., CVPR 2024); deformation is switched on after a static warm-up.
--mode cluster: canonical Gaussians moved by per-cluster rigid motion: k-means cells of the warm-up
                Gaussians at branch and leaf scale, each with its own SE(3) trajectory (Fourier series
                in time up to --f-max Hz, unlocked coarse-to-fine). Independent cells cannot collapse
                into one smooth global shift the way an MLP(x, t) can.
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
    ap.add_argument("--mode", default="deform", choices=["static", "deform", "robust", "cluster"],
                    help="robust: static 3DGS with a trimmed photometric loss (mitigation)")
    ap.add_argument("--trim", type=float, default=0.2,
                    help="robust mode: fraction of highest-residual pixels ignored in the L1 term")
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--warmup", type=int, default=3000, help="static iterations before deformation (deform mode)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-frames", type=int, default=4)
    ap.add_argument("--train-stride", type=int, default=1,
                    help="train on every k-th video frame only (4 on a 30 fps sequence = 7.5 fps); test views unchanged")
    ap.add_argument("--train-offset", type=int, default=2,
                    help="phase of the stride; 2 keeps 7.5 fps frames off the every-8th-frame test views")
    ap.add_argument("--leaf-clusters", type=int, default=2048, help="cluster mode: fine (leaf-scale) cells")
    ap.add_argument("--branch-clusters", type=int, default=32, help="cluster mode: coarse (branch-scale) cells")
    ap.add_argument("--grouping", default="kmeans", choices=["kmeans", "oracle"],
                    help="cluster mode: leaf cells from k-means, or one cell per GT leaf (upper bound for grouping)")
    ap.add_argument("--f-max", type=float, default=10.0, help="cluster mode: highest motion frequency, Hz")
    ap.add_argument("--lam-leaf", type=float, default=0.1, help="cluster mode: acceleration penalty, leaf cells")
    ap.add_argument("--lam-branch", type=float, default=1.0, help="cluster mode: acceleration penalty, branch cells")
    ap.add_argument("--lam-amp", type=float, default=0.01, help="cluster mode: motion amplitude penalty")
    ap.add_argument("--motion-lr", type=float, default=2e-5,
                    help="cluster mode: Adam lr of the Fourier coefficients; Adam moves all ~2J coefficients by ~lr "
                         "per step and they add coherently at a given time, so this must be ~1e-3 / (2J)")
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
    if a.train_stride > 1:  # lower temporal sampling from the same renders (sampling-vs-modelling study)
        tr.frames = [i for i in tr.frames if i % a.train_stride == a.train_offset % a.train_stride]
        assert tr.frames and not set(tr.frames) & set(te.frames), "strided train frames overlap the test views"
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
        # ft=10 bands: with t normalised to [0,1] over a ~20 s orbit, the highest band is
        # 2^9 cycles/2 per sequence (~13 Hz) - above the 3.75 Hz Nyquist limit of 7.5 fps frames.
        # (Deformable 3DGS's ft=6 tops out at ~0.8 Hz here and cannot represent sway or flutter.)
        def __init__(self, fx=10, ft=10, width=256, depth=8):
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
    has_motion = a.mode in ("deform", "cluster")

    # ---------------------------------------------------------------- cluster motion bases
    def nearest(x, c, chunk=65536):
        return torch.cat([torch.cdist(x[i:i + chunk], c).argmin(1) for i in range(0, len(x), chunk)])

    def kmeans(x, k, iters=15):
        c = x[torch.randperm(len(x), device=x.device)[:k]].clone()
        for _ in range(iters):
            lab = nearest(x, c)
            cnt = torch.bincount(lab, minlength=len(c)).float()[:, None]
            s = torch.zeros_like(c).index_add_(0, lab, x)
            c = torch.where(cnt > 0, s / cnt.clamp_min(1), c)  # empty cells keep their old centre
        return c

    def so3_exp(w):  # (K,3) axis-angle -> (K,3,3); Taylor branch keeps gradients finite at w = 0
        th2 = (w * w).sum(-1, keepdim=True)[..., None]
        th = torch.sqrt(th2 + 1e-12)
        A = torch.where(th2 < 1e-6, 1 - th2 / 6, torch.sin(th) / th)
        B = torch.where(th2 < 1e-6, 0.5 - th2 / 24, (1 - torch.cos(th)) / th2.clamp_min(1e-12))
        Wx = torch.zeros(len(w), 3, 3, device=w.device)
        Wx[:, 0, 1], Wx[:, 0, 2], Wx[:, 1, 2] = -w[:, 2], w[:, 1], -w[:, 0]
        Wx = Wx - Wx.transpose(1, 2)
        return torch.eye(3, device=w.device) + A * Wx + B * (Wx @ Wx)

    def rotmat_to_quat(R):  # wxyz; motions stay well below 180 deg, so w > 0 is safe
        w = torch.sqrt((1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]).clamp_min(1e-8)) / 2
        return torch.stack([w, (R[:, 2, 1] - R[:, 1, 2]) / (4 * w), (R[:, 0, 2] - R[:, 2, 0]) / (4 * w),
                            (R[:, 1, 0] - R[:, 0, 1]) / (4 * w)], -1)

    def quat_mul(p, q):
        pw, px, py, pz = p.unbind(-1)
        qw, qx, qy, qz = q.unbind(-1)
        return torch.stack([pw * qw - px * qx - py * qy - pz * qz, pw * qx + px * qw + py * qz - pz * qy,
                            pw * qy - px * qz + py * qw + pz * qx, pw * qz + px * qy - py * qx + pz * qw], -1)

    class ClusterMotion(torch.nn.Module):
        # Per-cell SE(3) trajectories x -> R (x - p) + p + v, with (rotation vector, v) a Fourier series in
        # seconds on a 1/(2T) grid (no wrap-around over the orbit) up to f_max. No DC term: the canonical
        # Gaussians are the time-average pose. Frequencies are unlocked coarse-to-fine through f_cut.
        TRANS_SCALE = 0.05  # metres per unit coefficient, so rotation and translation share one learning rate

        def __init__(self, c_branch, c_leaf, T_seq, f_max):
            super().__init__()
            J = max(1, int(round(f_max * 2 * T_seq)))
            f = torch.arange(1, J + 1, device=c_leaf.device, dtype=torch.float32) / (2 * T_seq)
            self.register_buffer("f", f)
            self.register_buffer("w4", ((f / f_max) ** 4).repeat(2))  # Sobolev weights ~ acceleration energy
            self.register_buffer("c_branch", c_branch)
            self.register_buffer("c_leaf", c_leaf)
            self.f_max, self.f_cut = f_max, 0.5
            self.pb = torch.nn.Parameter(c_branch.clone())
            self.pl = torch.nn.Parameter(c_leaf.clone())
            self.coef_b = torch.nn.Parameter(torch.zeros(len(c_branch), 6, 2 * J, device=f.device))
            self.coef_l = torch.nn.Parameter(torch.zeros(len(c_leaf), 6, 2 * J, device=f.device))

        def basis(self, t_sec):
            w = 0.5 - 0.5 * torch.cos(math.pi * ((self.f_cut - self.f) / 1.0 + 1).clamp(0, 1))  # 1 Hz cosine ramp
            ph = 2 * math.pi * self.f * t_sec
            return torch.cat([w * torch.cos(ph), w * torch.sin(ph)])

        def se3(self, coef, piv, bas):  # -> R (K,3,3), t (K,3) with x -> R x + t
            xi = coef @ bas
            R = so3_exp(xi[:, :3])
            return R, piv + self.TRANS_SCALE * xi[:, 3:] - torch.einsum("kij,kj->ki", R, piv)

        def sobolev(self, coef):
            return (coef.pow(2) * self.w4).sum(-1).mean()

    motion = None
    cells = {"n": -1, "bid": None, "lid": None}  # Gaussian -> cell, recomputed after densification
    oracle = {}  # --grouping oracle: GT leaf points at the time-average pose + cells for everything else

    def nearest_d(x, c, chunk=4096):  # c can hold ~10^4-10^5 leaf vertices
        d, i = zip(*[torch.cdist(x[j:j + chunk], c).min(1) for j in range(0, len(x), chunk)])
        return torch.cat(d), torch.cat(i)

    def leaf_cells(m):
        if not oracle:
            return nearest(m, motion.c_leaf)
        d, i = nearest_d(m, oracle["P"])  # a Gaussian within 1 cm of a leaf follows that leaf, the rest k-means cells
        return torch.where(d < 0.01, oracle["vl"][i], oracle["L"] + nearest(m, oracle["c_other"]))

    def oracle_init(x):
        z = te.leaf["T"].astype(np.float64)  # (frames, leaves, 3, 4) from the canonical pose
        A, tt = z[..., :3].mean(0), z[..., 3].mean(0)  # time-average pose ~ the static warm-up geometry
        ct = load_tree(os.path.join(a.tree_dir, "canonical", "canonical_tree.npz"))
        vl = ct["leaf_vert_leaf"]
        P = np.einsum("vij,vj->vi", A[vl], ct["leaf_verts"]) + tt[vl]
        L = len(A)
        cen = np.zeros((L, 3))
        np.add.at(cen, vl, P)
        cen /= np.maximum(np.bincount(vl, minlength=L), 1)[:, None]
        oracle.update(P=torch.tensor(P, device=dev, dtype=torch.float32), vl=torch.tensor(vl, device=dev), L=L)
        d, _ = nearest_d(x, oracle["P"])
        other = x[d >= 0.01]
        oracle["c_other"] = kmeans(other, min(256, len(other)))
        return torch.cat([torch.tensor(cen, device=dev, dtype=torch.float32), oracle["c_other"]])

    def gaussians_at(t, use_deform):
        m, q, s = params["means"], params["quats"], params["scales"]
        if use_deform and deform is not None:
            dx, dq, ds = deform(m.detach(), t.view(1, 1))
            m = m + dx
            q = q + dq
            s = s + ds
        if use_deform and motion is not None:
            if cells["n"] != len(m):
                with torch.no_grad():
                    cells.update(n=len(m), bid=nearest(m, motion.c_branch), lid=leaf_cells(m))
            bas = motion.basis(t * tmax)
            Rb, tb = motion.se3(motion.coef_b, motion.pb, bas)
            Rl, tl = motion.se3(motion.coef_l, motion.pl, bas)
            bid, lid = cells["bid"], cells["lid"]
            m = torch.einsum("nij,nj->ni", Rb[bid], torch.einsum("nij,nj->ni", Rl[lid], m) + tl[lid]) + tb[bid]
            q = quat_mul(rotmat_to_quat(Rb[bid] @ Rl[lid]), F.normalize(q, dim=-1))
        return m, F.normalize(q, dim=-1), torch.exp(s), torch.sigmoid(params["opacities"]), torch.sigmoid(params["colors"])

    def render(t, vm, use_deform):
        m, q, s, o, c = gaussians_at(t, use_deform)
        img, alpha, info = rasterization(m, q, s, o, c, vm[None], K[None], W, H, near_plane=0.01, far_plane=100.0,
                                         backgrounds=bg[None], packed=False)  # must match strategy packed=False
        return img[0].clamp(0, 1), info

    # ---------------------------------------------------------------- training
    for step in range(a.iters):
        k = int(torch.randint(len(tr_img), (1,)))
        use_def = has_motion and step >= a.warmup
        if a.mode == "cluster" and use_def:
            if motion is None:  # cells from the warm-up geometry, then motion starts from identity
                with torch.no_grad():
                    x = params["means"].detach()
                    c_leaf = oracle_init(x) if a.grouping == "oracle" else kmeans(x, min(a.leaf_clusters, len(x)))
                    motion = ClusterMotion(kmeans(x, a.branch_clusters), c_leaf, tmax, a.f_max)
                opt_def = torch.optim.Adam([{"params": [motion.coef_b, motion.coef_l], "lr": a.motion_lr},
                                            {"params": [motion.pb, motion.pl], "lr": 1e-5}])
            frac = (step - a.warmup) / max(0.6 * (a.iters - a.warmup), 1)  # all frequencies open at 60 %
            motion.f_cut = 0.5 + (a.f_max + 1.0 - 0.5) * min(frac, 1.0)
        img, info = render(tr_t[k], tr_vm[k], use_def)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        gt = tr_img[k]
        res = (img - gt).abs()
        if a.mode == "robust" and step >= 1000:
            # trimmed L1 (in the spirit of RobustNeRF): pixels whose residual is in the top `trim`
            # fraction - mostly moving leaves the static model cannot explain - are ignored, so they
            # do not pull Gaussians into blurred, floater-prone compromises.
            r = res.mean(-1).detach()
            thr = torch.quantile(r.flatten()[torch.randint(r.numel(), (65536,), device=r.device)], 1 - a.trim)
            keep = (r <= thr).float()[..., None]
            l1 = (res * keep).sum() / (keep.sum() * 3).clamp_min(1)
        else:
            l1 = res.mean()
        ss = ssim_torch(img.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], torch, F)
        loss = 0.8 * l1 + 0.2 * (1 - ss)
        if use_def and motion is not None:  # stiff branches, freer leaves, small-motion tiebreak
            loss = loss + a.lam_branch * motion.sobolev(motion.coef_b) + a.lam_leaf * motion.sobolev(motion.coef_l) \
                + a.lam_amp * (motion.coef_b.pow(2).mean() + motion.coef_l.pow(2).mean())
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if use_def:
            for g in opt_def.param_groups:  # exponential decay to 2 % of the base rate over the deformation phase
                g["lr"] = g.setdefault("base_lr", g["lr"]) * (0.02 ** ((step - a.warmup) / max(a.iters - a.warmup, 1)))
            opt_def.step()
            opt_def.zero_grad(set_to_none=True)
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        if step % 100 == 0:  # densification/pruning may reorder Gaussians even when their number is unchanged
            cells["n"] = -1
        if step % 1000 == 0 or step == a.iters - 1:
            print(f"step {step:6d}  loss {loss.item():.4f}  gaussians {len(params['means'])}  "
                  f"deform {'on' if use_def else 'off'}  {time.time() - t_start:.0f}s", flush=True)

    # ---------------------------------------------------------------- evaluation
    tree = load_tree(os.path.join(a.tree_dir, "canonical", "canonical_tree.npz"))
    rng = np.random.default_rng(0)

    def geometry(t, leaf_T):
        with torch.no_grad():
            m, q, s, o, _ = gaussians_at(t, leaf_T is not None and has_motion)
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
    out["posed_psnr"] = psnr(has_motion)
    if has_motion:  # diagnostic: how far does the learned deformation move Gaussians?
        with torch.no_grad():
            mags = torch.cat([(gaussians_at(te_t[j], True)[0] - params["means"]).norm(dim=-1) for j in sel])
        out["deform_offset_mm_median"] = float(mags.median()) * 1000
        out["deform_offset_mm_p95"] = float(torch.quantile(mags[torch.randperm(len(mags))[:100000]], 0.95)) * 1000
    if motion is not None:
        out.update({"leaf_clusters": len(motion.c_leaf), "branch_clusters": len(motion.c_branch), "f_max": a.f_max,
                    "grouping": a.grouping})
    out.update({"mode": a.mode, "variant": a.variant, "iters": a.iters, "n_gaussians": int(len(params["means"])),
                "train_stride": a.train_stride, "n_train_views": len(tr_img),
                "seconds": round(time.time() - t_start, 1)})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("fscore", "chamfer_mm", "posed_fscore", "posed_chamfer_mm", "psnr", "posed_psnr",
                                          "n_gaussians", "seconds") if k in out}), flush=True)


if __name__ == "__main__":
    main()
