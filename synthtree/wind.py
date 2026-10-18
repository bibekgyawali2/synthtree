"""Exact, repeatable procedural wind (pure NumPy).

Why not Geometry Nodes: the same closed-form function must (a) move the rendered
geometry and (b) produce the exported per-leaf ground-truth transforms. Writing
it once in NumPy and pushing vertices into Blender guarantees (a) == (b) bit for
bit, avoids duplicating formulas in a node graph, and is immune to GN API churn
across Blender 4.x/5.x. Cost: ~10 ms/frame of vertex upload. Motion blur and
rolling shutter are therefore done by sub-frame rendering (see render_sequence).

Model
-----
Branch sway (skeletal forward kinematics). Every node j gets a bend rotation
    Delta_j(t) = Rot(b, theta_j(t)) @ Rot(w, psi_j(t))
with w the horizontal wind direction and b = up x w (rotation about b by +theta
bends downwind). Rotations accumulate root -> tip:
    A_j = A_parent @ Delta_j,  p_j = p_parent + A_parent (p_j0 - p_parent0)
so displacement grows with height and with distance along the hierarchy.
    theta_j = c_j * g_b(U_eff) * [static + osc * sin(2 pi f_B t + phi_B)]
    psi_j   = c_j * g_b(U_eff) * cross * sin(2 pi f_B r_c t + phi'_B)
f_B and phi_B are per *branch* (coherent along the branch). Compliance
c_j ~ (r_ref / r_j)^p * seg_len_j, normalised so the trunk path sums to 1, i.e.
`static_deg` is the accumulated trunk-tip bend at U = u_ref.

Leaf flutter: rigid rotation of each leaf about its petiole attachment point,
    R_leaf = R_mean @ Rot(X_leaf, twist) @ Rot(Y_leaf, pitch)
with per-leaf random frequency/phase, then carried by its node: A_leaf = A_node @ R_leaf.

Amplitude law g(U) = (U / u_ref)^exponent (optionally soft-saturated with tanh).
Defaults: exponent 2 for branches (drag ~ rho Cd A U^2 and small linear-elastic
deflection ~ force), exponent 1 + saturation for leaves (flutter angle grows
~linearly at low U and saturates). These are plausible, NOT calibrated: fit
them against your real fan videos (see README, "Calibrating wind").

U = 0 is exactly the rest pose: all angles are exactly 0, `rodrigues` returns
exactly I, and positions are computed in displacement form v = v0 + delta so no
round-off is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rng import substream
from .transforms import normalize, rodrigues

DEFAULT_WIND = {
    "direction_deg": 0.0,  # azimuth the wind blows TOWARDS, from +X towards +Y
    "gust": {"intensity": 0.35, "freq_hz": [0.05, 0.6], "n_components": 5},
    "branch": {"law": {"exponent": 2.0, "u_ref": 5.0, "saturation": None},
               "static_deg": 6.0, "osc_deg": 3.0, "cross_deg": 1.5, "cross_freq_ratio": 1.37,
               "freq_hz": [0.6, 2.0], "freq_jitter": 0.12, "compliance_exp": 1.0},
    "leaf": {"law": {"exponent": 1.0, "u_ref": 5.0, "saturation": 1.5},
             "pitch_deg": 25.0, "twist_deg": 18.0, "mean_deg": 15.0,
             "freq_hz": [2.5, 9.0], "twist_freq_ratio": [1.1, 1.6]},
}


def amp_gain(U, law: dict):
    U = np.maximum(np.asarray(U, float), 0.0)
    x = (U / law["u_ref"]) ** law["exponent"]
    sat = law.get("saturation")
    if sat:
        x = sat * np.tanh(x / sat)
    return x


@dataclass
class WindState:
    t: float
    U_eff: float
    node_A: np.ndarray  # (N,3,3)
    node_disp: np.ndarray  # (N,3)
    leaf_A: np.ndarray  # (L,3,3) rotation from canonical
    leaf_t: np.ndarray  # (L,3) translation: x_t = A x_0 + t
    leaf_attach: np.ndarray  # (L,3) current attachment point
    wood_verts: np.ndarray
    leaf_verts: np.ndarray

    def leaf_T34(self, dtype=np.float32):
        return np.concatenate([self.leaf_A, self.leaf_t[:, :, None]], axis=2).astype(dtype)


class WindModel:
    def __init__(self, tree: dict, wind_cfg: dict, U: float, seed: int):
        from .tree import deep_merge
        self.cfg = deep_merge(DEFAULT_WIND, wind_cfg or {})
        self.U = float(U)
        self.tree = tree
        az = np.radians(self.cfg["direction_deg"])
        self.w = np.array([np.cos(az), np.sin(az), 0.0])
        self.b = normalize(np.cross([0.0, 0.0, 1.0], self.w))

        p0, par = tree["nodes_p0"], tree["nodes_parent"]
        N = len(p0)
        depth = np.zeros(N, np.int64)
        for j in range(N):  # parents always precede children
            depth[j] = 0 if par[j] < 0 else depth[par[j]] + 1
        self.levels = [np.nonzero(depth == d)[0] for d in range(depth.max() + 1)]
        self.par = par
        self.off = np.zeros_like(p0)
        has_par = par >= 0
        self.off[has_par] = p0[has_par] - p0[par[has_par]]

        # compliance: thinner and longer segments bend more
        bc = self.cfg["branch"]
        seglen = np.zeros(N)
        children_first = {}
        for j in range(N):
            if par[j] >= 0 and tree["nodes_branch"][par[j]] == tree["nodes_branch"][j]:
                children_first.setdefault(par[j], j)
        for j in range(N):
            if j in children_first:
                seglen[j] = np.linalg.norm(p0[children_first[j]] - p0[j])
        r = tree["nodes_radius"]
        c = (r[0] / r) ** bc["compliance_exp"] * seglen
        trunk = tree["nodes_branch"] == 0
        self.compliance = c / max(c[trunk].sum(), 1e-12)

        rng = substream(seed, "wind", "phases")  # tree seed => identical across wind levels
        B = len(tree["branch_level"])
        lvl = tree["branch_level"]
        f_lo, f_hi = bc["freq_hz"]
        frac = lvl / max(lvl.max(), 1)
        self.f_branch = (f_lo + (f_hi - f_lo) * frac) * np.exp(rng.normal(0, bc["freq_jitter"], B))
        self.ph_branch = rng.uniform(0, 2 * np.pi, B)
        self.ph_cross = rng.uniform(0, 2 * np.pi, B)

        lc = self.cfg["leaf"]
        L = len(tree["leaf_attach"])
        self.f_leaf = rng.uniform(*lc["freq_hz"], L)
        self.r_twist = rng.uniform(*lc["twist_freq_ratio"], L)
        self.ph_pitch = rng.uniform(0, 2 * np.pi, L)
        self.ph_twist = rng.uniform(0, 2 * np.pi, L)
        Fr = tree["leaf_frame"]
        self.leaf_X, self.leaf_Y = Fr[:, :, 0], Fr[:, :, 1]
        n = tree["leaf_normal"]
        m = np.cross(n, self.w)
        bad = np.linalg.norm(m, axis=1) < 1e-6
        m[bad] = self.leaf_Y[bad]
        self.leaf_mean_axis = normalize(m)

        g = self.cfg["gust"]
        self.g_f = rng.uniform(*g["freq_hz"], g["n_components"])
        self.g_ph = rng.uniform(0, 2 * np.pi, g["n_components"])
        self.g_a = 1.0 / np.sqrt(self.g_f)

    def gust_noise(self, t):
        """Smooth zero-mean noise in [-1, 1], analytic in t (consistent at sub-frames)."""
        return float(np.sum(self.g_a * np.sin(2 * np.pi * self.g_f * t + self.g_ph)) / np.sum(self.g_a))

    def U_eff(self, t):
        return self.U * max(0.0, 1.0 + self.cfg["gust"]["intensity"] * self.gust_noise(t))

    def evaluate(self, t: float) -> WindState:
        tr = self.tree
        Ue = self.U_eff(t)
        bc, lc = self.cfg["branch"], self.cfg["leaf"]
        gb = float(amp_gain(Ue, bc["law"]))
        gl = float(amp_gain(Ue, lc["law"]))

        nb = tr["nodes_branch"]
        osc = np.sin(2 * np.pi * self.f_branch[nb] * t + self.ph_branch[nb])
        crs = np.sin(2 * np.pi * self.f_branch[nb] * bc["cross_freq_ratio"] * t + self.ph_cross[nb])
        theta = self.compliance * gb * (np.radians(bc["static_deg"]) + np.radians(bc["osc_deg"]) * osc)
        psi = self.compliance * gb * np.radians(bc["cross_deg"]) * crs
        Delta = rodrigues(self.b, theta) @ rodrigues(self.w, psi)

        N = len(nb)
        A = np.empty((N, 3, 3))
        disp = np.zeros((N, 3))
        I = np.eye(3)
        for d, idx in enumerate(self.levels):
            if d == 0:
                A[idx] = Delta[idx]
                continue
            q = self.par[idx]
            A[idx] = A[q] @ Delta[idx]
            disp[idx] = disp[q] + np.einsum("nij,nj->ni", A[q] - I, self.off[idx])

        # wood: rings follow their node rigidly
        p0 = tr["nodes_p0"]
        vn = tr["wood_vert_node"]
        v0 = tr["wood_verts"]
        wood = v0 + disp[vn] + np.einsum("nij,nj->ni", A[vn] - I, v0 - p0[vn])

        # leaves: carried by the node, flutter about the petiole
        ln = tr["leaf_node"]
        a0 = tr["leaf_attach"]
        tt = 2 * np.pi * self.f_leaf * t
        pitch = gl * np.radians(lc["pitch_deg"]) * np.sin(tt + self.ph_pitch)
        twist = gl * np.radians(lc["twist_deg"]) * np.sin(tt * self.r_twist + self.ph_twist)
        mean = gl * np.radians(lc["mean_deg"]) * np.ones_like(pitch)
        R_rel = rodrigues(self.leaf_mean_axis, mean) @ rodrigues(self.leaf_X, twist) @ rodrigues(self.leaf_Y, pitch)
        A_leaf = A[ln] @ R_rel
        da = disp[ln] + np.einsum("nij,nj->ni", A[ln] - I, a0 - p0[ln])
        a_t = a0 + da
        t_leaf = a_t - np.einsum("nij,nj->ni", A_leaf, a0)
        vl = tr["leaf_vert_leaf"]
        lv0 = tr["leaf_verts"]
        leaf = lv0 + da[vl] + np.einsum("nij,nj->ni", A_leaf[vl] - I, lv0 - a0[vl])
        return WindState(t, Ue, A, disp, A_leaf, t_leaf, a_t, wood, leaf)
