"""Chapter 11 — symmetry suffices.

Two facts about a table fix its default geometry. Rows are *exchangeable* — an iid sample with
no canonical order — so the operator over them should be permutation-invariant, which forces a
*symmetric* Gram k(x,x') = k(x',x). And Bochner (Ch. 8) closes the loop: the symmetric stationary
class already contains every stationary geometry, so committing to symmetry sacrifices no
expressivity. Asymmetry is therefore extra capacity; on an exchangeable table it has nothing to
represent, and an asymmetric model must spend data to drive that capacity back toward zero (or pay
its variance). Asymmetry is earned only where the data carries a genuinely *directed* relation —
time, causality, transport, user->item.

This module makes the argument runnable on the running example, in four pieces.

1. **Head-to-head (Taiwan Credit).** A scaled-dot-product attention smoother in two forms that
   differ *only* in symmetry: ``asym`` (separate W_Q, W_K) and ``sym`` (shared W = W_Q = W_K, a PSD
   Gram — the one-line "linear-attention" retrofit). The symmetric form matches the asymmetric one
   on held-out AUC with fewer parameters and lower seed-to-seed variance.

2. **The first-order law.** Decompose any pairwise score into symmetric and antisymmetric parts
   under the swap x<->x'. The directional content is Delta = k - k^T (antisymmetric); the response
   gradient is h_a(x,x') = m(x) - m(x') (antisymmetric). The first-order change in risk along a
   path k_s + eps*Delta is governed by -<Delta, h_a>: the gain from asymmetry is *proportional to*
   the alignment of the kernel's directional content with the response gradient, and vanishes
   exactly when Delta is orthogonal to h_a. We verify the proportionality directly (corr ~ -0.98).

3. **Orthogonality on exchangeable data.** On Taiwan Credit, <Delta, h_a> over random admissible
   antisymmetric directions concentrates at zero: the directional content of a generic asymmetric
   kernel is orthogonal-in-expectation to the response gradient, so asymmetry buys nothing to
   first order.

4. **A directed task where asymmetry is earned.** A lagged-signal task — predict s(pos - lag) by
   smoothing labels s(pos) — carries a directed relation. Its aligned antisymmetric direction has
   <Delta, h_a> far from zero, and the asymmetric smoother beats the symmetric one by a wide margin.

The decision rule: symmetric by default for exchangeable rows; reach for an antisymmetric geometry
only when a directed mechanism is present and the screening alignment <Delta, h_a> is non-zero.

    python -m lkbook.chapters.ch11 --out-prefix fig11
"""
from __future__ import annotations

import argparse
import functools

import numpy as np

from lkbook import load_taiwan, set_style

SEED = 0


def _single_thread(fn):
    @functools.wraps(fn)
    def wrap(*a, **k):
        try:
            from threadpoolctl import threadpool_limits
            with threadpool_limits(4):
                return fn(*a, **k)
        except Exception:
            return fn(*a, **k)
    return wrap


def _subset(X, y, n, seed):
    idx = np.random.default_rng(seed).choice(len(X), min(n, len(X)), replace=False)
    return X[idx], y[idx]


# =============================================================================
# 1. Head-to-head: an attention smoother whose only knob is symmetry
# =============================================================================

class KernelAttention:
    """Scaled-dot-product attention as a Nadaraya--Watson smoother over a support set,
    with a learned linear encoder. The geometry is the bilinear form B = W_Q^T W_K applied
    in the encoded space; the prediction at a query is the softmax-weighted average of support
    labels. The *only* structural difference between the two modes is symmetry of B:

    - ``asym``: separate W_Q, W_K  ->  B asymmetric whenever W_Q != W_K.
    - ``sym`` : shared W (W_Q = W_K)  ->  B = W^T W is symmetric PSD (a Gram). This is the
      one-line retrofit of standard self-attention (linear attention with shared Q = K).

    Implemented in torch only to get gradients; trains in seconds on CPU.
    """

    def __init__(self, p, r=16, mode="asym", kernel="dot", seed=SEED):
        import torch
        torch.set_default_dtype(torch.float64)
        g = torch.Generator().manual_seed(seed)
        self.enc = torch.nn.Parameter(0.5 * torch.randn(p, r, generator=g))
        self.Wq = torch.nn.Parameter(torch.eye(r) + 0.1 * torch.randn(r, r, generator=g))
        self.mode = mode
        self.kernel = kernel
        if mode == "asym":
            g2 = torch.Generator().manual_seed(seed + 7)
            self.Wk = torch.nn.Parameter(torch.eye(r) + 0.1 * torch.randn(r, r, generator=g2))
        else:  # shared projection: a symmetric kernel (PSD Gram for dot, symmetric distance for gauss)
            self.Wk = self.Wq
        self.log_sigma = torch.nn.Parameter(torch.zeros(()))
        self.r = r

    def params(self):
        ps = [self.enc, self.Wq]
        if self.mode == "asym":
            ps.append(self.Wk)
        if self.kernel == "gauss":
            ps.append(self.log_sigma)
        return ps

    def n_params(self):
        return int(sum(p.numel() for p in self.params()))

    def scores(self, Xq, Xk):
        import numpy as _np
        import torch
        q = (Xq @ self.enc) @ self.Wq
        k = (Xk @ self.enc) @ self.Wk
        if self.kernel == "gauss":
            d2 = (q * q).sum(1, keepdim=True) - 2 * q @ k.t() + (k * k).sum(1)[None, :]
            return -d2 / (2 * torch.exp(2 * self.log_sigma))
        return (q @ k.t()) / _np.sqrt(self.r)

    def predict(self, Xq, Xk, yk, mask_diag=False):
        import torch
        S = self.scores(Xq, Xk)
        if mask_diag:
            S = S - torch.diag(torch.full((S.shape[0],), 1e9, dtype=S.dtype))
        return torch.softmax(S, 1) @ yk


@_single_thread
def _fit_attention(Xtr, ytr, mode, seed, steps=300, lr=0.02, wd=1e-3, task="cls"):
    import torch
    Xt = torch.tensor(Xtr); yt = torch.tensor(np.asarray(ytr, float))
    m = KernelAttention(Xtr.shape[1], mode=mode, seed=seed)
    opt = torch.optim.Adam(m.params(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        pred = m.predict(Xt, Xt, yt, mask_diag=True)
        if task == "cls":
            p = pred.clamp(1e-6, 1 - 1e-6)
            loss = -(yt * torch.log(p) + (1 - yt) * torch.log(1 - p)).mean()
        else:
            loss = ((pred - yt) ** 2).mean()
        loss.backward()
        opt.step()
    return m


def run_taiwan_headtohead(seeds=range(6), n_train=800, n_test=800):
    """Train ``asym`` and ``sym`` attention smoothers on Taiwan Credit across seeds.
    Returns matched held-out AUC, parameter counts and seed-to-seed variance."""
    import torch
    from sklearn.metrics import roc_auc_score
    d = load_taiwan()
    out = {"asym": [], "sym": [], "n_params": {}}
    for seed in seeds:
        Xtr, ytr = _subset(d.Xtr, d.ytr, n_train, seed)
        Xte, yte = _subset(d.Xte, d.yte, n_test, seed + 100)
        Xte_t = torch.tensor(Xte); Xtr_t = torch.tensor(Xtr)
        ytr_t = torch.tensor(ytr.astype(float))
        for mode in ("asym", "sym"):
            m = _fit_attention(Xtr, ytr, mode, seed)
            with torch.no_grad():
                p = torch.softmax(m.scores(Xte_t, Xtr_t), 1) @ ytr_t
            out[mode].append(roc_auc_score(yte, p.numpy()))
            out["n_params"][mode] = m.n_params()
    for k in ("asym", "sym"):
        out[k] = np.array(out[k])
    return out


# =============================================================================
# 2-3. The first-order law and orthogonality on exchangeable data  (pure numpy)
# =============================================================================

def gauss_logits(Xq, Xk, ell):
    d2 = ((Xq[:, None, :] - Xk[None, :, :]) ** 2).sum(-1)
    return -d2 / (2 * ell ** 2)


def softmax_rows(S):
    S = S - S.max(1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(1, keepdims=True)


def skew(r, seed):
    """A unit-norm antisymmetric matrix: an admissible directional perturbation direction."""
    g = np.random.default_rng(seed)
    A = g.normal(0, 1, (r, r))
    A = A - A.T
    return A / (np.linalg.norm(A) + 1e-12)


def delta_score(Xq, Xk, A):
    """The antisymmetric bilinear score Delta(x,x') = x^T A x' for skew A (Delta(x',x) = -Delta(x,x'))."""
    return Xq @ A @ Xk.T


def ip_delta_ha(X, y, A, npairs=40000, seed=1):
    """<Delta, h_a> = E[ Delta(x_i, x_j) * (m(x_i) - m(x_j)) ], estimated over random pairs.
    The response gradient h_a(x,x') = m(x) - m(x') is estimated from the (centered) labels."""
    g = np.random.default_rng(seed)
    n = len(X)
    i = g.integers(0, n, npairs)
    j = g.integers(0, n, npairs)
    dij = np.sum((X[i] @ A) * X[j], axis=1)
    ha = y[i] - y[j]
    return float(np.mean(dij * ha))


def _risk(Xq, Xk, yk, yq, ell, A, eps):
    S = gauss_logits(Xq, Xk, ell) + eps * delta_score(Xq, Xk, A)
    yhat = softmax_rows(S) @ yk
    return float(np.mean((yhat - yq) ** 2))


def _risk_derivative(Xq, Xk, yk, yq, ell, A, h=1e-2):
    return (_risk(Xq, Xk, yk, yq, ell, A, +h) - _risk(Xq, Xk, yk, yq, ell, A, -h)) / (2 * h)


def make_directed(n, seed, lag=0.18, freq=2.0, noise=0.05):
    """A directed (lagged-signal) task. Each point has a position; the support label at a point
    is the signal s(pos); the query target is the *lagged* signal s(pos - lag). Predicting the
    target by smoothing support labels requires attending to points *upstream* by a fixed offset
    — a directed relation no symmetric (distance) kernel can represent."""
    g = np.random.default_rng(seed)
    pos = np.sort(g.uniform(0, 1, n))
    X = np.column_stack([pos, g.normal(0, 1, (n, 3))])
    s = lambda t: np.sin(2 * np.pi * freq * t)
    y_label = s(pos) + noise * g.normal(0, 1, n)
    y_target = s(pos - lag)
    return X.astype(float), y_label.astype(float), y_target.astype(float)


def run_first_order(n_dirs=30):
    """Verify the first-order law on the directed task: the measured directional risk gain is
    proportional to -<Delta, h_a> (corr ~ -1). The family interpolates each random direction
    toward the aligned (directed) direction so <Delta, h_a> sweeps a range."""
    Xtr, ylab, ytgt = make_directed(900, 0)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd
    Xte, _, ytgt_te = make_directed(900, 100)
    Xte = (Xte - mu) / sd
    # augment a constant column so a pure positional offset is an antisymmetric bilinear direction
    Xa = np.column_stack([Xtr, np.ones(len(Xtr))])
    Xa_te = np.column_stack([Xte, np.ones(len(Xte))])
    ra = Xa.shape[1]
    ell = np.sqrt(ra)
    A_align = np.zeros((ra, ra))
    A_align[0, ra - 1] = 1.0
    A_align[ra - 1, 0] = -1.0
    A_align /= np.linalg.norm(A_align)

    ips, ders = [], []
    for s in range(n_dirs):
        Ar = skew(ra, s)
        for w in np.linspace(0, 1, 5):
            A = (1 - w) * Ar + w * A_align
            A = A / (np.linalg.norm(A) + 1e-12)
            ips.append(ip_delta_ha(Xa, ytgt, A, seed=s))
            ders.append(_risk_derivative(Xa_te, Xa, ylab, ytgt_te, ell, A))
    ips = np.array(ips); ders = np.array(ders)
    corr = float(np.corrcoef(ips, ders)[0, 1])
    return {"ip": ips, "deriv": ders, "corr": corr,
            "ip_align": ip_delta_ha(Xa, ytgt, A_align)}


def run_orthogonality(n_dirs=60, n=1500):
    """On exchangeable Taiwan Credit, <Delta, h_a> over random antisymmetric directions
    concentrates at zero. Returns the distribution, plus the directed-task aligned value for
    contrast (rescaled to a comparable per-direction-norm basis)."""
    d = load_taiwan()
    X, y = _subset(d.Xtr, d.ytr.astype(float), n, 0)
    y = y - y.mean()
    r = X.shape[1]
    ips = np.array([ip_delta_ha(X, y, skew(r, s), seed=s) for s in range(n_dirs)])
    # directed-task contrast on the same kind of axis: aligned vs random spread there
    Xd, _, ytgt = make_directed(1500, 0)
    mu, sd = Xd.mean(0), Xd.std(0) + 1e-9
    Xd = (Xd - mu) / sd
    Xa = np.column_stack([Xd, np.ones(len(Xd))])
    ra = Xa.shape[1]
    A_align = np.zeros((ra, ra)); A_align[0, ra - 1] = 1.0; A_align[ra - 1, 0] = -1.0
    A_align /= np.linalg.norm(A_align)
    ip_align = ip_delta_ha(Xa, ytgt, A_align)
    ips_dir_rand = np.array([ip_delta_ha(Xa, ytgt, skew(ra, s), seed=s + 500) for s in range(n_dirs)])
    return {"taiwan_ips": ips, "directed_aligned": ip_align,
            "directed_random_std": float(ips_dir_rand.std())}


# =============================================================================
# 4. Directed task: asymmetry is earned
# =============================================================================

def directed_one_lag(lag, seed=0, n=400, steps=300):
    """Symmetric vs asymmetric held-out RMSE on the directed task at one lag (fast, one seed).
    At lag 0 the task is undirected and the two match; as the lag grows the symmetric kernel,
    forced to center its weight at the query's own position, falls behind. Used by the explorer."""
    import torch
    Xtr, ylab, ytgt = make_directed(n, seed, lag=lag)
    Xte, _, ytgt_te = make_directed(n, seed + 100, lag=lag)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    out = {}
    for mode in ("sym", "asym"):
        m = KernelAttention(Xtr.shape[1], r=8, mode=mode, kernel="gauss", seed=seed)
        Xt = torch.tensor(Xtr); yl = torch.tensor(ylab); yt = torch.tensor(ytgt)
        opt = torch.optim.Adam(m.params(), lr=0.02, weight_decay=1e-4)
        for _ in range(steps):
            opt.zero_grad()
            (((m.predict(Xt, Xt, yl, mask_diag=True) - yt) ** 2).mean()).backward()
            opt.step()
        with torch.no_grad():
            pf = m.predict(torch.tensor(Xte), Xt, yl).numpy()
        out[mode] = float(np.sqrt(((pf - ytgt_te) ** 2).mean()))
    return out


def run_directed_headtohead(seeds=range(4), n=400, steps=400):
    """Head-to-head on the directed lag task: the asymmetric smoother captures the offset; the
    symmetric one cannot. Also computes the audit gain D = (L_sym - L_full)/L_full on the trained
    asymmetric model (symmetrizing its logits destroys the lag, so D >> 0)."""
    import torch
    out = {"asym": [], "sym": [], "D": []}
    for seed in seeds:
        Xtr, ylab, ytgt = make_directed(n, seed)
        Xte, _, ytgt_te = make_directed(n, seed + 100)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        for mode in ("asym", "sym"):
            m = KernelAttention(Xtr.shape[1], r=8, mode=mode, kernel="gauss", seed=seed)
            Xt = torch.tensor(Xtr); yl = torch.tensor(ylab); yt = torch.tensor(ytgt)
            opt = torch.optim.Adam(m.params(), lr=0.02, weight_decay=1e-4)
            for _ in range(steps):
                opt.zero_grad()
                pred = m.predict(Xt, Xt, yl, mask_diag=True)
                (((pred - yt) ** 2).mean()).backward()
                opt.step()
            with torch.no_grad():
                pf = m.predict(torch.tensor(Xte), Xt, yl)
                rmse = float(np.sqrt(((pf.numpy() - ytgt_te) ** 2).mean()))
            out[mode].append(rmse)
            if mode == "asym":
                with torch.no_grad():
                    Xte_t = torch.tensor(Xte)
                    S = m.scores(Xte_t, Xt)
                    Sba = m.scores(Xt, Xte_t).t()
                    A = torch.softmax(S, 1); Asym = torch.softmax(0.5 * (S + Sba), 1)
                    yfa = (A @ yl).numpy(); ysa = (Asym @ yl).numpy()
                    Lf = float(np.mean((yfa - ytgt_te) ** 2))
                    Ls = float(np.mean((ysa - ytgt_te) ** 2))
                    out["D"].append((Ls - Lf) / (abs(Lf) + 1e-9))
    out["asym"] = np.array(out["asym"]); out["sym"] = np.array(out["sym"]); out["D"] = np.array(out["D"])
    return out


# =============================================================================
# Aggregate
# =============================================================================

def run_all():
    return {
        "taiwan": run_taiwan_headtohead(),
        "law": run_first_order(),
        "ortho": run_orthogonality(),
        "directed": run_directed_headtohead(),
    }


# =============================================================================
# Figures
# =============================================================================

def make_law_figure(res=None):
    """Figure 11.1 --- three panels.
    (A) the first-order law: measured directional risk gain proportional to -<Delta, h_a>.
    (B) orthogonality on exchangeable Taiwan Credit: <Delta, h_a> concentrates at zero.
    (C) Taiwan head-to-head: symmetric matches asymmetric AUC with fewer parameters."""
    import matplotlib.pyplot as plt
    res = res or {"law": run_first_order(), "ortho": run_orthogonality(),
                  "taiwan": run_taiwan_headtohead()}
    law, ortho, tw = res["law"], res["ortho"], res["taiwan"]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # (A) first-order law
    ip, der = law["ip"], law["deriv"]
    ax[0].axhline(0, color="0.8", lw=0.8); ax[0].axvline(0, color="0.8", lw=0.8)
    ax[0].scatter(ip, der, s=18, alpha=0.6, color="#3b6fb6", edgecolor="none")
    xs = np.linspace(ip.min(), ip.max(), 50)
    b, a = np.polyfit(ip, der, 1)
    ax[0].plot(xs, b * xs + a, color="#b6403b", lw=1.8)
    ax[0].set_xlabel(r"alignment $\langle\Delta,\,h_a\rangle$")
    ax[0].set_ylabel(r"measured first-order gain $dL/d\epsilon$")
    ax[0].set_title(f"First-order law: gain $\\propto -\\langle\\Delta,h_a\\rangle$\n"
                    f"(corr $= {law['corr']:+.2f}$; vanishes iff $\\Delta\\perp h_a$)", fontsize=10)

    # (B) orthogonality histogram on Taiwan
    ti = ortho["taiwan_ips"]
    ax[1].hist(ti, bins=18, color="#6b9bd1", edgecolor="white")
    ax[1].axvline(0, color="0.3", lw=1.0, ls="--")
    ax[1].axvline(ti.mean(), color="#b6403b", lw=1.6,
                  label=f"mean ${ti.mean():+.4f}$\n(std ${ti.std():.4f}$)")
    ax[1].set_xlabel(r"$\langle\Delta,\,h_a\rangle$ over random directions")
    ax[1].set_ylabel("count")
    ax[1].set_title("Exchangeable rows (Taiwan Credit):\ndirectional content $\\perp$ response gradient",
                    fontsize=10)
    ax[1].legend(fontsize=8, loc="upper right")

    # (C) Taiwan head-to-head AUC
    a_auc, s_auc = tw["asym"], tw["sym"]
    means = [s_auc.mean(), a_auc.mean()]
    stds = [s_auc.std(), a_auc.std()]
    labels = [f"symmetric\n(shared $W$, {tw['n_params']['sym']} params)",
              f"asymmetric\n($W_Q\\neq W_K$, {tw['n_params']['asym']} params)"]
    bars = ax[2].bar([0, 1], means, yerr=stds, width=0.6,
                     color=["#4a7ab5", "#c98a3b"], capsize=5, edgecolor="white")
    ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(labels, fontsize=8.5)
    lo = min(means) - max(stds) - 0.01
    ax[2].set_ylim(lo, max(means) + max(stds) + 0.01)
    ax[2].set_ylabel("held-out AUC (mean $\\pm$ std over seeds)")
    ax[2].set_title("Head-to-head on Taiwan Credit:\nsymmetry matches, with fewer parameters",
                    fontsize=10)
    for x, mu in zip([0, 1], means):
        ax[2].text(x, mu, f" {mu:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


def make_decision_figure(directed=None, taiwan=None):
    """Figure 11.2 --- the decision. (left) the rule as a flowchart; (right) the two branches
    realized: Taiwan Credit (symmetric matches asymmetric) and the directed task (asymmetric
    earned, symmetric strictly suboptimal)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    directed = directed if directed is not None else run_directed_headtohead()
    tw = taiwan if taiwan is not None else run_taiwan_headtohead()
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 4.3))

    # (left) flowchart
    axf = ax[0]; axf.set_xlim(0, 10); axf.set_ylim(0, 10); axf.axis("off")

    def box(x, y, w, h, text, fc):
        axf.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                      boxstyle="round,pad=0.1,rounding_size=0.15", fc=fc, ec="0.3", lw=1.2))
        axf.text(x, y, text, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2, label=None):
        axf.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                      mutation_scale=14, color="0.4", lw=1.3))
        if label:
            axf.text((x1 + x2) / 2 + 0.35, (y1 + y2) / 2, label, fontsize=8.5, color="0.3")

    box(5, 9, 6.4, 1.1, "A pairwise operator over the rows", "#eef3f9")
    box(5, 7, 6.4, 1.1, "Is there a directed mechanism?\n(time, causality, transport, user$\\to$item)", "#eef3f9")
    arrow(5, 8.45, 5, 7.55)
    box(2.3, 4.6, 3.8, 1.3, "No\nexchangeable rows", "#dbe7d4")
    box(7.6, 4.6, 3.8, 1.3, "Yes\ndirected relation", "#f3e2cc")
    arrow(3.7, 6.55, 2.5, 5.3, "no")
    arrow(6.3, 6.55, 7.5, 5.3, "yes")
    box(2.3, 2.0, 4.0, 1.5, "Symmetric kernel\n(default; Bochner: no\nexpressivity lost)", "#cfe0c6")
    box(7.6, 2.0, 4.4, 1.5, "Screen $\\langle\\Delta,h_a\\rangle$;\nadopt asymmetry only if\nit clears the threshold", "#efd9bd")
    arrow(2.3, 3.95, 2.3, 2.75)
    arrow(7.6, 3.95, 7.6, 2.75)
    axf.set_title("The decision rule", fontsize=11)

    # (right) the two branches realized
    axb = ax[1]
    # normalize each task's losses to its symmetric baseline so both fit one axis (relative)
    tw_rel = [1.0, tw["asym"].mean() / tw["sym"].mean()]   # AUC ratio (higher better) -> use directly
    d_rel = [1.0, directed["asym"].mean() / directed["sym"].mean()]  # RMSE ratio (lower better)
    x = np.array([0, 1])
    w = 0.35
    # Taiwan: plot AUC directly (twin not needed; show as text). Use grouped bars of relative metric.
    axb.bar(x - w / 2, [1.0, 1.0], w, color="#4a7ab5", label="symmetric (baseline = 1)", edgecolor="white")
    axb.bar(x + w / 2, [tw_rel[1], d_rel[1]], w, color="#c98a3b", label="asymmetric", edgecolor="white")
    axb.axhline(1.0, color="0.6", lw=0.8, ls="--")
    axb.set_xticks(x)
    axb.set_xticklabels([f"Taiwan Credit\n(AUC ratio; $\\approx$1 = matched)\nsym {tw['sym'].mean():.3f} / asym {tw['asym'].mean():.3f}",
                         f"Directed lag task\n(RMSE ratio; $<$1 = asym wins)\nsym {directed['sym'].mean():.2f} / asym {directed['asym'].mean():.2f}"],
                        fontsize=8.5)
    axb.set_ylabel("asymmetric metric / symmetric metric")
    axb.set_title("The two branches realized:\nexchangeable $\\Rightarrow$ matched; directed $\\Rightarrow$ asymmetry earned",
                  fontsize=10)
    axb.legend(fontsize=8, loc="center left")
    axb.set_ylim(0, 1.3)
    fig.tight_layout()
    return fig


# =============================================================================
# CLI
# =============================================================================

def main(argv=None):
    p = argparse.ArgumentParser(description="Chapter 11 — symmetry suffices")
    p.add_argument("--out-prefix", default=None)
    args = p.parse_args(argv)
    set_style()

    res = run_all()
    tw, law, ortho, di = res["taiwan"], res["law"], res["ortho"], res["directed"]
    print("=" * 74)
    print("HEAD-TO-HEAD (Taiwan Credit): symmetric vs asymmetric attention smoother")
    print(f"  asym AUC {tw['asym'].mean():.3f} +/- {tw['asym'].std():.3f}  ({tw['n_params']['asym']} params)")
    print(f"  sym  AUC {tw['sym'].mean():.3f} +/- {tw['sym'].std():.3f}  ({tw['n_params']['sym']} params)")
    print("-" * 74)
    print(f"FIRST-ORDER LAW (directed family): corr(<Delta,h_a>, dL/deps) = {law['corr']:+.3f}")
    print(f"ORTHOGONALITY (Taiwan): <Delta,h_a> mean {ortho['taiwan_ips'].mean():+.4f} "
          f"std {ortho['taiwan_ips'].std():.4f}")
    print(f"  directed aligned <Delta,h_a> = {ortho['directed_aligned']:+.3f} "
          f"(vs random std {ortho['directed_random_std']:.3f})")
    print("-" * 74)
    print(f"DIRECTED TASK (asymmetry earned): asym RMSE {di['asym'].mean():.3f}, "
          f"sym RMSE {di['sym'].mean():.3f}, audit D median {np.median(di['D']):+.2f}")

    if args.out_prefix:
        import matplotlib
        matplotlib.use("Agg")
        make_law_figure(res).savefig(f"{args.out_prefix}1_symmetry_law.pdf")
        make_decision_figure(di, tw).savefig(f"{args.out_prefix}2_decision.pdf")
        print("wrote figures with prefix", args.out_prefix)


if __name__ == "__main__":
    main()
