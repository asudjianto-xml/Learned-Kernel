"""Chapter 11 — symmetry suffices.

Two facts about a table fix its default geometry. Rows are *exchangeable* — an iid sample with no
canonical order — so a permutation-invariant operator over them factors through a symmetric
statistic, which forces a *symmetric* Gram k(x,x') = k(x',x). Bochner (Ch. 8) closes the loop: the
symmetric stationary class already contains every stationary geometry, so committing to symmetry
sacrifices no expressivity. Asymmetry is extra capacity; on an exchangeable table it has nothing to
represent, and — as this module measures — it carries a real cost.

The chapter is built around the book's *sophisticated* kernels, not a toy.

1. **The first-order law.** Decompose any pairwise score into symmetric + antisymmetric parts under
   the swap x<->x'. The directional content Delta = k - k^T does predictive work only insofar as it
   aligns with the response gradient h_a(x,x') = m(x) - m(x'): the first-order risk change along
   k_s + eps*Delta is -2<Delta, h_a>, so the gain is proportional to that alignment and vanishes
   when Delta _|_ h_a. On exchangeable Taiwan Credit the alignment over random directions
   concentrates at zero.

2. **Asymmetrizing a sophisticated PSD kernel costs KRR.** Give a spectral (Bochner) kernel's
   feature map separate query/key transforms (M_Q != M_K) and it is no longer symmetric PSD, so it
   falls out of the RKHS: KRR's (K + lambda I)^-1 solve no longer applies and the predictor drops to
   row-normalization (NW). On Taiwan the asymmetric arm gains nothing AND forfeits KRR, so it trails
   the symmetric kernel by the full KRR-vs-NW gap.

3. **The book's symmetric kernels + KRR beat asymmetric attention.** The real spectral (Ch. 8) and
   tree leaf (Ch. 4) kernels, paired with KRR (which symmetry unlocks), beat an asymmetric-attention
   NW smoother on held-out AUC.

4. **Use both, and read the diagnostic.** A two-channel fusion (Ch. 10) of a symmetric PSD channel
   (KRR) and an asymmetric channel (NW), with the asymmetry weight rho chosen leakage-free on a
   query fold, contains both endpoints. The earned weight rho* is the diagnostic: rho* -> 0 on an
   exchangeable table (symmetry suffices), rho* > 0 on a directed task (asymmetry earned).

The directed/earned case (a lagged-signal task) is the situation taken up by the temporal kernels of
Ch. 23 and the asymmetric-kernel theory of Ch. 24.

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
# Attention as a kernel: a Nadaraya--Watson smoother whose only knob is symmetry
# =============================================================================

class KernelAttention:
    """Attention as a Nadaraya--Watson smoother over a support set, with a learned linear encoder.
    The geometry is the bilinear form applied in the encoded space; the prediction is the
    softmax-weighted average of support labels. The *only* structural difference between modes is
    symmetry:

    - ``asym``: separate W_Q, W_K  -> asymmetric whenever W_Q != W_K.
    - ``sym`` : shared W (W_Q = W_K) -> a symmetric kernel (PSD Gram for ``dot``, symmetric distance
      for ``gauss``).

    Fed raw features it is self-attention; fed a spectral feature map it asymmetrizes that kernel.
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
        else:
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
def _fit_attention(Xtr, ytr, mode, seed, kernel="dot", steps=300, lr=0.02, wd=1e-3, task="cls",
                   r=16, target=None):
    """Train an attention smoother. The support LABELS being smoothed are ``ytr``; the loss is
    against ``target`` (defaults to ``ytr``). For a directed task the two differ — smooth the
    signal s(pos) but fit the lagged target s(pos-lag)."""
    import torch
    Xt = torch.tensor(Xtr); yt = torch.tensor(np.asarray(ytr, float))
    tgt = yt if target is None else torch.tensor(np.asarray(target, float))
    m = KernelAttention(Xtr.shape[1], r=r, mode=mode, kernel=kernel, seed=seed)
    opt = torch.optim.Adam(m.params(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        pred = m.predict(Xt, Xt, yt, mask_diag=True)
        if task == "cls":
            p = pred.clamp(1e-6, 1 - 1e-6)
            loss = -(tgt * torch.log(p) + (1 - tgt) * torch.log(1 - p)).mean()
        else:
            loss = ((pred - tgt) ** 2).mean()
        loss.backward()
        opt.step()
    return m


def _attn_predict(m, Xte, Xtr, ytr):
    import torch
    with torch.no_grad():
        S = m.scores(torch.tensor(Xte), torch.tensor(Xtr))
        return (torch.softmax(S, 1) @ torch.tensor(np.asarray(ytr, float))).numpy()


def run_taiwan_headtohead(seeds=range(6), n_train=800, n_test=800):
    """Isolate symmetry: ``asym`` (W_Q!=W_K) vs ``sym`` (shared W) attention smoothers, NW both,
    everything else fixed. Matched held-out AUC at fewer parameters."""
    from sklearn.metrics import roc_auc_score
    d = load_taiwan()
    out = {"asym": [], "sym": [], "n_params": {}}
    for seed in seeds:
        Xtr, ytr = _subset(d.Xtr, d.ytr, n_train, seed)
        Xte, yte = _subset(d.Xte, d.yte, n_test, seed + 100)
        for mode in ("asym", "sym"):
            m = _fit_attention(Xtr, ytr, mode, seed)
            out[mode].append(roc_auc_score(yte, _attn_predict(m, Xte, Xtr, ytr.astype(float))))
            out["n_params"][mode] = m.n_params()
    for k in ("asym", "sym"):
        out[k] = np.array(out[k])
    return out


# =============================================================================
# Spectral (Bochner) features and KRR; asymmetrizing a sophisticated PSD kernel
# =============================================================================

class SpectralFeatures:
    """Bochner random Fourier features: a Gaussian spectral measure read out as cos features
    approximates an RBF kernel (Ch. 8). phi(x) = sqrt(2/D) cos(Omega x + b), Omega ~ N(0, 2 gamma).
    The symmetric spectral kernel uses the SAME map on both arguments; asymmetrizing gives the map
    separate query/key transforms."""

    def __init__(self, p, D=64, gamma=0.05, seed=0):
        g = np.random.default_rng(seed)
        self.Omega = g.normal(0.0, np.sqrt(2 * gamma), (D, p))
        self.b = g.uniform(0, 2 * np.pi, D)
        self.D = D

    def __call__(self, X):
        X = np.asarray(X, float)
        return np.sqrt(2.0 / self.D) * np.cos(X @ self.Omega.T + self.b)


def _krr_predict(Ptr, ytr, Pte, T, lam):
    from scipy.spatial.distance import cdist
    Ks = np.exp(-cdist(Ptr, Ptr) ** 2 / (2 * T ** 2))
    Kq = np.exp(-cdist(Pte, Ptr) ** 2 / (2 * T ** 2))
    ybar = ytr.mean()
    alpha = np.linalg.solve(Ks + lam * np.eye(len(Ks)), ytr - ybar)
    return Kq @ alpha + ybar


def _krr_select(Ptr, ytr, Pva, yva, Pte, task,
                Tgrid=np.geomspace(0.3, 6, 7), lamgrid=np.geomspace(1e-3, 1, 6)):
    from sklearn.metrics import roc_auc_score
    best = None
    for T in Tgrid:
        for lam in lamgrid:
            pv = _krr_predict(Ptr, ytr, Pva, T, lam)
            s = roc_auc_score(yva, pv) if task == "cls" else -np.sqrt(np.mean((pv - yva) ** 2))
            if best is None or s > best[0]:
                best = (s, T, lam)
    return _krr_predict(Ptr, ytr, Pte, best[1], best[2])


def run_spectral_cost(seeds=range(4), n=800, D=64, gamma=0.05):
    """Asymmetrize the spectral kernel and price it. Three arms on the SAME spectral feature map:
    (1) sym + KRR (the RKHS predictor symmetry unlocks), (2) sym + NW, (3) asym + NW (M_Q != M_K,
    so KRR no longer applies). On exchangeable Taiwan the asymmetric arm gains nothing AND forfeits
    KRR. Also reports the audit gain D and the alignment <Delta,h_a> for the asymmetric arm."""
    from sklearn.metrics import roc_auc_score
    d = load_taiwan()
    phi = SpectralFeatures(d.Xtr.shape[1], D=D, gamma=gamma, seed=0)
    out = {"sym_krr": [], "sym_nw": [], "asym_nw": [], "D": [], "ip": []}
    for seed in seeds:
        Xtr, ytr = _subset(d.Xtr, d.ytr, n, seed)
        Xte, yte = _subset(d.Xte, d.yte, n, seed + 100)
        Ptr, Pte = phi(Xtr), phi(Xte)
        nv = n // 4
        out["sym_krr"].append(roc_auc_score(yte, _krr_select(
            Ptr[nv:], ytr[nv:].astype(float), Ptr[:nv], ytr[:nv].astype(float), Pte, "cls")))
        ms = _fit_attention(Ptr, ytr, "sym", seed, kernel="gauss")
        out["sym_nw"].append(roc_auc_score(yte, _attn_predict(ms, Pte, Ptr, ytr.astype(float))))
        ma = _fit_attention(Ptr, ytr, "asym", seed, kernel="gauss")
        out["asym_nw"].append(roc_auc_score(yte, _attn_predict(ma, Pte, Ptr, ytr.astype(float))))
        out["D"].append(_audit_gain(ma, Pte, Ptr, ytr.astype(float), yte.astype(float), "cls"))
        out["ip"].append(np.mean([ip_delta_ha(Ptr, ytr.astype(float) - ytr.mean(), skew(D, s), seed=s)
                                  for s in range(20)]))
    for k in ("sym_krr", "sym_nw", "asym_nw"):
        out[k] = np.array(out[k])
    return out


def run_real_headtohead(seeds=range(3), n_train=800):
    """The book's symmetric kernels (spectral Ch. 8, tree leaf Ch. 4) + KRR vs asymmetric attention
    + NW, on Taiwan. Reproduces the spectral / tree / fused KRR AUCs by driving Ch. 10's
    leakage-free fusion per seed. The symmetric kernels, using KRR (which symmetry unlocks), beat
    asymmetric attention."""
    from lkbook.chapters import ch10
    out = {"spectral_krr": [], "tree_krr": [], "fused_krr": []}
    for s in seeds:
        r = ch10.run_taiwan(seed=s, n_train=n_train)
        out["spectral_krr"].append(r["single_auc"]["spectral"])
        out["tree_krr"].append(r["single_auc"]["tree"])
        out["fused_krr"].append(r["fused_auc"])
    th = run_taiwan_headtohead(seeds=seeds, n_train=min(n_train, 800))
    out["asym_attn_nw"] = th["asym"]
    for k in ("spectral_krr", "tree_krr", "fused_krr"):
        out[k] = np.array(out[k])
    return out


# =============================================================================
# The first-order law and orthogonality on exchangeable data  (pure numpy)
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
    """<Delta, h_a> = E[ Delta(x_i, x_j) * (m(x_i) - m(x_j)) ], estimated over random pairs."""
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
    """A directed (lagged-signal) task. The support label at a point is the signal s(pos); the query
    target is the *lagged* signal s(pos - lag). Predicting the target requires attending upstream by
    a fixed offset — a directed relation no symmetric distance kernel can represent. This is a
    positional/temporal mechanism, the regime taken up in Ch. 23."""
    g = np.random.default_rng(seed)
    pos = np.sort(g.uniform(0, 1, n))
    X = np.column_stack([pos, g.normal(0, 1, (n, 3))])
    s = lambda t: np.sin(2 * np.pi * freq * t)
    return X.astype(float), (s(pos) + noise * g.normal(0, 1, n)).astype(float), s(pos - lag).astype(float)


def run_first_order(n_dirs=30):
    """Verify the first-order law on the directed task: the measured directional risk gain is
    proportional to -<Delta, h_a> (corr ~ -1). The family interpolates each random direction toward
    the aligned (directed) direction so <Delta, h_a> sweeps a range."""
    Xtr, ylab, ytgt = make_directed(900, 0)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd
    Xte, _, ytgt_te = make_directed(900, 100)
    Xte = (Xte - mu) / sd
    Xa = np.column_stack([Xtr, np.ones(len(Xtr))])
    Xa_te = np.column_stack([Xte, np.ones(len(Xte))])
    ra = Xa.shape[1]
    ell = np.sqrt(ra)
    A_align = np.zeros((ra, ra)); A_align[0, ra - 1] = 1.0; A_align[ra - 1, 0] = -1.0
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
    return {"ip": ips, "deriv": ders, "corr": float(np.corrcoef(ips, ders)[0, 1]),
            "ip_align": ip_delta_ha(Xa, ytgt, A_align)}


def run_orthogonality(n_dirs=60, n=1500):
    """On exchangeable Taiwan Credit, <Delta, h_a> over random antisymmetric directions concentrates
    at zero. Returns the distribution and the directed-task aligned value for contrast."""
    d = load_taiwan()
    X, y = _subset(d.Xtr, d.ytr.astype(float), n, 0)
    y = y - y.mean()
    r = X.shape[1]
    ips = np.array([ip_delta_ha(X, y, skew(r, s), seed=s) for s in range(n_dirs)])
    Xd, _, ytgt = make_directed(1500, 0)
    mu, sd = Xd.mean(0), Xd.std(0) + 1e-9
    Xd = (Xd - mu) / sd
    Xa = np.column_stack([Xd, np.ones(len(Xd))])
    ra = Xa.shape[1]
    A_align = np.zeros((ra, ra)); A_align[0, ra - 1] = 1.0; A_align[ra - 1, 0] = -1.0
    A_align /= np.linalg.norm(A_align)
    return {"taiwan_ips": ips, "directed_aligned": ip_delta_ha(Xa, ytgt, A_align),
            "directed_random_std": float(np.array(
                [ip_delta_ha(Xa, ytgt, skew(ra, s), seed=s + 500) for s in range(n_dirs)]).std())}


# =============================================================================
# The audit gain D and the directed head-to-head
# =============================================================================

def _audit_gain(m, Xte, Xtr, ytr, yte, task):
    """D = (L_sym - L_full)/L_full on a trained asymmetric model: symmetrizing its logits (role
    swap) and measuring the held-out loss change. ~0 on exchangeable data, large on directed."""
    import torch
    with torch.no_grad():
        Xt = torch.tensor(Xtr); Xe = torch.tensor(Xte); y = torch.tensor(np.asarray(ytr, float))
        S = m.scores(Xe, Xt); Sba = m.scores(Xt, Xe).t()
        A = torch.softmax(S, 1); Asym = torch.softmax(0.5 * (S + Sba), 1)
        yf = (A @ y).numpy(); ys = (Asym @ y).numpy()
        if task == "cls":
            cl = lambda p: np.clip(p, 1e-6, 1 - 1e-6)
            L = lambda p: float(-np.mean(yte * np.log(cl(p)) + (1 - yte) * np.log(1 - cl(p))))
        else:
            L = lambda p: float(np.mean((p - yte) ** 2))
        Lf = L(yf)
        return (L(ys) - Lf) / (abs(Lf) + 1e-9)


def run_directed_headtohead(seeds=range(4), n=400, steps=400):
    """Head-to-head on the directed lag task: the asymmetric smoother captures the offset; the
    symmetric one cannot. Reports the audit gain D on the trained asymmetric model."""
    out = {"asym": [], "sym": [], "D": []}
    for seed in seeds:
        Xtr, ylab, ytgt = make_directed(n, seed)
        Xte, _, ytgt_te = make_directed(n, seed + 100)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        for mode in ("asym", "sym"):
            m = _fit_attention(Xtr, ylab, mode, seed, kernel="gauss", steps=steps, wd=1e-4,
                               task="reg", r=8, target=ytgt)
            pf = _attn_predict(m, Xte, Xtr, ylab)
            out[mode].append(float(np.sqrt(np.mean((pf - ytgt_te) ** 2))))
            if mode == "asym":
                out["D"].append(_audit_gain(m, Xte, Xtr, ylab, ytgt_te, "reg"))
    out["asym"] = np.array(out["asym"]); out["sym"] = np.array(out["sym"]); out["D"] = np.array(out["D"])
    return out


def directed_one_lag(lag, seed=0, n=400, steps=450):
    """Symmetric vs asymmetric held-out RMSE on the directed task at one lag (fast, one seed) — for
    the explorer. At lag 0 the task is undirected and the two match; the gap grows with the lag."""
    out = {}
    Xtr, ylab, ytgt = make_directed(n, seed, lag=lag)
    Xte, _, ytgt_te = make_directed(n, seed + 100, lag=lag)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    for mode in ("sym", "asym"):
        m = _fit_attention(Xtr, ylab, mode, seed, kernel="gauss", steps=steps, wd=1e-4,
                           task="reg", r=8, target=ytgt)
        out[mode] = float(np.sqrt(np.mean((_attn_predict(m, Xte, Xtr, ylab) - ytgt_te) ** 2)))
    return out


# =============================================================================
# Use both: fuse a symmetric (KRR) and an asymmetric (NW) channel; read rho*
# =============================================================================

def _fuse_rho(ysym_va, yasym_va, yva, ysym_te, yasym_te, task):
    from sklearn.metrics import roc_auc_score
    best = None
    for rho in np.linspace(0, 1, 21):
        pv = (1 - rho) * ysym_va + rho * yasym_va
        s = roc_auc_score(yva, pv) if task == "cls" else -np.sqrt(np.mean((pv - yva) ** 2))
        if best is None or s > best[0]:
            best = (s, rho)
    rho = best[1]
    return rho, (1 - rho) * ysym_te + rho * yasym_te


def run_fusion_diagnostic(seeds=range(3), n_taiwan=700, n_directed=400):
    """Fuse a symmetric PSD channel (RBF + KRR) and an asymmetric channel (attention + NW); choose
    the asymmetry weight rho leakage-free on a query fold. The earned rho* is the diagnostic:
    rho* -> 0 on exchangeable Taiwan, rho* large on the directed task."""
    from scipy.spatial.distance import cdist
    from sklearn.metrics import roc_auc_score

    def rbf_krr(Xtr, ytr, Xte, ell, lam):
        K = np.exp(-cdist(Xtr, Xtr) ** 2 / (2 * ell ** 2))
        Kq = np.exp(-cdist(Xte, Xtr) ** 2 / (2 * ell ** 2))
        yb = ytr.mean()
        return Kq @ np.linalg.solve(K + lam * np.eye(len(K)), ytr - yb) + yb

    d = load_taiwan()
    out = {"taiwan_rho": [], "directed_rho": [], "taiwan_fused_auc": [], "directed_fused_rmse": []}
    # exchangeable
    for seed in seeds:
        Xtr, ytr = _subset(d.Xtr, d.ytr, n_taiwan, seed)
        Xte, yte = _subset(d.Xte, d.yte, n_taiwan, seed + 100)
        nv = n_taiwan // 4
        ell = np.sqrt(Xtr.shape[1])
        ys_v = rbf_krr(Xtr[nv:], ytr[nv:].astype(float), Xtr[:nv], ell, 0.1)
        ys_t = rbf_krr(Xtr[nv:], ytr[nv:].astype(float), Xte, ell, 0.1)
        m = _fit_attention(Xtr[nv:], ytr[nv:], "asym", seed, kernel="gauss")
        ya_v = _attn_predict(m, Xtr[:nv], Xtr[nv:], ytr[nv:].astype(float))
        ya_t = _attn_predict(m, Xte, Xtr[nv:], ytr[nv:].astype(float))
        rho, yf = _fuse_rho(ys_v, ya_v, ytr[:nv], ys_t, ya_t, "cls")
        out["taiwan_rho"].append(rho)
        out["taiwan_fused_auc"].append(roc_auc_score(yte, yf))
    # directed
    for seed in seeds:
        Xtr, ylab, ytgt = make_directed(n_directed, seed)
        Xte, _, ytgt_te = make_directed(n_directed, seed + 100)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        nv = n_directed // 4
        ell = np.sqrt(Xtr.shape[1])
        ys_v = rbf_krr(Xtr[nv:], ylab[nv:], Xtr[:nv], ell, 0.05)
        ys_t = rbf_krr(Xtr[nv:], ylab[nv:], Xte, ell, 0.05)
        m = _fit_attention(Xtr[nv:], ylab[nv:], "asym", seed, kernel="gauss", steps=400, wd=1e-4,
                           task="reg", r=8, target=ytgt[nv:])
        ya_v = _attn_predict(m, Xtr[:nv], Xtr[nv:], ylab[nv:])
        ya_t = _attn_predict(m, Xte, Xtr[nv:], ylab[nv:])
        rho, yf = _fuse_rho(ys_v, ya_v, ytgt[:nv], ys_t, ya_t, "reg")
        out["directed_rho"].append(rho)
        out["directed_fused_rmse"].append(float(np.sqrt(np.mean((yf - ytgt_te) ** 2))))
    for k in list(out):
        out[k] = np.array(out[k])
    return out


# =============================================================================
# Aggregate
# =============================================================================

def run_all(real_seeds=range(3), real_n_train=800):
    return {
        "law": run_first_order(),
        "ortho": run_orthogonality(),
        "spectral_cost": run_spectral_cost(),
        "real": run_real_headtohead(seeds=real_seeds, n_train=real_n_train),
        "fusion": run_fusion_diagnostic(),
        "directed": run_directed_headtohead(),
    }


# =============================================================================
# Figures
# =============================================================================

def make_law_figure(res=None):
    """Figure 11.1 --- the law and the cost.
    (A) the first-order law: directional risk gain proportional to -<Delta,h_a>.
    (B) orthogonality on exchangeable Taiwan Credit: <Delta,h_a> concentrates at zero.
    (C) asymmetrizing the spectral kernel: sym+KRR vs sym+NW vs asym+NW (the cost of losing KRR)."""
    import matplotlib.pyplot as plt
    res = res or {"law": run_first_order(), "ortho": run_orthogonality(),
                  "spectral_cost": run_spectral_cost()}
    law, ortho, sc = res["law"], res["ortho"], res["spectral_cost"]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))

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

    means = [sc["sym_krr"].mean(), sc["sym_nw"].mean(), sc["asym_nw"].mean()]
    stds = [sc["sym_krr"].std(), sc["sym_nw"].std(), sc["asym_nw"].std()]
    labels = ["symmetric\n+ KRR", "symmetric\n+ NW", "asymmetric\n+ NW"]
    cols = ["#2e7d32", "#7aa6c2", "#c98a3b"]
    ax[2].bar([0, 1, 2], means, yerr=stds, width=0.62, color=cols, capsize=4, edgecolor="white")
    ax[2].set_xticks([0, 1, 2]); ax[2].set_xticklabels(labels, fontsize=9)
    lo = min(means) - max(stds) - 0.01
    ax[2].set_ylim(lo, max(means) + max(stds) + 0.02)
    ax[2].set_ylabel("held-out AUC")
    ax[2].set_title("Asymmetrizing the spectral kernel:\nasymmetry gains nothing and forfeits KRR",
                    fontsize=10)
    for x, mu in zip([0, 1, 2], means):
        ax[2].text(x, mu, f"{mu:.3f}", ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout()
    return fig


def make_kernels_figure(res=None):
    """Figure 11.2 --- the book's symmetric kernels beat asymmetric attention, and the fusion
    diagnostic. (left) spectral / tree / fused + KRR vs asymmetric attention + NW on Taiwan.
    (right) the earned asymmetry weight rho*: ~0 on exchangeable Taiwan, large on the directed task."""
    import matplotlib.pyplot as plt
    res = res or {"real": run_real_headtohead(), "fusion": run_fusion_diagnostic()}
    real, fu = res["real"], res["fusion"]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))

    names = ["spectral\n+ KRR", "tree\n+ KRR", "fused\n+ KRR", "attention\n+ NW"]
    vals = [real["spectral_krr"].mean(), real["tree_krr"].mean(),
            real["fused_krr"].mean(), real["asym_attn_nw"].mean()]
    errs = [real["spectral_krr"].std(), real["tree_krr"].std(),
            real["fused_krr"].std(), real["asym_attn_nw"].std()]
    cols = ["#2e7d32", "#2e7d32", "#2e7d32", "#c98a3b"]
    ax[0].bar(range(4), vals, yerr=errs, width=0.62, color=cols, capsize=4, edgecolor="white")
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(names, fontsize=9)
    ax[0].set_ylim(min(vals) - max(errs) - 0.02, max(vals) + max(errs) + 0.02)
    ax[0].set_ylabel("held-out AUC")
    ax[0].set_title("Symmetric (book kernels + KRR) vs asymmetric attention\non Taiwan Credit",
                    fontsize=10)
    for x, v in zip(range(4), vals):
        ax[0].text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

    tr, dr = fu["taiwan_rho"], fu["directed_rho"]
    ax[1].bar([0, 1], [tr.mean(), dr.mean()], yerr=[tr.std(), dr.std()], width=0.5,
              color=["#4a7ab5", "#c98a3b"], capsize=5, edgecolor="white")
    ax[1].axhline(0.0, color="0.6", lw=0.8)
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["Taiwan Credit\n(exchangeable)", "directed lag task\n(temporal, Ch. 23)"],
                          fontsize=9)
    ax[1].set_ylabel(r"earned asymmetry weight $\rho^*$")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("The diagnostic: $\\rho^*$ from the symmetric+asymmetric fusion\n"
                    "$\\rho^*\\!\\to\\!0$ exchangeable, $\\rho^*\\!>\\!0$ directed", fontsize=10)
    for x, v in zip([0, 1], [tr.mean(), dr.mean()]):
        ax[1].text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


def make_decision_figure():
    """Figure 11.3 --- the decision rule as a flowchart."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, axf = plt.subplots(figsize=(7.2, 5.0))
    axf.set_xlim(0, 10); axf.set_ylim(0, 10); axf.axis("off")

    def box(x, y, w, h, text, fc):
        axf.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                      boxstyle="round,pad=0.1,rounding_size=0.15", fc=fc, ec="0.3", lw=1.2))
        axf.text(x, y, text, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2, label=None):
        axf.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                      mutation_scale=14, color="0.4", lw=1.3))
        if label:
            axf.text((x1 + x2) / 2 + 0.4, (y1 + y2) / 2, label, fontsize=8.5, color="0.3")

    box(5, 9.2, 7.2, 1.0, "Fuse a symmetric (PSD + KRR) and an\nasymmetric (NW) channel; fit $\\rho$ leakage-free", "#eef3f9")
    box(5, 6.9, 4.6, 1.0, "Read the earned weight $\\rho^*$", "#eef3f9")
    arrow(5, 8.7, 5, 7.45)
    box(2.4, 4.3, 4.0, 1.3, "$\\rho^*\\approx 0$\nno directed mechanism", "#dbe7d4")
    box(7.6, 4.3, 4.2, 1.3, "$\\rho^*>0$\ndirected mechanism", "#f3e2cc")
    arrow(3.9, 6.4, 2.7, 5.0, "$\\approx 0$")
    arrow(6.1, 6.4, 7.4, 5.0, "$>0$")
    box(2.4, 1.6, 4.4, 1.5, "Symmetric kernel + KRR\n(default; Bochner loses\nnothing, KRR is unlocked)", "#cfe0c6")
    box(7.6, 1.6, 4.6, 1.5, "Keep the asymmetric channel\n(time / causality / transport;\nCh. 23, Ch. 24)", "#efd9bd")
    arrow(2.4, 3.65, 2.4, 2.35)
    arrow(7.6, 3.65, 7.6, 2.35)
    axf.set_title("Symmetry by default, asymmetry on evidence", fontsize=11)
    fig.tight_layout()
    return fig


# =============================================================================
# CLI
# =============================================================================

def main(argv=None):
    p = argparse.ArgumentParser(description="Chapter 11 — symmetry suffices")
    p.add_argument("--out-prefix", default=None)
    p.add_argument("--real-seeds", type=int, default=3)
    args = p.parse_args(argv)
    set_style()

    res = run_all(real_seeds=range(args.real_seeds))
    law, ortho, sc, real, fu, di = (res["law"], res["ortho"], res["spectral_cost"],
                                    res["real"], res["fusion"], res["directed"])
    print("=" * 76)
    print(f"FIRST-ORDER LAW: corr(<Delta,h_a>, dL/deps) = {law['corr']:+.3f}")
    print(f"ORTHOGONALITY (Taiwan): <Delta,h_a> mean {ortho['taiwan_ips'].mean():+.4f} "
          f"std {ortho['taiwan_ips'].std():.4f}")
    print("-" * 76)
    print("ASYMMETRIZING THE SPECTRAL KERNEL (Taiwan):")
    print(f"  sym + KRR {sc['sym_krr'].mean():.3f}   sym + NW {sc['sym_nw'].mean():.3f}   "
          f"asym + NW {sc['asym_nw'].mean():.3f}   (audit D {np.median(sc['D']):+.3f})")
    print("-" * 76)
    print("BOOK KERNELS + KRR vs ASYMMETRIC ATTENTION + NW (Taiwan):")
    print(f"  spectral {real['spectral_krr'].mean():.3f}   tree {real['tree_krr'].mean():.3f}   "
          f"fused {real['fused_krr'].mean():.3f}   |   attention {real['asym_attn_nw'].mean():.3f}")
    print("-" * 76)
    print(f"FUSION DIAGNOSTIC rho*: Taiwan {fu['taiwan_rho'].mean():.2f}   "
          f"directed {fu['directed_rho'].mean():.2f}")
    print(f"DIRECTED HEAD-TO-HEAD: asym RMSE {di['asym'].mean():.3f}, sym RMSE {di['sym'].mean():.3f}")

    if args.out_prefix:
        import matplotlib
        matplotlib.use("Agg")
        make_law_figure(res).savefig(f"{args.out_prefix}1_symmetry_law.pdf")
        make_kernels_figure(res).savefig(f"{args.out_prefix}2_kernels_diagnostic.pdf")
        make_decision_figure().savefig(f"{args.out_prefix}3_decision.pdf")
        print("wrote figures with prefix", args.out_prefix)


if __name__ == "__main__":
    main()
