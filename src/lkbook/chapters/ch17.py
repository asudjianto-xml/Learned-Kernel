"""Chapter 17 — linear-in-n: Nystrom and landmarks.

Matrix-free (Ch. 16) removed the memory wall exactly, but the flop wall -- ``O(n^2)``
-- still stands: every matvec touches all ``n`` columns. To go *linear* in ``n`` the
kernel itself must be approximated. The spectral kernel is effectively low rank, so
most of ``K`` is redundant: pick ``m`` landmark points ``S``, build the kernel's own
Nystrom feature map ``phi~(x) = Lambda_r^{-1/2} V_r^T k(x,S)`` on them, and the dual
KRR becomes a primal ridge in that ``r``-dimensional map at ``O(n m^2 + m^3)`` cost --
linear in ``n``. Both walls fall at once, at the price of an approximation floor set by
the discarded tail of the spectrum.

Inducing-point sparse GPs and random Fourier features are the same low-rank idea; the
Nystrom map doubles as a reusable supervised embedding. Faithful port of
``skm/linalg.py`` (``nystrom_factor``, ``nystrom_features``, ``nystrom_solve``).

    python -m lkbook.chapters.ch17 --out-prefix fig17
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from lkbook.chapters.ch15 import (MultiScaleKernel, load_california_scaled, tile_jitter,
                                   dense_krr_solve, predict, r2_score, dense_gram_gb,
                                   _peak_gb, LAM, SEED)


def _robust_cholesky(A, base_jitter=1e-6, tries=8):
    """Cholesky with escalating diagonal jitter (as in ``skm.decoders``)."""
    import torch
    jit = base_jitter * float(A.diagonal().mean())
    eye = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    for _ in range(tries):
        try:
            return torch.linalg.cholesky(A + jit * eye)
        except Exception:
            jit *= 10
    return torch.linalg.cholesky(A + jit * eye)


# =============================================================================
# The Nystrom feature map on m landmarks (path B: linear in n)
# =============================================================================

def nystrom_factor(emb_S, kmat, rank=None, tol=1e-10):
    """The kernel's Nystrom feature map on the ``m`` landmark points ``S``.

    Eigendecomposes the landmark Gram ``K_mm = V Lambda V^T`` and returns the
    projector ``P = V_r Lambda_r^{-1/2}`` (keeping the top ``rank`` eigenpairs, or all
    above ``tol * max``). The induced map ``phi~(x) = k(x,S) @ P`` reproduces the
    Nystrom approximation of the learned kernel. Returns ``(P, kept_eigenvalues)``.
    Faithful port of ``skm.linalg.nystrom_factor``."""
    import torch
    Kmm = kmat(emb_S, emb_S)
    Kmm = 0.5 * (Kmm + Kmm.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(Kmm)
    evals = evals.clamp_min(0.0)
    keep = evals > tol * evals.max().clamp_min(tol)
    evals, evecs = evals[keep], evecs[:, keep]
    if rank is not None and evals.numel() > rank:
        idx = torch.argsort(evals, descending=True)[:rank]
        evals, evecs = evals[idx], evecs[:, idx]
    P = evecs * evals.clamp_min(tol).rsqrt()
    return P, evals


def nystrom_features(emb_q, emb_S, P, kmat, block=None):
    """The r-dimensional Nystrom embedding ``phi~(q) = k(q,S) @ P``, query-blocked.
    Faithful port of ``skm.linalg.nystrom_features``."""
    import torch
    if block is None:
        return kmat(emb_q, emb_S) @ P
    q = emb_q[0].shape[0]
    out = torch.empty(q, P.shape[1], dtype=P.dtype, device=P.device)
    for i in range(0, q, block):
        sl = slice(i, min(i + block, q))
        out[sl] = kmat([e[sl] for e in emb_q], emb_S) @ P
    return out


def nystrom_solve(embeds, emb_S, P, kmat, y, lam=LAM, block=8192):
    """Primal KRR in the Nystrom map -- linear in n, no ``n x n`` Gram.

    Streams the train set in row-blocks, accumulating the ``r x r`` normal matrix
    ``G = Phi~^T Phi~`` and the ``r x C`` right-hand side ``Phi~^T y``, then solves the
    small ridge ``(G + lam I) beta = Phi~^T y``. This is exactly the Nystrom
    (subset-of-regressors) estimator. Cost ``O(n r^2 + r^3)``. Faithful port of
    ``skm.linalg.nystrom_solve``."""
    import torch
    n, r, C = embeds[0].shape[0], P.shape[1], y.shape[1]
    G = torch.zeros(r, r, dtype=P.dtype, device=P.device)
    rhs = torch.zeros(r, C, dtype=P.dtype, device=P.device)
    for i in range(0, n, block):
        sl = slice(i, min(i + block, n))
        Phi = kmat([e[sl] for e in embeds], emb_S) @ P         # (b, r)
        G += Phi.transpose(-1, -2) @ Phi
        rhs += Phi.transpose(-1, -2) @ y[sl]
    A = G + lam * torch.eye(r, dtype=P.dtype, device=P.device)
    return torch.cholesky_solve(rhs, _robust_cholesky(A))       # (r, C)


def select_landmarks(X, m, method="uniform", seed=SEED):
    """Choose ``m`` landmark rows: ``uniform`` random, or ``kmeans`` centroids that
    spread over the input geometry. Leverage-score sampling is discussed in the text;
    uniform is the cheap default when the spectrum decays fast."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    if method == "uniform":
        idx = rng.choice(X.shape[0], size=min(m, X.shape[0]), replace=False)
        return X[idx]
    if method == "kmeans":
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=min(m, X.shape[0]), n_init=3, random_state=seed).fit(X)
        return km.cluster_centers_
    raise ValueError(method)


# =============================================================================
# Random Fourier features: the explicit-map cousin (exact primal decode)
# =============================================================================

def rff_map(X, M=512, ell=2.0, seed=SEED, device=None, dtype=None):
    """Random Fourier features for a Gaussian kernel of length scale ``ell``:
    ``Psi(x) = sqrt(1/M) [cos(x W + b), sin(x W)]`` with ``W ~ N(0, 1/ell^2)``. An
    *explicit* finite map, so KRR is the exact primal ridge -- the ``kernel='gram'``
    idea, no landmarks, a resolution cost in ``M``."""
    import torch
    dtype = dtype or torch.float64
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.as_tensor(np.asarray(X), dtype=dtype, device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    d = X.shape[1]
    W = (torch.randn(d, M, generator=gen, dtype=dtype) / ell).to(device)
    arg = X @ W
    return torch.cat([torch.cos(arg), torch.sin(arg)], dim=1) * (1.0 / M) ** 0.5


def rff_solve(Xtr, y, Xte, M=512, ell=2.0, lam=LAM, seed=SEED):
    """Exact primal ridge in the RFF map: ``beta = (Psi^T Psi + lam I)^{-1} Psi^T y``,
    linear in n in both compute and memory, no Nystrom approximation."""
    import torch
    Psi = rff_map(Xtr, M, ell, seed)
    yt = torch.as_tensor(np.asarray(y), dtype=Psi.dtype, device=Psi.device).reshape(-1, 1)
    r = Psi.shape[1]
    A = Psi.transpose(0, 1) @ Psi + lam * torch.eye(r, dtype=Psi.dtype, device=Psi.device)
    beta = torch.cholesky_solve(Psi.transpose(0, 1) @ yt, _robust_cholesky(A))
    Pq = rff_map(Xte, M, ell, seed)
    return (Pq @ beta).reshape(-1)


# =============================================================================
# Demonstrations
# =============================================================================

def nystrom_vs_exact(n=8000, m=512, lam=LAM, seed=SEED):
    """Nystrom primal solve vs the exact dense solve at one n: accuracy and the memory
    saved (r x r normal matrix instead of n x n Gram)."""
    import torch
    Xtr, ytr, Xte, yte, ym, ys = load_california_scaled(seed=seed)
    Xb, yb = tile_jitter(Xtr, ytr, n, seed=seed)
    ker = MultiScaleKernel(); emb = ker.embed(Xb)
    yt = torch.as_tensor(yb, dtype=ker.dtype, device=ker.device).reshape(-1, 1)
    S = select_landmarks(Xb, m, "uniform", seed); embS = ker.embed(S)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    K = ker.gram(Xb, Xb); a = dense_krr_solve(K, yt, lam)
    r2_exact = r2_score(predict(ker, a, Xb, Xte), yte, ym, ys)
    exact_peak = _peak_gb()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    P, ev = nystrom_factor(embS, ker.kmat, rank=m)
    beta = nystrom_solve(emb, embS, P, ker.kmat, yt, lam, block=4096)
    embTe = ker.embed(Xte)
    pred = nystrom_features(embTe, embS, P, ker.kmat) @ beta
    r2_nys = r2_score(pred, yte, ym, ys)
    nys_peak = _peak_gb()
    return {"n": n, "m": m, "rank_kept": int(ev.numel()), "r2_exact": r2_exact,
            "r2_nystrom": r2_nys, "exact_peak_gb": exact_peak, "nystrom_peak_gb": nys_peak,
            "gram_gb_f64": dense_gram_gb(n)}


def scaling_curve(ns=(8000, 16000, 32000, 64000, 128000), m=512, lam=LAM, seed=SEED):
    """Wall-clock and peak memory of the Nystrom solve as n grows: linear time, flat
    memory (the n x n Gram never exists)."""
    import torch
    Xtr, ytr, Xte, yte, ym, ys = load_california_scaled(seed=seed)
    ker = MultiScaleKernel()
    S = select_landmarks(Xtr, m, "uniform", seed); embS = ker.embed(S)
    P, _ = nystrom_factor(embS, ker.kmat, rank=m)
    embTe = ker.embed(Xte)
    rows = []
    for n in ns:
        Xb, yb = tile_jitter(Xtr, ytr, n, seed=seed)
        emb = ker.embed(Xb)
        yt = torch.as_tensor(yb, dtype=ker.dtype, device=ker.device).reshape(-1, 1)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        beta = nystrom_solve(emb, embS, P, ker.kmat, yt, lam, block=8192)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        pred = nystrom_features(embTe, embS, P, ker.kmat) @ beta
        rows.append({"n": int(n), "fit_s": dt, "peak_gb": _peak_gb(),
                     "r2": r2_score(pred, yte, ym, ys)})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def landmark_curve(n=8000, fracs=(0.01, 0.02, 0.04, 0.08, 0.12, 0.2, 0.4), lam=LAM,
                   method="uniform", seed=SEED):
    """Accuracy and fit-time vs the landmark fraction m/n: graceful degradation as the
    landmark budget shrinks, against the exact-decode reference."""
    import torch
    Xtr, ytr, Xte, yte, ym, ys = load_california_scaled(seed=seed)
    Xb, yb = tile_jitter(Xtr, ytr, n, seed=seed)
    ker = MultiScaleKernel(); emb = ker.embed(Xb); embTe = ker.embed(Xte)
    yt = torch.as_tensor(yb, dtype=ker.dtype, device=ker.device).reshape(-1, 1)
    K = ker.gram(Xb, Xb)
    r2_exact = r2_score(predict(ker, dense_krr_solve(K, yt, lam), Xb, Xte), yte, ym, ys)
    rows = []
    for f in fracs:
        m = max(8, int(f * n))
        S = select_landmarks(Xb, m, method, seed); embS = ker.embed(S)
        t0 = time.perf_counter()
        P, ev = nystrom_factor(embS, ker.kmat, rank=m)
        beta = nystrom_solve(emb, embS, P, ker.kmat, yt, lam, block=8192)
        dt = time.perf_counter() - t0
        pred = nystrom_features(embTe, embS, P, ker.kmat) @ beta
        rows.append({"frac": f, "m": m, "rank_kept": int(ev.numel()),
                     "r2": r2_score(pred, yte, ym, ys), "fit_s": dt})
    return {"rows": rows, "r2_exact": r2_exact, "n": n}


def spectral_decay(n=2000, m=600, seed=SEED):
    """The landmark-Gram eigenvalue spectrum and its cumulative tail mass: the decay
    that sets how many landmarks are needed (pick m past the knee)."""
    import torch
    Xtr, _, _, _, _, _ = load_california_scaled(seed=seed)
    ker = MultiScaleKernel()
    S = select_landmarks(Xtr, m, "uniform", seed)
    Kmm = ker.gram(S, S)
    ev = torch.linalg.eigvalsh(0.5 * (Kmm + Kmm.transpose(0, 1))).clamp_min(0.0)
    ev = torch.sort(ev, descending=True).values
    total = float(ev.sum())
    tail = 1.0 - torch.cumsum(ev, 0) / total
    return {"eigs": ev.cpu().numpy(), "tail_mass": tail.cpu().numpy()}


def auto_ladder(n, n_dense=50_000, matfree_max=1_000_000):
    """The ``solver='auto'`` decision: dense up to ``n_dense``, matrix-free (exact) up
    to ``matfree_max``, then Nystrom. Returns the solver name for a given n."""
    if n <= n_dense:
        return "dense"
    if n <= matfree_max:
        return "matfree"
    return "nystrom"


# =============================================================================
# Figures
# =============================================================================

def make_ladder_figure():
    """Fig 17.1 — the ``solver='auto'`` ladder picking dense -> matrix-free -> Nystrom
    as n grows, annotated with each solver's wall."""
    import matplotlib.pyplot as plt
    from lkbook.plotting import set_style
    set_style()
    ns = np.logspace(3, 6.4, 400)
    color = {"dense": "#b2182b", "matfree": "#4393c3", "nystrom": "#1a9850"}
    solv = [auto_ladder(n) for n in ns]
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    for name in ("dense", "matfree", "nystrom"):
        seg = np.array([s == name for s in solv])
        ax.fill_between(ns, 0, 1, where=seg, color=color[name], alpha=0.8,
                        step="mid", label=name)
    ax.axvline(50_000, ls=":", color="0.3"); ax.axvline(1_000_000, ls=":", color="0.3")
    ax.text(7e3, 1.06, "dense\n(exact, O(n²) mem)", fontsize=8, ha="center")
    ax.text(2e5, 1.06, "matrix-free\n(exact, O(n²) flops)", fontsize=8, ha="center")
    ax.text(3.5e6, 1.06, "Nystrom\n(linear in n)", fontsize=8, ha="center")
    ax.set_xscale("log"); ax.set_yticks([]); ax.set_ylim(0, 1.25)
    ax.set_xlabel("n (train rows)")
    ax.set_title('solver="auto": one trained kernel, three decoders')
    fig.tight_layout()
    return fig


def make_landmark_figure(lc=None, n=8000, seed=SEED):
    """Fig 17.2 — accuracy and fit-time vs landmark fraction m/n on California:
    graceful degradation below the exact-decode reference."""
    import matplotlib.pyplot as plt
    from lkbook.plotting import set_style
    set_style()
    if lc is None:
        lc = landmark_curve(n=n, seed=seed)
    rows = lc["rows"]
    fr = np.array([r["frac"] for r in rows], float) * 100
    r2 = np.array([r["r2"] for r in rows], float)
    ft = np.array([r["fit_s"] for r in rows], float)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(fr, r2, "o-", color="#1a9850", label="Nystrom test $R^2$")
    ax.axhline(lc["r2_exact"], ls="--", color="0.4", label=f"exact decode ({lc['r2_exact']:.3f})")
    ax.set_xlabel("landmark fraction m/n (%)"); ax.set_ylabel("test $R^2$", color="#1a9850")
    ax.tick_params(axis="y", labelcolor="#1a9850")
    ax2 = ax.twinx()
    ax2.plot(fr, ft, "s:", color="#b2182b", label="fit time (s)")
    ax2.set_ylabel("fit time (s)", color="#b2182b")
    ax2.tick_params(axis="y", labelcolor="#b2182b")
    ax.set_xscale("log")
    ax.set_title("Nystrom: graceful degradation as landmarks shrink")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    p = argparse.ArgumentParser()
    p.add_argument("--out-prefix", default="fig17")
    args = p.parse_args(argv)
    print(nystrom_vs_exact())
    for r in scaling_curve():
        print(r)
    lc = landmark_curve()
    print("exact", lc["r2_exact"])
    for r in lc["rows"]:
        print(r)
    make_ladder_figure().savefig(f"{args.out_prefix}_1.pdf", bbox_inches="tight")
    make_landmark_figure(lc).savefig(f"{args.out_prefix}_2.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
