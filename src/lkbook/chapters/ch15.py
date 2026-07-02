"""Chapter 15 — the scaling wall.

The fitted kernel machine trains cheaply on a small support subset (Ch. 8), but a
prediction is the kernel-ridge solve ``alpha = (K + lam I)^{-1} y`` over the *full*
train set of ``n`` rows. Two costs grow with ``n`` and they are different walls:
forming the dense Gram ``K in R^{n x n}`` is ``O(n^2)`` **memory**, and factoring
``K + lam I`` is ``O(n^3)`` **flops**. They fail at different ``n`` and yield to
different fixes -- the memory wall to never forming ``K`` (Ch. 16), the flop wall to
approximation (Ch. 17) or more hardware (Ch. 18).

This module carries the running kernel used by all of Part V: a *fixed* multi-scale
Laplace kernel ``k(x,x') = sum_h w_h exp(-||x-x'|| / T_h)`` over standardized inputs,
in the ``embeds`` / ``kmat`` shape the decoders in ``skm.linalg`` expect (a list of
per-bank embeddings and a fused Gram between two embedding sets). The Laplace kernel
is the Bochner kernel of a Cauchy spectral density, so this is a spectral kernel; the
bandwidths are fixed, so the kernel is deterministic and the chapters are about the
*linear algebra of the decode*, not about learning the kernel. The distance is never
primitive -- it is formed from the embedding via
``||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>`` -- which is exactly what lets Ch. 16
evaluate ``K @ v`` without materializing ``K``. Ch. 16 imports this kernel for the
exact matrix-free solve; Ch. 17 for the linear-in-n Nystrom solve.

The running example is California Housing (Ch. 8's dataset), tiled with small jitter
to synthesize the large ``n`` at which the wall arrives, following the benchmark
``skm/benchmarks/scale_decode.py``.

    python -m lkbook.chapters.ch15 --out-prefix fig15
"""
from __future__ import annotations

import argparse
import time

import numpy as np

SEED = 0
ELLS = (1.0, 2.0, 4.0)          # log-spaced bandwidths (the multi-scale bank grid)
LAM = 0.1                        # validated ridge for the fixed kernel on California


def _device(device=None):
    import torch
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# The running example: California Housing, standardized, tiled to large n
# =============================================================================

def load_california_scaled(seed=SEED):
    """Standardized California Housing: ``(Xtr, ytr_std, Xte, yte, y_mean, y_std)``.

    The features come pre-standardized from :func:`lkbook.data.load_california`; the
    target is standardized here so the ridge scale is dataset-independent. ``yte`` is
    left in raw units so R^2 is reported on the natural target.
    """
    from lkbook.data import load_california
    D = load_california(seed=seed)
    ym, ys = float(D.ytr.mean()), float(D.ytr.std())
    return D.Xtr, (D.ytr - ym) / ys, D.Xte, D.yte, ym, ys


def tile_jitter(X, y, n_target, jitter=0.03, seed=SEED):
    """Tile ``(X, y)`` with small Gaussian jitter up to ``n_target`` rows.

    The stand-in for genuinely large data: repeats the California rows and perturbs
    the features by ``jitter`` standard deviations so the Gram is full and the decode
    is non-trivial, letting the wall arrive at controllable ``n``.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float); y = np.asarray(y, float).ravel()
    reps = int(np.ceil(n_target / X.shape[0]))
    Xb = np.tile(X, (reps, 1))[:n_target]
    yb = np.tile(y, reps)[:n_target]
    Xb = Xb + jitter * rng.standard_normal(Xb.shape)
    return Xb, yb


# =============================================================================
# The fixed multi-scale Laplace kernel (the kernel of record for Part V)
# =============================================================================

class MultiScaleKernel:
    """Fixed finite multi-scale Laplace kernel over standardized inputs.

    ``k(x,x') = sum_h w_h exp(-||x-x'|| / T_h)``, a convex combination over ``H``
    log-spaced bandwidths with unit diagonal. The embedding is the identity on the
    d-dimensional standardized inputs, so ``embed(X)`` returns the one-element list
    ``[X]`` and ``kmat(A_embeds, B_embeds)`` forms the fused Gram from the single
    squared-distance ``||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>`` -- the distance is
    never primitive, which is what Ch. 16's matrix-free ``K @ v`` exploits.
    """

    def __init__(self, ells=ELLS, w=None, device=None, dtype=None):
        import torch
        self.device = _device(device)
        self.dtype = dtype or torch.float64
        self.T = torch.as_tensor(ells, dtype=self.dtype, device=self.device)
        H = len(ells)
        self.H = H
        self.w = (torch.full((H,), 1.0 / H, dtype=self.dtype, device=self.device)
                  if w is None else torch.as_tensor(w, dtype=self.dtype, device=self.device))

    def embed(self, X):
        """The (single) embedding: standardized inputs as a one-element list."""
        import torch
        return [torch.as_tensor(np.asarray(X), dtype=self.dtype, device=self.device)]

    def kmat(self, A_embeds, B_embeds):
        """Fused Gram ``sum_h w_h exp(-||a-b|| / T_h)`` from the embeddings."""
        import torch
        A, B = A_embeds[0], B_embeds[0]
        a2 = (A * A).sum(1, keepdim=True)
        b2 = (B * B).sum(1, keepdim=True).transpose(0, 1)
        dist = (a2 + b2 - 2.0 * (A @ B.transpose(0, 1))).clamp_min(0.0).sqrt()
        K = None
        for h in range(self.H):
            Kh = torch.exp(-dist / self.T[h])
            K = self.w[h] * Kh if K is None else K + self.w[h] * Kh
        return K

    def gram(self, A, B):
        """Convenience: full Gram between raw inputs ``A`` and ``B``."""
        return self.kmat(self.embed(A), self.embed(B))


# =============================================================================
# Dense kernel-ridge decode (the teaching path -- exact, O(n^2) mem, O(n^3) flops)
# =============================================================================

def dense_krr_solve(K, y, lam=LAM):
    """Dense kernel-ridge coefficients ``alpha = (K + lam I)^{-1} y`` by Cholesky.

    The teaching path: forms ``K + lam I`` and factors it. Exact, and the reference
    every other solver reproduces. ``O(n^2)`` memory to hold ``K``, ``O(n^3)`` flops
    to factor.
    """
    import torch
    n = K.shape[0]
    A = K + lam * torch.eye(n, dtype=K.dtype, device=K.device)
    L = torch.linalg.cholesky(A)
    return torch.cholesky_solve(y.reshape(n, -1), L).reshape(n, -1)


def predict(kernel, alpha, X_train, X_query, block=8192):
    """Prediction ``K(query, train) @ alpha`` in query row-blocks."""
    import torch
    et = kernel.embed(X_train)
    Xq = torch.as_tensor(np.asarray(X_query), dtype=kernel.dtype, device=kernel.device)
    q = Xq.shape[0]
    out = torch.zeros(q, alpha.shape[1], dtype=kernel.dtype, device=kernel.device)
    for i in range(0, q, block):
        sl = slice(i, min(i + block, q))
        out[sl] = kernel.kmat([Xq[sl]], et) @ alpha
    return out


def r2_score(pred, y, y_mean=0.0, y_std=1.0):
    """R^2 of a standardized prediction against a raw target."""
    import torch
    pred = pred.reshape(-1) * y_std + y_mean
    y = torch.as_tensor(np.asarray(y), dtype=pred.dtype, device=pred.device).reshape(-1)
    ss_res = ((pred - y) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


# =============================================================================
# Cost accounting: the two walls, in bytes and in flops
# =============================================================================

def dense_gram_gb(n, bytes_per=8):
    """Memory to hold a dense ``n x n`` Gram, in GB (float64 by default).
    Accepts a scalar or an array of ``n``."""
    n = np.asarray(n, dtype=float)
    return bytes_per * n * n / 1e9


def cost_table(ns, bytes_per=8):
    """The exact-KRR cost decomposition: Gram memory (GB) and the leading flop terms
    (form ``K`` at ~n^2, then Cholesky factor at n^3/3)."""
    rows = []
    for n in ns:
        rows.append({"n": int(n), "gram_GB": dense_gram_gb(n, bytes_per),
                     "form_flops": float(n) ** 2, "chol_flops": float(n) ** 3 / 3.0})
    return rows


# =============================================================================
# Conditioning and the ridge: lambda lower-bounds the spectrum of K + lam I
# =============================================================================

def condition_number(K, lam):
    """Condition number ``kappa(K + lam I)`` and the Ch. 15 bound.

    Since ``K >= 0``, every eigenvalue of ``K + lam I`` is at least ``lam``, so
    ``lambda_min(K + lam I) >= lam`` and ``kappa <= (lambda_max(K) + lam)/lam``.
    Returns the measured ``kappa``, its upper bound and ``sqrt(kappa)``.
    """
    import torch
    ev = torch.linalg.eigvalsh(0.5 * (K + K.transpose(0, 1)))
    lmax, lmin = float(ev.max()), float(ev.min())
    kappa = (lmax + lam) / (lmin + lam)
    return {"kappa": kappa, "bound": (lmax + lam) / lam,
            "lmax": lmax, "lmin_K": lmin, "sqrt_kappa": kappa ** 0.5}


def conditioning_sweep(n=1500, lams=(1e-6, 1e-4, 1e-2, 1e-1, 1.0), seed=SEED):
    """Measured ``kappa`` and the ridge bound as ``lam`` varies, on a California
    subset, with the sqrt-kappa iteration proxy Ch. 16 uses."""
    Xtr, ytr, _, _, _, _ = load_california_scaled(seed=seed)
    ker = MultiScaleKernel()
    K = ker.gram(Xtr[:n], Xtr[:n])
    rows = []
    for lam in lams:
        c = condition_number(K, lam); c["lam"] = lam
        rows.append(c)
    return rows


# =============================================================================
# Timing harness: watch the dense wall arrive
# =============================================================================

def _peak_gb():
    import torch
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return float("nan")


def time_dense(ns, lam=LAM, seed=SEED, max_n=None):
    """Time and memory-profile the *dense* decode on California tiled to each ``n``.

    For each ``n`` (up to ``max_n``, else skipped as the benchmark does rather than
    running to OOM) forms the Gram, factors it, predicts on the real test set, and
    records test R^2, wall-clock and peak GPU memory. Returns one row per ``n``.
    """
    import torch
    Xtr, ytr, Xte, yte, ym, ys = load_california_scaled(seed=seed)
    ker = MultiScaleKernel()
    rows = []
    for n in ns:
        if max_n is not None and n > max_n:
            rows.append({"n": int(n), "skipped": True}); continue
        Xb, yb = tile_jitter(Xtr, ytr, n, seed=seed)
        yt = torch.as_tensor(yb, dtype=ker.dtype, device=ker.device).reshape(-1, 1)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        K = ker.gram(Xb, Xb)
        alpha = dense_krr_solve(K, yt, lam)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        solve_s = time.perf_counter() - t0
        pred = predict(ker, alpha, Xb, Xte)
        rows.append({"n": int(n), "solve_s": solve_s, "peak_gb": _peak_gb(),
                     "gram_gb_f64": dense_gram_gb(n), "r2": r2_score(pred, yte, ym, ys),
                     "skipped": False})
        del K, alpha
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


# The measured GB10 reference (float64, benchmarks/scale_decode.py; skm README).
# Carried so the book's roadmap figure shows the real large-n reaches without a
# multi-hour rerun. Peak memory is torch max_memory_allocated (GB10 unified memory).
GB10_REFERENCE = [
    {"n": 20_000,   "lanczos": (38, 10.5),  "matfree": (232, 6.3),   "nystrom": (41, 2.1),   "nystrom_r2": None},
    {"n": 50_000,   "lanczos": (152, 64.0), "matfree": (2378, 16.0), "nystrom": (319, 2.2),  "nystrom_r2": 0.901},
    {"n": 100_000,  "lanczos": None,        "matfree": None,         "nystrom": (421, 2.4),  "nystrom_r2": 0.905},
    {"n": 200_000,  "lanczos": None,        "matfree": None,         "nystrom": (586, 3.6),  "nystrom_r2": 0.909},
    {"n": 500_000,  "lanczos": None,        "matfree": None,         "nystrom": (1461, 8.8), "nystrom_r2": 0.907},
    {"n": 1_000_000,"lanczos": None,        "matfree": None,         "nystrom": (1814, 17.5),"nystrom_r2": 0.906},
]
# solver -> largest n to attempt before skipping (from benchmarks/scale_decode.py)
SOLVER_MAX_N = {"dense": 50_000, "lanczos": 50_000, "matfree": 50_000, "nystrom": 2_000_000}


# =============================================================================
# Figures
# =============================================================================

def make_wall_figure(timing=None, ns=(2000, 4000, 8000, 16000, 24000), seed=SEED):
    """Fig 15.1 — wall-clock and peak memory vs n for the dense solve (live small-n),
    with the O(n^2) float64 Gram-memory curve extrapolated to the GB10 OOM cliff."""
    import matplotlib.pyplot as plt
    from lkbook.plotting import set_style
    set_style()
    if timing is None:
        timing = time_dense(ns, seed=seed)
    live = [r for r in timing if not r.get("skipped")]
    nn = np.array([r["n"] for r in live], float)
    secs = np.array([r["solve_s"] for r in live], float)
    peak = np.array([r["peak_gb"] for r in live], float)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axL.loglog(nn, secs, "o-", color="#b2182b", label="dense solve (measured)")
    axL.loglog(nn, secs[0] * (nn / nn[0]) ** 3, "--", color="0.6", label=r"$O(n^3)$ slope")
    axL.set_xlabel("n (train rows)"); axL.set_ylabel("solve wall-clock (s)")
    axL.set_title("Dense decode time"); axL.legend(frameon=False, fontsize=9)

    ncurve = np.logspace(np.log10(2000), np.log10(90000), 200)
    axR.loglog(ncurve, dense_gram_gb(ncurve), "-", color="#2166ac",
               label=r"float64 Gram $8n^2/10^9$ GB")
    if np.isfinite(peak).any():
        axR.loglog(nn, peak, "o", color="#b2182b", label="measured peak (GB)")
    axR.axhline(64, ls=":", color="0.4")
    axR.text(2.2e3, 74, "dense OOM (~64 GB peak at n=50k, GB10)", fontsize=8, color="0.3")
    axR.axvline(50_000, ls=":", color="0.4")
    axR.set_xlabel("n (train rows)"); axR.set_ylabel("memory (GB)")
    axR.set_title("The memory wall"); axR.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def make_roadmap_figure():
    """Fig 15.2 — the three-solver roadmap on one n-axis: the reach of dense,
    matrix-free exact and Nystrom, annotated with the wall each attacks."""
    import matplotlib.pyplot as plt
    from lkbook.plotting import set_style
    set_style()
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    bars = [
        ("dense (form K)",        5e4, "#b2182b", "memory wall\n~5e4"),
        ("matrix-free (exact)",   1e6, "#4393c3", "flop wall\n~1e6"),
        ("Nystrom (linear in n)", 2e6, "#1a9850", "approximation\nfloor"),
    ]
    for i, (name, reach, color, note) in enumerate(bars):
        ax.barh(i, np.log10(reach) - 3, left=3, color=color, alpha=0.85, height=0.55)
        ax.text(3.05, i, name, va="center", ha="left", color="white",
                fontsize=10, fontweight="bold")
        ax.text(np.log10(reach) + 0.05, i, note, va="center", fontsize=8, color="0.3")
    ax.set_yticks([]); ax.set_xlim(3, 7.4)
    ax.set_xticks([3, 4, 5, 6, 7])
    ax.set_xticklabels(["$10^3$", "$10^4$", "$10^5$", "$10^6$", "$10^7$"])
    ax.set_xlabel("n (train rows) reachable")
    ax.set_title("Three decoders through the scaling wall (Ch. 16-18)")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    p = argparse.ArgumentParser()
    p.add_argument("--out-prefix", default="fig15")
    args = p.parse_args(argv)
    timing = time_dense((2000, 4000, 8000, 16000, 24000))
    for r in timing:
        print(r)
    for row in conditioning_sweep():
        print(f"lam={row['lam']:.0e}  kappa={row['kappa']:.3e}  "
              f"bound={row['bound']:.3e}  sqrt_kappa={row['sqrt_kappa']:.1f}")
    make_wall_figure(timing).savefig(f"{args.out_prefix}_1.pdf", bbox_inches="tight")
    make_roadmap_figure().savefig(f"{args.out_prefix}_2.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
