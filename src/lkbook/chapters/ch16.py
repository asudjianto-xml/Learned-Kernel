"""Chapter 16 — exact but cheaper: matrix-free Krylov.

The memory wall of Ch. 15 came from forming the ``n x n`` Gram. A Krylov solver never
asks for ``K``; it asks for the product ``K @ v``. And ``K @ v`` does not need ``K`` to
exist: the Gram factors through the embedding, so the product can be accumulated in
row-blocks, at ``O(block * n)`` memory instead of ``O(n^2)``. Lanczos then recovers the
*exact* ridge solution ``alpha = (K + lam I)^{-1} y`` from a small Krylov basis. The
memory wall falls; the answer is the dense answer to machine precision. The flop wall
(``O(n^2 * rank)``) remains -- that is the next chapter's trade.

This is a faithful port of ``skm/linalg.py`` (``kernel_matvec``, ``lanczos``,
``krr_solve``, ``krr_solve_matfree``, ``krr_solve_sweep``), driven by the fixed
multi-scale Laplace kernel of Ch. 15.

    python -m lkbook.chapters.ch16 --out-prefix fig16
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from lkbook.chapters.ch15 import (MultiScaleKernel, load_california_scaled, tile_jitter,
                                   dense_krr_solve, predict, r2_score, dense_gram_gb,
                                   _peak_gb, LAM, SEED)


# =============================================================================
# The matrix-free operator v -> K @ v (path A: never materialize the Gram)
# =============================================================================

def kernel_matvec(embeds, kmat, block):
    """A matrix-free operator ``v -> K @ v`` for the fused kernel.

    ``embeds`` is the list of per-bank train embeddings and ``kmat`` the model's fused
    kernel between two embedding sets. The product is accumulated in row-blocks: block
    ``i`` forms only the ``(block, n)`` slab ``kmat(embeds[i], embeds)`` and applies it
    to ``v``, so peak memory is ``O(block * n)`` and the full Gram is never held.
    Faithful port of ``skm.linalg.kernel_matvec``.
    """
    import torch
    n = embeds[0].shape[0]

    def mv(v):
        out = torch.empty((n,) + tuple(v.shape[1:]), dtype=v.dtype, device=v.device)
        for i in range(0, n, block):
            sl = slice(i, min(i + block, n))
            Kb = kmat([e[sl] for e in embeds], embeds)         # (b, n) slab only
            out[sl] = Kb @ v
        return out

    return mv


def lanczos(matvec, b, k, device=None, dtype=None):
    """Lanczos tridiagonalization of a symmetric operator given by ``matvec``.

    Builds an orthonormal Krylov basis ``Q (n, m)`` and tridiagonal coefficients
    ``(alpha, beta)`` with ``Q^T A Q = tridiag(beta, alpha, beta)``, from start vector
    ``b``. Full reorthogonalization (twice) keeps ``Q`` orthonormal in float
    arithmetic; the Krylov space deflates early if ``beta`` collapses. Returns
    ``(Q, alpha, beta, ||b||)``. Faithful port of ``skm.linalg.lanczos``.
    """
    import torch
    device = device or b.device
    dtype = dtype or b.dtype
    n = b.shape[0]
    Q = torch.zeros(n, k, dtype=dtype, device=device)
    alpha = torch.zeros(k, dtype=dtype, device=device)
    beta = torch.zeros(max(k - 1, 1), dtype=dtype, device=device)
    nb = torch.linalg.norm(b)
    Q[:, 0] = b / nb
    last = k
    for j in range(k):
        w = matvec(Q[:, j])
        alpha[j] = Q[:, j] @ w
        w = w - alpha[j] * Q[:, j] - (beta[j - 1] * Q[:, j - 1] if j > 0 else 0.0)
        w = w - Q[:, :j + 1] @ (Q[:, :j + 1].t() @ w)          # reorthogonalize twice
        w = w - Q[:, :j + 1] @ (Q[:, :j + 1].t() @ w)
        if j < k - 1:
            beta[j] = torch.linalg.norm(w)
            if beta[j] < 1e-10:                                # Krylov space exhausted
                last = j + 1
                break
            Q[:, j + 1] = w / beta[j]
    return Q[:, :last], alpha[:last], beta[:last - 1], nb


def _solve_from_lanczos(Q, al, be, nb, lam):
    """Recover ``alpha(lam) = Q U (||b|| U[0,:] / (Theta + lam))`` from the tridiagonal
    factors -- a length-m solve, not an n x n one. ``(Q, Theta, U)`` do not depend on
    ``lam``, which is what licenses the ridge sweep."""
    import torch
    Tk = torch.diag(al) + torch.diag(be, 1) + torch.diag(be, -1)
    theta, U = torch.linalg.eigh(Tk)
    return Q @ (nb * (U @ (U[0, :] / (theta + lam)))), theta, U


def krr_solve_dense_lanczos(K, y, lam=LAM, rank=150):
    """Exact KRR on the *dense* Gram via Lanczos (the ``solver='lanczos'`` path).

    Each output column is solved on its own Lanczos basis with ``matvec = K @ v``.
    Faithful port of ``skm.linalg.krr_solve``."""
    import torch
    cols = []
    for c in range(y.shape[1]):
        Q, al, be, nb = lanczos(lambda v: K @ v, y[:, c].contiguous(), rank)
        a, _, _ = _solve_from_lanczos(Q, al, be, nb, lam)
        cols.append(a)
    return torch.stack(cols, 1)


def krr_solve_matfree(embeds, kmat, y, lam=LAM, rank=150, block=8192):
    """Exact KRR ``alpha = (K + lam I)^{-1} y`` with no ``n x n`` materialization.

    Identical math to the dense Lanczos solve but the matvec is the row-blocked
    operator, so peak memory is ``O(block * n)`` instead of ``O(n^2)``. Faithful port
    of ``skm.linalg.krr_solve_matfree``."""
    import torch
    mv = kernel_matvec(embeds, kmat, block)
    cols = []
    for c in range(y.shape[1]):
        Q, al, be, nb = lanczos(mv, y[:, c].contiguous(), rank)
        a, _, _ = _solve_from_lanczos(Q, al, be, nb, lam)
        cols.append(a)
    return torch.stack(cols, 1)


def krr_solve_sweep(embeds, kmat, y, lambdas, rank=150, block=8192):
    """One Lanczos basis reused across a ridge sweep (the ``lam``-selection pattern).

    ``(Q, Theta, U)`` are built once per output column; each ``lam`` only re-solves the
    length-m diagonal system ``(Theta + lam)^{-1}``. Returns ``coefs[i]`` for every
    ``lam`` in ``lambdas``. Faithful port of ``skm.linalg.krr_solve_sweep``."""
    import torch
    mv = kernel_matvec(embeds, kmat, block)
    per_col = []
    for c in range(y.shape[1]):
        Q, al, be, nb = lanczos(mv, y[:, c].contiguous(), rank)
        Tk = torch.diag(al) + torch.diag(be, 1) + torch.diag(be, -1)
        theta, U = torch.linalg.eigh(Tk)
        per_col.append((Q, theta, U, U[0, :], nb))
    coefs = []
    for lam in lambdas:
        coef = torch.stack([Q @ (nb * (U @ (u0 / (theta + lam))))
                            for (Q, theta, U, u0, nb) in per_col], 1)
        coefs.append(coef)
    return coefs


# =============================================================================
# Demonstrations: matfree == dense, memory profile, rank-as-budget, sweep reuse
# =============================================================================

def matfree_vs_dense(n=6000, lam=LAM, rank=150, block=2048, seed=SEED):
    """Confirm the matrix-free solve reproduces the dense solve to machine precision,
    and report the memory each path used."""
    import torch
    Xtr, ytr, Xte, yte, ym, ys = load_california_scaled(seed=seed)
    Xb, yb = tile_jitter(Xtr, ytr, n, seed=seed)
    ker = MultiScaleKernel()
    emb = ker.embed(Xb)
    yt = torch.as_tensor(yb, dtype=ker.dtype, device=ker.device).reshape(-1, 1)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    K = ker.gram(Xb, Xb)
    a_dense = krr_solve_dense_lanczos(K, yt, lam, rank)
    dense_peak = _peak_gb()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    a_mf = krr_solve_matfree(emb, ker.kmat, yt, lam, rank, block)
    mf_peak = _peak_gb()

    max_diff = float((a_dense - a_mf).abs().max())
    r2_dense = r2_score(predict(ker, a_dense, Xb, Xte), yte, ym, ys)
    r2_mf = r2_score(predict(ker, a_mf, Xb, Xte), yte, ym, ys)
    return {"n": n, "coef_max_diff": max_diff, "dense_peak_gb": dense_peak,
            "matfree_peak_gb": mf_peak, "gram_gb_f64": dense_gram_gb(n),
            "r2_dense": r2_dense, "r2_matfree": r2_mf, "block": block, "rank": rank}


def memory_profile(ns=(2000, 4000, 8000, 16000, 24000), block=2048, rank=150, seed=SEED):
    """Peak memory of dense vs matrix-free across n: dense rises as n^2, matrix-free is
    flat (set by ``block``). ``rank`` sets the Krylov budget (lower is faster; peak
    memory is dominated by the ``(block, n)`` slab, not the rank)."""
    import torch
    Xtr, ytr, _, _, _, _ = load_california_scaled(seed=seed)
    ker = MultiScaleKernel()
    rows = []
    for n in ns:
        Xb, yb = tile_jitter(Xtr, ytr, n, seed=seed)
        emb = ker.embed(Xb)
        yt = torch.as_tensor(yb, dtype=ker.dtype, device=ker.device).reshape(-1, 1)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        krr_solve_matfree(emb, ker.kmat, yt, rank=rank, block=block)
        mf = _peak_gb()
        rows.append({"n": int(n), "matfree_peak_gb": mf, "dense_gram_gb_f64": dense_gram_gb(n)})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def rank_budget(n=6000, ranks=(5, 10, 20, 40, 80, 120, 160, 200), lam=LAM,
                block=2048, seed=SEED):
    """Coefficient error vs the dense solution and test R^2 as the Krylov rank grows:
    the rank is a compute budget, exact (to machine precision) by modest rank."""
    import torch
    Xtr, ytr, Xte, yte, ym, ys = load_california_scaled(seed=seed)
    Xb, yb = tile_jitter(Xtr, ytr, n, seed=seed)
    ker = MultiScaleKernel(); emb = ker.embed(Xb)
    yt = torch.as_tensor(yb, dtype=ker.dtype, device=ker.device).reshape(-1, 1)
    K = ker.gram(Xb, Xb)
    a_ref = dense_krr_solve(K, yt, lam)
    ref_norm = float(a_ref.norm())
    rows = []
    for r in ranks:
        a = krr_solve_matfree(emb, ker.kmat, yt, lam, r, block)
        rows.append({"rank": int(r),
                     "rel_coef_err": float((a - a_ref).norm()) / ref_norm,
                     "r2": r2_score(predict(ker, a, Xb, Xte), yte, ym, ys)})
    return rows


def sweep_demo(n=6000, lambdas=(1e-3, 1e-2, 1e-1, 1.0), rank=150, block=2048, seed=SEED):
    """One Lanczos basis, an entire ridge sweep: pick ``lam`` on a validation fold
    without re-factorizing. Confirms each swept coefficient matches a from-scratch
    matfree solve at that ``lam``."""
    import torch
    Xtr, ytr, Xte, yte, ym, ys = load_california_scaled(seed=seed)
    Xb, yb = tile_jitter(Xtr, ytr, n, seed=seed)
    # validation split of the tiled train
    nb = Xb.shape[0]; nval = nb // 5
    Xf, yf = Xb[:-nval], yb[:-nval]
    Xv, yv = Xb[-nval:], yb[-nval:]
    ker = MultiScaleKernel(); emb = ker.embed(Xf)
    yt = torch.as_tensor(yf, dtype=ker.dtype, device=ker.device).reshape(-1, 1)
    coefs = krr_solve_sweep(emb, ker.kmat, yt, lambdas, rank, block)
    yv_t = torch.as_tensor(yv, dtype=ker.dtype, device=ker.device).reshape(-1)  # standardized
    rows = []
    for lam, coef in zip(lambdas, coefs):
        a_scratch = krr_solve_matfree(emb, ker.kmat, yt, lam, rank, block)
        pv = predict(ker, coef, Xf, Xv).reshape(-1)                            # standardized
        ss_res = float(((pv - yv_t) ** 2).sum())
        ss_tot = float(((yv_t - yv_t.mean()) ** 2).sum())
        rows.append({"lam": lam,
                     "reuse_vs_scratch_max_diff": float((coef - a_scratch).abs().max()),
                     "val_r2": 1.0 - ss_res / ss_tot})
    best = max(rows, key=lambda r: r["val_r2"])
    return {"rows": rows, "chosen_lam": best["lam"]}


# =============================================================================
# Figures
# =============================================================================

def make_memory_figure(prof=None, ns=(2000, 4000, 8000, 16000, 24000), block=2048,
                       rank=150, seed=SEED):
    """Fig 16.1 — peak memory vs n: dense (rising as n^2, extrapolated to the OOM
    cliff) against matrix-free (flat, set by ``block``)."""
    import matplotlib.pyplot as plt
    from lkbook.plotting import set_style
    set_style()
    if prof is None:
        prof = memory_profile(ns, block=block, rank=rank, seed=seed)
    nn = np.array([r["n"] for r in prof], float)
    mf = np.array([r["matfree_peak_gb"] for r in prof], float)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ncurve = np.logspace(np.log10(nn.min()), np.log10(90000), 200)
    ax.loglog(ncurve, dense_gram_gb(ncurve), "-", color="#2166ac",
              label=r"dense Gram $8n^2/10^9$ GB")
    if np.isfinite(mf).any():
        ax.loglog(nn, mf, "o-", color="#1a9850", label=f"matrix-free peak (block={block})")
    ax.axhline(64, ls=":", color="0.4")
    ax.text(nn.min() * 1.05, 74, "GB10 dense OOM (~64 GB at n=50k)", fontsize=8, color="0.3")
    ax.axvline(50_000, ls=":", color="0.4")
    ax.set_xlabel("n (train rows)"); ax.set_ylabel("peak memory (GB)")
    ax.set_title("Matrix-free removes the memory wall")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def make_rank_figure(rb=None, n=6000, seed=SEED):
    """Fig 16.2 — coefficient error vs the dense solution and test R^2 as the Krylov
    rank grows: exact by modest rank, the rank a compute budget."""
    import matplotlib.pyplot as plt
    from lkbook.plotting import set_style
    set_style()
    if rb is None:
        rb = rank_budget(n=n, seed=seed)
    ranks = np.array([r["rank"] for r in rb], float)
    err = np.array([r["rel_coef_err"] for r in rb], float)
    r2 = np.array([r["r2"] for r in rb], float)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.semilogy(ranks, err, "o-", color="#b2182b", label="rel. coefficient error vs dense")
    ax.set_xlabel("Krylov rank (matvecs)")
    ax.set_ylabel("relative coefficient error", color="#b2182b")
    ax.tick_params(axis="y", labelcolor="#b2182b")
    ax2 = ax.twinx()
    ax2.plot(ranks, r2, "s--", color="#1a9850", label="test $R^2$")
    ax2.set_ylabel("test $R^2$", color="#1a9850")
    ax2.tick_params(axis="y", labelcolor="#1a9850")
    ax.set_title("Rank as a compute budget: exact by modest rank")
    fig.tight_layout()
    return fig


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    p = argparse.ArgumentParser()
    p.add_argument("--out-prefix", default="fig16")
    args = p.parse_args(argv)
    print(matfree_vs_dense())
    for r in rank_budget():
        print(r)
    print(sweep_demo())
    make_memory_figure().savefig(f"{args.out_prefix}_1.pdf", bbox_inches="tight")
    make_rank_figure().savefig(f"{args.out_prefix}_2.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
