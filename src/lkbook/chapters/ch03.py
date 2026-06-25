"""Chapter 3 — from chosen to learned kernels.

The central inversion: the geometry is the output, not the input. We show on the running
data that (a) choosing the RBF bandwidth *is* the modeling — test error is sharply
U-shaped in ℓ; (b) making the geometry a learnable object (per-feature ARD length scales,
fit on a held-out fold) beats the best hand-chosen isotropic bandwidth and ranks the
features; and (c) the (K, λ) → (αK, αλ) degeneracy leaves predictions invariant, which the
unit-diagonal convention removes so λ becomes identifiable.

The ARD fit uses scipy's L-BFGS (ships with scikit-learn) so the package carries no torch
dependency. Companion to `fuse-kernel/fusekernel/kernels.py:mix`, whose docstring states the
same degeneracy ("removes the (K, lambda) -> (cK, c*lambda) degeneracy").

    python -m lkbook.chapters.ch03 --out-prefix fig3
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from lkbook import load_california, load_taiwan, set_style

N_SUP, N_QRY, LAM, SEED = 1000, 1000, 1e-2, 0


# --- kernels ------------------------------------------------------------------

def iso_gram(A, B, ell):
    """Isotropic RBF: one bandwidth ℓ for all features. Unit diagonal."""
    d2 = np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)
    return np.exp(-d2 / (2.0 * ell * ell))


def ard_gram(A, B, ell_vec):
    """ARD RBF: per-feature length scales. k(x,x')=exp(-½ Σ_j (x_j-x'_j)²/ℓ_j²).
    The diagonal is exp(0)=1 automatically, so the family is unit-diagonal by construction."""
    Aw = A / ell_vec
    Bw = B / ell_vec
    d2 = np.maximum((Aw * Aw).sum(1)[:, None] + (Bw * Bw).sum(1)[None, :] - 2 * Aw @ Bw.T, 0.0)
    return np.exp(-0.5 * d2)


def _rmse(pred, y):
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def _split(d, n_sup=N_SUP, n_qry=N_QRY, seed=SEED):
    rng = np.random.RandomState(seed)
    idx = rng.choice(d.n, n_sup + n_qry, replace=False)
    s, q = idx[:n_sup], idx[n_sup:]
    return d.Xtr[s], d.ytr[s], d.Xtr[q], d.ytr[q]


# --- (a) the cost of choosing: bandwidth sweep --------------------------------

def bandwidth_sweep(d, ells=None, lam=LAM, seed=SEED):
    """Test RMSE as a function of the isotropic bandwidth ℓ. Returns (ells, rmses)."""
    if ells is None:
        ells = np.logspace(-1.0, 1.5, 24)
    Xs, ys, Xq, yq = _split(d, seed=seed)
    ybar = ys.mean()
    out = []
    for ell in ells:
        K = iso_gram(Xs, Xs, ell)
        a = np.linalg.solve(K + lam * np.eye(len(ys)), ys - ybar)
        out.append(_rmse(iso_gram(Xq, Xs, ell) @ a + ybar, yq))
    return np.asarray(ells), np.asarray(out)


# --- (b) learning the geometry: ARD fit on a held-out fold --------------------

def fit_ard(d, lam=LAM, seed=SEED, maxiter=60):
    """Fit per-feature log length-scales by minimizing held-out (query-fold) MSE with
    L-BFGS. The support fold builds K; the query fold scores it — leakage-free selection,
    the discipline Chapter 7 formalizes. Returns dict with ℓ_j, query/test RMSE, etc."""
    Xs, ys, Xq, yq = _split(d, seed=seed)
    ybar = ys.mean()

    def objective(log_ell):
        K = ard_gram(Xs, Xs, np.exp(log_ell))
        a = np.linalg.solve(K + lam * np.eye(len(ys)), ys - ybar)
        return np.mean((ard_gram(Xq, Xs, np.exp(log_ell)) @ a + ybar - yq) ** 2)

    x0 = np.zeros(d.d)                         # start isotropic, ℓ_j = 1
    res = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": maxiter})
    ell = np.exp(res.x)

    # held-out test RMSE of the learned ARD geometry, vs the best isotropic bandwidth
    K = ard_gram(Xs, Xs, ell)
    a = np.linalg.solve(K + lam * np.eye(len(ys)), ys - ybar)
    ard_rmse = _rmse(ard_gram(d.Xte, Xs, ell) @ a + ybar, d.yte)
    ells, sweep = bandwidth_sweep(d, lam=lam, seed=seed)
    best_iso_ell = float(ells[np.argmin(sweep)])
    Ki = iso_gram(Xs, Xs, best_iso_ell)
    ai = np.linalg.solve(Ki + lam * np.eye(len(ys)), ys - ybar)
    iso_rmse = _rmse(iso_gram(d.Xte, Xs, best_iso_ell) @ ai + ybar, d.yte)
    return {"ell": ell, "relevance": 1.0 / ell, "names": d.names,
            "ard_test_rmse": ard_rmse, "best_iso_ell": best_iso_ell,
            "iso_test_rmse": iso_rmse}


# --- (c) the degeneracy and the unit-diagonal fix -----------------------------

def degeneracy_demo(d, ell=2.0, lam=LAM, alpha=2.0, seed=SEED, q=7):
    """Show KRR predictions are invariant under (K, λ) → (αK, αλ), and that the
    unit-diagonal kernel removes the free scale so λ is identifiable."""
    Xs, ys, _, _ = _split(d, seed=seed)
    ybar = ys.mean()
    x = d.Xte[q]
    K = iso_gram(Xs, Xs, ell); kq = iso_gram(x[None], Xs, ell)[0]
    n = len(ys)
    base = float(kq @ np.linalg.solve(K + lam * np.eye(n), ys - ybar) + ybar)
    scaled = float((alpha * kq) @ np.linalg.solve(alpha * K + alpha * lam * np.eye(n), ys - ybar) + ybar)
    return {"pred_base": base, "pred_scaled": scaled, "abs_diff": abs(base - scaled),
            "unit_diagonal": float(np.diag(K).mean()), "alpha": alpha}


# --- figures ------------------------------------------------------------------

def make_bandwidth_figure(d):
    ells, rmse = bandwidth_sweep(d)
    i_best, i_worst = int(np.argmin(rmse)), int(np.argmax(rmse))
    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    ax.semilogx(ells, rmse, "-o", ms=4, color="#3b6ea5")
    ax.scatter([ells[i_best]], [rmse[i_best]], c="#2ca02c", s=90, zorder=5,
               label=fr"best $\ell$={ells[i_best]:.2f}, RMSE={rmse[i_best]:.3f}")
    ax.scatter([ells[i_worst]], [rmse[i_worst]], c="#c44e52", s=90, zorder=5,
               label=fr"worst $\ell$={ells[i_worst]:.2f}, RMSE={rmse[i_worst]:.3f}")
    ax.set_xlabel(r"RBF bandwidth $\ell$ (log scale)"); ax.set_ylabel("test RMSE ($100k)")
    ax.set_title("The 'choice' of bandwidth was the model.\n"
                 fr"Best beats worst by {rmse[i_worst]-rmse[i_best]:.2f} in RMSE — "
                 r"$\ell$ is a parameter, not a setting.", fontsize=10)
    ax.legend(fontsize=9)
    return fig


def make_ard_figure(d):
    r = fit_ard(d)
    order = np.argsort(-r["relevance"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axes[0]
    ax.bar(np.array(r["names"])[order], r["relevance"][order], color="#3b6ea5")
    ax.set_ylabel(r"learned relevance  $1/\ell_j$"); ax.tick_params(axis="x", rotation=40)
    ax.set_title(r"Fitted ARD relevances $1/\ell_j$ — per-feature local sensitivity"
                 "\n(geography decays fast; Population switched off)", fontsize=10)
    ax2 = axes[1]
    bars = ax2.bar(["best isotropic\n(chosen)", "ARD\n(learned)"],
                   [r["iso_test_rmse"], r["ard_test_rmse"]], color=["#c44e52", "#2ca02c"])
    for b, v in zip(bars, [r["iso_test_rmse"], r["ard_test_rmse"]]):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=10)
    ax2.set_ylabel("test RMSE ($100k)")
    ax2.set_title("Fitting the geometry beats choosing it", fontsize=10)
    return fig


# --- CLI ----------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Chapter 3 — from chosen to learned kernels")
    p.add_argument("--out-prefix", default=None)
    args = p.parse_args(argv)
    set_style()
    cal = load_california()

    ells, rmse = bandwidth_sweep(cal)
    print("=" * 70, f"\nBANDWIDTH SWEEP: best RMSE {rmse.min():.3f} at ℓ={ells[np.argmin(rmse)]:.2f}; "
          f"worst {rmse.max():.3f} at ℓ={ells[np.argmax(rmse)]:.2f}")

    r = fit_ard(cal)
    print(f"\nARD fit: test RMSE {r['ard_test_rmse']:.3f}  vs best isotropic "
          f"{r['iso_test_rmse']:.3f} (ℓ={r['best_iso_ell']:.2f})")
    rank = sorted(zip(r["names"], r["relevance"]), key=lambda t: -t[1])
    print("  relevances 1/ℓ_j:", ", ".join(f"{n}={v:.2f}" for n, v in rank))

    deg = degeneracy_demo(cal)
    print(f"\nDEGENERACY: pred {deg['pred_base']:.6f} vs scaled (αK,αλ) {deg['pred_scaled']:.6f}; "
          f"|Δ|={deg['abs_diff']:.2e}; unit diagonal = {deg['unit_diagonal']:.3f}")

    if args.out_prefix:
        import matplotlib
        matplotlib.use("Agg")
        make_bandwidth_figure(cal).savefig(f"{args.out_prefix}1_bandwidth.pdf")
        make_ard_figure(cal).savefig(f"{args.out_prefix}2_ard.pdf")
        print("wrote figures with prefix", args.out_prefix)


if __name__ == "__main__":
    main()
