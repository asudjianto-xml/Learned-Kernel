"""Chapter 8 — spectral kernels and Bochner.

By Bochner's theorem a continuous stationary kernel is the Fourier transform of a finite
nonnegative spectral measure on frequencies, so choosing a stationary geometry is choosing
a measure on how fast the function may wiggle, and learning the kernel is learning that
measure.

Two design decisions hide in one kernel. The **readout** — the outer function turning
embedding distance into similarity — sets the *roughness* of the function class. The
**spectrum** below it sets only the *geometry*. The Laplace readout exp(-r/T) has a cusp at
the origin that places the RKHS at Sobolev order (d+1)/2 (the rough Matern-1/2 / ReLU-NTK
class tabular targets inhabit); the Gaussian readout exp(-r^2/T^2) is C-infinity and
oversmooths (Theorem A).

This module builds the **multi-scale spectral kernel (MS-SKM)** in its finite form, a NumPy
mirror of the torch `skm.SpectralMixture` (finite/`spectral="free"` mode): a spectral
embedding of cosine/sine features at a finite set of frequencies, read out through a
**Laplace** exponential, with H banks fused convexly,

    K(x,x') = sum_h w_h exp(-||phi_h(x) - phi_h(x')|| / T_h),   w on the simplex.

A single bank (H=1) fixes one bandwidth; fusing banks spans several scales. The continuous
(density / Gauss-Hermite) parameterization and the "continuity is free" capacity statement
are a later chapter; here we stay with the finite single and fused banks.

    python -m lkbook.chapters.ch08 --out-prefix fig8
"""
from __future__ import annotations

import argparse
import functools

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from lkbook import load_california, set_style

SEED = 0


def _single_thread(fn):
    """Pin BLAS to one thread inside small-Gram solves (this box oversubscribes)."""
    @functools.wraps(fn)
    def wrap(*a, **k):
        try:
            from threadpoolctl import threadpool_limits
            with threadpool_limits(1):
                return fn(*a, **k)
        except Exception:
            return fn(*a, **k)
    return wrap


# =============================================================================
# The spectral embedding and the Laplace-readout kernel
# =============================================================================

def spectral_embedding(x, omegas, amps=None, s=1.0):
    """Map 1-D inputs to the cosine/sine spectral feature map at frequencies `omegas`:
        psi(x)_k = a_k [cos(2 pi s omega_k x), sin(2 pi s omega_k x)].
    The atomic spectral measure puts mass a_k^2 at frequency s*omega_k (Bochner)."""
    x = np.asarray(x, float).ravel()
    omegas = np.asarray(omegas, float)
    a = np.ones_like(omegas) if amps is None else np.asarray(amps, float)
    arg = 2.0 * np.pi * s * np.outer(x, omegas)                 # (n, K)
    return np.concatenate([a * np.cos(arg), a * np.sin(arg)], axis=1)   # (n, 2K)


def laplace_gram(Pa, Pb, T):
    """Laplace readout over the embedding: k = exp(-||phi(a)-phi(b)|| / T). Unit diagonal."""
    return np.exp(-cdist(Pa, Pb) / T)


class SpectralLaplaceKernel:
    """Finite multi-scale spectral kernel: H banks of a Laplace readout over a shared
    spectral embedding, fused convexly. K(x,x') = sum_h w_h exp(-||phi(x)-phi(x')||/T_h)."""

    def __init__(self, omegas, amps, T, w=None):
        self.omegas = np.asarray(omegas, float)
        self.amps = np.asarray(amps, float)
        self.T = np.atleast_1d(np.asarray(T, float))            # one bandwidth per bank
        H = len(self.T)
        self.w = np.full(H, 1.0 / H) if w is None else np.asarray(w, float)

    def gram(self, A, B):
        Pa = spectral_embedding(A, self.omegas, self.amps)
        Pb = spectral_embedding(B, self.omegas, self.amps)
        return sum(wh * laplace_gram(Pa, Pb, Th) for wh, Th in zip(self.w, self.T))

    def spectral_density(self, grid):
        """The atomic spectral measure as mass a_k^2 at each frequency (for plotting)."""
        return self.omegas, self.amps ** 2


def rbf_kernel(A, B, ell):
    """The RBF (Gaussian) kernel: the single-Gaussian-at-zero special case of Bochner,
    and the C-infinity rung of the roughness ladder."""
    tau = cdist(np.asarray(A, float).reshape(-1, 1), np.asarray(B, float).reshape(-1, 1))
    return np.exp(-0.5 * tau ** 2 / ell ** 2)


def laplace_readout(A, B, ell):
    """A plain Laplace kernel exp(-|x-x'|/ell) on the raw input — the rough rung of the
    ladder (same spectrum as a single scale, but a cusp readout)."""
    tau = cdist(np.asarray(A, float).reshape(-1, 1), np.asarray(B, float).reshape(-1, 1))
    return np.exp(-tau / ell)


# =============================================================================
# Reading the spectral measure from data (least-squares periodogram)
# =============================================================================

def periodogram(X, y, freqs):
    """Per-frequency variance explained by a sinusoid over a smooth trend: the data's
    spectral density. This is the measure the spectral kernel places mass on."""
    X = np.asarray(X, float).ravel(); y = np.asarray(y, float).ravel()
    trend = np.vstack([np.ones_like(X), X, X ** 2]).T
    r = y - trend @ np.linalg.lstsq(trend, y, rcond=None)[0]
    tot = float(r @ r) + 1e-12
    power = np.empty(len(freqs))
    for i, f in enumerate(freqs):
        B = np.vstack([np.cos(2 * np.pi * f * X), np.sin(2 * np.pi * f * X)]).T
        resid = r - B @ np.linalg.lstsq(B, r, rcond=None)[0]
        power[i] = 1.0 - float(resid @ resid) / tot          # fraction of variance explained
    return power


def dominant_frequency(X, y, mu_max=6.0, n_grid=600):
    """The frequency whose sinusoid best explains the detrended signal."""
    freqs = np.linspace(0.1, mu_max, n_grid)
    return float(freqs[np.argmax(periodogram(X, y, freqs))])


# =============================================================================
# Fitting the finite spectral kernel: data-chosen frequency support, query-fold T
# =============================================================================

@_single_thread
def fit_spectral_laplace(Xtr, ytr, mu_max=6.0, lam=1e-3, seed=SEED):
    """Fit a finite spectral-Laplace kernel. The frequency support is read from the data
    (periodogram: a trend atom near zero plus the dominant frequency and its first
    harmonic), so it is reproducible; the bandwidth T is selected on a held-out query fold
    (the Chapter 7 discipline). Returns (kernel, predict, true_freq_support)."""
    Xtr = np.asarray(Xtr, float).ravel(); ytr = np.asarray(ytr, float).ravel()
    f_star = dominant_frequency(Xtr, ytr, mu_max)
    omegas = np.array([0.0, 0.5, f_star, 2.0 * f_star])        # trend atoms + periodic + harmonic
    amps = np.array([1.0, 1.0, 1.0, 1.0])

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(Xtr)); ns = len(Xtr) // 2
    s_idx, q_idx = perm[:ns], perm[ns:]
    Xs, ys, Xq, yq = Xtr[s_idx], ytr[s_idx], Xtr[q_idx], ytr[q_idx]
    ybar = ys.mean()
    Ps, Pq = spectral_embedding(Xs, omegas, amps), spectral_embedding(Xq, omegas, amps)

    best_T, best_loss = None, np.inf
    for T in np.logspace(-1.0, 1.5, 40):                       # query-fold bandwidth scan
        Kss = laplace_gram(Ps, Ps, T)
        alpha = np.linalg.solve(Kss + lam * np.eye(ns), ys - ybar)
        pred = laplace_gram(Pq, Ps, T) @ alpha + ybar
        loss = float(np.mean((pred - yq) ** 2))
        if loss < best_loss:
            best_loss, best_T = loss, T

    kernel = SpectralLaplaceKernel(omegas, amps, best_T)
    K = kernel.gram(Xtr, Xtr)
    alpha = np.linalg.solve(K + lam * np.eye(len(Xtr)), ytr - ytr.mean())

    def predict(Xnew):
        return kernel.gram(np.asarray(Xnew, float).ravel(), Xtr) @ alpha + ytr.mean()

    return kernel, predict, f_star


@_single_thread
def fit_rbf(Xtr, ytr, lam=1e-3, seed=SEED):
    """Best single RBF bandwidth by the same query-fold criterion — the one-scale baseline."""
    Xtr = np.asarray(Xtr, float).ravel(); ytr = np.asarray(ytr, float).ravel()
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(Xtr)); ns = len(Xtr) // 2
    s_idx, q_idx = perm[:ns], perm[ns:]
    Xs, ys, Xq, yq = Xtr[s_idx], ytr[s_idx], Xtr[q_idx], ytr[q_idx]
    ybar = ys.mean()
    best_ell, best_loss = None, np.inf
    for ell in np.logspace(-2.5, 1.0, 60):
        Kss = rbf_kernel(Xs, Xs, ell)
        alpha = np.linalg.solve(Kss + lam * np.eye(ns), ys - ybar)
        pred = rbf_kernel(Xq, Xs, ell) @ alpha + ybar
        loss = float(np.mean((pred - yq) ** 2))
        if loss < best_loss:
            best_loss, best_ell = loss, ell

    K = rbf_kernel(Xtr, Xtr, best_ell)
    alpha = np.linalg.solve(K + lam * np.eye(len(Xtr)), ytr - ytr.mean())

    def predict(Xnew):
        return rbf_kernel(np.asarray(Xnew, float).ravel(), Xtr, best_ell) @ alpha + ytr.mean()

    return best_ell, predict


# =============================================================================
# Demonstrations
# =============================================================================

def smooth_plus_periodic(n=120, x_max=1.0, freq=3.0, noise=0.05, seed=SEED):
    """A 1-D target at two separated scales: a smooth quadratic trend plus a periodic
    oscillation. No single RBF bandwidth serves both. periodic = 0.5 sin(2 pi freq x)."""
    rng = np.random.RandomState(seed)
    X = np.sort(rng.rand(n) * x_max)
    trend = 0.8 * (X - 0.5) ** 2
    periodic = 0.5 * np.sin(2.0 * np.pi * freq * X)
    return X, trend + periodic + noise * rng.randn(n), float(freq)


def _rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def periodic_extrapolation_demo(freq=3.0, n=80, x_max=1.0, x_test_max=2.0, seed=SEED):
    """The headline demonstration: the finite spectral-Laplace kernel carries the oscillation
    beyond the data range (it learned a frequency); the RBF reverts to the mean (it learned
    only a length scale)."""
    X, y, true_freq = smooth_plus_periodic(n=n, x_max=x_max, freq=freq, seed=seed)
    sm_kernel, sm_pred, f_star = fit_spectral_laplace(X, y, mu_max=2.0 * freq, seed=seed)
    ell, rbf_pred = fit_rbf(X, y, seed=seed)

    Xg = np.linspace(0, x_test_max, 400)
    truth = 0.8 * (Xg - 0.5) ** 2 + 0.5 * np.sin(2.0 * np.pi * true_freq * Xg)
    in_hull, extrap = Xg <= x_max, Xg > x_max
    smg, rbfg = sm_pred(Xg), rbf_pred(Xg)
    return {
        "X": X, "y": y, "Xg": Xg, "truth": truth, "sm": smg, "rbf": rbfg,
        "x_max": x_max, "true_freq": true_freq, "recovered_freq": f_star, "ell": ell,
        "sm_test_rmse": _rmse(smg[in_hull], truth[in_hull]),
        "rbf_test_rmse": _rmse(rbfg[in_hull], truth[in_hull]),
        "sm_extrap_rmse": _rmse(smg[extrap], truth[extrap]),
        "rbf_extrap_rmse": _rmse(rbfg[extrap], truth[extrap]),
    }


def _ou_path(grid, ell=0.05, seed=SEED):
    """A sample path of the Ornstein-Uhlenbeck process (the Laplace/Matern-1/2 GP) on `grid`
    — an H^{1/2} rough function: continuous but nowhere smooth, the regularity tabular
    targets carry and the Gaussian RKHS excludes."""
    rng = np.random.RandomState(seed)
    g = np.sort(grid)
    K = np.exp(-cdist(g.reshape(-1, 1), g.reshape(-1, 1)) / ell)   # Laplace covariance
    L = np.linalg.cholesky(K + 1e-8 * np.eye(len(g)))
    return g, L @ rng.randn(len(g))


def roughness_ladder_demo(n=200, ell_truth=0.04, seed=SEED):
    """The readout sets the order, not the spectrum: fit a genuinely rough (H^{1/2}) target —
    an Ornstein-Uhlenbeck sample path — with a Gaussian (RBF, C-infinity) readout and a
    Laplace (cusp, H^{(d+1)/2}) readout. The Gaussian RKHS does not contain the rough
    component, so it oversmooths; the Laplace readout reaches it (Theorem A)."""
    rng = np.random.RandomState(seed)
    Xg, fg = _ou_path(np.linspace(0, 1, 400), ell=ell_truth, seed=seed)   # dense truth path
    idx = np.sort(rng.choice(len(Xg), n, replace=False))                  # observe a subset
    X, y = Xg[idx], fg[idx] + 0.02 * rng.randn(n)
    out = {"X": X, "y": y, "Xg": Xg, "truth": fg}
    for name, kfun in [("rbf", rbf_kernel), ("laplace", laplace_readout)]:
        best = min(np.logspace(-2, 0.5, 40),
                   key=lambda e: _fit_eval(kfun, X, y, e))
        K = kfun(X, X, best); a = np.linalg.solve(K + 1e-3 * np.eye(n), y - y.mean())
        out[name] = kfun(Xg, X, best) @ a + y.mean()
        out[name + "_rmse"] = _rmse(out[name], fg)
    return out


def _fit_eval(kfun, X, y, ell, lam=1e-3, seed=SEED):
    rng = np.random.RandomState(seed); perm = rng.permutation(len(X)); ns = len(X) // 2
    s, q = perm[:ns], perm[ns:]; ybar = y[s].mean()
    K = kfun(X[s], X[s], ell); a = np.linalg.solve(K + lam * np.eye(ns), y[s] - ybar)
    return _rmse(kfun(X[q], X[s], ell) @ a + ybar, y[q])


def california_spectral_density(features=("MedInc", "Latitude"), n=1500, seed=SEED):
    """Read the per-feature spectral measure off California by periodogram: which features
    carry a smooth trend (low-frequency mass) versus structure at a definite scale (a peak)."""
    d = load_california()
    rng = np.random.RandomState(seed)
    idx = rng.choice(d.n, min(n, d.n), replace=False)
    freqs = np.linspace(0.05, 3.0, 120)
    out = {}
    for f in features:
        xj = d.Xtr[idx, d.col(f)]
        order = np.argsort(xj)
        power = periodogram(xj[order], d.ytr[idx][order], freqs)
        power = np.clip(power, 0, None); power = power / (power.sum() + 1e-12)
        low_mass = float(power[freqs < 0.5].sum())
        out[f] = {"freqs": freqs, "power": power, "peak": float(freqs[np.argmax(power)]),
                  "low_freq_mass": low_mass}
    return out


# =============================================================================
# Figures
# =============================================================================

def make_roughness_figure(seed=SEED):
    r = roughness_ladder_demo(seed=seed)
    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    ax.scatter(r["X"], r["y"], s=10, c="#999999", label="data (rough target)", zorder=1)
    ax.plot(r["Xg"], r["truth"], "k--", lw=1, label="truth", zorder=2)
    ax.plot(r["Xg"], r["rbf"], color="#3b6ea5", lw=2,
            label=fr"RBF readout ($C^\infty$), RMSE {r['rbf_rmse']:.3f}")
    ax.plot(r["Xg"], r["laplace"], color="#c44e52", lw=2,
            label=fr"Laplace readout ($H^{{(d+1)/2}}$), RMSE {r['laplace_rmse']:.3f}")
    ax.set_title("The readout sets the roughness: same data, two outer functions.\n"
                 "The Gaussian oversmooths the kinks; the Laplace cusp reaches them.", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=8)
    return fig


def make_periodic_figure(seed=SEED):
    d = periodic_extrapolation_demo(seed=seed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ax.axvspan(d["x_max"], d["Xg"].max(), color="#f0f0f0", zorder=0, label="extrapolation")
    ax.scatter(d["X"], d["y"], s=12, c="#999999", zorder=2, label="training data")
    ax.plot(d["Xg"], d["truth"], "k--", lw=1, zorder=3, label="truth")
    ax.plot(d["Xg"], d["sm"], color="#c44e52", lw=2, zorder=4,
            label=f"spectral-Laplace (extrap RMSE {d['sm_extrap_rmse']:.2f})")
    ax.plot(d["Xg"], d["rbf"], color="#3b6ea5", lw=2, zorder=4,
            label=f"RBF (extrap RMSE {d['rbf_extrap_rmse']:.2f})")
    ax.set_title("A geometry that knows the frequency extrapolates the oscillation;\n"
                 "one that knows only a length scale flattens.", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=8)

    ax2 = axes[1]
    freqs = np.linspace(0.1, 6.0, 400)
    ax2.plot(freqs, periodogram(d["X"], d["y"], freqs), color="#c44e52")
    ax2.axvline(d["true_freq"], ls="--", color="k", lw=1, label=f"true {d['true_freq']:.1f}")
    ax2.axvline(d["recovered_freq"], ls=":", color="#c44e52", lw=1.5,
                label=f"recovered {d['recovered_freq']:.2f}")
    ax2.set_title("Recovered spectral measure", fontsize=10)
    ax2.set_xlabel("frequency ω"); ax2.set_ylabel("variance explained"); ax2.legend(fontsize=8)
    return fig


def make_california_density_figure(features=("MedInc", "Latitude"), seed=SEED):
    dens = california_spectral_density(features=features, seed=seed)
    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    for f, c in zip(features, ["#3b6ea5", "#c44e52"]):
        D = dens[f]
        ax.plot(D["freqs"], D["power"], color=c,
                label=f"{f}: peak ω={D['peak']:.2f}, low-freq mass {D['low_freq_mass']:.0%}")
    ax.set_title("Per-feature spectral measure on California:\n"
                 "income is a low-frequency trend; latitude carries scale-specific structure",
                 fontsize=10)
    ax.set_xlabel("frequency ω"); ax.set_ylabel("normalized spectral mass"); ax.legend(fontsize=9)
    return fig


# =============================================================================
# CLI
# =============================================================================

def main(argv=None):
    p = argparse.ArgumentParser(description="Chapter 8 — spectral kernels and Bochner")
    p.add_argument("--out-prefix", default=None)
    args = p.parse_args(argv)
    set_style()

    d = periodic_extrapolation_demo()
    print("=" * 70, "\nSPECTRAL-LAPLACE (finite MS-SKM) — smooth + periodic")
    print(f"  true frequency           {d['true_freq']:.2f}")
    print(f"  recovered peak frequency {d['recovered_freq']:.2f}")
    print(f"  RBF bandwidth (selected) {d['ell']:.3f}")
    print(f"  in-hull test RMSE   spectral {d['sm_test_rmse']:.3f}   RBF {d['rbf_test_rmse']:.3f}")
    print(f"  extrapolation RMSE  spectral {d['sm_extrap_rmse']:.3f}   RBF {d['rbf_extrap_rmse']:.3f}")
    print("  -> the spectral kernel carries the oscillation beyond the data; the RBF flattens.")

    r = roughness_ladder_demo()
    print(f"\nROUGHNESS LADDER (kinked target): RBF (C-inf) {r['rbf_rmse']:.3f}, "
          f"Laplace (H^(d+1)/2) {r['laplace_rmse']:.3f} -- the readout sets the order.")

    dens = california_spectral_density()
    print("\nCALIFORNIA per-feature spectral measure:")
    for f, D in dens.items():
        print(f"  {f:10s} peak ω={D['peak']:.2f}, low-frequency mass {D['low_freq_mass']:.0%}")

    if args.out_prefix:
        import matplotlib
        matplotlib.use("Agg")
        make_roughness_figure().savefig(f"{args.out_prefix}1_roughness.pdf")
        make_periodic_figure().savefig(f"{args.out_prefix}2_spectral_measure.pdf")
        print("wrote figures with prefix", args.out_prefix)


if __name__ == "__main__":
    main()
