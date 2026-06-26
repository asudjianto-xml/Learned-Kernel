"""Chapter 8 — spectral kernels and Bochner.

By Bochner's theorem a continuous stationary kernel is the Fourier transform of a finite
nonnegative spectral measure,

    k(tau) = integral exp(i omega . tau) d mu(omega),   tau = x - x',

so *choosing a stationary geometry is choosing a measure on frequencies*, and learning the
kernel is learning that measure. A single RBF bandwidth is one Gaussian bump in frequency
space — one scale, one knee at |omega| ~ 1/T. A target that carries a smooth trend AND a
periodic oscillation lives at two separated scales, and no single bandwidth can serve both.

The cure is a measure flexible enough to be a universal geometry yet cheap to learn: the
**Gaussian spectral-mixture (SM) kernel**, the Fourier transform of a mixture of Q Gaussians
in frequency (Wilson & Adams 2013). With weights w_q, frequency means mu_q and variances
v_q, the 1-D stationary kernel is

    k(tau) = sum_q w_q exp(-2 pi^2 tau^2 v_q) cos(2 pi mu_q tau),

each term a cosine at frequency mu_q under a Gaussian envelope of bandwidth set by v_q. A
point mass at mu_q = 0 is the constant kernel; a single Gaussian bump at mu = 0 is the RBF;
a bump away from 0 is a periodic component the RBF cannot carry. The measure IS the geometry.

This module mirrors the torch `skm.SpectralMixture` / `measure.gram` in NumPy/SciPy only.
It demonstrates:

  (i)  **recover periodic + smooth structure a single RBF cannot** — on a 1-D smooth-plus-
       periodic target the SM kernel fits and *extrapolates the periodicity* while the RBF
       goes flat outside the data; plus a California demonstration of the learned per-scale
       spectral content;
  (ii) **the roughness ladder** — the *readout* sets the regularity order, not the spectrum.
       The Gaussian (RBF) readout exp(-r^2/T^2) is analytic at the origin and gives a C-infinity
       class that oversmooths; the Laplace readout exp(-r/T) has a cusp and drops the RKHS to
       H^{(d+1)/2}, the Matern-1/2 / ReLU-NTK roughness tabular targets inhabit (Theorem A).

    python -m lkbook.chapters.ch08 --out-prefix fig8
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from lkbook import load_california, set_style

SEED = 0


def _single_thread(fn):
    """The spectral-measure fit is thousands of tiny Gram solves; on a many-core box BLAS
    thread oversubscription makes those small solves orders of magnitude slower. Pin BLAS to
    one thread for the duration of the call."""
    import functools

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with threadpool_limits(limits=1):
            return fn(*args, **kwargs)

    return wrapped


# =============================================================================
# The Gaussian spectral-mixture (Bochner) kernel — NumPy mirror of skm
# =============================================================================

class SpectralMixtureKernel:
    """Gaussian spectral-mixture kernel in 1-D (Wilson & Adams 2013), the Fourier transform
    of a mixture of Q Gaussians in frequency:

        k(tau) = sum_q w_q exp(-2 pi^2 tau^2 v_q) cos(2 pi mu_q tau),   tau = x - x'.

    Parameters are the spectral measure: weights w_q >= 0 (mass), means mu_q >= 0
    (frequency), variances v_q > 0 (bandwidth). The kernel is PSD by Bochner (a nonnegative
    mixture of Gaussians in frequency is a nonnegative measure) and unit-diagonal when the
    weights sum to one (k(0) = sum_q w_q). It is the complete learnable stationary
    parameterization: as Q grows the mixture approximates any spectral density.
    """

    def __init__(self, w, mu, v):
        self.w = np.asarray(w, float)
        self.mu = np.asarray(mu, float)
        self.v = np.asarray(v, float)

    def profile(self, tau):
        """k as a function of the lag tau (1-D array in, 1-D array out)."""
        tau = np.asarray(tau, float)[..., None]                     # (..., 1)
        env = np.exp(-2.0 * np.pi ** 2 * tau ** 2 * self.v)         # Gaussian envelope
        cos = np.cos(2.0 * np.pi * self.mu * tau)                   # periodic carrier
        return (self.w * env * cos).sum(-1)

    def gram(self, A, B):
        """Gram matrix between 1-D inputs A (n,) and B (m,)."""
        A = np.asarray(A, float).ravel()
        B = np.asarray(B, float).ravel()
        return self.profile(A[:, None] - B[None, :])

    def spectral_density(self, omega):
        """The spectral measure as a density on (one-sided) frequency omega: a sum of Q
        Gaussians, mass w_q at mean mu_q with variance v_q. This is the geometry plotted."""
        omega = np.asarray(omega, float)[..., None]
        g = np.exp(-0.5 * (omega - self.mu) ** 2 / self.v) / np.sqrt(2.0 * np.pi * self.v)
        return (self.w * g).sum(-1)


def rbf_kernel(A, B, ell):
    """The RBF (squared-exponential) kernel — the single-Gaussian-at-zero special case of
    Bochner. Its spectral measure is one Gaussian bump centered at omega = 0, so it carries
    exactly one scale and cannot place mass at a nonzero frequency."""
    A = np.asarray(A, float).ravel()
    B = np.asarray(B, float).ravel()
    tau = A[:, None] - B[None, :]
    return np.exp(-0.5 * tau ** 2 / ell ** 2)


# =============================================================================
# Fitting the spectral measure (least squares on a held-out fold; Ch. 7 discipline)
# =============================================================================

def _unpack(p, Q):
    """Map an unconstrained vector to (w, mu, v) with w on the simplex, mu, v > 0."""
    lw, lmu, lv = p[:Q], p[Q:2 * Q], p[2 * Q:]
    w = np.exp(lw - lw.max()); w = w / w.sum()              # softmax -> simplex (unit diag)
    mu = np.exp(lmu)                                        # frequency > 0
    v = np.exp(lv)                                          # bandwidth variance > 0
    return w, mu, v


def _sm_profile_from_lag(T, w, mu, v):
    """k(T) for a cached lag matrix T (any shape) and SM parameters."""
    T = T[..., None]
    return (w * np.exp(-2.0 * np.pi ** 2 * T ** 2 * v) * np.cos(2.0 * np.pi * mu * T)).sum(-1)


@_single_thread
def fit_sm_kernel(Xtr, ytr, Q=4, lam=1e-4, seed=SEED, n_restarts=3, mu_max=4.0):
    """Fit the spectral measure of a Q-component SM kernel by minimizing held-out (query-fold)
    KRR error, the leakage-free selection criterion of Chapter 7. The support fold fits the
    ridge solution; the query fold scores the spectral measure. Returns the fitted kernel and
    the selected ridge solver bound to the full training set."""
    Xtr = np.asarray(Xtr, float).ravel()
    ytr = np.asarray(ytr, float).ravel()
    rng = np.random.RandomState(seed)
    ns = len(Xtr) // 2
    perm = rng.permutation(len(Xtr))
    s_idx, q_idx = perm[:ns], perm[ns:]                     # support / query split
    Xs, ys, Xq, yq = Xtr[s_idx], ytr[s_idx], Xtr[q_idx], ytr[q_idx]
    ybar = ys.mean()
    Tss = Xs[:, None] - Xs[None, :]                         # cache the lag matrices once
    Tqs = Xq[:, None] - Xs[None, :]
    Iss = lam * np.eye(len(Xs))

    def query_loss(p):
        w, mu, v = _unpack(p, Q)
        Kss = _sm_profile_from_lag(Tss, w, mu, v)
        alpha = np.linalg.solve(Kss + Iss, ys - ybar)
        pred = _sm_profile_from_lag(Tqs, w, mu, v) @ alpha + ybar
        return float(np.mean((pred - yq) ** 2))

    best = None
    for r in range(n_restarts):
        rs = np.random.RandomState(seed + r)
        # spread initial frequency means across [0, mu_max]; small bandwidths
        mu0 = np.linspace(0.05, mu_max, Q) * (0.5 + rs.rand(Q))
        p0 = np.concatenate([np.zeros(Q),                  # equal weights
                             np.log(np.clip(mu0, 1e-2, None)),
                             np.log(0.02 + 0.05 * rs.rand(Q))])
        res = minimize(query_loss, p0, method="L-BFGS-B",
                       options=dict(maxiter=300, ftol=1e-10, gtol=1e-8))
        if best is None or res.fun < best.fun:
            best = res

    w, mu, v = _unpack(best.x, Q)
    kernel = SpectralMixtureKernel(w, mu, v)
    # refit ridge on the full training set with the selected measure
    K = kernel.gram(Xtr, Xtr)
    alpha = np.linalg.solve(K + lam * np.eye(len(Xtr)), ytr - ytr.mean())

    def predict(Xnew):
        Xnew = np.asarray(Xnew, float).ravel()
        return kernel.gram(Xnew, Xtr) @ alpha + ytr.mean()

    return kernel, predict, float(best.fun)


@_single_thread
def fit_rbf(Xtr, ytr, lam=1e-4, seed=SEED, n_ell=40):
    """Fit a single RBF bandwidth by the same query-fold criterion, so the comparison to the
    SM kernel is on equal footing: one bandwidth chosen as well as cross-validation allows."""
    Xtr = np.asarray(Xtr, float).ravel()
    ytr = np.asarray(ytr, float).ravel()
    rng = np.random.RandomState(seed)
    ns = len(Xtr) // 2
    perm = rng.permutation(len(Xtr))
    s_idx, q_idx = perm[:ns], perm[ns:]
    Xs, ys, Xq, yq = Xtr[s_idx], ytr[s_idx], Xtr[q_idx], ytr[q_idx]
    ybar = ys.mean()

    ells = np.logspace(-2.0, 1.0, n_ell)
    best_ell, best_loss = None, np.inf
    for ell in ells:
        Kss = rbf_kernel(Xs, Xs, ell)
        alpha = np.linalg.solve(Kss + lam * np.eye(len(Xs)), ys - ybar)
        pred = rbf_kernel(Xq, Xs, ell) @ alpha + ybar
        loss = float(np.mean((pred - yq) ** 2))
        if loss < best_loss:
            best_ell, best_loss = ell, loss

    K = rbf_kernel(Xtr, Xtr, best_ell)
    alpha = np.linalg.solve(K + lam * np.eye(len(Xtr)), ytr - ytr.mean())

    def predict(Xnew):
        Xnew = np.asarray(Xnew, float).ravel()
        return rbf_kernel(Xnew, Xtr, best_ell) @ alpha + ytr.mean()

    return best_ell, predict, best_loss


# =============================================================================
# The smooth-plus-periodic target (generated INSIDE ch08, not in data.py)
# =============================================================================

def smooth_plus_periodic(n=120, x_max=1.0, freq=3.0, noise=0.05, seed=SEED):
    """A 1-D target with two separated scales: a smooth quadratic trend plus a periodic
    component at a definite frequency. The smooth part needs a long correlation length; the
    oscillation needs a short one. No single RBF bandwidth can fit both — the motivating
    failure of one bandwidth, and the clean case for a spectral measure with two peaks.

    Returns (Xtr, ytr, true_freq); the periodic component is sin(2 pi freq x)."""
    rng = np.random.RandomState(seed)
    X = np.sort(rng.uniform(0.0, x_max, n))
    trend = 0.8 * (X - 0.5) ** 2                            # smooth low-frequency trend
    periodic = 0.5 * np.sin(2.0 * np.pi * freq * X)         # periodic at `freq`
    y = trend + periodic + noise * rng.randn(n)
    return X, y, float(freq)


def periodic_extrapolation_demo(freq=3.0, n=80, x_max=1.0, x_test_max=2.0,
                                Q=4, seed=SEED):
    """The headline demonstration. Fit the SM kernel and a single RBF on a smooth-plus-periodic
    target observed on [0, x_max]; evaluate on [0, x_test_max] which runs BEYOND the data hull.
    The SM kernel carries the oscillation forward (the periodic component continues); the RBF
    reverts to the mean and goes flat. Returns a dict of arrays and recovered numbers."""
    X, y, true_freq = smooth_plus_periodic(n=n, x_max=x_max, freq=freq, seed=seed)
    kernel, sm_pred, sm_qloss = fit_sm_kernel(X, y, Q=Q, seed=seed)
    ell, rbf_pred, rbf_qloss = fit_rbf(X, y, seed=seed)

    Xg = np.linspace(0.0, x_test_max, 400)
    trend_g = 0.8 * (Xg - 0.5) ** 2
    periodic_g = 0.5 * np.sin(2.0 * np.pi * true_freq * Xg)
    truth_g = trend_g + periodic_g
    sm_g, rbf_g = sm_pred(Xg), rbf_pred(Xg)

    # test region strictly beyond the data hull: [x_max, x_test_max]
    out = Xg > x_max
    sm_extrap_rmse = float(np.sqrt(np.mean((sm_g[out] - truth_g[out]) ** 2)))
    rbf_extrap_rmse = float(np.sqrt(np.mean((rbf_g[out] - truth_g[out]) ** 2)))

    # in-hull test points (fresh draw on [0, x_max])
    Xte, yte, _ = smooth_plus_periodic(n=400, x_max=x_max, freq=freq, noise=0.0,
                                       seed=seed + 99)
    sm_test_rmse = float(np.sqrt(np.mean((sm_pred(Xte) - yte) ** 2)))
    rbf_test_rmse = float(np.sqrt(np.mean((rbf_pred(Xte) - yte) ** 2)))

    # recovered dominant frequency: the SM mean carrying the most off-DC mass
    nonzero = kernel.mu > 0.3                               # drop the trend/DC component
    if nonzero.any():
        dom = kernel.mu[nonzero][np.argmax(kernel.w[nonzero])]
    else:
        dom = kernel.mu[np.argmax(kernel.w)]
    recovered_freq = float(dom)

    return dict(X=X, y=y, Xg=Xg, truth_g=truth_g, sm_g=sm_g, rbf_g=rbf_g,
                x_max=x_max, x_test_max=x_test_max, true_freq=true_freq,
                recovered_freq=recovered_freq, ell=float(ell),
                sm_test_rmse=sm_test_rmse, rbf_test_rmse=rbf_test_rmse,
                sm_extrap_rmse=sm_extrap_rmse, rbf_extrap_rmse=rbf_extrap_rmse,
                kernel=kernel)


# =============================================================================
# The roughness ladder — the readout sets the order, not the spectrum
# =============================================================================

def laplace_readout(A, B, ell):
    """Laplace readout exp(-|tau|/ell): a cusp at tau = 0, RKHS H^{(d+1)/2} (Matern-1/2)."""
    A = np.asarray(A, float).ravel()
    B = np.asarray(B, float).ravel()
    tau = A[:, None] - B[None, :]
    return np.exp(-np.abs(tau) / ell)


def _krr_fit_predict(Kss, ys, Kqs, ybar, lam):
    alpha = np.linalg.solve(Kss + lam * np.eye(len(ys)), ys - ybar)
    return Kqs @ alpha + ybar


@_single_thread
def roughness_ladder_demo(n=90, x_max=1.0, noise=0.03, seed=SEED, lam=1e-5):
    """The readout sets the roughness. On a rough target (a Brownian-like sample path, which
    lives in the low-Sobolev class), fit KRR under three readouts with the bandwidth chosen by
    a query fold: Gaussian/RBF (C-infinity, oversmooths), Laplace (cusp, H^{(d+1)/2}, the
    Matern-1/2 / ReLU-NTK class), and a finer view. The Laplace readout tracks the kinks the
    RBF rounds off. Returns curves and test RMSEs."""
    rng = np.random.RandomState(seed)
    X = np.sort(rng.uniform(0.0, x_max, n))
    # a rough target: integrated white noise (a discretized Brownian path), in H^{1/2-}
    grid = np.linspace(0, x_max, 600)
    incr = rng.randn(len(grid)) * np.sqrt(grid[1] - grid[0])
    path = np.cumsum(incr)
    path = path - path.mean()
    f_true = np.interp(X, grid, path)
    y = f_true + noise * rng.randn(n)

    ns = len(X) // 2
    perm = rng.permutation(len(X))
    s_idx, q_idx = perm[:ns], perm[ns:]
    Xs, ys, Xq, yq = X[s_idx], y[s_idx], X[q_idx], y[q_idx]
    ybar = ys.mean()
    ells = np.logspace(-2.5, 0.0, 50)

    def select(kfun):
        best, bl = None, np.inf
        for ell in ells:
            pred = _krr_fit_predict(kfun(Xs, Xs, ell), ys, kfun(Xq, Xs, ell), ybar, lam)
            l = np.mean((pred - yq) ** 2)
            if l < bl:
                best, bl = ell, l
        return best

    Xg = np.linspace(0, x_max, 400)
    truth_g = np.interp(Xg, grid, path)
    out = {}
    rmse = {}
    for name, kfun in [("RBF (Gaussian, C-inf)", rbf_kernel),
                       ("Laplace (cusp, H^{(d+1)/2})", laplace_readout)]:
        ell = select(kfun)
        alpha = np.linalg.solve(kfun(X, X, ell) + lam * np.eye(len(X)), y - y.mean())
        out[name] = kfun(Xg, X, ell) @ alpha + y.mean()
        rmse[name] = float(np.sqrt(np.mean((out[name] - truth_g) ** 2)))

    return dict(X=X, y=y, Xg=Xg, truth_g=truth_g, curves=out, rmse=rmse)


# =============================================================================
# California demonstration: the learned per-feature spectral density
# =============================================================================

def california_spectral_density(feature="MedInc", Q=4, n_train=400, seed=SEED):
    """Fit the SM kernel on a single standardized California feature against the target, and
    read off the learned spectral density. A feature carrying a smooth trend shows
    low-frequency mass; structure at a definite scale shows a peak away from zero. Returns the
    fitted kernel, the recovered density on a frequency grid, and the SM-vs-RBF query loss."""
    cal = load_california()
    j = cal.col(feature)
    rng = np.random.RandomState(seed)
    idx = rng.choice(cal.n, min(n_train, cal.n), replace=False)
    X = cal.Xtr[idx, j]
    y = cal.ytr[idx]
    kernel, sm_pred, sm_qloss = fit_sm_kernel(X, y, Q=Q, seed=seed, mu_max=3.0)
    _, _, rbf_qloss = fit_rbf(X, y, seed=seed)
    omega = np.linspace(0, 3.0, 400)
    dens = kernel.spectral_density(omega)
    lowfreq_mass = float(kernel.w[kernel.mu < 0.5].sum())       # share of mass near DC
    return dict(kernel=kernel, omega=omega, density=dens, feature=feature,
                lowfreq_mass=lowfreq_mass,
                sm_qloss=float(sm_qloss), rbf_qloss=float(rbf_qloss))


# =============================================================================
# Density parameterization: Gauss-Hermite quadrature of a log-frequency density
# =============================================================================

def density_atoms_gauss_hermite(mu_log, gamma, G):
    """Discretize one Gaussian component on log-frequency (mean mu_log, std gamma) into G
    atoms by Gauss-Hermite quadrature, mirroring skm.measure.density_to_atoms. The trainable
    size is the 2 numbers (mu_log, gamma); G is a numerical resolution, not a parameter
    (Corollary B.1, 'continuity is free'). Returns (omega_atoms, weights) summing to one."""
    nodes, wts = hermegauss(G)                              # probabilists' Hermite (weight e^{-x^2/2})
    wts = wts / wts.sum()                                   # normalize to a probability rule
    omega = np.exp(mu_log + gamma * nodes)                  # log-normal frequency atoms
    return omega, wts


# =============================================================================
# Figures (return a Figure; no Agg at import)
# =============================================================================

def make_periodic_figure(seed=SEED):
    """Figure 8.1 companion: the SM kernel recovers and EXTRAPOLATES the periodic component a
    single RBF cannot. Left: the fit and the extrapolation beyond the data hull. Right: the
    learned spectral measure (a sharp peak at the true frequency) vs the RBF's single bump at
    zero."""
    d = periodic_extrapolation_demo(seed=seed)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), constrained_layout=True)

    ax = axes[0]
    ax.scatter(d["X"], d["y"], s=14, c="#444", zorder=4, label="data (observed on [0,1])")
    ax.plot(d["Xg"], d["truth_g"], color="#888", lw=1.4, ls=":", label="true signal")
    ax.plot(d["Xg"], d["sm_g"], color="#2ca02c", lw=2.0,
            label=f"spectral mixture (extrap RMSE {d['sm_extrap_rmse']:.2f})")
    ax.plot(d["Xg"], d["rbf_g"], color="#c44e52", lw=2.0,
            label=f"single RBF (extrap RMSE {d['rbf_extrap_rmse']:.2f})")
    ax.axvspan(d["x_max"], d["x_test_max"], color="#f0f0f0", zorder=0)
    ax.text(0.5 * (d["x_max"] + d["x_test_max"]), ax.get_ylim()[0], " extrapolation",
            va="bottom", ha="center", fontsize=9, color="#555")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Smooth + periodic: the spectral kernel carries the\noscillation forward; "
                 "the RBF reverts to the mean", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    k = d["kernel"]
    omega = np.linspace(0, 5.0, 500)
    ax.plot(omega, k.spectral_density(omega), color="#2ca02c", lw=2.0,
            label="learned spectral measure")
    ax.axvline(d["true_freq"], color="#888", ls=":", lw=1.4,
               label=f"true frequency = {d['true_freq']:.1f}")
    ax.axvline(0.0, color="#c44e52", ls="--", lw=1.4, label="RBF mass (at 0 only)")
    ax.set_xlabel("frequency $\\omega$"); ax.set_ylabel("spectral density")
    ax.set_title(f"The measure is the geometry. Recovered peak\n at "
                 f"$\\mu \\approx$ {d['recovered_freq']:.2f} (true {d['true_freq']:.1f})",
                 fontsize=10)
    ax.legend(fontsize=8)
    return fig


def make_roughness_figure(seed=SEED):
    """Figure 8.2 companion: the readout sets the roughness. Three radial profiles with their
    RKHS orders, and a fit under the RBF (oversmooths) vs the Laplace readout (tracks the
    kinks) on a rough target."""
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), constrained_layout=True)

    ax = axes[0]
    r = np.linspace(0, 3, 400)
    ax.plot(r, np.exp(-r ** 2), color="#3b6ea5", lw=2.0, label="RBF $e^{-r^2/T^2}$  ($C^\\infty$)")
    ax.plot(r, np.exp(-r), color="#2ca02c", lw=2.0,
            label="Laplace $e^{-r/T}$  ($H^{(d+1)/2}$)")
    step = np.where(r < 1e-9, 1.0, 0.0)
    ax.plot([0, 0, 3], [1, 0, 0], color="#c44e52", lw=2.0,
            label="tree (step, discontinuous)")
    ax.scatter([0], [1], color="#c44e52", zorder=5, s=20)
    ax.set_xlabel("radial distance $r$"); ax.set_ylabel("$k(r)$")
    ax.set_title("The readout sets the roughness order.\nLaplace has a cusp at $r=0$; "
                 "RBF is analytic", fontsize=10)
    ax.legend(fontsize=8.5)

    d = roughness_ladder_demo(seed=seed)
    ax = axes[1]
    ax.scatter(d["X"], d["y"], s=14, c="#444", zorder=4, label="data (rough target)")
    ax.plot(d["Xg"], d["truth_g"], color="#888", lw=1.2, ls=":", label="true path")
    colors = {"RBF (Gaussian, C-inf)": "#3b6ea5", "Laplace (cusp, H^{(d+1)/2})": "#2ca02c"}
    names = {"RBF (Gaussian, C-inf)": "RBF readout", "Laplace (cusp, H^{(d+1)/2})": "Laplace readout"}
    for name, curve in d["curves"].items():
        ax.plot(d["Xg"], curve, color=colors[name], lw=1.8,
                label=f"{names[name]} (RMSE {d['rmse'][name]:.2f})")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Same spectrum, different readout: the Laplace fit\ntracks kinks the RBF "
                 "rounds off", fontsize=10)
    ax.legend(fontsize=8.5)
    return fig


def make_california_density_figure(features=("MedInc", "Latitude"), seed=SEED):
    """The measure is the geometry, on the running data. Fit the SM kernel per California
    feature and read off the learned spectral density: a smooth-trend feature concentrates
    mass near zero frequency, a feature carrying structure at a definite scale puts a peak
    away from zero. Returns the figure and a dict of recovered numbers per feature."""
    fig, axes = plt.subplots(1, len(features), figsize=(6.1 * len(features), 4.4),
                             constrained_layout=True)
    if len(features) == 1:
        axes = [axes]
    out = {}
    for ax, feat in zip(axes, features):
        c = california_spectral_density(feature=feat, seed=seed)
        out[feat] = c
        ax.plot(c["omega"], c["density"], color="#3b6ea5", lw=2.0)
        ax.fill_between(c["omega"], c["density"], color="#3b6ea5", alpha=0.18)
        peak = c["omega"][np.argmax(c["density"])]
        ax.axvline(peak, color="#c44e52", ls="--", lw=1.2, label=f"peak $\\omega$ = {peak:.2f}")
        kind = "smooth trend (low-frequency mass)" if c["lowfreq_mass"] > 0.5 \
            else "structure at a scale (peak away from 0)"
        ax.set_title(f"{feat}: {kind}\nlow-freq mass {c['lowfreq_mass']:.2f}", fontsize=10)
        ax.set_xlabel("frequency $\\omega$"); ax.set_ylabel("learned spectral density")
        ax.legend(fontsize=9)
    fig.suptitle("California: the learned spectral measure per feature is the geometry",
                 fontsize=11)
    return fig, out


# =============================================================================
# CLI
# =============================================================================

def main(argv=None):
    p = argparse.ArgumentParser(description="Chapter 8 — spectral kernels and Bochner")
    p.add_argument("--out-prefix", default=None)
    args = p.parse_args(argv)
    set_style()

    print("=" * 72, "\nSPECTRAL-MIXTURE (BOCHNER) KERNEL — smooth + periodic")
    d = periodic_extrapolation_demo()
    print(f"  true frequency           {d['true_freq']:.2f}")
    print(f"  recovered peak frequency {d['recovered_freq']:.2f}")
    print(f"  RBF bandwidth (selected) {d['ell']:.3f}")
    print(f"  in-hull test RMSE   spectral {d['sm_test_rmse']:.3f}   RBF {d['rbf_test_rmse']:.3f}")
    print(f"  extrapolation RMSE  spectral {d['sm_extrap_rmse']:.3f}   RBF {d['rbf_extrap_rmse']:.3f}")
    print("  -> the spectral kernel carries the oscillation beyond the data; the RBF flattens.")

    print("\nROUGHNESS LADDER — the readout sets the order")
    r = roughness_ladder_demo()
    for name, val in r["rmse"].items():
        print(f"  {name:32s} fit RMSE {val:.3f}")

    print("\nCALIFORNIA — learned per-feature spectral density")
    for feat in ("MedInc", "Latitude"):
        c = california_spectral_density(feature=feat)
        peak = c["omega"][np.argmax(c["density"])]
        print(f"  {feat:9s} peak omega {peak:.2f}, low-freq mass {c['lowfreq_mass']:.2f}, "
              f"query loss spectral {c['sm_qloss']:.3f} / RBF {c['rbf_qloss']:.3f}")

    print("\nDENSITY PARAMETERIZATION — Gauss-Hermite quadrature (continuity is free)")
    for G in (4, 8, 16):
        omega, wts = density_atoms_gauss_hermite(mu_log=0.0, gamma=0.4, G=G)
        print(f"  G={G:2d} atoms: mass sum {wts.sum():.4f}, mean freq {(omega*wts).sum():.3f} "
              f"(2 trainable numbers, independent of G)")

    if args.out_prefix:
        import matplotlib
        matplotlib.use("Agg")
        # fig 8.1 — the readout sets the roughness order; fig 8.2 — a kernel and its
        # spectral measure (periodic recovery, plus the California per-feature densities)
        make_roughness_figure().savefig(f"{args.out_prefix}1_roughness.pdf")
        make_periodic_figure().savefig(f"{args.out_prefix}2_spectral_measure.pdf")
        make_california_density_figure()[0].savefig(f"{args.out_prefix}2_california.pdf")
        print("\nwrote figures with prefix", args.out_prefix)


if __name__ == "__main__":
    main()
