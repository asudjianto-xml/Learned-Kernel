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
# The canonical learned spectral-Laplace kernel and its three training modes
#
# This is the model the rest of the book builds on (Chapter 9 imports it directly). It is the
# finite "free" mirror of torch `skm.SpectralMixture`: per-feature/per-bank cosine/sine atoms
# at frequencies omega_{h,j,k} with amplitudes a_{h,j,k}, a per-feature ARD relevance s_j, and
# H convexly fused Laplace banks (weights w_h, bandwidths T_h),
#     K(x,x') = sum_h w_h exp(-||phi_h(x)-phi_h(x')|| / T_h),
#     phi_h(x)_j = sqrt(s_j) a_{h,j,:} [cos(2 pi omega_{h,j,:} x_j), sin(2 pi omega_{h,j,:} x_j)].
# The ARD relevance scales each feature's BLOCK (so s_j -> 0 retires a nuisance feature),
# decoupled from the frequency. The measure is split into ANCHOR atoms (frozen frequency,
# amplitude floored so they cannot be optimized away) and FREE atoms (frequency + amplitude
# learned). The three training modes are three settings of that split, with a real tradeoff:
#
#   estimate    : ANCHOR only. The periodogram ESTIMATES the support; NLML learns only the
#                 readout geometry (ARD s_j, bandwidth T_h, weights w_h, noise). Interpretable,
#                 reproducible, EXTRAPOLATES a periodic signal -- but underfits real data.
#   learned     : FREE only. Every spectral parameter is LEARNED by gradient descent on the
#                 marginal likelihood (skm's NLML), periodogram-seeded. Wins on real data and
#                 interactions, but in-hull fitting drops the periodic atom, so it does NOT
#                 extrapolate; it is higher-variance run to run.
#   constrained : ANCHOR + FREE. The geometric constraint that the learned measure must RETAIN
#                 the periodogram atoms with non-vanishing mass keeps the extrapolating
#                 structure (anchor) while the free atoms + ARD + bandwidths deliver accuracy.
#                 It gets BOTH -- extrapolation like estimate and accuracy near learned -- at a
#                 small accuracy cost and the learned mode's variance.
# =============================================================================

_DTYPE = None


def _torch():
    global _DTYPE
    import torch
    if _DTYPE is None:
        _DTYPE = torch.float64
    return torch


def _device():
    torch = _torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _estimate_support(Xn, y, K, fmax=6.0, peak_thresh=0.08):
    """Parsimonious periodogram support for the estimate mode: two trend atoms plus, where a
    feature carries a significant periodic component, its dominant frequency and first harmonic.
    Uniform unit mass on the support atoms, zero on the padding. Frequencies are fixed."""
    d = Xn.shape[1]; grid = np.linspace(0.3, fmax, 80)
    om = np.zeros((d, K)); amp = np.zeros((d, K))
    for j in range(d):
        p = periodogram(Xn[:, j], y, grid); i = int(np.argmax(p))
        f = float(grid[i]) if p[i] > peak_thresh else None
        atoms = ([0.1, 0.5] + ([f, 2 * f] if f else [1.0, 2.0]) + [0.05] * K)[:K]
        om[j] = atoms
        amp[j, :min(4, K)] = 1.0
    return om, amp


def _seed_support(Xn, y, K, fmax=6.0, n_low=3, peak_thresh=0.08):
    """Periodogram SEED for the learned mode: low trend atoms, the significant peaks, and a
    low-band log-uniform fill; uniform mass. Every atom is refined by gradient afterward."""
    d = Xn.shape[1]; low = list(np.linspace(0.1, 0.6, n_low)); grid = np.linspace(0.3, fmax, 80)
    om = np.zeros((d, K))
    for j in range(d):
        p = periodogram(Xn[:, j], y, grid); order = np.argsort(p)[::-1]
        peaks = [float(grid[i]) for i in order if p[i] > peak_thresh][:2]
        om[j] = np.array((low + peaks + list(np.geomspace(0.3, 2.5, K)))[:K])
    return om, np.ones((d, K))


class LearnedSpectralLaplace:
    """The canonical learned spectral-Laplace kernel (general-d). The spectral measure is split
    into ANCHOR atoms (`anchor_omega`: frozen frequency, amplitude floored at `anchor_floor` so
    they cannot be optimized away) and FREE atoms (`free_omega`: frequency and amplitude learned
    by gradient). The three training modes (see the section header) are three settings of this
    split — anchor-only (estimate), free-only (learned), both (constrained). ARD relevance s_j
    scales each feature's block; H banks fuse convexly. PSD and unit-diagonal.
    `interaction="additive"` is Chapter 9's order-one control (sum of per-feature Laplace)."""

    def __init__(self, d, H, anchor_omega=None, free_omega=None, anchor_floor=1.0,
                 interaction="full", seed=SEED, device=None):
        torch = _torch()
        self.d, self.H, self.interaction, self.anchor_floor = d, H, interaction, anchor_floor
        dev = device if device is not None else _device(); self._dev = dev
        g = torch.Generator(device="cpu").manual_seed(seed)
        # anchor atoms: frozen frequency, amplitude = floor + softplus(0) (fixed, never optimized away)
        if anchor_omega is not None:
            oma = torch.as_tensor(anchor_omega, dtype=_DTYPE)[None].repeat(H, 1, 1)
            self.Ka = oma.shape[-1]
            self.log_oma = torch.log(oma.clamp_min(1e-3)).to(dev, _DTYPE)              # frozen (no grad)
            self.log_ampa = torch.zeros(H, d, self.Ka, device=dev, dtype=_DTYPE)      # frozen
        else:
            self.Ka, self.log_oma, self.log_ampa = 0, None, None
        # free atoms: learned frequency + amplitude
        if free_omega is not None:
            omf = torch.as_tensor(free_omega, dtype=_DTYPE)[None].repeat(H, 1, 1)
            self.Kf = omf.shape[-1]
            omf = omf * torch.exp(0.05 * torch.randn(H, d, self.Kf, generator=g))     # tiny per-bank jitter
            self.log_omf = torch.log(omf.clamp_min(1e-3)).to(dev, _DTYPE).requires_grad_(True)
            amp1 = float(np.log(np.expm1(1.0)))                                       # init free mass at 1.0
            self.log_ampf = torch.full((H, d, self.Kf), amp1, device=dev, dtype=_DTYPE).requires_grad_(True)
        else:
            self.Kf, self.log_omf, self.log_ampf = 0, None, None
        self.log_s = torch.zeros(d, device=dev, dtype=_DTYPE).requires_grad_(True)    # ARD relevance
        self.log_T = torch.zeros(H, device=dev, dtype=_DTYPE).requires_grad_(True)
        self.w_logit = torch.zeros(H, device=dev, dtype=_DTYPE).requires_grad_(True)
        self.log_sig2 = torch.tensor(np.log(0.1), device=dev, dtype=_DTYPE).requires_grad_(True)

    def free_params(self):
        return [self.log_omf, self.log_ampf] if self.Kf else []

    def geometry_params(self):
        return [self.log_s, self.log_T, self.w_logit, self.log_sig2]

    def _om_amp(self):
        torch = _torch(); F = torch.nn.functional
        oms, amps = [], []
        if self.Ka:
            oms.append(torch.exp(self.log_oma)); amps.append(self.anchor_floor + F.softplus(self.log_ampa))
        if self.Kf:
            oms.append(torch.exp(self.log_omf)); amps.append(F.softplus(self.log_ampf))
        return torch.cat(oms, -1), torch.cat(amps, -1)               # (H,d,Ka+Kf)

    def _embed(self, X):
        torch = _torch()
        s = torch.nn.functional.softplus(self.log_s)
        om, amp = self._om_amp(); K = om.shape[-1]
        arg = 2.0 * np.pi * X.view(1, X.shape[0], self.d, 1) * om.view(self.H, 1, self.d, K)
        a = amp.view(self.H, 1, self.d, K)
        block = torch.cat([a * torch.cos(arg), a * torch.sin(arg)], dim=-1)   # (H,n,d,2K)
        return block * torch.sqrt(s).view(1, 1, self.d, 1)                    # ARD weights the block

    def gram_torch(self, A, B):
        torch = _torch(); F = torch.nn.functional
        T = F.softplus(self.log_T).clamp_min(1e-4); w = torch.softmax(self.w_logit, 0)
        Ea, Eb = self._embed(A), self._embed(B); H, na, d, k2 = Ea.shape
        out = A.new_zeros(na, B.shape[0])
        if self.interaction == "additive":
            for h in range(H):
                acc = A.new_zeros(na, B.shape[0])
                for j in range(d):
                    acc = acc + torch.exp(-torch.cdist(Ea[h, :, j, :], Eb[h, :, j, :]) / T[h])
                out = out + w[h] * acc / d
        else:
            Pa = Ea.reshape(H, na, d * k2); Pb = Eb.reshape(H, B.shape[0], d * k2)
            for h in range(H):
                out = out + w[h] * torch.exp(-torch.cdist(Pa[h], Pb[h]) / T[h])
        return out

    def gram(self, A, B):
        torch = _torch()
        with torch.no_grad():
            At = torch.as_tensor(np.atleast_2d(np.asarray(A, float)), device=self._dev, dtype=_DTYPE)
            Bt = torch.as_tensor(np.atleast_2d(np.asarray(B, float)), device=self._dev, dtype=_DTYPE)
            return self.gram_torch(At, Bt).cpu().numpy()

    def learned_density(self, j):
        """Read off the spectral measure for feature j: (frequencies, mass = a^2) over all atoms
        (anchor + free), flattened across banks. The inspectable geometry of the standing frame."""
        torch = _torch()
        with torch.no_grad():
            om, amp = self._om_amp()
            return om[:, j, :].cpu().numpy().ravel(), (amp[:, j, :] ** 2).cpu().numpy().ravel()


@_single_thread
def fit_spectral(Xtr, ytr, mode="learned", objective="nlml", H=2, K=8, steps=500, lr=0.05,
                 seed=SEED, interaction="full", n_fit=1200, standardize=True):
    """Fit the canonical learned spectral-Laplace kernel in one of the three training modes
    (`mode="estimate"`, `"learned"` or `"constrained"`; see the section header). `objective`
    is `"nlml"` (the Chapter-5 marginal likelihood, skm's default) or `"query"` (the Chapter-7
    leakage-free query-fold risk). Returns (kernel, predict)."""
    torch = _torch()
    Xtr = np.atleast_2d(np.asarray(Xtr, float)); ytr = np.asarray(ytr, float).ravel()
    if len(ytr) > n_fit:
        sub = np.random.RandomState(seed).choice(len(ytr), n_fit, replace=False)
        Xtr, ytr = Xtr[sub], ytr[sub]
    mu, sd = ((Xtr.mean(0), Xtr.std(0) + 1e-12) if standardize
              else (np.zeros(Xtr.shape[1]), np.ones(Xtr.shape[1])))
    Xn = (Xtr - mu) / sd
    d = Xtr.shape[1]; dev = _device()
    Xt = torch.as_tensor(Xn, device=dev, dtype=_DTYPE); yt = torch.as_tensor(ytr, device=dev, dtype=_DTYPE)
    ybar = yt.mean(); yc = yt - ybar; n = len(ytr)

    if mode == "estimate":                                   # anchor only (parsimonious, single bank)
        om0, _ = _estimate_support(Xn, ytr, 4)
        ker = LearnedSpectralLaplace(d, 1, anchor_omega=om0, free_omega=None,
                                     interaction=interaction, seed=seed, device=dev)
    elif mode == "constrained":                              # anchor (frozen) + free (learned)
        oma, _ = _estimate_support(Xn, ytr, 4); omf, _ = _seed_support(Xn, ytr, max(K - 4, 4))
        ker = LearnedSpectralLaplace(d, H, anchor_omega=oma, free_omega=omf,
                                     interaction=interaction, seed=seed, device=dev)
    else:                                                    # learned: free only
        om0, _ = _seed_support(Xn, ytr, K)
        ker = LearnedSpectralLaplace(d, H, anchor_omega=None, free_omega=om0,
                                     interaction=interaction, seed=seed, device=dev)

    groups = [{"params": ker.geometry_params(), "lr": lr}]
    if ker.Kf:                                               # free frequencies refine at a smaller LR
        groups += [{"params": [ker.log_omf], "lr": lr * 0.3}, {"params": [ker.log_ampf], "lr": lr}]
    opt = torch.optim.Adam(groups)

    I = torch.eye(n, device=dev, dtype=_DTYPE)
    if objective == "query":
        rng = np.random.RandomState(seed); perm = rng.permutation(n); ns = n // 2
        si = torch.as_tensor(perm[:ns], device=dev); qi = torch.as_tensor(perm[ns:], device=dev)
        Is = 1e-3 * torch.eye(ns, device=dev, dtype=_DTYPE)
    for _ in range(steps):
        opt.zero_grad()
        if objective == "query":
            Kss = ker.gram_torch(Xt[si], Xt[si]); al = torch.linalg.solve(Kss + Is, yc[si])
            loss = torch.mean((ker.gram_torch(Xt[qi], Xt[si]) @ al - yc[qi]) ** 2)
        else:
            Kf = ker.gram_torch(Xt, Xt); sig2 = torch.nn.functional.softplus(ker.log_sig2).clamp_min(1e-5)
            L = torch.linalg.cholesky(Kf + sig2 * I); al = torch.cholesky_solve(yc.unsqueeze(1), L)
            loss = (0.5 * (yc.unsqueeze(1) * al).sum() + torch.log(torch.diagonal(L)).sum()) / n
        loss.backward(); opt.step()

    with torch.no_grad():
        Kf = ker.gram_torch(Xt, Xt)
        sig2 = (float(torch.nn.functional.softplus(ker.log_sig2).clamp_min(1e-5))
                if objective == "nlml" else 1e-3)
        alpha = torch.linalg.solve(Kf + sig2 * I, yc)

    def predict(Xnew):
        Xnew = (np.atleast_2d(np.asarray(Xnew, float)) - mu) / sd
        with torch.no_grad():
            g = ker.gram_torch(torch.as_tensor(Xnew, device=dev, dtype=_DTYPE), Xt)
            return (g @ alpha + ybar).cpu().numpy()

    return ker, predict


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
# The two-mode comparison: variability and reproducibility across seeds
# =============================================================================

def _interaction_target(n=1200, d=8, seed=SEED):
    """A degree-four product x0 x1 x2 x3 (pure high-order interaction, zero main effects) — the
    case the estimate mode cannot reach and the learned mode can (Chapter 9's S9)."""
    X = np.random.RandomState(seed).uniform(-1, 1, (n, d))
    y = 4.0 * X[:, 0] * X[:, 1] * X[:, 2] * X[:, 3] + 0.05 * np.random.RandomState(seed + 1).randn(n)
    return X, y


def _rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


_MODES = ("estimate", "learned", "constrained")


def compare_training_modes(seeds=range(8)):
    """Run all three training modes over several seeds on three regimes that separate them, and
    report mean +/- std test RMSE — the reproducibility readout. The estimate mode (periodogram
    measure, frozen frequencies) is tight and extrapolates the periodic signal but cannot reach
    the interaction; the learned mode (gradient NLML) wins on real data and the interaction at
    higher variance; the constrained mode (anchor + free) gets both — extrapolation like estimate
    and accuracy near learned. Returns a dict keyed by regime."""
    seeds = list(seeds)
    cal = load_california()
    res = {"periodic": {m: [] for m in _MODES},
           "california": {m: [] for m in _MODES + ("tree",)},
           "interaction": {m: [] for m in _MODES}}
    for s in seeds:
        # periodic extrapolation (the geometry that knows a frequency vs one that does not)
        rng = np.random.RandomState(s); n = 120; Xp = np.sort(rng.rand(n))
        yp = 0.8 * (Xp - 0.5) ** 2 + 0.5 * np.sin(2 * np.pi * 3.0 * Xp) + 0.05 * rng.randn(n)
        Xg = np.linspace(0, 2, 400); truth = 0.8 * (Xg - 0.5) ** 2 + 0.5 * np.sin(2 * np.pi * 3.0 * Xg)
        ex = Xg > 1.0
        for mode in _MODES:
            _, pr = fit_spectral(Xp[:, None], yp, mode=mode, standardize=False, steps=500, seed=s)
            res["periodic"][mode].append(_rmse(pr(Xg[:, None])[ex], truth[ex]))

        # California (real multivariate data) vs the gradient-boosted-forest leaf kernel
        rng = np.random.RandomState(s); idx = rng.choice(cal.n, 2500, replace=False)
        Xtr, ytr = cal.Xtr[idx], cal.ytr[idx]
        muy, sdy = ytr.mean(), ytr.std() + 1e-12
        ytr_s, yte_s = (ytr - muy) / sdy, (cal.yte - muy) / sdy
        for mode in _MODES:
            _, pr = fit_spectral(Xtr, ytr_s, mode=mode, steps=400, seed=s)
            res["california"][mode].append(_rmse(pr(cal.Xte), yte_s))
        from sklearn.ensemble import GradientBoostingRegressor
        from lkbook.chapters import ch04
        m = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.1,
                                      random_state=s).fit(Xtr, ytr_s)
        lk = ch04.LeafKernel().fit(m); Kt = lk.gram(Xtr, Xtr)
        al = np.linalg.solve(Kt + 1e-3 * np.eye(len(Xtr)), ytr_s - ytr_s.mean())
        res["california"]["tree"].append(_rmse(lk.gram(cal.Xte, Xtr) @ al + ytr_s.mean(), yte_s))

        # degree-four interaction
        X, y = _interaction_target(seed=s)
        perm = np.random.RandomState(s + 5).permutation(len(y)); nte = len(y) // 4
        te, tr = perm[:nte], perm[nte:]; mm, ss = y[tr].mean(), y[tr].std() + 1e-12
        yi_tr, yi_te = (y[tr] - mm) / ss, (y[te] - mm) / ss
        for mode in _MODES:
            _, pr = fit_spectral(X[tr], yi_tr, mode=mode, steps=400, seed=s)
            res["interaction"][mode].append(_rmse(pr(X[te]), yi_te))
    return res


def _meanstd(v):
    v = np.asarray(v, float); return float(v.mean()), float(v.std())


# =============================================================================
# Figures
# =============================================================================

def make_modes_comparison_figure(res=None, seeds=range(8)):
    """(8.3) The three training modes across seeds: mean +/- std test RMSE on the periodic
    extrapolation, California, and the degree-four interaction. Estimate is tight and
    extrapolates but cannot reach the interaction; learned wins real data and the interaction at
    higher variance; constrained gets both — extrapolation like estimate and accuracy near
    learned."""
    if res is None:
        res = compare_training_modes(seeds=seeds)
    panels = [("periodic", "Periodic\nextrapolation", ("estimate", "learned", "constrained")),
              ("california", "California\n(real data)", ("estimate", "learned", "constrained", "tree")),
              ("interaction", "Degree-4\ninteraction", ("estimate", "learned", "constrained"))]
    color = {"estimate": "#3b6ea5", "learned": "#c44e52", "constrained": "#55a868", "tree": "#555555"}
    name = {"estimate": "estimate (periodogram)", "learned": "learned (gradient NLML)",
            "constrained": "constrained (anchor+free)", "tree": "tree (leaf kernel)"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    for ax, (key, title, modes) in zip(axes, panels):
        for i, mode in enumerate(modes):
            mu, sdv = _meanstd(res[key][mode])
            ax.bar(i, mu, 0.7, yerr=sdv, capsize=4, color=color[mode], label=name[mode])
            ax.text(i, mu + sdv + 0.01, f"{mu:.3f}\n±{sdv:.3f}", ha="center", fontsize=7.5)
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels([m[:5] for m in modes], fontsize=8)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("test RMSE (mean ± std over seeds)")
    axes[1].legend(fontsize=7, loc="upper right")
    fig.suptitle("Three ways to fit the spectral measure, across seeds: estimate extrapolates but "
                 "underfits;\nlearned wins real data at higher variance; constrained gets both",
                 fontsize=10.5)
    return fig


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
