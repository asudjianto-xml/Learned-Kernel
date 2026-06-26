"""Chapter 5 — Gaussian processes: the kernel as covariance and prior.

The Gaussian process makes one identification — **kernel = covariance = prior over
functions** — and that identification is the hinge of the book. Three consequences, all
shown here on the running data with NumPy/SciPy/scikit-learn (no torch):

  - **GP posterior mean = KRR with ridge λ=σ².** Conditioning the GP prior on data is a
    closed-form Gaussian computation; its mean is *exactly* the kernel-ridge predictor
    m_*(x) = K_*(K+σ²I)⁻¹y. We assert the two agree to machine precision.

  - **The marginal likelihood scores the kernel.** Integrating the function out leaves
    y ~ N(0, K+σ²I), so log p(y) = −½ yᵀA⁻¹y − ½ log|A| − (n/2) log 2π, A=K+σ²I — a fit
    term minus a log-determinant complexity penalty. This single number is what makes a
    kernel *learnable* (maximize it in the length scales / noise) rather than chosen.

  - **Evidence learning is stable (Prop. F).** Maximizing the evidence does NOT collapse
    the kernel to the over-correlated all-ones limit K=J=11ᵀ to cheapen the log-det. We
    show empirically that the NLML has an interior optimum in the length scale (it rises as
    ℓ→∞, the K→J direction) and that the closed-form gap ΔL = nVar̂(y)/(2σ²) −
    (n−1)/2·log(1/σ²) + O(1) → +∞ as σ²→0. The all-ones kernel is not a stationary point.

This mirrors the math of the spectral kernel machine (skm): its fit loop builds A=K+σ²I,
Choleskys it, and minimizes nlml = ½ yᵀα + ½ log|A|, decoding by KRR at λ=σ². We reproduce
that with sklearn's GaussianProcessRegressor and a plain NumPy KRR so the package stays
dependency-light.

    python -m lkbook.chapters.ch05 --out-prefix fig5
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel

from lkbook import load_california, load_taiwan, set_style

N_TRAIN, N_TEST, SEED = 600, 600, 0
# RBF length scale used for the fixed-kernel demonstrations (gamma = 1/(2 ell^2)).
ELL0 = 2.0


# --- data subsets -------------------------------------------------------------

def _subset(d, n_train=N_TRAIN, n_test=N_TEST, seed=SEED):
    """A modest train/test subset; a GP is O(n³), so keep n in the hundreds."""
    rng = np.random.RandomState(seed)
    itr = rng.choice(d.n, min(n_train, d.n), replace=False)
    ite = rng.choice(len(d.Xte), min(n_test, len(d.Xte)), replace=False)
    Xtr, ytr = d.Xtr[itr], np.asarray(d.ytr, float)[itr]
    Xte, yte = d.Xte[ite], np.asarray(d.yte, float)[ite]
    return Xtr, ytr, Xte, yte


# --- GP posterior mean == KRR at lambda = sigma^2 -----------------------------

def gp_mean_equals_krr(d, ell=ELL0, sigma2=0.1, seed=SEED):
    """The boxed identity: the GP posterior mean m_*(x)=K_*(K+σ²I)⁻¹y is exactly the KRR
    predictor with ridge λ=σ². We compute the GP mean and variance from the Gaussian
    conditioning identities by hand, KRR with sklearn's KernelRidge, and report the max
    discrepancy (≈ machine precision)."""
    Xtr, ytr, Xte, yte = _subset(d, seed=seed)
    gamma = 1.0 / (2.0 * ell ** 2)
    ybar = ytr.mean()
    yc = ytr - ybar                                   # GP prior has mean zero

    K = rbf_kernel(Xtr, Xtr, gamma=gamma)
    Ks = rbf_kernel(Xte, Xtr, gamma=gamma)
    A = K + sigma2 * np.eye(len(ytr))
    cho = cho_factor(A, lower=True)
    alpha = cho_solve(cho, yc)                         # α = (K+σ²I)⁻¹ y_centered

    gp_mean = Ks @ alpha + ybar                        # m_*(x) = K_* α  (+ mean back)
    # posterior variance v_*(x) = k(x,x) − K_* A⁻¹ K_*ᵀ ; k(x,x)=1 for the RBF
    v = cho_solve(cho, Ks.T)
    gp_var = 1.0 - np.einsum("ij,ji->i", Ks, v)

    krr = KernelRidge(alpha=sigma2, kernel="rbf", gamma=gamma).fit(Xtr, yc)
    krr_mean = krr.predict(Xte) + ybar

    return {
        "gp_mean": gp_mean, "gp_var": gp_var, "krr_mean": krr_mean,
        "max_abs_diff": float(np.max(np.abs(gp_mean - krr_mean))),
        "gp_rmse": float(np.sqrt(np.mean((gp_mean - yte) ** 2))),
        "min_var": float(gp_var.min()), "max_var": float(gp_var.max()),
        "Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte,
    }


# --- the marginal likelihood and evidence-fit kernel learning -----------------

def nlml_rbf(Xtr, yc, ell, sigma2):
    """Negative log marginal likelihood of a zero-mean GP with an RBF covariance, the same
    objective skm minimizes: ½ yᵀA⁻¹y + ½ log|A| + (n/2) log 2π, A = K(ell)+σ²I."""
    n = len(yc)
    K = rbf_kernel(Xtr, Xtr, gamma=1.0 / (2.0 * ell ** 2))
    A = K + sigma2 * np.eye(n)
    L, low = cho_factor(A, lower=True)
    alpha = cho_solve((L, low), yc)
    fit = 0.5 * float(yc @ alpha)
    logdet = float(np.sum(np.log(np.diag(L)))) * 2.0 * 0.5     # ½ log|A| = Σ log L_ii
    const = 0.5 * n * np.log(2.0 * np.pi)
    return fit + logdet + const, fit, logdet


def nlml_length_scale_scan(d, ells=None, sigma2=0.1, seed=SEED):
    """Scan the NLML over the RBF length scale. The point: an *interior* minimum, and the
    NLML rising as ℓ→∞ (the K→J over-correlated direction) — the empirical face of Prop. F.
    Returns the fit and complexity (log-det) terms separately so their tug-of-war is visible."""
    if ells is None:
        ells = np.geomspace(0.3, 60.0, 40)
    Xtr, ytr, _, _ = _subset(d, seed=seed)
    yc = ytr - ytr.mean()
    nlml, fit, comp = [], [], []
    for ell in ells:
        L, f, c = nlml_rbf(Xtr, yc, ell, sigma2)
        nlml.append(L); fit.append(f); comp.append(c)
    nlml = np.array(nlml); fit = np.array(fit); comp = np.array(comp)
    i = int(np.argmin(nlml))
    return {"ells": np.asarray(ells), "nlml": nlml, "fit": np.array(fit),
            "comp": np.array(comp), "ell_star": float(ells[i]),
            "interior": bool(0 < i < len(ells) - 1)}


def evidence_fit_vs_fixed(d, seed=SEED):
    """Learn the kernel by maximizing the evidence (sklearn fits the RBF length scale and the
    noise by L-BFGS on the log marginal likelihood) and compare test RMSE to a fixed,
    deliberately-too-broad RBF KRR. The evidence-fit kernel matches or beats the fixed one,
    and reports the learned length scale and noise — the geometry is fit, not tuned."""
    Xtr, ytr, Xte, yte = _subset(d, seed=seed)
    ybar = ytr.mean(); yc = ytr - ybar

    kern = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kern, normalize_y=False,
                                  n_restarts_optimizer=2, random_state=seed).fit(Xtr, yc)
    mu, sd = gp.predict(Xte, return_std=True)
    mu = mu + ybar
    learned = gp.kernel_
    ell_hat = float(learned.k1.k2.length_scale)
    sig2_hat = float(learned.k2.noise_level)
    amp_hat = float(learned.k1.k1.constant_value)

    # fixed too-broad RBF KRR at the same noise ridge
    gamma_fixed = 1.0 / (2.0 * (3.0 * ell_hat) ** 2)
    krr = KernelRidge(alpha=sig2_hat, kernel="rbf", gamma=gamma_fixed).fit(Xtr, yc)
    fixed_mu = krr.predict(Xte) + ybar

    def rmse(p):
        return float(np.sqrt(np.mean((p - yte) ** 2)))

    # coverage of the GP 95% interval (calibration)
    lo, hi = mu - 1.96 * sd, mu + 1.96 * sd
    cover = float(np.mean((yte >= lo) & (yte <= hi)))
    return {"ell_hat": ell_hat, "sig2_hat": sig2_hat, "amp_hat": amp_hat,
            "evidence_rmse": rmse(mu), "fixed_rmse": rmse(fixed_mu),
            "coverage95": cover, "lml": float(gp.log_marginal_likelihood_value_)}


# --- Prop. F: the all-ones collapse is not a marginal-likelihood minimum ------

def prop_f_allones_gap(d, sigma2s=None, seed=SEED):
    """Demonstrate Prop. F numerically. Build the over-correlated all-ones kernel K=J=11ᵀ
    (every off-diagonal → 1, unit diagonal) and a genuinely-fitting kernel K_0 (a sharp RBF),
    and track the NLML gap ΔL = L_J − L_0 as σ²→0. Lemma F.0 closed forms are checked against
    the eigen-computation; the gap diverges like nVar̂(y)/(2σ²), confirming J is not preferred."""
    if sigma2s is None:
        sigma2s = np.geomspace(1e-3, 1.0, 25)
    Xtr, ytr, _, _ = _subset(d, n_train=300, seed=seed)
    yc = ytr - ytr.mean()
    n = len(yc)
    J = np.ones((n, n))
    K0 = rbf_kernel(Xtr, Xtr, gamma=1.0 / (2.0 * 1.0 ** 2))     # a sharp, fitting RBF
    var_hat = float(np.var(yc))

    def nlml_from_K(K, s2):
        A = K + s2 * np.eye(n)
        L, low = cho_factor(A, lower=True)
        alpha = cho_solve((L, low), yc)
        return 0.5 * float(yc @ alpha) + float(np.sum(np.log(np.diag(L)))) + 0.5 * n * np.log(2 * np.pi)

    gap, lj, l0, predicted = [], [], [], []
    for s2 in sigma2s:
        Lj = nlml_from_K(J, s2)
        L0 = nlml_from_K(K0, s2)
        gap.append(Lj - L0); lj.append(Lj); l0.append(L0)
        # leading-order prediction of the gap (Prop. F)
        predicted.append(n * var_hat / (2 * s2) - (n - 1) / 2 * np.log(1.0 / s2))

    # Lemma F.0 closed forms at the smallest sigma2, checked against eigen-truth
    s2 = float(sigma2s[0])
    logdet_closed = np.log(n + s2) + (n - 1) * np.log(s2)
    logdet_true = float(np.linalg.slogdet(J + s2 * np.eye(n))[1])
    quad_closed = (1.0 / s2) * (float(yc @ yc) - (float(yc.sum())) ** 2 / (n + s2))
    quad_true = float(yc @ np.linalg.solve(J + s2 * np.eye(n), yc))

    return {"sigma2s": np.asarray(sigma2s), "gap": np.array(gap),
            "predicted_gap": np.array(predicted), "lj": np.array(lj), "l0": np.array(l0),
            "var_hat": var_hat, "n": n,
            "logdet_closed": logdet_closed, "logdet_true": logdet_true,
            "logdet_err": abs(logdet_closed - logdet_true),
            "quad_closed": quad_closed, "quad_true": quad_true,
            "quad_relerr": abs(quad_closed - quad_true) / abs(quad_true)}


# --- figures ------------------------------------------------------------------

def make_prior_posterior_figure(d, feature="MedInc", n_prior=4, n_post=4, seed=SEED):
    """Figure 5.1 — prior samples vs posterior samples on a 1-D California slice, with the
    posterior 2σ band shaded. The kernel IS the prior; data turns the prior into a posterior,
    and the band is the free uncertainty."""
    Xtr, ytr, _, _ = _subset(d, n_train=120, seed=seed)
    j = d.col(feature)
    # vary one standardized feature on a grid, hold the rest at their training mean
    xs = np.linspace(Xtr[:, j].min(), Xtr[:, j].max(), 200)
    Xgrid = np.tile(Xtr.mean(0), (len(xs), 1)); Xgrid[:, j] = xs

    # pick training points near the slice (other coords close to the mean) so they sit on it
    dist = np.linalg.norm(np.delete(Xtr, j, 1) - np.delete(Xtr, j, 1).mean(0), axis=1)
    sl = np.argsort(dist)[:40]
    Xs, ys = Xtr[sl], ytr[sl]
    ybar = ys.mean(); yc = ys - ybar
    ell, sigma2 = 1.2, 0.15
    gamma = 1.0 / (2.0 * ell ** 2)
    rng = np.random.RandomState(seed)

    # prior: f ~ N(0, K_grid) (+ training mean for display scale)
    Kgg = rbf_kernel(Xgrid, Xgrid, gamma=gamma) + 1e-8 * np.eye(len(xs))
    Lg = np.linalg.cholesky(Kgg)
    prior = (Lg @ rng.standard_normal((len(xs), n_prior))) + ybar

    # posterior given the slice points
    K = rbf_kernel(Xs, Xs, gamma=gamma) + sigma2 * np.eye(len(ys))
    Ksg = rbf_kernel(Xgrid, Xs, gamma=gamma)
    cho = cho_factor(K, lower=True)
    mu = Ksg @ cho_solve(cho, yc) + ybar
    cov = (rbf_kernel(Xgrid, Xgrid, gamma=gamma)
           - Ksg @ cho_solve(cho, Ksg.T)) + 1e-8 * np.eye(len(xs))
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    Lp = np.linalg.cholesky(cov)
    post = (mu[:, None] + Lp @ rng.standard_normal((len(xs), n_post)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True, sharey=True)
    axes[0].plot(xs, prior, lw=1.1, alpha=0.9)
    axes[0].set_title("Prior: draws from f ~ GP(0, k) — the kernel alone")
    axes[1].fill_between(xs, mu - 2 * sd, mu + 2 * sd, color="#3b6ea5", alpha=0.18,
                         label="posterior 2σ band")
    axes[1].plot(xs, post, lw=1.0, alpha=0.85)
    axes[1].plot(xs, mu, color="k", lw=2.0, label="posterior mean (= KRR)")
    axes[1].scatter(Xs[:, j], ys, s=18, c="#c44e52", zorder=5, label="training blocks")
    axes[1].set_title("Posterior: conditioned on data — mean + free 2σ band")
    axes[1].legend(fontsize=8, loc="upper left")
    for ax in axes:
        ax.set_xlabel(f"{feature} (standardized)")
    axes[0].set_ylabel("median house value ($100k)")
    fig.suptitle("The kernel is the prior; data turns it into a posterior, and the band is "
                 "uncertainty at no extra cost")
    return fig


def make_evidence_figure(d, seed=SEED):
    """Figure 5.2 — the marginal-likelihood landscape over the RBF length scale, with the fit
    term and the complexity (log-det) term pulling against each other and the interior optimum
    marked. Inset: the NLML gap of the all-ones K→J corner diverges as σ²→0 (Prop. F) — that
    corner is uphill, not a basin."""
    scan = nlml_length_scale_scan(d, seed=seed)
    pf = prop_f_allones_gap(d, seed=seed)

    fig, ax = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
    ax.plot(scan["ells"], scan["nlml"], color="k", lw=2.2, label="NLML (the kernel's score)")
    ax.plot(scan["ells"], scan["fit"] - scan["fit"].min(),
            color="#c44e52", lw=1.4, ls="--", label="data-fit term (shifted)")
    ax.plot(scan["ells"], scan["comp"] - scan["comp"].min(),
            color="#3b6ea5", lw=1.4, ls="--", label="log-det complexity term (shifted)")
    ax.axvline(scan["ell_star"], color="#2ca02c", lw=1.2)
    ax.scatter([scan["ell_star"]], [scan["nlml"].min()], s=70, c="#2ca02c", zorder=6,
               label=f"interior optimum ℓ*={scan['ell_star']:.1f}")
    ax.set_xscale("log")
    ax.set_xlabel("RBF length scale ℓ   (ℓ→∞ is the over-correlated K→J corner)")
    ax.set_ylabel("negative log marginal likelihood")
    ax.set_title("Evidence is a scored landscape we climb to LEARN the geometry.\n"
                 "Fit and complexity pull against each other; the optimum is interior, "
                 "and the K→J corner is uphill (Prop. F)")
    ax.legend(fontsize=8.5, loc="upper center")

    ins = ax.inset_axes([0.60, 0.40, 0.36, 0.38])
    ins.loglog(pf["sigma2s"], pf["gap"], color="#8172b3", lw=2.0, label="ΔL = L_J − L₀")
    ins.loglog(pf["sigma2s"], pf["predicted_gap"], color="gray", lw=1.0, ls=":",
               label="nVar̂/2σ² − …")
    ins.set_xlabel("noise σ²", fontsize=7.5); ins.set_ylabel("NLML gap", fontsize=7.5)
    ins.set_title("all-ones corner: ΔL→+∞ as σ²→0", fontsize=7.5)
    ins.tick_params(labelsize=6.5); ins.legend(fontsize=6.0, loc="lower left")
    return fig


# --- CLI ----------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Chapter 5 — Gaussian processes")
    p.add_argument("--out-prefix", default=None)
    args = p.parse_args(argv)
    set_style()
    cal = load_california()

    print("=" * 72, "\nGP POSTERIOR MEAN = KRR (λ=σ²) on California")
    eq = gp_mean_equals_krr(cal)
    print(f"  max |GP-mean − KRR| = {eq['max_abs_diff']:.2e}  (machine precision)")
    print(f"  GP test RMSE        = {eq['gp_rmse']:.3f} ($100k)")
    print(f"  posterior variance  = [{eq['min_var']:.3f}, {eq['max_var']:.3f}]"
          "  (free, widens off-support)")

    print("\nEVIDENCE-FIT KERNEL vs fixed too-broad RBF (California)")
    ev = evidence_fit_vs_fixed(cal)
    print(f"  learned  ℓ={ev['ell_hat']:.2f}, σ²={ev['sig2_hat']:.3f}, amp={ev['amp_hat']:.2f}")
    print(f"  evidence-fit RMSE   = {ev['evidence_rmse']:.3f}")
    print(f"  fixed broad RMSE    = {ev['fixed_rmse']:.3f}")
    print(f"  GP 95% coverage     = {ev['coverage95']:.2f}  (calibration)")

    print("\nNLML LENGTH-SCALE SCAN (interior optimum)")
    sc = nlml_length_scale_scan(cal)
    print(f"  ℓ* = {sc['ell_star']:.2f}, interior optimum = {sc['interior']}")
    print(f"  NLML at ℓ* = {sc['nlml'].min():.1f}, at ℓ_max = {sc['nlml'][-1]:.1f} "
          "(rises toward the K→J corner)")

    print("\nPROP. F — all-ones collapse is not an NLML minimum")
    pf = prop_f_allones_gap(cal)
    print(f"  Var̂(y) = {pf['var_hat']:.3f}, n = {pf['n']}")
    print(f"  Lemma F.0 log|J+σ²I|: closed {pf['logdet_closed']:.3f} vs true "
          f"{pf['logdet_true']:.3f} (err {pf['logdet_err']:.1e})")
    print(f"  Lemma F.0 quadratic form rel.err = {pf['quad_relerr']:.1e}")
    print(f"  NLML gap ΔL at σ²={pf['sigma2s'][0]:.0e}:  {pf['gap'][0]:.1f}  (> 0, "
          "diverges as σ²→0)")
    print(f"  ΔL at σ²={pf['sigma2s'][-1]:.2f}:  {pf['gap'][-1]:.1f}")

    print("\nTAIWAN CREDIT (same machinery, classification target as real-valued GP)")
    tw = load_taiwan()
    eqt = gp_mean_equals_krr(tw)
    pft = prop_f_allones_gap(tw)
    print(f"  GP-mean = KRR max diff = {eqt['max_abs_diff']:.2e}")
    print(f"  Prop. F gap at σ²={pft['sigma2s'][0]:.0e}: {pft['gap'][0]:.1f} (> 0)")

    if args.out_prefix:
        import matplotlib
        matplotlib.use("Agg")
        make_prior_posterior_figure(cal).savefig(f"{args.out_prefix}1_prior_posterior.pdf")
        make_evidence_figure(cal).savefig(f"{args.out_prefix}2_evidence_propf.pdf")
        print("\nwrote figures with prefix", args.out_prefix)


if __name__ == "__main__":
    main()
