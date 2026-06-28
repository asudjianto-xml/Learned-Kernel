"""Smoke tests for Chapter 8: the finite spectral-Laplace (MS-SKM) kernel."""
import numpy as np

from lkbook.chapters import ch08


def test_spectral_laplace_unit_diagonal_and_psd():
    X = np.linspace(0, 1, 60)
    k = ch08.SpectralLaplaceKernel(omegas=[0.0, 1.0, 3.0], amps=[1, 1, 1], T=1.5)
    K = k.gram(X, X)
    assert np.allclose(np.diag(K), 1.0)                       # unit diagonal
    assert np.linalg.eigvalsh(K).min() >= -1e-8               # PSD


def test_fused_bank_unit_diagonal():
    X = np.linspace(0, 1, 50)
    k = ch08.SpectralLaplaceKernel(omegas=[0.0, 2.0], amps=[1, 1], T=[0.3, 3.0],
                                   w=[0.5, 0.5])              # two fused banks
    assert np.allclose(np.diag(k.gram(X, X)), 1.0)


def test_recovers_frequency_and_extrapolates():
    d = ch08.periodic_extrapolation_demo(freq=3.0)
    assert abs(d["recovered_freq"] - d["true_freq"]) < 0.3    # reads the frequency from data
    assert d["sm_test_rmse"] < 0.15                           # fits in-hull
    assert d["sm_extrap_rmse"] < d["rbf_extrap_rmse"]         # wins on extrapolation


def test_readout_sets_roughness():
    # on a genuinely rough (H^{1/2}) OU path the Laplace readout beats the Gaussian RBF
    r = ch08.roughness_ladder_demo()
    assert r["laplace_rmse"] < r["rbf_rmse"]


def test_california_density_reads_trend_vs_scale():
    dens = ch08.california_spectral_density()
    # income carries more low-frequency (trend) mass than latitude
    assert dens["MedInc"]["low_freq_mass"] > dens["Latitude"]["low_freq_mass"]


def test_canonical_kernel_modes_psd_and_unit_diagonal():
    import numpy as np
    rng = np.random.RandomState(0); X = rng.uniform(-1, 1, size=(40, 4)); y = X[:, 0] + X[:, 1]
    for mode in ("estimate", "learned", "constrained"):
        k, _ = ch08.fit_spectral(X, y, mode=mode, steps=20)   # few steps: this is a structural smoke test
        K = k.gram(X, X)
        assert np.allclose(np.diag(K), 1.0)                   # unit diagonal
        assert np.linalg.eigvalsh(K).min() >= -1e-8           # PSD


def test_constrained_mode_extrapolates_like_estimate():
    # the geometric constraint keeps the periodic atom, so it extrapolates where learned does not
    import numpy as np
    rng = np.random.RandomState(0); n = 120; X = np.sort(rng.rand(n))
    y = 0.8 * (X - 0.5) ** 2 + 0.5 * np.sin(2 * np.pi * 3.0 * X) + 0.05 * rng.randn(n)
    Xg = np.linspace(0, 2, 400); truth = 0.8 * (Xg - 0.5) ** 2 + 0.5 * np.sin(2 * np.pi * 3.0 * Xg)
    ex = Xg > 1.0

    def extrap(mode):
        _, pr = ch08.fit_spectral(X[:, None], y, mode=mode, standardize=False, steps=400)
        return float(np.sqrt(np.mean((pr(Xg[:, None])[ex] - truth[ex]) ** 2)))

    e, lrn, c = extrap("estimate"), extrap("learned"), extrap("constrained")
    assert c < lrn - 0.03            # constrained extrapolates markedly better than learned
    assert c < e + 0.06              # and close to the estimate mode
