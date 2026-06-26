"""Smoke tests for Chapter 8: the spectral-mixture (Bochner) kernel and the roughness ladder."""
import numpy as np

from lkbook.chapters import ch08


def test_sm_kernel_psd_and_unit_diagonal():
    # the SM kernel is PSD by Bochner and unit-diagonal when the weights sum to one
    rng = np.random.RandomState(0)
    w = np.array([0.4, 0.35, 0.25]); mu = np.array([0.0, 1.5, 3.0]); v = np.array([0.05, 0.1, 0.2])
    k = ch08.SpectralMixtureKernel(w, mu, v)
    X = np.sort(rng.uniform(0, 1, 60))
    K = k.gram(X, X)
    assert np.allclose(np.diag(K), w.sum())                      # k(0) = sum_q w_q = 1
    assert np.allclose(np.diag(K), 1.0)
    assert np.linalg.eigvalsh(K).min() >= -1e-8                  # PSD


def test_rbf_is_sm_special_case():
    # the RBF is the single-Gaussian-at-zero special case of Bochner: mu=0, v=1/(4 pi^2 ell^2)
    ell = 0.3
    x = np.linspace(0, 1, 40)
    v0 = 1.0 / (4.0 * np.pi ** 2 * ell ** 2)
    sm = ch08.SpectralMixtureKernel([1.0], [0.0], [v0])
    K_sm = sm.gram(x, x)
    K_rbf = ch08.rbf_kernel(x, x, ell)
    assert np.allclose(K_sm, K_rbf, atol=1e-10)


def test_spectral_beats_rbf_on_periodic():
    # the spectral kernel extrapolates the periodicity beyond the data hull, where a single
    # RBF reverts to the mean and goes flat (the structural win)
    d = ch08.periodic_extrapolation_demo(seed=0)
    assert d["sm_test_rmse"] < 0.1                               # fits in-hull
    assert d["sm_extrap_rmse"] < d["rbf_extrap_rmse"]            # wins on extrapolation
    assert d["sm_extrap_rmse"] < 0.6 * d["rbf_extrap_rmse"]      # decisively so


def test_recovered_frequency_matches_truth():
    # the recovered dominant frequency is close to the true periodic frequency
    d = ch08.periodic_extrapolation_demo(freq=3.0, seed=0)
    assert abs(d["recovered_freq"] - d["true_freq"]) < 0.4


def test_density_atoms_normalized_and_g_independent_count():
    # Gauss-Hermite quadrature: 2 trainable numbers, mass sums to one for any node count G
    for G in (4, 8, 16):
        omega, wts = ch08.density_atoms_gauss_hermite(mu_log=0.0, gamma=0.4, G=G)
        assert len(omega) == G
        assert abs(wts.sum() - 1.0) < 1e-10
        assert np.all(omega > 0)


def test_figures_build():
    fig1 = ch08.make_periodic_figure(seed=0)
    fig2 = ch08.make_roughness_figure(seed=0)
    assert fig1 is not None and fig2 is not None
