"""Smoke tests for Chapter 9: the head-to-head between Chapter 8's LEARNED spectral-Laplace
kernel (imported, not re-implemented) and the leaf kernel.

These assert what the companion ACTUALLY produces (see the chapter's reconciled numbers):
the learned spectral kernel wins the smooth, periodic and interaction targets and California;
the tree wins the discontinuous target; the additive-Laplace control (order one) cannot
represent the interaction.
"""
import numpy as np

from lkbook.chapters import ch08, ch09


def _kernel(d, interaction="full", seed=0):
    X = np.random.RandomState(seed).uniform(-1, 1, size=(40, d))
    om, _ = ch08._seed_support((X - X.mean(0)) / (X.std(0) + 1e-9), X[:, 0], 6)
    return X, ch08.LearnedSpectralLaplace(d, H=2, free_omega=om,
                                          interaction=interaction, seed=seed)


def test_spectral_kernel_psd_and_unit_diagonal():
    X, k = _kernel(5)                                          # unfit (random) measure
    K = k.gram(X, X)
    assert np.allclose(np.diag(K), 1.0)                        # unit diagonal (bank weights sum to 1)
    assert np.linalg.eigvalsh(K).min() >= -1e-8                # PSD


def test_additive_laplace_psd_and_unit_diagonal():
    X, k = _kernel(5, interaction="additive", seed=1)
    K = k.gram(X, X)
    assert np.allclose(np.diag(K), 1.0)
    assert np.linalg.eigvalsh(K).min() >= -1e-8


def test_spectral_beats_tree_on_smooth():
    r = ch09.head_to_head_target("S1")
    assert r["winner"] == "spectral"
    assert r["spectral"] < r["tree"]


def test_spectral_beats_tree_on_periodic():
    r = ch09.head_to_head_target("S2")
    assert r["winner"] == "spectral"
    assert r["spectral"] < r["tree"]


def test_spectral_beats_tree_on_interaction_and_additive_cannot_represent_it():
    r = ch09.head_to_head_target("S9")
    assert r["winner"] == "spectral"
    # the order-one additive control cannot represent the pure interaction: it trails spectral badly
    assert r["additive"] > r["spectral"]


def test_tree_beats_or_ties_spectral_on_discontinuous():
    r = ch09.head_to_head_target("S10")
    assert r["tree"] <= r["spectral"] + 1e-9


def test_figures_build():
    import matplotlib
    matplotlib.use("Agg")
    rows = [ch09.head_to_head_target(k) for k in ("S1", "S2")]
    assert ch09.make_winloss_figure(rows=rows) is not None
    assert ch09.make_smoothness_ladder_figure() is not None
