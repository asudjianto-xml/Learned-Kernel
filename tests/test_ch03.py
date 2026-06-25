"""Smoke tests for Chapter 3: the inversion, the ARD win, and the scale degeneracy."""
import numpy as np

from lkbook import load_california
from lkbook.chapters import ch03


def test_bandwidth_is_not_flat():
    # if choosing ℓ were free, the test-error curve would be flat; it is not
    ells, rmse = ch03.bandwidth_sweep(load_california())
    assert rmse.max() - rmse.min() > 0.2


def test_ard_beats_or_matches_best_isotropic():
    r = ch03.fit_ard(load_california())
    assert r["ard_test_rmse"] <= r["iso_test_rmse"] + 1e-6
    assert r["relevance"].shape == (8,)


def test_scale_degeneracy_and_unit_diagonal():
    deg = ch03.degeneracy_demo(load_california())
    assert deg["abs_diff"] < 1e-9              # (K,λ) and (αK,αλ) give the same prediction
    assert abs(deg["unit_diagonal"] - 1.0) < 1e-9


def test_ard_kernel_is_unit_diagonal():
    X = load_california().Xtr[:50]
    K = ch03.ard_gram(X, X, np.ones(X.shape[1]) * 1.7)
    assert np.allclose(np.diag(K), 1.0)
