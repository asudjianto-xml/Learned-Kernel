"""Smoke tests for Chapter 4: the leaf kernel and the GNW exact recovery."""
import numpy as np

from lkbook import load_california, load_taiwan
from lkbook.chapters import ch04


def test_leaf_kernel_unit_diagonal_and_range():
    d = load_california()
    model, Xtr, ytr = ch04.fit_forest(d)
    lk = ch04.LeafKernel().fit(model)
    K = lk.gram(Xtr[:300], Xtr[:300])
    assert np.allclose(np.diag(K), 1.0)               # unit diagonal
    assert K.min() >= -1e-12 and K.max() <= 1.0 + 1e-9
    assert np.linalg.eigvalsh(K).min() >= -1e-8       # PSD


def test_gnw_exact_recovery():
    # the forest IS a GNW operator: leaf-score values reproduce model.predict exactly
    d = load_california()
    model, Xtr, ytr = ch04.fit_forest(d)
    recon = ch04.gbdt_leaf_value_prediction(model, d.Xte)
    assert np.max(np.abs(recon - model.predict(d.Xte))) < 1e-9


def test_value_axis_ordering():
    vm = ch04.value_mechanisms(load_california())
    assert vm["exact_recovery_err"] < 1e-9
    assert abs(vm["exact_rmse"] - vm["forest_rmse"]) < 1e-9      # exact == forest
    assert vm["rawlabel_nw_rmse"] > vm["forest_rmse"]           # raw labels are crude


def test_taiwan_classification_forest():
    d = load_taiwan()
    model, Xtr, ytr = ch04.fit_forest(d, classifier=True)
    lk = ch04.LeafKernel().fit(model)
    assert np.allclose(np.diag(lk.gram(Xtr[:200], Xtr[:200])), 1.0)
