"""Smoke tests: the weighted-vote identity that Chapter 1 rests on, on real data."""
import numpy as np

from lkbook import load_california, load_taiwan
from lkbook.chapters import ch01


def test_weighted_vote_california():
    d = load_california()
    ridge, tree, knn = ch01.fit_models(d)
    x = d.Xte[7]
    checks = {
        "ridge": (ch01.ridge_weights(d.Xtr, x),
                  float(ridge.predict(np.append(x, 1.0)[None])[0])),
        "tree": (ch01.tree_weights(tree, d.Xtr, x), float(tree.predict(x[None])[0])),
        "k-NN": (ch01.knn_weights(knn, d.Xtr, x), float(knn.predict(x[None])[0])),
    }
    for name, (w, pred) in checks.items():
        assert abs(float(w @ d.ytr) - pred) < 1e-6, name


def test_weighted_vote_taiwan():
    d = load_taiwan()
    assert d.task == "classification" and d.d == 23
    _, tree, knn = ch01.fit_models(d)
    x = d.Xte[3]
    for w, pred in [(ch01.tree_weights(tree, d.Xtr, x), float(tree.predict(x[None])[0])),
                    (ch01.knn_weights(knn, d.Xtr, x), float(knn.predict(x[None])[0]))]:
        assert abs(float(w @ d.ytr) - pred) < 1e-6


def test_figure_builds():
    fig = ch01.make_influence_figure(load_california(), res=60)
    assert len(fig.axes) >= 3
