"""Smoke tests for Chapter 10: fusing geometries.

The fused kernel is a convex simplex mixture of unit-diagonal channels (a tuned CatBoost leaf
kernel + the Chapter-8 learned spectral-Laplace kernel), with weights selected leakage-free on
a held-out query fold. These tests assert the structural guarantees the chapter rests on:
unit-diagonal preservation, the EXACT additive decomposition, the Loewner domination of fusion
over output-averaging, the soft-gate -> hard-leaf-kernel limit, and that the earned weight
tracks the target's geometry (spectral on a smooth target, tree on a sharp one).
"""
import numpy as np

from lkbook.chapters import ch10


def _psd_unit_block(n, seed):
    rng = np.random.RandomState(seed)
    A = rng.randn(n, n)
    K = A @ A.T
    d = np.sqrt(np.diag(K))
    return K / np.outer(d, d)                                   # correlation: PSD, unit diagonal


def test_mix_n_preserves_psd_and_unit_diagonal():
    K1, K2 = _psd_unit_block(20, 0), _psd_unit_block(20, 1)
    K = ch10.mix_n([K1, K2], np.array([0.3, 0.7]))
    assert np.allclose(np.diag(K), 1.0)                         # unit diagonal in => unit diagonal out
    assert np.linalg.eigvalsh((K + K.T) / 2).min() >= -1e-8     # convex combo of PSD is PSD


def test_simplex_grid_is_on_the_simplex_and_contains_vertices():
    grid = list(ch10._simplex_grid(2, 10))
    assert all(abs(w.sum() - 1.0) < 1e-12 and (w >= 0).all() for w in grid)
    vertices = {tuple(np.eye(2)[i]) for i in range(2)}
    assert vertices <= {tuple(w) for w in grid}                 # every vertex is in the grid


def test_soft_gate_converges_to_hard_leaf_kernel():
    fid = ch10.soft_tree_fidelity(taus=(1, 20, 500))
    gaps = [g for _, g in fid]
    assert gaps[0] > gaps[-1]                                   # sharpening closes the gap
    assert gaps[-1] < 0.15                                      # tau=500 is close to the hard kernel


def test_exact_additive_decomposition():
    X, y = ch10.smooth_to_sharp(0.3, n=400, seed=0)
    reps = ch10.same_reps(X, ["tree", "spectral"])
    fm = ch10.fit_fused(reps, y, n_fit=160)
    contribs, intercept = fm.channel_contributions(reps)
    recon = intercept + sum(np.atleast_1d(v) for v in contribs.values())
    assert np.allclose(recon, fm.predict(reps), atol=1e-8)      # components sum to the fit, exactly


def test_fusion_dominates_averaging_in_loewner_order():
    X, y = ch10.smooth_to_sharp(0.5, n=400, seed=0)
    n = len(y); perm = np.random.RandomState(7).permutation(n); nte = n // 4
    te, tr = perm[:nte], perm[nte:]
    out = ch10.fusion_vs_averaging(ch10.same_reps(X[tr], ["tree", "spectral"]), y[tr],
                                   ch10.same_reps(X[te], ["tree", "spectral"]), y[te], n_fit=200)
    # the theorem is the operator domination S_fuse >= S_avg (Loewner order); the lower test RMSE
    # is an empirical observation at the book's full settings, not a small-sample guarantee
    assert out["eig_min"] >= -1e-6                              # S_fuse - S_avg is PSD
    assert out["fused_rmse"] > 0 and out["avg_rmse"] > 0


def test_earned_weight_tracks_geometry():
    # smooth/periodic target -> spectral; sharp partition -> tree
    Xs, ys = ch10.smooth_to_sharp(0.0, n=500, seed=0)
    fm_s = ch10.fit_fused(ch10.same_reps(Xs, ["tree", "spectral"]), ys, n_fit=240)
    ws = dict(zip(fm_s.names, fm_s.w))
    Xh, yh = ch10.smooth_to_sharp(1.0, n=500, seed=0)
    fm_h = ch10.fit_fused(ch10.same_reps(Xh, ["tree", "spectral"]), yh, n_fit=240)
    wh = dict(zip(fm_h.names, fm_h.w))
    assert ws["spectral"] >= ws["tree"]                         # smooth target favors spectral
    assert wh["tree"] >= wh["spectral"]                         # sharp target favors the tree


def test_figures_build():
    import matplotlib
    matplotlib.use("Agg")
    bike = {"bank_weights": [0.01, 0.22, 0.77], "bank_T": [1.93, 2.19, 2.87],
            "weights": {"spectral": 0.9, "tree": 0.1}, "shares": {"spectral": 0.84, "tree": 0.16},
            "decomp": {"tree": 0.174, "spectral": -2.422}, "intercept": 4.592,
            "pred0": 2.344, "recon0": 2.344}
    tw = {"weights": {"spectral": 0.5, "tree": 0.5}}
    sweep = [{"t": 0.0, "spectral": 1.0, "tree": 0.0}, {"t": 1.0, "spectral": 0.0, "tree": 1.0}]
    assert ch10.make_weights_figure(bike=bike, tw=tw, sweep=sweep) is not None
    assert ch10.make_decomposition_figure(bike=bike) is not None
