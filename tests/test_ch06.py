"""Smoke tests for Chapter 6: the NW evidence ledger and the empirical/teacher/blended heads.

The identities checked: the weights are a probability distribution over training cases
(nonnegative, sum to one); N_eff = (Σw)²/Σw² counts uniform = n and one-hot = 1; the
Cauchy--Schwarz fidelity radius bounds the teacher-fidelity gap G_q; and the validation
finding ρ*(California) < ρ*(Taiwan) — the same machine, opposite blends.
"""
import numpy as np

from lkbook import load_california, load_taiwan
from lkbook.chapters import ch06


def test_weights_are_a_distribution():
    rng = np.random.RandomState(0)
    K = rng.rand(20, 50)
    W = ch06.nw_weights(K, topk=None, power=1.0)
    assert (W >= 0).all()
    assert np.allclose(W.sum(axis=1), 1.0)               # convex weights


def test_prediction_in_convex_hull():
    # f(x) = Σ w_i y_i must lie within [min y, max y]
    rng = np.random.RandomState(1)
    K = rng.rand(15, 40)
    y = rng.randn(40)
    W = ch06.nw_weights(K, topk=None)
    f = W @ y
    assert (f >= y.min() - 1e-9).all() and (f <= y.max() + 1e-9).all()


def test_neff_uniform_and_onehot():
    n = 50
    K_uniform = np.ones((1, n))
    neff_u = 1.0 / np.sum(ch06.nw_weights(K_uniform, topk=None) ** 2)
    assert abs(neff_u - n) < 1e-6                          # uniform weights → N_eff = n

    K_onehot = np.zeros((1, n)); K_onehot[0, 7] = 1.0
    neff_1 = 1.0 / np.sum(ch06.nw_weights(K_onehot, topk=None) ** 2)
    assert abs(neff_1 - 1.0) < 1e-6                        # one-hot weight → N_eff = 1


def test_topk_truncation_caps_neff():
    rng = np.random.RandomState(2)
    K = rng.rand(8, 300)
    W = ch06.nw_weights(K, topk=50, power=1.0)
    assert (W > 0).sum(axis=1).max() <= 50
    assert np.allclose(W.sum(axis=1), 1.0)


def test_fidelity_radius_bounds_gap():
    # |q(x) - Σ w_i q_i| ≤ (Σ w_i (q_i - q(x))²)^{1/2}  (Cauchy–Schwarz)
    rng = np.random.RandomState(3)
    K = rng.rand(30, 60)
    q_t = rng.randn(60)
    q_query = rng.randn(30)
    W = ch06.nw_weights(K, topk=None)
    g_q = np.abs(q_query - W @ q_t)
    radius = ch06.fidelity_radius(K, q_t, q_query, topk=None)
    assert (g_q <= radius + 1e-9).all()


def test_evidence_ledger_fields():
    d = load_california()
    m = ch06.CaseBasedModel().fit(d)
    ev = m.ledger(d.Xte[:100], top_k=8)
    for key in ("neff", "delta_y", "g_q", "c_cal", "delta_K"):
        assert ev[key].shape == (100,)
    assert ev["top_idx"].shape == (100, 8)
    assert (ev["neff"] > 0).all()
    assert (ev["delta_y"] >= 0).all()
    # witnesses are sorted by weight, descending
    assert (np.diff(ev["top_w"], axis=1) <= 1e-12).all()
    # kernel distance d_K² = 2(1-k)
    assert np.allclose(ev["top_dist"] ** 2, 2 * (1 - ev["top_K"]), atol=1e-9)


def test_rho_ordering_california_below_taiwan():
    cal = ch06.run_dataset(load_california())
    tw = ch06.run_dataset(load_taiwan())
    # the headline finding: smooth labels reward label averaging (ρ*→0); noisy binary
    # defaults reward the teacher's smoother score (ρ*→1)
    assert cal["rho_star"] < tw["rho_star"]
    assert cal["rho_star"] <= 0.2
    assert tw["rho_star"] >= 0.8


def test_runs_on_both_datasets():
    for d in (load_california(), load_taiwan()):
        run = ch06.run_dataset(d)
        for h in ("empirical", "teacher", "blended"):
            assert np.isfinite(run["head_loss"][h])
        assert run["forest_loss"] > 0
