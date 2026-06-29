"""Smoke + quality tests for Chapter 12: meta-learning a prior over kernels.

Claims, as directional facts with loose tolerances (a tiny emitter trained for few steps;
exact values are not the point):
  - the self-consistent prior produces predictable (non-white-noise) tasks: bandwidth
    calibration keeps the Gram off-diagonal alive, so Bayes beats predict-the-mean;
  - the Bayes posterior is the exact GP posterior mean and is decoded by the SAME KRR code
    the emitter uses, so regret is a one-line subtraction;
  - regret over Bayes is nonnegative;
  - a briefly-trained emitter beats predict-the-mean and trails Bayes (a bounded gap);
  - binarize threshold gives balanced 0/1 labels without leaving the prior;
  - the gram has unit diagonal and is symmetric.
"""
import numpy as np

from lkbook.chapters import ch12


def _net(steps=120, device=None):
    return ch12.train_emitter(steps=steps, B=24, device=ch12._device(device), seed=0)


def test_gram_unit_diagonal_and_symmetric():
    import torch
    g = torch.Generator().manual_seed(0)
    W = 0.1 * torch.randn(64, 8 * 2 * 18, generator=g)
    m, s2 = ch12.sample_measure_prior(3, 8, 4, 3,
                                      *_quad(), W, g, device="cpu")
    X = torch.rand(3, 10, 8, generator=g) * 2 - 1
    K = ch12.gram(m, X, X)
    assert K.shape == (3, 10, 10)
    diag = torch.diagonal(K, dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-4)
    assert torch.allclose(K, K.transpose(-1, -2), atol=1e-4)


def _quad():
    import torch
    nodes, wts = np.polynomial.hermite.hermgauss(6)
    return (torch.as_tensor(nodes, dtype=torch.float32),
            torch.as_tensor(wts / np.sqrt(np.pi), dtype=torch.float32))


def test_calibration_keeps_tasks_learnable_bayes_beats_mean():
    """Without bandwidth calibration the Gram collapses to the identity and the draw is white
    noise; with it, Bayes improves on the mean by a wide margin."""
    import torch
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(1)
    gh = _quad()
    W = 0.1 * torch.randn(64, 8 * 2 * 18, generator=g)
    m, s2 = ch12.sample_measure_prior(40, 8, 4, 3, *gh, W, g, device="cpu")
    Xc, yc, Xq, yq = ch12.sample_gp_tasks(m, 64, 64, s2, g, device="cpu")
    bayes = ch12.bayes_posterior(m, Xc, yc, Xq, s2)
    mse_bayes = F.mse_loss(bayes, yq).item()
    mse_mean = F.mse_loss(torch.zeros_like(yq), yq).item()
    assert mse_bayes < 0.85 * mse_mean


def test_binarize_balanced():
    import torch
    g = torch.Generator().manual_seed(2)
    gh = _quad()
    W = 0.1 * torch.randn(64, 8 * 2 * 18, generator=g)
    m, s2 = ch12.sample_measure_prior(8, 8, 4, 3, *gh, W, g, device="cpu")
    Xc, yc, Xq, yq = ch12.sample_gp_tasks(m, 64, 64, s2, g, device="cpu")
    bc, bq = ch12.binarize(yc, yq)
    assert set(np.unique(bc.numpy())) <= {0.0, 1.0}
    frac = bc.mean().item()
    assert 0.35 < frac < 0.65  # thresholded at the context median -> ~balanced


def test_regret_nonnegative_and_emitter_beats_mean():
    net = _net(steps=150)
    rows = ch12.eval_regret_vs_k(net, 8, 4, 3, ks=(16, 64, 256), n_q=64, n_tasks=120,
                                 device=ch12._device())
    for k, r in rows.items():
        assert r["regret"] >= -1e-3, (k, r)            # Bayes is the minimizer
        assert r["emitter"] < r["mean"], (k, r)        # the emitter is informative
        assert r["bayes"] <= r["emitter"] + 1e-3, (k, r)


def test_explore_task_shapes():
    net = _net(steps=120)
    e = ch12.explore_task(net, k=64, n_q=80, device=ch12._device())
    assert e["emitter"].shape == (80,) and e["bayes"].shape == (80,)
    assert e["mse_bayes"] <= e["mse_emitter"] + 1e-2


def test_amortization_cost_is_small_in_distribution():
    """In-distribution the one-pass emitter is not much worse than a per-dataset gradient fit on the
    same context (amortization is nearly free) — and both trail Bayes (the finite-context cost)."""
    net = _net(steps=400)
    d = ch12.decompose_amortization(net, ks=(256,), n_tasks=10, pd_steps=150, device=ch12._device())
    r = d[256]
    assert r["emitter"] >= r["per_dataset"] - 0.10       # emitter ~matches per-dataset fitting
    assert r["bayes"] >= r["emitter"] - 1e-6             # Bayes is the ceiling


def test_recover_vs_chapter8_usable_predictor():
    """The one-pass emitter recovers a usable predictor on California, below the Ch8 per-dataset fit
    under the synthetic prior (the prior-reality gap)."""
    net = _net(steps=400)
    rec = ch12.recover_vs_chapter8(net, seed=0, device=ch12._device(), ch8_steps=200)
    assert rec["r2_emitter"] > 0.0                       # better than predicting the mean
    assert rec["r2_ch8"] > rec["r2_emitter"]             # per-dataset training is stronger here
    assert len(rec["rel_ch8"]) == len(rec["rel_emitter"]) == 8


def test_ceiling_incontext_real_matches_per_dataset():
    """Meta-training in-context on real California sub-tasks closes the gap to the per-dataset fit:
    the residual was prior-misspecification, not amortization."""
    res = ch12.ceiling_incontext_real(steps=800, ctx_caps=(512,), seed=0, device=ch12._device())
    r2 = res[512][0]
    assert r2 > 0.65                                     # well above the synthetic-prior ~0.55
    assert r2 > res["ch8"] - 0.15                        # approaches the per-dataset fit (~0.80)
