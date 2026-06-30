"""Smoke + quality tests for Chapter 13: invariant encoders.

Claims, as directional facts with loose tolerances (small emitters, few steps):
  - the encoder's job is invariance: the feature-token emitter is row-permutation invariant and
    invariant to the content of padded columns (exact);
  - one checkpoint serves several feature counts (beats predict-the-mean at each);
  - the symmetry bottleneck: a symmetric (Mahalanobis) and an asymmetric attention encoder reach
    close regret (the encoder's attention symmetry is immaterial to the prediction);
  - mean pooling is the uniform-attention special case of PMA (both invariant);
  - the emitted measure builds a symmetric, unit-diagonal Gram.
"""
import numpy as np

from lkbook.chapters import ch13


def _ft(steps=200, device=None):
    return ch13.train_feature_token(steps=steps, B=24, device=ch13._device(device), seed=0)


def test_feature_token_invariances():
    ft = _ft(steps=150)
    inv = ch13.invariance_checks(ft, device=ch13._device())
    assert inv["row_perm"] < 1e-2          # permutation-invariant pooling
    assert inv["pad_content"] < 1e-5       # padded columns enter nowhere (exact)


def test_one_checkpoint_serves_widths():
    ft = _ft(steps=250)
    w = ch13.width_check(ft, widths=(3, 6), n_tasks=80, device=ch13._device())
    for da in (3, 6):
        assert w[da]["ratio"] < 1.0        # emitter beats predict-the-mean at each width


def test_feature_token_gram_symmetric_unit_diagonal():
    import torch
    ft = ch13.FeatureTokenEmitter(8, H=4, Q=3, seed=0)
    g = torch.Generator().manual_seed(0)
    X = torch.rand(2, 10, 8, generator=g) * 2 - 1
    y = torch.randn(2, 10, 1, generator=g)
    with torch.no_grad():
        m = ft.emit(X, y)
        K = ch13.gram(m, X, X)
    diag = torch.diagonal(K, dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-4)
    assert torch.allclose(K, K.transpose(-1, -2), atol=1e-4)


def test_bottleneck_sym_asym_close():
    """The encoder's attention symmetry is immaterial: symmetric and asymmetric attention encoders
    reach close regret over Bayes (gap small relative to the regret)."""
    sa = ch13.run_sym_asym_ab(steps=300, n_tasks=120, ks=(64, 256), device=ch13._device())
    for k in (64, 256):
        rs, ra = sa["symmetric"][k]["regret"], sa["asymmetric"][k]["regret"]
        assert rs >= -1e-3 and ra >= -1e-3
        assert abs(rs - ra) < 0.1          # nearly identical, not an order of magnitude apart


def test_mean_is_uniform_attention_special_case():
    """Both pools are permutation-invariant; mean pooling is the uniform-attention special case.
    Here we just check both MetaMSSKM pools run and produce finite regret."""
    import torch
    net_mean = ch13.train_emitter(steps=120, pool="mean", device=ch13._device(), seed=0)
    net_pma = ch13.train_emitter(steps=120, pool="pma", device=ch13._device(), seed=0)
    rows_m = ch13.eval_regret_vs_k(net_mean, 8, 4, 3, ks=(64,), n_tasks=60, device=ch13._device())
    rows_p = ch13.eval_regret_vs_k(net_pma, 8, 4, 3, ks=(64,), n_tasks=60, device=ch13._device())
    assert np.isfinite(rows_m[64]["regret"]) and np.isfinite(rows_p[64]["regret"])
    assert rows_m[64]["emitter"] < rows_m[64]["mean"]      # both are informative
    assert rows_p[64]["emitter"] < rows_p[64]["mean"]
