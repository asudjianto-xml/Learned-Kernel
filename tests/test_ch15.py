"""Smoke + quality tests for Chapter 15: the scaling wall.

Claims:
  - the dense decode fits California (fixed kernel, R2 ~ 0.7-0.8) and its peak memory tracks the
    O(n^2) Gram while solve time grows with n;
  - the float64 Gram-memory law crosses ~16 GB near n=45k (the memory wall);
  - the ridge conditioning bound holds: lambda_min(K+lam I) >= lam, kappa <= (lmax+lam)/lam, and a
    larger ridge lowers both kappa and sqrt(kappa).
"""
import numpy as np

from lkbook.chapters import ch15


def test_dense_decode_fits_and_scales():
    rows = ch15.time_dense((2000, 4000, 8000))
    r2 = [r["r2"] for r in rows]
    peak = [r["peak_gb"] for r in rows]
    assert all(0.6 < v < 0.9 for v in r2)                 # fixed kernel fits California
    assert peak[-1] > peak[0]                             # memory grows with n
    # float64 Gram term is exactly O(n^2)
    g = [r["gram_gb_f64"] for r in rows]
    assert abs(g[1] / g[0] - 4.0) < 1e-6                  # 2x n -> 4x memory


def test_gram_memory_law():
    assert ch15.dense_gram_gb(45_000) > 16.0             # crosses 16 GB near 45k
    assert ch15.dense_gram_gb(50_000) == 8 * 50_000 ** 2 / 1e9
    # array-safe
    v = ch15.dense_gram_gb(np.array([1000.0, 2000.0]))
    assert v.shape == (2,) and abs(v[1] / v[0] - 4.0) < 1e-9


def test_ridge_conditioning_bound():
    rows = ch15.conditioning_sweep(n=1200)
    for r in rows:
        assert r["kappa"] <= r["bound"] + 1e-6           # measured kappa within the bound
        assert r["lmin_K"] >= -1e-8                       # K is PSD
    # larger ridge -> smaller condition number and iteration proxy
    small_lam = min(rows, key=lambda r: r["lam"])
    large_lam = max(rows, key=lambda r: r["lam"])
    assert large_lam["kappa"] < small_lam["kappa"]
    assert large_lam["sqrt_kappa"] < small_lam["sqrt_kappa"]


def test_cost_table_leading_terms():
    t = ch15.cost_table((1000, 2000))
    assert abs(t[1]["form_flops"] / t[0]["form_flops"] - 4.0) < 1e-6   # ~n^2
    assert abs(t[1]["chol_flops"] / t[0]["chol_flops"] - 8.0) < 1e-6   # ~n^3
