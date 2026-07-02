"""Smoke + quality tests for Chapter 16: matrix-free Krylov.

Claims:
  - the row-blocked matvec equals the dense K @ v for any block size;
  - the matrix-free KRR solve reproduces the dense solve to machine precision, at lower peak memory;
  - the Krylov rank is a compute budget: coefficient error vs dense falls to ~0 by modest rank;
  - one Lanczos basis serves an entire ridge sweep (reuse is exact).
"""
import numpy as np
import torch

from lkbook.chapters import ch15, ch16


def test_matvec_equals_dense_Kv():
    Xtr, ytr, _, _, _, _ = ch15.load_california_scaled()
    Xb, _ = ch15.tile_jitter(Xtr, ytr, 1500)
    ker = ch15.MultiScaleKernel()
    emb = ker.embed(Xb)
    K = ker.gram(Xb, Xb)
    v = torch.randn(1500, dtype=ker.dtype, device=ker.device)
    for block in (256, 512, 4096):
        mv = ch16.kernel_matvec(emb, ker.kmat, block)
        assert torch.allclose(mv(v), K @ v, atol=1e-9)


def test_matfree_reproduces_dense():
    r = ch16.matfree_vs_dense(n=3000, block=1024)
    assert r["coef_max_diff"] < 1e-8                        # exact to machine precision
    assert abs(r["r2_dense"] - r["r2_matfree"]) < 1e-9
    assert r["matfree_peak_gb"] <= r["dense_peak_gb"] + 1e-6


def test_rank_is_a_compute_budget():
    rb = ch16.rank_budget(n=3000, ranks=(10, 40, 120))
    errs = [r["rel_coef_err"] for r in rb]
    assert errs[0] > errs[-1]                               # error falls with rank
    assert errs[-1] < 1e-6                                  # exact by modest rank
    assert rb[-1]["r2"] > 0.6


def test_ridge_sweep_basis_reuse_exact():
    sw = ch16.sweep_demo(n=3000, lambdas=(1e-2, 1e-1, 1.0))
    assert all(r["reuse_vs_scratch_max_diff"] < 1e-10 for r in sw["rows"])
    assert sw["chosen_lam"] in (1e-2, 1e-1, 1.0)
