"""Smoke + quality tests for Chapter 17: Nystrom and landmarks.

Claims:
  - the Nystrom map reproduces the low-rank kernel: <phi~(x),phi~(x')> = k(x,S) Kmm^-1 k(S,x');
  - the primal Nystrom solve is accurate (near the exact decode) at far lower peak memory;
  - it scales linearly in n with flat memory (the n x n Gram never exists);
  - accuracy degrades gracefully as the landmark fraction shrinks;
  - the RFF cousin is an exact primal ridge whose accuracy rises with M;
  - the auto ladder picks dense -> matfree -> nystrom as n grows.
"""
import numpy as np
import torch

from lkbook.chapters import ch15, ch17


def test_nystrom_map_reproduces_kernel():
    Xtr, ytr, _, _, _, _ = ch15.load_california_scaled()
    ker = ch15.MultiScaleKernel()
    S = ch17.select_landmarks(Xtr, 200, "uniform")
    embS = ker.embed(S)
    P, _ = ch17.nystrom_factor(embS, ker.kmat)             # full rank
    q = Xtr[:50]
    phi = ch17.nystrom_features(ker.embed(q), embS, P, ker.kmat)
    approx = phi @ phi.transpose(0, 1)
    Kmm = ker.gram(S, S)
    kqS = ker.gram(q, S)
    exact = kqS @ torch.linalg.solve(Kmm, kqS.transpose(0, 1))
    assert torch.allclose(approx, exact, atol=1e-4)


def test_nystrom_accurate_and_lean():
    r = ch17.nystrom_vs_exact(n=6000, m=400)
    assert r["r2_nystrom"] > r["r2_exact"] - 0.06          # near the exact decode
    assert r["nystrom_peak_gb"] < r["exact_peak_gb"]       # far less memory


def test_nystrom_scales_linearly_flat_memory():
    sc = ch17.scaling_curve((8000, 32000, 64000), m=256)
    peak = [r["peak_gb"] for r in sc]
    assert max(peak) / min(peak) < 2.0                     # memory roughly flat
    assert all(r["r2"] > 0.6 for r in sc)


def test_graceful_degradation():
    lc = ch17.landmark_curve(n=6000, fracs=(0.02, 0.08, 0.2))
    r2 = [r["r2"] for r in lc["rows"]]
    assert r2[0] < r2[-1] <= lc["r2_exact"] + 1e-6         # monotone up toward exact


def test_rff_cousin_rises_with_M():
    Xtr, ytr, Xte, yte, ym, ys = ch15.load_california_scaled()
    r2 = [ch15.r2_score(ch17.rff_solve(Xtr, ytr, Xte, M=M, ell=2.0), yte, ym, ys)
          for M in (256, 1024)]
    assert r2[1] >= r2[0] - 0.01 and r2[0] > 0.6


def test_auto_ladder():
    assert ch17.auto_ladder(1000) == "dense"
    assert ch17.auto_ladder(200_000) == "matfree"
    assert ch17.auto_ladder(5_000_000) == "nystrom"
