"""Smoke + quality tests for Chapter 11: symmetry suffices.

The chapter's claims are operational and the tests assert them as directional facts with loose
tolerances (the demos are deterministic but the exact values are not the point):
  - the symmetric/antisymmetric split is exact and a shared-W kernel is symmetric;
  - on exchangeable Taiwan Credit a symmetric attention smoother MATCHES the asymmetric one at
    fewer parameters;
  - the first-order law holds: the measured directional gain is proportional to -<Delta, h_a>;
  - on an exchangeable table <Delta, h_a> over random directions concentrates at zero;
  - on a constructed DIRECTED task asymmetry is earned (asym RMSE << sym RMSE).
"""
import numpy as np

from lkbook.chapters import ch11


def test_skew_is_antisymmetric_unit_norm():
    A = ch11.skew(6, 0)
    assert np.allclose(A, -A.T)                                  # antisymmetric
    assert abs(np.linalg.norm(A) - 1.0) < 1e-9                   # unit norm


def test_shared_projection_kernel_is_symmetric():
    import torch
    rng = np.random.RandomState(0)
    X = torch.tensor(rng.randn(12, 5))
    m = ch11.KernelAttention(5, r=8, mode="sym", seed=0)         # shared W => symmetric Gram
    with torch.no_grad():
        S = m.scores(X, X).numpy()
    assert np.allclose(S, S.T, atol=1e-8)
    ma = ch11.KernelAttention(5, r=8, mode="asym", seed=0)       # separate W => generically not
    with torch.no_grad():
        Sa = ma.scores(X, X).numpy()
    assert not np.allclose(Sa, Sa.T, atol=1e-6)


def test_taiwan_symmetric_matches_asymmetric_at_fewer_params():
    tw = ch11.run_taiwan_headtohead(seeds=range(3))
    assert tw["n_params"]["sym"] < tw["n_params"]["asym"]        # shared W has fewer parameters
    # matched accuracy: the symmetric model is within seed noise of the asymmetric one
    assert tw["sym"].mean() >= tw["asym"].mean() - 0.03


def test_first_order_law_proportional_to_alignment():
    law = ch11.run_first_order(n_dirs=12)
    assert law["corr"] < -0.8                                    # gain ~ -<Delta, h_a>


def test_exchangeable_data_directional_content_orthogonal():
    o = ch11.run_orthogonality(n_dirs=30, n=800)
    ti = o["taiwan_ips"]
    assert abs(ti.mean()) / (ti.std() + 1e-12) < 0.6            # centered at zero
    # the directed-task aligned direction is far off zero on the same kind of axis
    assert abs(o["directed_aligned"]) > 3 * o["directed_random_std"]


def test_directed_task_earns_asymmetry():
    di = ch11.run_directed_headtohead(seeds=range(2))
    assert di["asym"].mean() < 0.6 * di["sym"].mean()           # asymmetry strictly helps
    assert np.median(di["D"]) > 1.0                              # symmetrizing destroys the lag


def test_figures_build():
    res = {"taiwan": ch11.run_taiwan_headtohead(seeds=range(2)),
           "law": ch11.run_first_order(n_dirs=8),
           "ortho": ch11.run_orthogonality(n_dirs=12, n=400)}
    fig = ch11.make_law_figure(res)
    assert len(fig.axes) == 3
    di = ch11.run_directed_headtohead(seeds=range(1))
    fig2 = ch11.make_decision_figure(di, res["taiwan"])
    assert len(fig2.axes) == 2
