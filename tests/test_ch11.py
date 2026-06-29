"""Smoke + quality tests for Chapter 11: symmetry suffices.

Claims, as directional facts with loose tolerances (deterministic demos, exact values not the point):
  - a shared-W kernel is symmetric; separate W_Q,W_K is not;
  - the first-order law holds (gain proportional to -<Delta,h_a>), and on exchangeable Taiwan the
    alignment concentrates at zero;
  - asymmetrizing a spectral kernel forfeits KRR: sym+KRR > sym+NW ~ asym+NW (asymmetry adds nothing);
  - the book's symmetric kernels + KRR beat asymmetric attention + NW;
  - the fusion diagnostic rho* separates exchangeable (~0) from directed (>0);
  - on the directed task asymmetry is earned (asym RMSE << sym RMSE).
"""
import numpy as np

from lkbook.chapters import ch11


def test_skew_is_antisymmetric_unit_norm():
    A = ch11.skew(6, 0)
    assert np.allclose(A, -A.T)
    assert abs(np.linalg.norm(A) - 1.0) < 1e-9


def test_shared_projection_kernel_is_symmetric():
    import torch
    X = torch.tensor(np.random.RandomState(0).randn(12, 5))
    ms = ch11.KernelAttention(5, r=8, mode="sym", seed=0)
    with torch.no_grad():
        S = ms.scores(X, X).numpy()
    assert np.allclose(S, S.T, atol=1e-8)
    ma = ch11.KernelAttention(5, r=8, mode="asym", seed=0)
    with torch.no_grad():
        Sa = ma.scores(X, X).numpy()
    assert not np.allclose(Sa, Sa.T, atol=1e-6)


def test_spectral_features_shape_and_bounded():
    phi = ch11.SpectralFeatures(4, D=32, gamma=0.1, seed=0)
    P = phi(np.random.RandomState(1).randn(7, 4))
    assert P.shape == (7, 32)
    assert np.all(np.abs(P) <= np.sqrt(2.0 / 32) + 1e-9)


def test_first_order_law_proportional_to_alignment():
    law = ch11.run_first_order(n_dirs=12)
    assert law["corr"] < -0.8                                   # gain ~ -<Delta, h_a>


def test_exchangeable_data_directional_content_orthogonal():
    o = ch11.run_orthogonality(n_dirs=30, n=800)
    ti = o["taiwan_ips"]
    assert abs(ti.mean()) / (ti.std() + 1e-12) < 0.6           # centered at zero
    assert abs(o["directed_aligned"]) > 3 * o["directed_random_std"]


def test_asymmetrizing_spectral_kernel_forfeits_krr():
    sc = ch11.run_spectral_cost(seeds=range(2))
    # KRR (which symmetry unlocks) clearly beats NW; asymmetry adds nothing over the symmetric NW
    assert sc["sym_krr"].mean() > sc["sym_nw"].mean() + 0.02
    assert abs(sc["asym_nw"].mean() - sc["sym_nw"].mean()) < 0.03


def test_book_symmetric_kernels_beat_asymmetric_attention():
    real = ch11.run_real_headtohead(seeds=range(2), n_train=600)
    assert real["spectral_krr"].mean() > real["asym_attn_nw"].mean()
    assert real["tree_krr"].mean() > real["asym_attn_nw"].mean()


def test_fusion_diagnostic_separates_regimes():
    fu = ch11.run_fusion_diagnostic(seeds=range(2))
    assert fu["directed_rho"].mean() > fu["taiwan_rho"].mean() + 0.2


def test_directed_task_earns_asymmetry():
    di = ch11.run_directed_headtohead(seeds=range(2))
    assert di["asym"].mean() < 0.6 * di["sym"].mean()
    assert np.median(di["D"]) > 1.0


def test_figures_build():
    law = ch11.run_first_order(n_dirs=8)
    ortho = ch11.run_orthogonality(n_dirs=12, n=400)
    sc = ch11.run_spectral_cost(seeds=range(1))
    fig = ch11.make_law_figure({"law": law, "ortho": ortho, "spectral_cost": sc})
    assert len(fig.axes) == 3
    real = ch11.run_real_headtohead(seeds=range(1), n_train=500)
    fu = ch11.run_fusion_diagnostic(seeds=range(1))
    fig2 = ch11.make_kernels_figure({"real": real, "fusion": fu})
    assert len(fig2.axes) == 2
    fig3 = ch11.make_decision_figure()
    assert len(fig3.axes) == 1
