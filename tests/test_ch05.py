"""Smoke tests for Chapter 5: GP-mean = KRR, the evidence interior optimum, and Prop. F."""
import numpy as np

from lkbook import load_california, load_taiwan
from lkbook.chapters import ch05


def test_gp_mean_equals_krr_machine_precision():
    # the boxed identity: GP posterior mean == KRR with ridge λ=σ²
    for d in (load_california(), load_taiwan()):
        eq = ch05.gp_mean_equals_krr(d)
        assert eq["max_abs_diff"] < 1e-6           # to machine precision
        assert eq["min_var"] >= -1e-9              # variance is nonnegative
        assert eq["max_var"] <= 1.0 + 1e-9         # unit-diagonal RBF prior


def test_nlml_has_interior_optimum():
    # Prop. F, empirical face: the NLML minimum in the length scale is interior, and the
    # NLML rises toward ℓ→∞ (the over-correlated K→J corner) — it is not driven there.
    sc = ch05.nlml_length_scale_scan(load_california())
    assert sc["interior"]
    assert sc["nlml"][-1] > sc["nlml"].min()       # K→J corner is uphill


def test_prop_f_lemma_closed_forms_and_divergent_gap():
    pf = ch05.prop_f_allones_gap(load_california())
    # Lemma F.0 closed forms match the eigen/Sherman–Morrison truth
    assert pf["logdet_err"] < 1e-6
    assert pf["quad_relerr"] < 1e-6
    # the all-ones kernel has strictly larger NLML than a fitting kernel, and the gap grows
    assert np.all(pf["gap"] > 0)
    assert pf["gap"][0] > pf["gap"][-1]            # diverges as σ²→0 (smaller σ² ⇒ larger gap)


def test_evidence_fit_runs_on_both():
    for d in (load_california(), load_taiwan()):
        ev = ch05.evidence_fit_vs_fixed(d)
        assert ev["ell_hat"] > 0 and ev["sig2_hat"] > 0
        assert 0.0 <= ev["coverage95"] <= 1.0


def test_figures_build():
    cal = load_california()
    fig1 = ch05.make_prior_posterior_figure(cal)
    fig2 = ch05.make_evidence_figure(cal)
    assert fig1 is not None and fig2 is not None
