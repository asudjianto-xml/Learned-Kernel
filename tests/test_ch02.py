"""Smoke tests for Chapter 2: the exact identities that make the kernel the normal form."""
from lkbook import load_california, load_taiwan
from lkbook.chapters import ch02


def test_krr_equals_gp_and_attention_equals_nw():
    preds, info = ch02.one_machine(load_california())
    # the two algebraic identities the chapter rests on
    assert abs(preds["KRR"] - preds["GP mean"]) < 1e-9
    assert abs(preds["attention"] - preds["NW"]) < 1e-9
    assert ch02.exact_identities(preds)
    assert 0 < info["n_support"] <= info["n_train"]


def test_taiwan_krr_score_agrees_with_svm_sign():
    out = ch02.taiwan_decision(load_taiwan())
    assert out["krr_class"] == out["svm_class"]


def test_figures_build():
    assert len(ch02.make_reproducing_figure().axes) >= 1
    assert len(ch02.make_predictions_figure(load_california()).axes) >= 1
