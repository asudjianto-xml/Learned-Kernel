"""Smoke tests for Chapter 7: leakage-free selection and SURE.

The over-credit problem: the supervised leaf kernel nearly interpolates its support, so an
in-sample criterion over-credits it; query-fold selection ranks honestly. SURE for a fixed
smoother tracks the true denoising risk; the in-sample residual does not.
"""
import numpy as np

from lkbook import load_california
from lkbook.chapters import ch07


def test_leaf_kernel_over_credited_in_sample():
    # the supervised tree kernel nearly interpolates its support (support R2 ~ 1) but its
    # held-out query R2 is much lower -- the over-credit gap is large for the leaf channel
    rows = ch07.per_channel_credit(load_california())
    leaf = next(r for r in rows if r["name"].startswith("leaf"))
    rbf = next(r for r in rows if r["name"].startswith("RBF"))
    assert leaf["support_r2"] > 0.95                  # near-interpolation
    assert leaf["query_r2"] < leaf["support_r2"] - 0.2
    assert leaf["gap"] > rbf["gap"]                   # the tree is over-credited the most


def test_in_sample_overcredits_vs_query():
    # in-sample (SURE on support) routes more weight to the leaf channel than query selection,
    # and the query-selected fusion does not generalize worse
    sel = ch07.select_in_sample_vs_query(load_california())
    leaf = "leaf / tree (Ch. 4)"
    assert sel["in_sample"]["weights"][leaf] > sel["query"]["weights"][leaf]
    assert sel["in_sample"]["weights"][leaf] > 0.8    # in-sample hands the tree the lion's share
    assert sel["query"]["test_rmse"] <= sel["in_sample"]["test_rmse"] + 1e-6


def test_sure_tracks_true_risk():
    sr = ch07.sure_tracks_risk()
    # SURE is (approximately) unbiased for the true denoising risk across the sweep
    assert sr["corr_sure"] > 0.95
    assert sr["corr_sure"] > sr["corr_insample"]
    # SURE selects near the oracle; the in-sample residual picks (near-)interpolation, worse
    assert sr["true_at_sure"] < 1.2 * sr["true_min"]
    assert sr["true_at_insample"] > 1.5 * sr["true_min"]
    assert sr["lam_insample"] < sr["lam_sure"]        # in-sample drives lambda toward zero


def test_capacity_map_free_grows_banks_flat():
    cap = ch07.capacity_map(load_california())
    free_gaps = [r["gap"] for r in cap["free"]]
    bank_gaps = [r["gap"] for r in cap["banks"]]
    # free-atom capacity (tree depth K) widens the train-test gap monotone-ish
    assert free_gaps[-1] > free_gaps[0] + 0.1
    # convex banks are near-free: the gap is flat once a useful scale is in the bank
    assert max(bank_gaps[2:]) - min(bank_gaps[2:]) < 0.1
