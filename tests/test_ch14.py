"""Smoke + quality tests for Chapter 14: zero-shot tabular foundation models.

Claims, as directional facts (small emitters, few steps; absolute zero-shot numbers on small data
are seed-noisy, so tolerances are loose):
  - a frozen emitter predicts real tables zero-shot (finite R2/AUC), competitive on breast-cancer;
  - the probe: meta-training the same architecture in-context on California reaches well above the
    synthetic-prior zero-shot and near the fitted ceiling (the gap is the prior, not the meta-learner);
  - the context lever is monotone-ish: more context does not hurt California zero-shot;
  - fitted baselines run and beat the frozen emitter on interaction-heavy California.
"""
import numpy as np

from lkbook.chapters import ch14


def _net(steps=300, device=None):
    return ch14.train_zeroshot_emitter(d_max=32, steps=steps, B=16, device=ch14._device(device),
                                       seed=0)


def test_load_real_shapes():
    for name, kind in (("california", "reg"), ("diabetes", "reg"), ("breast_cancer", "clf")):
        Xtr, ytr, Xte, yte, k, names = ch14._load_real(name, seed=0)
        assert Xtr.ndim == 2 and Xtr.shape[1] == len(names)
        assert k == kind and len(ytr) == len(Xtr) and len(yte) == len(Xte)


def test_zeroshot_predicts_real_tables():
    net = _net(steps=300)
    dev = ch14._device()
    # California (regression): finite R2
    Xtr, ytr, Xte, yte, kind, _ = ch14._load_real("california", seed=0)
    pred, jdx = ch14.zeroshot_predict(net, Xtr, ytr, Xte, kind, d_max=32, device=dev)
    ye = yte[jdx] if not isinstance(jdx, slice) else yte
    assert np.isfinite(ch14._score(pred, ye, kind)["score"])
    # breast-cancer (classification): competitive AUC
    Xtr, ytr, Xte, yte, kind, _ = ch14._load_real("breast_cancer", seed=0)
    pred, jdx = ch14.zeroshot_predict(net, Xtr, ytr, Xte, kind, d_max=32, device=dev)
    ye = yte[jdx] if not isinstance(jdx, slice) else yte
    assert ch14._score(pred, ye, kind)["auc"] > 0.85


def test_context_lever_monotone_ish():
    net = _net(steps=400)
    lv = ch14.context_lever(net, d_max=32, contexts=(128, 512, 2048), device=ch14._device())
    assert lv[2048] >= lv[128] - 0.05          # more context does not hurt California


def test_probe_gap_is_the_prior():
    """In-context-on-California reaches well above the synthetic-prior zero-shot and near the fitted
    ceiling: the gap is the prior, not the meta-learner."""
    pr = ch14.probe_california(0.40, steps=800, seed=0, device=ch14._device())
    assert pr["in_context_real_2048"] > 0.6            # matched prior recovers California
    assert pr["in_context_real_2048"] > pr["synthetic"] + 0.2   # well above synthetic prior
    assert pr["ch8_ceiling"] > 0.6 and pr["catboost"] > 0.6     # fitted baselines are strong


def test_fitted_baselines_beat_frozen_on_california():
    net = _net(steps=400)
    dev = ch14._device()
    Xtr, ytr, Xte, yte, kind, _ = ch14._load_real("california", seed=0)
    pred, jdx = ch14.zeroshot_predict(net, Xtr, ytr, Xte, kind, d_max=32, ctx_cap=512, device=dev)
    ye = yte[jdx] if not isinstance(jdx, slice) else yte
    zs = ch14._score(pred, ye, kind)["score"]
    base = ch14.fitted_baselines(Xtr, ytr, Xte, yte, kind, seed=0, ch8_steps=200, cb_iters=200)
    assert base["ch8"]["score"] > zs and base["catboost"]["score"] > zs


# --- Designing the prior (generative-simulator priors + ceiling-lift) ---------------------

def test_generators_are_learnable():
    """Each generative prior preserves X->y, so synthetic tasks are learnable (GBDT R2 > 0)."""
    Xtr, ytr, _, _ = ch14.load_ca8(seed=0)
    arf = ch14.make_arf_sampler(Xtr, ytr, seed=0, num_trees=20)
    cop = ch14.CopulaAdvGenerator(rounds=2, iters=80, seed=0).fit(Xtr, ytr).sample_fn()
    mc = ch14.MCMCAdvGenerator(rounds=2, mh_steps=40, walkers=2048, iters=80, seed=0).fit(Xtr, ytr).sample_fn()
    for fn in (arf, cop, mc):
        X, y = fn(200)
        assert X.shape[1] == 8 and len(y) == 200
        assert ch14.learnability(fn, seed=0) > 0.0


def test_generative_prior_beats_floor():
    """A generative prior trained zero-shot clears the GP/bandwidth floor on California (loose)."""
    dev = ch14._device()
    Xtr, ytr, Xte, yte = ch14.load_ca8(seed=0, device=dev)
    floor = ch14.train_gp_prior(steps=400, seed=0, device=dev)
    fl = ch14.eval_ca_zeroshot(floor, Xtr, ytr, Xte, yte, reps=2, seed=0, device=dev)[0]
    fn = ch14.make_arf_sampler(Xtr, ytr, seed=0, num_trees=20)
    net = ch14.train_on_generator(fn, steps=400, seed=0, device=dev)
    zs = ch14.eval_ca_zeroshot(net, Xtr, ytr, Xte, yte, reps=2, seed=0, device=dev)[0]
    assert np.isfinite(fl) and np.isfinite(zs)
    assert zs > fl - 0.05                          # generative prior is no worse than the floor


def test_ceiling_lift_context_helps():
    """The real-subtask ceiling does not fall when context grows 512 -> 2048."""
    res = ch14.ceiling_lift(train_w=False, H=4, steps=400, seed=0, caps=(512, 2048),
                            device=ch14._device())
    assert res[2048][0] >= res[512][0] - 0.05
