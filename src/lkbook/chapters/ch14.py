"""Chapter 14 — Zero-shot tabular foundation models.

Read through the kernel lens, TabPFN and TabICL do **amortized Bayesian inference**: a forward pass
that approximates a posterior predictive under a synthetic prior, with no fit step. The white-box
spectral emitter (Ch. 12-13) does the same inference, but with the right invariances and an
interpretable output -- a kernel whose frequencies, relevances and length scales can be read. Frozen
on a new table it competes with per-dataset-fitted models where the data resemble a draw from the
prior; where it trails, the cause is **prior<->reality mismatch**, a property of the task distribution
trained on, readable and fixable in the prior, not a failure of the meta-learner.

This module builds on Chapters 12-13 (it imports the prior, kernel and emitter from
``lkbook.chapters.ch12`` and the per-dataset spectral fit from ``ch08``) and adds:

  * ``zeroshot_table`` -- a frozen emitter zero-shot on real tables vs a per-dataset fitted kernel
    machine and gradient boosting (the Table-real comparison);
  * ``probe_california`` -- the meta-learner-vs-prior probe: the same architecture meta-trained on
    California resamples reaches the fitted ceiling, so the synthetic-prior gap is prior mismatch
    (reuses ``ch12.ceiling_incontext_real``);
  * ``geometry_lever`` -- frozen vs *trained* shared geometry W: training W overfits the prior and
    collapses transfer; a fixed random W is the better inductive bias;
  * ``read_relevances`` -- the emitted ARD relevances on a real context (the readable diagnostic);
  * **designing the prior** (the route to interactions, re-exported from ``ch14_priors``):
    ``run_designing_prior`` builds the prior as a generative simulator fit to the data
    (``make_arf_sampler``, ``CopulaAdvGenerator``, ``MCMCAdvGenerator``), meta-trains the same
    width-8 emitter on the synthetic tables (``train_on_generator``) and evaluates zero-shot on real
    California (``eval_ca_zeroshot``); ``ceiling_lift`` decomposes the residual with the architecture
    levers (context, trained W, head count). ``make_designing_figure`` draws the result.

    python -m lkbook.chapters.ch14 --out-prefix fig14
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from lkbook.chapters import ch12
from lkbook.chapters.ch12 import (MetaMSSKM, sample_measure_prior, sample_gp_tasks, binarize,
                                  bayes_posterior, _pad, _device)

# Designing the prior (the route to interactions). Implementation lives in ch14_priors; re-exported
# here so the chapter, its notebook and regen_figures import a single surface (``ch14.*``) and never
# re-implement. ch14_priors.run_all is exposed as run_designing_prior to avoid clashing with ch14's
# own run_all.
from lkbook.chapters.ch14_priors import (  # noqa: E402,F401
    run_all as run_designing_prior,
    make_designing_figure,
    make_arf_sampler, CopulaAdvGenerator, MCMCAdvGenerator,
    train_gp_prior, train_on_generator, eval_ca_zeroshot, ceiling_lift, learnability, load_ca8,
)

SEED = 0


# =============================================================================
# Real datasets
# =============================================================================

def _load_real(name, seed=0, test_size=0.4):
    """Return (Xtr, ytr, Xte, yte, kind, names), standardized by training-split statistics.
    Regression: california, diabetes. Classification: breast_cancer, taiwan."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    if name == "california":
        from sklearn.datasets import fetch_california_housing
        d = fetch_california_housing(); X, y, kind, names = d.data, d.target, "reg", list(d.feature_names)
    elif name == "diabetes":
        from sklearn.datasets import load_diabetes
        d = load_diabetes(); X, y, kind, names = d.data, d.target, "reg", list(d.feature_names)
    elif name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer(); X, y, kind, names = d.data, d.target, "clf", list(d.feature_names)
    elif name == "taiwan":
        from lkbook import load_taiwan
        t = load_taiwan(seed=seed)
        return (t.Xtr.astype(np.float32), np.asarray(t.ytr, float), t.Xte.astype(np.float32),
                np.asarray(t.yte, float), "clf", list(t.names))
    else:
        raise ValueError(name)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
    sc = StandardScaler().fit(Xtr)
    return (sc.transform(Xtr).astype(np.float32), np.asarray(ytr, float),
            sc.transform(Xte).astype(np.float32), np.asarray(yte, float), kind, names)


# =============================================================================
# A frozen zero-shot emitter trained on the broadened prior (any width via masking)
# =============================================================================

def train_zeroshot_emitter(*, d_max=32, H=4, Q=3, steps=4000, B=24, n_q=64, lr=2e-3, seed=0,
                           train_w=False, binary_frac=0.3, d_min=2, device=None, log_every=0):
    """Meta-train one row-token emitter on the broadened self-consistent prior at padded width
    ``d_max``: each task activates a random ``d_active`` of the columns (the rest masked), and a
    fraction of tasks are binarized, so one checkpoint serves regression and classification at any
    feature count. ``train_w=True`` trains the shared interaction geometry W on a slow timescale
    (the prior tracks the drifting W via ``net.W.detach()``, so it stays well specified); the
    default freezes W. Returns the trained net."""
    import torch
    import torch.nn.functional as F
    device = _device(device)
    torch.manual_seed(seed)
    net = MetaMSSKM(max_features=d_max, H=H, Q=Q, n_quad=6, d_phi=64, decode="krr", pool="pma",
                    seed=seed).to(device)
    net.W.requires_grad_(train_w)
    net.train()
    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.Adam((p for p in net.parameters() if p.requires_grad), lr=lr)
    choices = torch.as_tensor((32, 64, 128, 256), device=device)
    for step in range(steps):
        k = int(choices[torch.randint(len(choices), (1,), generator=gen, device=device)].item())
        measure, s2 = sample_measure_prior(B, d_max, H, Q, net.gh_nodes, net.gh_wts,
                                           net.W.detach(), gen, device=device)
        da = int(torch.randint(d_min, d_max + 1, (1,), generator=gen, device=device).item())
        fmask = (torch.arange(d_max, device=device) < da).float()[None, :].expand(B, d_max).contiguous()
        measure.scale = measure.scale * fmask
        Xc, yc, Xq, yq = sample_gp_tasks(measure, k, n_q, s2, gen, feature_mask=fmask, device=device)
        if torch.rand(1, generator=gen, device=device).item() < binary_frac:
            yc, yq = binarize(yc, yq)
        opt.zero_grad()
        loss = F.mse_loss(net(Xq, Xc, yc, feature_mask=fmask), yq)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in net.parameters() if p.requires_grad), 5.0)
        opt.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"  step {step+1:5d}  MSE {loss.item():.4f}  (k={k}, d_active={da}, train_w={train_w})")
    return net


# =============================================================================
# Zero-shot prediction and per-dataset fitted baselines
# =============================================================================

def zeroshot_predict(net, Xtr, ytr, Xte, kind, *, d_max, ctx_cap=512, n_test=3000, seed=0,
                     device=None):
    """One frozen forward pass: context = (subsample of) training rows, padded to d_max with a mask;
    emit a measure; decode the test queries by in-context KRR. Regression standardizes/destandardizes
    the target by context stats; classification feeds {0,1} and returns the continuous score."""
    import torch
    device = _device(device)
    net.eval()
    rng = np.random.default_rng(seed)
    if len(Xtr) > ctx_cap:
        idx = rng.choice(len(Xtr), ctx_cap, replace=False)
        Xc_np, yc_np = Xtr[idx], ytr[idx]
    else:
        Xc_np, yc_np = Xtr, ytr
    Xte2, yte2 = Xte, None
    if len(Xte) > n_test:
        jdx = rng.choice(len(Xte), n_test, replace=False)
        Xte2 = Xte[jdx]
    Xc_p, fmask = _pad(Xc_np, d_max)
    Xq_p, _ = _pad(Xte2, d_max)
    fm = torch.as_tensor(fmask, device=device)[None]
    Xc = torch.as_tensor(Xc_p, device=device)[None]
    Xq = torch.as_tensor(Xq_p, device=device)[None]
    yc = torch.as_tensor(yc_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        if kind == "reg":
            mu, sd = yc.mean(), yc.std().clamp_min(1e-6)
            pred = net(Xq, Xc, ((yc - mu) / sd)[None, :, None], feature_mask=fm)[0, :, 0]
            pred = pred.cpu().numpy() * sd.item() + mu.item()
        else:
            pred = net(Xq, Xc, yc[None, :, None], feature_mask=fm)[0, :, 0].cpu().numpy()
    return pred, (jdx if len(Xte) > n_test else slice(None))


def _score(pred, yte, kind):
    from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
    if kind == "reg":
        return {"metric": "R2", "score": float(r2_score(yte, pred))}
    return {"metric": "acc", "score": float(accuracy_score(yte, (pred > 0.5).astype(int))),
            "auc": float(roc_auc_score(yte, pred))}


def fitted_baselines(Xtr, ytr, Xte, yte, kind, *, seed=0, ch8_steps=500, cb_iters=400):
    """Per-dataset baselines that DO fit the data: a spectral-Laplace kernel machine (Ch. 8) and
    gradient-boosted trees (CatBoost). Returns {ch8, catboost} scores."""
    from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
    from lkbook.chapters import ch08
    out = {}
    # Chapter-8 fitted kernel machine (regression fit; classification fit on 0/1 then threshold)
    _, pred8 = ch08.fit_spectral(Xtr, ytr, mode="learned", H=2, K=8, steps=ch8_steps, seed=seed)
    p8 = pred8(Xte)
    out["ch8"] = ({"metric": "R2", "score": float(r2_score(yte, p8))} if kind == "reg"
                  else {"metric": "acc", "score": float(accuracy_score(yte, (p8 > 0.5).astype(int))),
                        "auc": float(roc_auc_score(yte, np.clip(p8, 0, 1)))})
    # CatBoost
    if kind == "reg":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=cb_iters, depth=6, learning_rate=0.05, verbose=0,
                              random_seed=seed)
        m.fit(Xtr, ytr)
        out["catboost"] = {"metric": "R2", "score": float(r2_score(yte, m.predict(Xte)))}
    else:
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=cb_iters, depth=6, learning_rate=0.05, verbose=0,
                               random_seed=seed)
        m.fit(Xtr, ytr)
        proba = m.predict_proba(Xte)[:, 1]
        out["catboost"] = {"metric": "acc", "score": float(accuracy_score(yte, (proba > 0.5).astype(int))),
                           "auc": float(roc_auc_score(yte, proba))}
    return out


def zeroshot_table(net, *, datasets=("diabetes", "breast_cancer", "california"), d_max=32,
                   seed=0, device=None):
    """The Table-real comparison: a frozen emitter zero-shot vs per-dataset fitted baselines, across
    several real datasets. Returns {dataset: {zeroshot, ch8, catboost}}."""
    device = _device(device)
    out = {}
    for name in datasets:
        Xtr, ytr, Xte, yte, kind, _ = _load_real(name, seed=seed)
        pred, jdx = zeroshot_predict(net, Xtr, ytr, Xte, kind, d_max=d_max, seed=seed, device=device)
        yte_eval = yte[jdx] if not isinstance(jdx, slice) else yte
        zs = _score(pred, yte_eval, kind)
        base = fitted_baselines(Xtr, ytr, Xte, yte, kind, seed=seed)
        out[name] = {"kind": kind, "zeroshot": zs, **base}
    return out


# =============================================================================
# The probe: meta-learner vs prior  (gap is the prior, not the architecture)
# =============================================================================

def probe_california(synthetic_r2, *, steps=3000, seed=0, device=None):
    """Resolve the California gap. The same architecture, meta-trained on resamples of the California
    training split (real labels, no per-dataset gradient steps at test), reaches the fitted ceiling;
    the synthetic-prior emitter does not. So the gap is prior mismatch, not a meta-learner limit.
    ``synthetic_r2`` is the frozen synthetic-prior emitter's California zero-shot R^2 (passed in from
    the same width-32 checkpoint used elsewhere, for consistency). Reuses
    ``ch12.ceiling_incontext_real`` for the in-context-on-real ceiling and fits CatBoost + a Ch. 8
    kernel machine as baselines."""
    from sklearn.metrics import r2_score
    from catboost import CatBoostRegressor
    from lkbook.chapters import ch08
    device = _device(device)
    ceil = ch12.ceiling_incontext_real(steps=steps, seed=seed, device=device)  # in-context-on-real
    Xtr, ytr, Xte, yte, _, _ = _load_real("california", seed=seed)
    cb = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, verbose=0, random_seed=seed)
    cb.fit(Xtr, ytr)
    return {"synthetic": float(synthetic_r2), "in_context_real": ceil[512][0],
            "in_context_real_2048": ceil[1024][0], "ch8_ceiling": ceil["ch8"],
            "catboost": float(r2_score(yte, cb.predict(Xte)))}


# =============================================================================
# The geometry lever: training W overfits the prior and collapses transfer
# =============================================================================

def context_lever(net=None, *, d_max=32, steps=4000, seed=0, device=None,
                  contexts=(128, 256, 512, 1024, 2048)):
    """California zero-shot R^2 of one frozen emitter as the context subsample grows. More context
    helps where the subsample is the binding constraint, but plateaus below the fitted ceiling --- the
    residual is the prior's functional class (too smooth and additive for California's interactions),
    not the amount of data. Returns {context: R2}."""
    device = _device(device)
    if net is None:
        net = train_zeroshot_emitter(d_max=d_max, steps=steps, seed=seed, device=device)
    Xtr, ytr, Xte, yte, kind, _ = _load_real("california", seed=seed)
    out = {}
    for cap in contexts:
        pred, jdx = zeroshot_predict(net, Xtr, ytr, Xte, kind, d_max=d_max, ctx_cap=cap,
                                     seed=seed, device=device)
        yte_eval = yte[jdx] if not isinstance(jdx, slice) else yte
        out[cap] = _score(pred, yte_eval, kind)["score"]
    return out


def read_relevances(net, *, dataset="california", d_max=32, ctx_cap=512, seed=0, device=None):
    """Emit a measure from a real context and read its per-feature ARD relevances --- the readable
    diagnostic. A smooth, near-uniform relevance profile on an interaction-heavy table is the visible
    signature of prior mismatch. Returns (names, normalized relevances over the active features)."""
    import torch
    device = _device(device)
    net.eval()
    Xtr, ytr, _, _, kind, names = _load_real(dataset, seed=seed)
    d = Xtr.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xtr), min(ctx_cap, len(Xtr)), replace=False)
    Xc_p, fmask = _pad(Xtr[idx], d_max)
    fm = torch.as_tensor(fmask, device=device)[None]
    Xc = torch.as_tensor(Xc_p, device=device)[None]
    yc = torch.as_tensor(ytr[idx], dtype=torch.float32, device=device)
    if kind == "reg":
        yc = (yc - yc.mean()) / yc.std().clamp_min(1e-6)
    with torch.no_grad():
        s = net.emit(Xc, yc[None, :, None], feature_mask=fm).scale[0].cpu().numpy()[:min(d, d_max)]
    s = s / (s.sum() + 1e-12)
    return names[:len(s)], s


# =============================================================================
# Aggregate driver
# =============================================================================

def run_all(*, steps=3000, zs_steps=4000, seed=0, device=None):
    """Train a width-32 zero-shot emitter (reused for the Table-real comparison and the context
    lever); run the California probe. Returns {table, probe, lever, zs_net}."""
    device = _device(device)
    net32 = train_zeroshot_emitter(d_max=32, steps=zs_steps, seed=seed, device=device)
    table = zeroshot_table(net32, d_max=32, seed=seed, device=device)
    lever = context_lever(net32, d_max=32, seed=seed, device=device)
    probe = probe_california(lever[512], steps=steps, seed=seed, device=device)
    return {"table": table, "probe": probe, "lever": lever, "zs_net": net32}


# =============================================================================
# Figures
# =============================================================================

def make_zeroshot_figure(table=None, **kw):
    """Figure 14.1 --- zero-shot (frozen) vs fitted kernel machine vs gradient boosting across real
    datasets: competitive where the prior matches, a clear gap on interaction-heavy California."""
    import matplotlib.pyplot as plt
    if table is None:
        table = run_all(**kw)["table"]
    names = list(table)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(names))
    w = 0.26

    def val(d, key):
        s = d[key]
        return s.get("auc", s["score"]) if d["kind"] == "clf" else s["score"]

    zs = [val(table[n], "zeroshot") for n in names]
    k8 = [val(table[n], "ch8") for n in names]
    cb = [val(table[n], "catboost") for n in names]
    ax.bar(x - w, zs, w, color="#c98a3b", label="zero-shot (frozen)", edgecolor="white")
    ax.bar(x, k8, w, color="#2e7d32", label="fitted kernel machine", edgecolor="white")
    ax.bar(x + w, cb, w, color="#3b6fb6", label="gradient boosting", edgecolor="white")
    labels = [f"{n}\n({table[n]['zeroshot']['metric'] if table[n]['kind']=='reg' else 'AUC'})"
              for n in names]
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("$R^2$ (regression) / AUC (classification)")
    ax.set_title("Zero-shot vs per-dataset-fitted models:\ncompetitive where the prior matches, a gap where it does not",
                 fontsize=10)
    ax.legend(fontsize=9)
    for xi, v in zip(np.r_[x - w, x, x + w], zs + k8 + cb):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout()
    return fig


def make_probe_figure(probe=None, lever=None, **kw):
    """Figure 14.2 --- the gap is the prior, not the architecture.
    (left) California R^2: the synthetic-prior emitter zero-shot, the same architecture meta-trained
    on California resamples (in-context, zero test-time fitting), the fitted-kernel ceiling and
    gradient boosting --- the probe jump to the ceiling. (right) the context lever: California
    zero-shot of the frozen synthetic-prior emitter rises with context but plateaus below the
    ceiling, so the residual is the prior's functional class, not the amount of data."""
    import matplotlib.pyplot as plt
    if probe is None or lever is None:
        r = run_all(**kw); probe = probe or r["probe"]; lever = lever or r["lever"]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))

    labels = ["synthetic\nprior", "in-context\non California", "fitted KM\n(ceiling)", "gradient\nboosting"]
    vals = [probe["synthetic"], probe["in_context_real_2048"], probe["ch8_ceiling"], probe["catboost"]]
    cols = ["#c98a3b", "#3b6fb6", "#2e7d32", "#7aa6c2"]
    ax[0].bar(range(4), vals, width=0.6, color=cols, edgecolor="white")
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(labels, fontsize=9)
    ax[0].set_ylabel("California held-out $R^2$")
    ax[0].set_ylim(0, max(vals) * 1.18)
    ax[0].set_title("The probe: the gap is the prior, not the meta-learner", fontsize=10)
    for xi, v in zip(range(4), vals):
        ax[0].text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    caps = sorted(lever)
    rs = [lever[c] for c in caps]
    ax[1].plot(caps, rs, "o-", color="#3b6fb6", label="frozen emitter, zero-shot")
    ax[1].axhline(probe["ch8_ceiling"], color="#2e7d32", ls="--", lw=1.2, label="fitted ceiling")
    ax[1].set_xscale("log", base=2); ax[1].set_xticks(caps); ax[1].set_xticklabels(caps)
    ax[1].set_xlabel("context size")
    ax[1].set_ylabel("California zero-shot $R^2$")
    ax[1].set_ylim(0, probe["ch8_ceiling"] * 1.1)
    ax[1].set_title("The context lever: more context helps,\nbut plateaus below the ceiling (prior-limited)",
                    fontsize=10)
    ax[1].legend(fontsize=9)
    for c, v in zip(caps, rs):
        ax[1].text(c, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig


# =============================================================================
# CLI
# =============================================================================

def main(argv=None):
    p = argparse.ArgumentParser(description="Chapter 14 — zero-shot tabular foundation models")
    p.add_argument("--out-prefix", default=None)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args(argv)
    from lkbook import set_style
    set_style()
    device = "cpu" if args.cpu else _device()
    res = run_all(steps=args.steps, zs_steps=args.steps + 1000, device=device)
    print("zero-shot vs fitted (Table real):")
    for n, d in res["table"].items():
        z = d["zeroshot"]; m = z["metric"]
        extra = f" AUC {z['auc']:.3f}" if d["kind"] == "clf" else ""
        print(f"  {n:14s} zero-shot {z['score']:.3f}{extra}   ch8 {d['ch8']['score']:.3f}   "
              f"catboost {d['catboost']['score']:.3f}")
    pr = res["probe"]
    print(f"probe (California R2): synthetic {pr['synthetic']:.3f}  in-context-real "
          f"{pr['in_context_real_2048']:.3f}  ch8 {pr['ch8_ceiling']:.3f}  catboost {pr['catboost']:.3f}")
    lv = res["lever"]
    print("context lever (California R2):", {c: round(v, 3) for c, v in lv.items()})
    if args.out_prefix:
        import matplotlib
        matplotlib.use("Agg")
        make_zeroshot_figure(res["table"]).savefig(f"{args.out_prefix}1_zeroshot.pdf")
        make_probe_figure(res["probe"], res["lever"]).savefig(f"{args.out_prefix}2_probe.pdf")
        print("wrote figures with prefix", args.out_prefix)


if __name__ == "__main__":
    main()
