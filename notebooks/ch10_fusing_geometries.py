# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Chapter 10 — Fusing geometries
#
# *Companion notebook to **The Learned Kernel**, Ch. 10. Run top to bottom.*
#
# Chapter 9 found the spectral kernel and the tree leaf kernel **complementary**: the spectral
# kernel owns smooth/periodic/high-order structure, the tree owns sharp axis-aligned thresholds,
# and a real dataset carries some of each. So you do not *pick* — you **fuse**, and let the data
# weigh the geometries on a held-out fold.
#
# Fusion shows up at **two levels**, and we build both:
# 1. **Within a geometry** — the spectral kernel is itself a convex mixture of *H* Laplace banks
#    at different bandwidths; learning the bank weights is fusion of *scales*.
# 2. **Across geometries** — blend that spectral kernel with a **tuned CatBoost** leaf kernel on
#    a simplex. Same mechanism, one level up.
#
# **The frame** — *what is learned · how scored · what you read off.* What: a point on the simplex
# (channel weights) plus one ridge. Scored: leakage-free query-fold risk (Ch. 7) — every channel
# fit on the support fold, the weights chosen on a held-out query fold. Read off: the earned
# weights and the per-channel **shares**, via an *exact* additive decomposition of the fit.

# %% [markdown]
# ## Setup

# %%
# On Google Colab (or any fresh env) install the companion package; no-op locally.
try:
    import lkbook  # noqa: F401
except ModuleNotFoundError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
        "learned-kernel[notebooks] @ git+https://github.com/asudjianto-xml/Learned-Kernel.git"],
        check=True)

# %%
# %matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lkbook import set_style
from lkbook.chapters import ch10

set_style()

# %% [markdown]
# ## 10.1  Fit the fusion on Bike Sharing
#
# Two channels: the **tuned CatBoost** tree (raw features) and the **spectral-Laplace** kernel
# (cyclical hour/month/weekday Fourier-encoded so a periodic kernel can see the cycles). Each is
# fit on the support fold; the cross-geometry weights and the ridge are selected on the held-out
# query fold. We run a single seed live (the book averages settings over a larger train set).

# %%
bike = ch10.run_bikeshare(n_train=1200)        # one leakage-free fit, single seed
print("cross-geometry weights :", {k: round(v, 3) for k, v in bike["weights"].items()})
print("fused R^2              :", round(bike["fused_r2"], 4))
print("pure-channel R^2       :", {k: round(v, 4) for k, v in bike["single_r2"].items()})
print("component shares rho_c :", {k: round(v, 3) for k, v in bike["shares"].items()})
# The tuned tree (a strong stand-alone model) earns only a sliver: given the spectral channel,
# more tree weight raises query error. The fused R^2 edges the best single channel.

# %% [markdown]
# ## 10.2  Level one — fusion *within* the spectral kernel
#
# The spectral channel is a convex mixture of *H* Laplace banks at learned bandwidths `T_h` with
# learned simplex weights `w_h` — fusion of *scales* inside one kernel (Ch. 8). This is the same
# convex-mixture move we make across geometries, one level down. The mass concentrates on the
# broadest bank, with finer banks filling in.

# %%
bw, bT = bike["bank_weights"], bike["bank_T"]
print("spectral bank fusion (within the spectral kernel):")
for i, (w, T) in enumerate(zip(bw, bT), 1):
    print(f"   bank {i}:  weight w_h = {w:.3f}   bandwidth T_h = {T:.3f}")
# Fusing banks is a *range* axis, not an order axis (Ch. 9): the Aronszajn sum is a union of
# function spaces, so it costs sqrt(log H / n) and raises no interaction order.

# %% [markdown]
# ## 10.3  The kernels and their mixture are PSD with unit diagonal
#
# Each channel block is symmetric PSD with a unit diagonal, so any convex mixture is too — which
# is what keeps the ridge `lambda` identifiable (Ch. 3 scale degeneracy).

# %%
fm = bike["model"]
Xs = {ch.name: fm.Xs_by[ch.name] for ch in fm.channels}
for ch in fm.channels:
    K = ch.block(Xs[ch.name], Xs[ch.name])
    print(f"{ch.name:9s}: diag mean {np.diag(K).mean():.3f}, "
          f"min eig {np.linalg.eigvalsh((K + K.T) / 2).min():.2e}  (>= 0 => PSD)")
Kmix = ch10.mix_n([ch.block(Xs[ch.name], Xs[ch.name]) for ch in fm.channels], fm.w)
print(f"mixture  : diag mean {np.diag(Kmix).mean():.3f}, "
      f"min eig {np.linalg.eigvalsh((Kmix + Kmix.T) / 2).min():.2e}")

# %% [markdown]
# ## 10.4  The exact additive decomposition
#
# Model `y = sum_c g_c + noise` with independent priors `g_c ~ N(0, alpha_c K_c)`. Each
# component's posterior mean is `g_c = alpha_c K_c (K_alpha + lambda I)^{-1} y`, and the
# components sum to the fused fit **exactly**. Take one held-out hour and decompose it.

# %%
print(f"intercept              : {bike['intercept']:.4f}")
print(f"tree contribution      : {bike['decomp']['tree']:+.4f}")
print(f"spectral contribution  : {bike['decomp']['spectral']:+.4f}")
recon = bike["intercept"] + sum(bike["decomp"].values())
print(f"reconstruction sum     : {recon:.6f}")
print(f"model prediction       : {bike['pred0']:.6f}")
assert abs(recon - bike["pred0"]) < 1e-6      # exact, not approximate
print("=> components reconstruct the prediction to machine precision")

# %%
ch10.make_decomposition_figure(bike=bike)
plt.show()

# %% [markdown]
# ## 10.5  Fusion is **not** output-averaging
#
# Averaging applies *M* independently-inverted ridge smoothers and combines the outputs; fusion
# applies *one* ridge to the summed kernel. Since the ridge filter `g(t)=t/(t+lambda)` is
# operator-concave, matrix Jensen gives the Loewner domination `S_fuse >= S_avg`: the fused
# smoother passes at least as much of every eigendirection. On a smooth-plus-sharp synthetic
# target the fused predictor beats averaging.

# %%
X, y = ch10.smooth_to_sharp(0.5)
n = len(y); perm = np.random.RandomState(7).permutation(n); nte = n // 4
te, tr = perm[:nte], perm[nte:]
fa = ch10.fusion_vs_averaging(ch10.same_reps(X[tr], ["tree", "spectral"]), y[tr],
                              ch10.same_reps(X[te], ["tree", "spectral"]), y[te])
print(f"min eig of (S_fuse - S_avg): {fa['eig_min']:.2e}   (>= 0 => Loewner domination)")
print(f"fused   test RMSE          : {fa['fused_rmse']:.4f}")
print(f"averaged test RMSE         : {fa['avg_rmse']:.4f}")

# %% [markdown]
# ## 10.6  Soft tree gates → the hard leaf kernel as τ → ∞
#
# The hard leaf kernel is a step function of the thresholds — no gradient. Replace each split by a
# sigmoid gate of sharpness τ; a leaf's membership is the path-conjunction product. As τ → ∞ the
# soft co-membership kernel converges to the exact leaf kernel, so finite τ is a differentiable
# relaxation around the hard anchor — the bridge to learning the tree geometry end to end.

# %%
fid = ch10.soft_tree_fidelity()
for tau, gap in fid:
    print(f"  tau = {tau:4d}   ||K_soft - K_hard||_F / ||K_hard||_F = {gap:.4f}")

# %% [markdown]
# ## 10.7  Explore: decompose any held-out hour
#
# The fusion is fit once, at the leakage-free setting selected in §10.1, and the exact additive
# decomposition then explains every prediction with no refitting. Pick a held-out Bike-Sharing
# hour and read its prediction as intercept + tree + spectral. (The book also sweeps a synthetic
# target from smooth to sharp to show the earned weight tracking the geometry; that sweep refits
# the fusion many times, so it is left to `lkbook.chapters.ch10.vertex_sweep` rather than run here.)

# %%
from ipywidgets import interact, IntSlider
from lkbook import load_bikeshare

d = load_bikeshare()
_, reps_te, _ = ch10._bike_reps(d)


def decompose_hour(i=0):
    reps_i = {nm: reps_te[nm][i:i + 1] for nm in fm.names}
    contribs, intercept = fm.channel_contributions(reps_i)
    parts = {c: float(np.atleast_1d(v)[0]) for c, v in contribs.items()}
    pred = float(fm.predict(reps_i)[0])
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    labels = ["intercept"] + list(parts) + ["= prediction"]
    running = intercept
    ax.bar(0, intercept, 0.7, color="#999999")
    for k, (c, v) in enumerate(parts.items(), 1):
        ax.bar(k, v, 0.7, bottom=running, color=ch10.TREE_C if c == "tree" else ch10.SPECTRAL_C)
        running += v
    ax.bar(len(parts) + 1, pred, 0.7, color="#222222")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("log rides/hr")
    ax.set_title(f"held-out hour {i}: intercept + tree + spectral = {pred:.3f}", fontsize=10)
    plt.show()


interact(decompose_hour, i=IntSlider(min=0, max=499, step=1, value=0, description="hour"));

# %% [markdown]
# ## Exercises
#
# Fill in each `# TODO`; the solution is one click away.

# %% [markdown]
# **(easy)** Confirm the fused fit is decomposed *exactly*. Sum the per-channel contributions plus
# the intercept on a batch of held-out rows and compare to `fm.predict`.

# %%
# TODO: build reps for the first few test rows, call fm.channel_contributions, and check the sum
recon = pred = None
print(recon, pred)

# %% [markdown]
# <details><summary>Solution</summary>
#
# ```python
# from lkbook import load_bikeshare
# d = load_bikeshare()
# _, reps_te, _ = ch10._bike_reps(d)
# reps5 = {nm: reps_te[nm][:5] for nm in fm.names}
# contribs, intercept = fm.channel_contributions(reps5)
# recon = intercept + sum(np.atleast_1d(v) for v in contribs.values())
# pred = fm.predict(reps5)
# print(np.max(np.abs(recon - pred)))   # ~1e-13: the decomposition IS the model
# ```
# The components are Gaussian posterior means whose sum is `K_alpha (K_alpha + lambda I)^{-1} y`,
# the fused smoother applied to `y` — so they reconstruct the prediction exactly, not approximately.
# </details>

# %% [markdown]
# **(⋆)** Verify the Loewner domination is **strict** at an interior mixture but an **equality**
# at a vertex. Re-run `fusion_vs_averaging` with `w=[1.0, 0.0]` (a vertex) and confirm the min
# eigenvalue of `S_fuse - S_avg` is ~0 there, while it is also >= 0 at the interior `w=[0.5,0.5]`.

# %%
# TODO: call ch10.fusion_vs_averaging twice with different w and compare eig_min
eig_vertex = eig_interior = None
print(eig_vertex, eig_interior)

# %% [markdown]
# <details><summary>Solution</summary>
#
# ```python
# args = (ch10.same_reps(X[tr], ["tree", "spectral"]), y[tr],
#         ch10.same_reps(X[te], ["tree", "spectral"]), y[te])
# eig_vertex   = ch10.fusion_vs_averaging(*args, w=[1.0, 0.0])["eig_min"]
# eig_interior = ch10.fusion_vs_averaging(*args, w=[0.5, 0.5])["eig_min"]
# print(f"vertex {eig_vertex:.2e}   interior {eig_interior:.2e}")
# # at a vertex S_fuse and S_avg coincide (a single channel), so the gap is exactly 0;
# # at an interior point the operator-concavity inequality is strict (>= 0, generically > 0).
# ```
# `g(t) = t/(t+lambda)` is operator-concave, so `g(sum_c w_c K_c) >= sum_c w_c g(K_c)` in the
# Loewner order, with equality only when the combination is degenerate (a vertex) or the kernels
# coincide. Averaging the *fits* throws away the cross-kernel similarity the *fused* kernel keeps.
# </details>

# %% [markdown]
# ---
# *Companion to Chapter 10 of **The Learned Kernel**. Everything here comes from
# `lkbook.chapters.ch10` — the same code the book's figures come from. The tree channel is a
# hyperparameter-tuned CatBoost read as the Ch. 4 leaf kernel; the spectral channel reuses Ch. 8's
# learned spectral-Laplace kernel (`ch08.fit_spectral`, learned mode). The book reports the Bike
# Sharing fit on a larger train set; here we run a single seed on a smaller sample live, so the
# exact weights may differ by a grid step.*
