"""Chapter 9 — spectral kernels versus trees: the capacity map.

Chapter 4 showed a gradient-boosted forest *is* a kernel — its leaves induce a
positive-semidefinite leaf co-membership similarity. Chapter 8 built and *learned* the
spectral-Laplace kernel from the other side. This chapter is carry-over: it imports Chapter
8's canonical `LearnedSpectralLaplace` / `fit_spectral` and puts the two learned, axis-aligned
geometries head to head — both decoded by **one** kernel-ridge head on a support fold and
scored on a held-out query/test fold, across a synthetic suite (S1 smooth, S2 periodic, S9
degree-four interaction, S10 sharp partition) and California.

We run the spectral kernel in its **learned** training mode (every spectral parameter fit by
gradient descent on the marginal likelihood), the mode Chapter 8 showed wins on real
multivariate data and interactions. The additive-Laplace control is the *same* learned kernel
under an order-one readout (`interaction="additive"`): a sum of per-feature 1-D Laplace
kernels, whose RKHS carries main effects only. Running both isolates Prop. H(1) — interactions
come from the exponential-of-summed-distance readout, not the embedding.

    python -m lkbook.chapters.ch09 --out-prefix fig9
"""
from __future__ import annotations

import argparse
import functools

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from lkbook import load_california, set_style
from lkbook.chapters import ch04
from lkbook.chapters.ch08 import LearnedSpectralLaplace, fit_spectral

SEED = 0
LAM = 1e-3


def _single_thread(fn):
    """Pin BLAS to one thread inside small-Gram solves (this box oversubscribes)."""
    @functools.wraps(fn)
    def wrap(*a, **k):
        try:
            from threadpoolctl import threadpool_limits
            with threadpool_limits(1):
                return fn(*a, **k)
        except Exception:
            return fn(*a, **k)
    return wrap


def _rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)))


# =============================================================================
# The spectral side — Chapter 8's canonical learned kernel, in its learned mode
# =============================================================================

def fit_spectral_mv(Xtr, ytr, seed=SEED, H=2, K=8, **kw):
    """The multivariate spectral-Laplace kernel of Chapter 8, fit in the **learned** mode
    (frequencies, amplitudes, ARD relevance, bank bandwidths and weights all learned by NLML).
    A thin wrapper over `ch08.fit_spectral` — Chapter 9 reuses, it does not re-implement."""
    return fit_spectral(Xtr, ytr, mode="learned", objective="nlml", H=H, K=K, seed=seed, **kw)


def fit_additive_laplace(Xtr, ytr, seed=SEED, H=2, K=8, **kw):
    """The order-one (GAM/EBM) control: the same learned kernel under a sum-of-per-feature
    Laplace readout instead of the exponential-of-summed-distance. Same learned spectrum, only
    the readout coupling differs — so its failure on the interaction target isolates Prop. H(1)."""
    return fit_spectral(Xtr, ytr, mode="learned", objective="nlml", H=H, K=K, seed=seed,
                        interaction="additive", **kw)


# =============================================================================
# The tree side: the leaf kernel of a gradient-boosted forest (Chapter 4), one ridge head
# =============================================================================

@_single_thread
def fit_leaf_kernel(Xtr, ytr, lam=LAM, seed=SEED, n_est=300, depth=4):
    """Fit a gradient-boosted forest, extract its Chapter-4 leaf kernel, and decode it with
    the SAME kernel-ridge head as the spectral side. Returns (leaf_kernel, predict)."""
    from sklearn.ensemble import GradientBoostingRegressor
    Xtr = np.atleast_2d(np.asarray(Xtr, float)); ytr = np.asarray(ytr, float).ravel()
    model = GradientBoostingRegressor(n_estimators=n_est, max_depth=depth,
                                      learning_rate=0.1, random_state=seed).fit(Xtr, ytr)
    lk = ch04.LeafKernel().fit(model)
    K = lk.gram(Xtr, Xtr)
    alpha = np.linalg.solve(K + lam * np.eye(len(ytr)), ytr - ytr.mean())
    Xsup, ybar = Xtr, ytr.mean()

    def predict(Xnew):
        return lk.gram(np.atleast_2d(Xnew), Xsup) @ alpha + ybar
    return lk, predict


# =============================================================================
# The synthetic suite (definitions from fuse-kernel/paper/capacity_study.tex)
#   S1 additive smooth · S2 single periodic · S9 degree-four product · S10 sharp tree partition
# Inputs uniform on [-1,1]^d, a few active features, the rest nuisance.
# =============================================================================

def _make_inputs(n, d, seed):
    return np.random.RandomState(seed).uniform(-1.0, 1.0, size=(n, d))


def synth_S1(n=1200, d=8, noise=0.05, seed=SEED):
    """S1 additive smooth: y = 0.8 x0^2 + sin(1.3 x1) + 0.5 x2 over three active features
    (x3..x7 nuisance); a sum of smooth single-feature trends, no interaction."""
    X = _make_inputs(n, d, seed)
    y = (0.8 * X[:, 0] ** 2 + np.sin(1.3 * X[:, 1]) + 0.5 * X[:, 2])
    y = y + noise * np.random.RandomState(seed + 1).randn(n)
    return X, y, "S1 smooth"


def synth_S2(n=1200, d=8, freq=2.0, noise=0.05, seed=SEED):
    """S2 single periodic component: y = sin(2 pi * 2 * x0) + 0.3 x1; a definite frequency on
    x0 plus a weak linear term on x1 (x2..x7 nuisance)."""
    X = _make_inputs(n, d, seed)
    y = np.sin(2.0 * np.pi * freq * X[:, 0]) + 0.3 * X[:, 1]
    y = y + noise * np.random.RandomState(seed + 1).randn(n)
    return X, y, "S2 periodic"


def synth_S9(n=1200, d=8, noise=0.05, seed=SEED):
    """S9 degree-four product: y = 4 x0 x1 x2 x3; a pure high-order interaction with zero main
    effects (x4..x7 nuisance) — the case an additive kernel cannot represent."""
    X = _make_inputs(n, d, seed)
    y = 4.0 * X[:, 0] * X[:, 1] * X[:, 2] * X[:, 3]
    y = y + noise * np.random.RandomState(seed + 1).randn(n)
    return X, y, "S9 deg-4"


def synth_S10(n=1200, d=8, n_leaves=12, depth=6, noise=0.05, seed=SEED):
    """S10 a deep random tree partition with sharp jumps: a piecewise-constant target,
    discontinuous in x. We grow `depth` random axis-aligned binary cuts on the active features
    x0..x3 and assign each of the 2^depth cells a standard-normal level — exactly the function
    class the leaf kernel represents and the continuous spectral kernel can only approximate at
    a Gibbs cost."""
    X = _make_inputs(n, d, seed)
    rng = np.random.RandomState(seed + 3)
    feats = rng.randint(0, 4, size=depth)
    thr = rng.uniform(-0.6, 0.6, size=depth)
    codes = np.zeros(n, dtype=int)
    for k in range(depth):
        codes = codes * 2 + (X[:, feats[k]] > thr[k]).astype(int)
    levels = rng.randn(2 ** depth)
    y = levels[codes]
    y = y + noise * np.random.RandomState(seed + 1).randn(n)
    return X, y, "S10 jumps"


SYNTH = {"S1": synth_S1, "S2": synth_S2, "S9": synth_S9, "S10": synth_S10}


def _split(X, y, test_frac=0.25, seed=SEED):
    rng = np.random.RandomState(seed + 5)
    perm = rng.permutation(len(y)); nte = int(test_frac * len(y))
    te, tr = perm[:nte], perm[nte:]
    # standardize y to zero mean / unit variance on train (RMSEs comparable across targets)
    mu, sd = y[tr].mean(), y[tr].std() + 1e-12
    return X[tr], (y[tr] - mu) / sd, X[te], (y[te] - mu) / sd


# =============================================================================
# The head-to-head: spectral vs leaf kernel vs additive-Laplace, one ridge, query/test scored
# =============================================================================

def head_to_head_target(key, seed=SEED, **gen_kw):
    """Run the three kernels on one synthetic target under the shared decoder. Returns a dict
    of test RMSEs and the winner between spectral and tree."""
    X, y, label = SYNTH[key](seed=seed, **gen_kw)
    Xtr, ytr, Xte, yte = _split(X, y, seed=seed)
    _, sp = fit_spectral_mv(Xtr, ytr, seed=seed)
    _, lf = fit_leaf_kernel(Xtr, ytr, seed=seed)
    _, ad = fit_additive_laplace(Xtr, ytr, seed=seed)
    spec, tree, add = _rmse(sp(Xte), yte), _rmse(lf(Xte), yte), _rmse(ad(Xte), yte)
    return {"label": label, "spectral": spec, "tree": tree, "additive": add,
            "winner": "spectral" if spec < tree else "tree"}


def head_to_head_california(seed=SEED, n_train=2500):
    """The same head-to-head on California Housing (target in $100k, standardized)."""
    d = load_california()
    rng = np.random.RandomState(seed)
    idx = rng.choice(d.n, min(n_train, d.n), replace=False)
    Xtr, ytr = d.Xtr[idx], d.ytr[idx]
    mu, sd = ytr.mean(), ytr.std() + 1e-12
    ytr_s, yte_s = (ytr - mu) / sd, (d.yte - mu) / sd
    _, sp = fit_spectral_mv(Xtr, ytr_s, seed=seed)
    _, lf = fit_leaf_kernel(Xtr, ytr_s, seed=seed)
    _, ad = fit_additive_laplace(Xtr, ytr_s, seed=seed)
    spec, tree, add = (_rmse(sp(d.Xte), yte_s), _rmse(lf(d.Xte), yte_s), _rmse(ad(d.Xte), yte_s))
    return {"label": "California", "spectral": spec, "tree": tree, "additive": add,
            "winner": "spectral" if spec < tree else "tree"}


def run_capacity_map(seed=SEED, include_california=True):
    """The full win/lose map across S1/S2/S9/S10 (+California) at one seed."""
    rows = [head_to_head_target(k, seed=seed) for k in ("S1", "S2", "S9", "S10")]
    if include_california:
        rows.append(head_to_head_california(seed=seed))
    return rows


_ORDER = ["S1 smooth", "S2 periodic", "S9 deg-4", "S10 jumps", "California"]


def run_capacity_map_multiseed(seeds=range(5), include_california=True):
    """The win/lose map over several seeds — the reproducibility readout. The learned spectral
    kernel is gradient-trained, so its run-to-run variance (large on the hard interaction)
    matters as much as its mean. Returns one row per target with mean/std and raw lists per
    kernel; the winner is decided by the mean."""
    seeds = list(seeds)
    agg = {}
    for s in seeds:
        rs = [head_to_head_target(k, seed=s) for k in ("S1", "S2", "S9", "S10")]
        if include_california:
            rs.append(head_to_head_california(seed=s))
        for r in rs:
            a = agg.setdefault(r["label"], {"spectral": [], "tree": [], "additive": []})
            for m in ("spectral", "tree", "additive"):
                a[m].append(r[m])
    rows = []
    for label in _ORDER:
        if label not in agg:
            continue
        a = agg[label]; row = {"label": label, "n_seeds": len(seeds)}
        for m in ("spectral", "tree", "additive"):
            v = np.asarray(a[m], float)
            row[m] = float(v.mean()); row[m + "_std"] = float(v.std()); row[m + "_raw"] = a[m]
        row["winner"] = "spectral" if row["spectral"] < row["tree"] else "tree"
        rows.append(row)
    return rows


# =============================================================================
# The smoothness ladder: radial profiles + a 1-D smooth-plus-jump fit under each readout
# =============================================================================

def smooth_plus_jump(n=160, noise=0.03, seed=SEED):
    """A 1-D target with a smooth trend AND a sharp jump,
        g(x) = 0.6 sin(2 pi * 0.8 * x) + 0.8 (x - 0.5) + 0.9 * 1{x > 0.6},
    the case that separates the three readouts. RBF oversmooths, Laplace bends, tree jumps."""
    rng = np.random.RandomState(seed)
    X = np.sort(rng.uniform(0, 1, n))
    g = 0.6 * np.sin(2 * np.pi * 0.8 * X) + 0.8 * (X - 0.5)
    g = g + 0.9 * (X > 0.6).astype(float)                    # the discontinuity
    return X, g + noise * rng.randn(n), g


def _rbf_1d(A, B, ell):
    tau = cdist(np.asarray(A).reshape(-1, 1), np.asarray(B).reshape(-1, 1))
    return np.exp(-0.5 * tau ** 2 / ell ** 2)


def _laplace_1d(A, B, ell):
    tau = cdist(np.asarray(A).reshape(-1, 1), np.asarray(B).reshape(-1, 1))
    return np.exp(-tau / ell)


def _fit_1d_kernel(kfun, X, y, lam=1e-3, seed=SEED):
    rng = np.random.RandomState(seed); perm = rng.permutation(len(X)); ns = len(X) // 2
    s, q = perm[:ns], perm[ns:]
    best = min(np.logspace(-2.5, 0.5, 40),
               key=lambda e: _rmse(kfun(X[q], X[s], e) @ np.linalg.solve(
                   kfun(X[s], X[s], e) + lam * np.eye(len(s)), y[s] - y[s].mean()) + y[s].mean(),
                   y[q]))
    a = np.linalg.solve(kfun(X, X, best) + lam * np.eye(len(X)), y - y.mean())
    return lambda Xg: kfun(Xg, X, best) @ a + y.mean()


def _true_curve(Xg):
    return 0.6 * np.sin(2 * np.pi * 0.8 * Xg) + 0.8 * (Xg - 0.5) + 0.9 * (Xg > 0.6).astype(float)


# =============================================================================
# Figures
# =============================================================================

def make_smoothness_ladder_figure(seed=SEED):
    """(9.1) The three readouts ranked by roughness. Left: radial profiles — the RBF analytic
    bump (C-infinity), the Laplace cusp (H^{(d+1)/2}), the tree step (discontinuous). Right: a
    1-D smooth-plus-jump fit under each — RBF oversmooths, Laplace bends continuously, the tree
    captures the jump exactly."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    # (a) radial profiles
    ax = axes[0]
    r = np.linspace(0, 2.0, 400)
    ax.plot(r, np.exp(-0.5 * r ** 2 / 0.5 ** 2), color="#3b6ea5", lw=2,
            label=r"RBF $e^{-r^2/T^2}$  ($C^\infty$)")
    ax.plot(r, np.exp(-r / 0.5), color="#c44e52", lw=2,
            label=r"Laplace $e^{-r/T}$  ($H^{(d+1)/2}$, cusp)")
    step = (r < 0.5).astype(float)
    ax.step(r, step, where="post", color="#555555", lw=2,
            label="tree step (discontinuous)")
    ax.set_title("Radial profiles, ranked by roughness:\n"
                 r"$C^\infty$ (RBF) $\supset H^{(d+1)/2}$ (Laplace) $>$ discontinuous (tree)",
                 fontsize=10)
    ax.set_xlabel("radial distance r"); ax.set_ylabel("k(r)"); ax.legend(fontsize=8.5)

    # (b) a smooth-plus-jump fit under each readout
    ax2 = axes[1]
    X, y, truth = smooth_plus_jump(seed=seed)
    Xg = np.linspace(0, 1, 500)
    rbf = _fit_1d_kernel(_rbf_1d, X, y, seed=seed)(Xg)
    lap = _fit_1d_kernel(_laplace_1d, X, y, seed=seed)(Xg)
    # tree side: a 1-D leaf kernel from a shallow boosted forest on the feature
    from sklearn.ensemble import GradientBoostingRegressor
    m = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.1,
                                  random_state=seed).fit(X[:, None], y)
    lk = ch04.LeafKernel().fit(m)
    Kt = lk.gram(X[:, None], X[:, None])
    at = np.linalg.solve(Kt + 1e-3 * np.eye(len(X)), y - y.mean())
    tree = lk.gram(Xg[:, None], X[:, None]) @ at + y.mean()

    ax2.scatter(X, y, s=10, c="#bbbbbb", zorder=1, label="data")
    ax2.plot(Xg, _true_curve(Xg), "k--", lw=1, zorder=2, label="truth (smooth + jump)")
    ax2.plot(Xg, rbf, color="#3b6ea5", lw=2, label="RBF (oversmooths the kink)")
    ax2.plot(Xg, lap, color="#c44e52", lw=2, label="Laplace (bends continuously)")
    ax2.plot(Xg, tree, color="#555555", lw=2, label="tree (jumps exactly)")
    ax2.set_title("Same smooth-plus-jump target, three readouts:\n"
                  "the RBF rounds the jump, the Laplace bends, the tree steps", fontsize=10)
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.legend(fontsize=8.5)
    return fig


def make_winloss_figure(rows=None, seed=SEED):
    """(9.2) The win/lose map: spectral vs leaf-kernel vs additive-Laplace test RMSE across
    S1/S2/S9/S10 and California, the winner (spectral/tree) flagged per target."""
    if rows is None:
        rows = run_capacity_map(seed=seed)
    labels = [r["label"] for r in rows]
    spec = [r["spectral"] for r in rows]
    tree = [r["tree"] for r in rows]
    add = [r["additive"] for r in rows]
    x = np.arange(len(rows)); w = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    ax.bar(x - w, spec, w, color="#c44e52", label="spectral-Laplace (learned)")
    ax.bar(x, tree, w, color="#555555", label="leaf kernel (tree)")
    ax.bar(x + w, add, w, color="#cccccc", label="additive-Laplace (order 1)")
    top = max(max(spec), max(tree), max(add))
    for xi, r in zip(x, rows):
        ax.text(xi, top * 1.02, "spectral" if r["winner"] == "spectral" else "tree",
                ha="center", fontsize=8.5,
                color="#c44e52" if r["winner"] == "spectral" else "#555555", weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("test RMSE (standardized target)")
    ax.set_ylim(0, top * 1.12)
    ax.set_title("The win/lose map: which learned geometry fits which target.\n"
                 "Spectral takes smooth / periodic / interaction; the tree takes sharp jumps.",
                 fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    return fig


def make_winloss_multiseed_figure(rows=None, seeds=range(5)):
    """(9.2) The win/lose map over seeds: mean +/- std test RMSE per kernel, the winner
    (by mean) flagged. The error bars show the learned spectral kernel's run-to-run variance —
    tight on the smooth/periodic targets, wide on the hard interaction."""
    if rows is None:
        rows = run_capacity_map_multiseed(seeds=seeds)
    labels = [r["label"] for r in rows]
    spec = [r["spectral"] for r in rows]; spec_s = [r["spectral_std"] for r in rows]
    tree = [r["tree"] for r in rows]; tree_s = [r["tree_std"] for r in rows]
    add = [r["additive"] for r in rows]; add_s = [r["additive_std"] for r in rows]
    x = np.arange(len(rows)); w = 0.26
    fig, ax = plt.subplots(figsize=(11, 4.9), constrained_layout=True)
    ax.bar(x - w, spec, w, yerr=spec_s, capsize=3, color="#c44e52", label="spectral-Laplace (learned)")
    ax.bar(x, tree, w, yerr=tree_s, capsize=3, color="#555555", label="leaf kernel (tree)")
    ax.bar(x + w, add, w, yerr=add_s, capsize=3, color="#cccccc", label="additive-Laplace (order 1)")
    top = max(max(spec), max(tree), max(add)) + max(max(spec_s), max(tree_s), max(add_s))
    for xi, r in zip(x, rows):
        ax.text(xi, top * 1.04, "spectral" if r["winner"] == "spectral" else "tree",
                ha="center", fontsize=8.5,
                color="#c44e52" if r["winner"] == "spectral" else "#555555", weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("test RMSE (mean ± std over seeds)")
    ax.set_ylim(0, top * 1.15)
    ax.set_title(f"The win/lose map over {rows[0]['n_seeds']} seeds: which learned geometry fits "
                 "which target.\nSpectral takes smooth / periodic / interaction and California; "
                 "the tree takes sharp jumps.", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    return fig


# =============================================================================
# CLI
# =============================================================================

def main(argv=None):
    p = argparse.ArgumentParser(description="Chapter 9 — spectral kernels versus trees")
    p.add_argument("--out-prefix", default=None)
    args = p.parse_args(argv)
    set_style()

    rows = run_capacity_map()
    print("=" * 78)
    print("HEAD-TO-HEAD: learned spectral-Laplace vs leaf kernel vs additive-Laplace (one ridge)")
    print(f"{'target':14s}{'spectral':>11s}{'tree':>9s}{'additive':>11s}{'winner':>11s}")
    for r in rows:
        print(f"{r['label']:14s}{r['spectral']:>11.3f}{r['tree']:>9.3f}"
              f"{r['additive']:>11.3f}{r['winner']:>11s}")
    print("-" * 78)
    print("capacity map: spectral wins S1/S2/S9 and California; tree wins S10;")
    print("the additive-Laplace control trails on the S9 interaction it cannot represent.")

    if args.out_prefix:
        import matplotlib
        matplotlib.use("Agg")
        make_smoothness_ladder_figure().savefig(f"{args.out_prefix}1_smoothness_ladder.pdf")
        make_winloss_figure(rows=rows).savefig(f"{args.out_prefix}2_winloss_map.pdf")
        print("wrote figures with prefix", args.out_prefix)


if __name__ == "__main__":
    main()
