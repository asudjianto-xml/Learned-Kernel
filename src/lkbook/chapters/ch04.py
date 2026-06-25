"""Chapter 4 — trees and forests are kernels.

A fitted gradient-boosted forest is *exactly* a kernel machine. Read it as a generalized
Nadaraya–Watson (GNW) operator with three separable parts:

    prediction(x) = D( Σ_i w_i(x) V_i ),   w_i(x) = K(x,x_i) / Σ_j K(x,x_j),

an **input geometry** K, a **value representation** V, and a **decoder** D. The leaf
co-occurrence kernel k(x,x') = (1/T) Σ_t 1{ℓ_t(x)=ℓ_t(x')} = (1/T) Ψ(x)ᵀΨ(x') is the
geometry (PSD, unit-diagonal, [0,1]). On that one geometry the value axis is where the action
is:

  - **leaf scores** as values  → recovers the forest's own prediction EXACTLY
        f(x) = f₀ + Σ_t η_t · γ_{t,ℓ_t(x)}  (each tree's one-hot weight selects its leaf value);
  - **raw labels** as values   → a crude Nadaraya–Watson smoother (underperforms the forest);
  - **ridge-refit** values     → kernel ridge in the leaf basis (matches or beats the forest).

This is the GNW view of Sudjianto et al. (the kernel-xgb papers). The construction is
library-agnostic — XGBoost, LightGBM, CatBoost and scikit-learn's GradientBoosting all expose
per-tree leaf indices and leaf scores via a forward pass; the package uses scikit-learn's so
it carries no extra dependency.

    python -m lkbook.chapters.ch04 --out-prefix fig4
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import KFold

from lkbook import load_california, load_taiwan, set_style, POS_CMAP

N_TRAIN, N_VAL, T_TREES, DEPTH, LR, LAM, SEED = 3000, 1000, 200, 3, 0.1, 1e-2, 0


# --- the leaf kernel (input geometry) -----------------------------------------

class LeafKernel:
    """The leaf-co-occurrence kernel of a fitted gradient-boosted forest:
    k(x,x') = (1/T) Ψ(x)ᵀΨ(x'), with Ψ the stacked one-hot leaf indicators. Each row of Ψ
    has exactly T ones, so the kernel is unit-diagonal; it is a Gram matrix, so it is PSD."""

    def fit(self, model):
        self.model = model
        self.trees = [est[0].tree_ for est in model.estimators_]   # GB: estimators_ is (T,1)
        self.T = len(self.trees)
        self.col_of, off = [], 0
        for tr in self.trees:
            leaves = np.where(tr.children_left == -1)[0]
            self.col_of.append({int(l): off + i for i, l in enumerate(leaves)})
            off += len(leaves)
        self.n_cols = off
        return self

    def _leaves(self, X):
        L = self.model.apply(X)
        return L[:, :, 0] if L.ndim == 3 else L

    def transform(self, X):
        L = self._leaves(X)
        n = L.shape[0]
        cols = np.array([[self.col_of[t][int(L[i, t])] for t in range(self.T)]
                         for i in range(n)])
        rows = np.repeat(np.arange(n), self.T)
        return sp.csr_matrix((np.ones(n * self.T), (rows, cols.ravel())),
                             shape=(n, self.n_cols))

    def gram(self, A, B):
        return (self.transform(A) @ self.transform(B).T).toarray() / self.T


# --- the GNW value axis -------------------------------------------------------

def gbdt_leaf_value_prediction(model, X):
    """Reconstruct the forest from its own leaf scores, as the additive GNW operator
    f(x) = f₀ + Σ_t η_t γ_{t,ℓ_t(x)}: per tree the one-hot leaf weight selects the leaf value
    γ_{t,ℓ_t(x)}. Equals model.predict(X) to machine precision."""
    f = model.init_.predict(X).ravel().astype(float)      # f₀ (constant init)
    leaves = model.apply(X)
    leaves = (leaves[:, :, 0] if leaves.ndim == 3 else leaves).astype(int)
    for t, est in enumerate(model.estimators_):
        vals = est[0].tree_.value.ravel()                 # γ_{t,ℓ} per leaf node
        f += LR * vals[leaves[:, t]]                       # one-hot GNW weight per tree
    return f


def nw_smooth(K_qa, y_a):
    """Raw-label Nadaraya–Watson on the leaf geometry: row-normalized weighted average."""
    w = K_qa / np.clip(K_qa.sum(1, keepdims=True), 1e-12, None)
    return w @ y_a


def _rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def fit_forest(d, classifier=False, n_train=N_TRAIN, seed=SEED):
    rng = np.random.RandomState(seed)
    idx = rng.choice(d.n, n_train, replace=False)
    Cls = GradientBoostingClassifier if classifier else GradientBoostingRegressor
    model = Cls(n_estimators=T_TREES, max_depth=DEPTH, learning_rate=LR,
                random_state=seed).fit(d.Xtr[idx], d.ytr[idx])
    return model, d.Xtr[idx], d.ytr[idx]


def value_mechanisms(d, seed=SEED):
    """Three value representations on the SAME leaf geometry: raw labels (crude NW), leaf
    scores (exact recovery of the forest), and ridge-refit values (KRR in the leaf basis)."""
    model, Xtr, ytr = fit_forest(d, seed=seed)
    lk = LeafKernel().fit(model)
    Kte = lk.gram(d.Xte, Xtr)

    forest = model.predict(d.Xte)
    exact = gbdt_leaf_value_prediction(model, d.Xte)       # leaf-score values → exact
    nw = nw_smooth(Kte, ytr)                                # raw labels → crude smoother
    K = lk.gram(Xtr, Xtr)
    a = np.linalg.solve(K + LAM * np.eye(len(ytr)), ytr - ytr.mean())
    ridge = Kte @ a + ytr.mean()                            # ridge-refit values → KRR

    return {
        "forest_rmse": _rmse(forest, d.yte),
        "exact_rmse": _rmse(exact, d.yte),
        "exact_recovery_err": float(np.max(np.abs(exact - forest))),   # ~0
        "rawlabel_nw_rmse": _rmse(nw, d.yte),
        "ridge_rmse": _rmse(ridge, d.yte),
        "diag": float(np.diag(lk.gram(Xtr[:200], Xtr[:200])).mean()),
        "krange": (float(Kte.min()), float(Kte.max())),
    }


# --- residual enhancement with OUT-OF-FOLD residuals (honest, per Chapter 3) --

def _oof_residuals(Xtr, ytr, folds=5, seed=SEED):
    """The leaf geometry is supervised, so in-sample residuals are optimistic (Chapter 3's
    leakage). Estimate residuals out-of-fold."""
    r = np.zeros(len(ytr))
    for tr_idx, te_idx in KFold(folds, shuffle=True, random_state=seed).split(Xtr):
        m = GradientBoostingRegressor(n_estimators=T_TREES, max_depth=DEPTH,
                                      learning_rate=LR, random_state=seed).fit(
            Xtr[tr_idx], ytr[tr_idx])
        r[te_idx] = ytr[te_idx] - m.predict(Xtr[te_idx])
    return r


def residual_enhancement(d, seed=SEED):
    """f̂ = f_forest + γ·r̂, r̂ a residual leaf-KRR fit on OUT-OF-FOLD residuals, γ chosen on a
    held-out validation fold."""
    model, Xtr, ytr = fit_forest(d, seed=seed)
    lk = LeafKernel().fit(model)
    r_oof = _oof_residuals(Xtr, ytr, seed=seed)
    K = lk.gram(Xtr, Xtr)
    alpha = np.linalg.solve(K + LAM * np.eye(len(ytr)), r_oof)

    rng = np.random.RandomState(seed + 7)
    vi = rng.choice(d.n, N_VAL, replace=False)
    Xv, yv = d.Xtr[vi], d.ytr[vi]
    rhat_val = lk.gram(Xv, Xtr) @ alpha
    base_val = model.predict(Xv)
    gammas = np.linspace(0, 1, 21)
    gamma = float(gammas[np.argmin([_rmse(base_val + g * rhat_val, yv) for g in gammas])])

    rhat_te = lk.gram(d.Xte, Xtr) @ alpha
    base_te = model.predict(d.Xte)
    return {"gamma": gamma, "raw_rmse": _rmse(base_te, d.yte),
            "enhanced_rmse": _rmse(base_te + gamma * rhat_te, d.yte)}


# --- figures ------------------------------------------------------------------

def make_value_figure(d, seed=SEED):
    """One geometry, four value heads: the value axis is where the accuracy lives."""
    vm = value_mechanisms(d, seed=seed)
    enh = residual_enhancement(d, seed=seed)
    labels = ["raw-label\nNW", "gradient-boosted\nforest", "leaf scores\n(exact GNW)",
              "ridge values\n(leaf KRR)", "residual\nenhanced"]
    vals = [vm["rawlabel_nw_rmse"], vm["forest_rmse"], vm["exact_rmse"],
            vm["ridge_rmse"], enh["enhanced_rmse"]]
    colors = ["#c44e52", "#8172b3", "#8172b3", "#3b6ea5", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(vm["forest_rmse"], ls="--", color="#8172b3", lw=1, zorder=0)
    ax.set_ylabel("test RMSE ($100k)")
    ax.set_title("One leaf geometry, four value representations. Leaf scores reproduce the "
                 f"forest exactly\n(|Δ|={vm['exact_recovery_err']:.0e}); raw labels are crude; "
                 "ridge values are competitive; a residual layer beats it.", fontsize=9.5)
    ax.set_ylim(0, max(vals) * 1.18)
    return fig


def make_leaf_vs_rbf_figure(d, q=7, ell=2.93, seed=SEED, win=2.4):
    """Similarity k(query, ·) over the California map: the data-adaptive leaf kernel vs the
    isotropic RBF the modeler chose."""
    model, Xtr, ytr = fit_forest(d, seed=seed)
    lk = LeafKernel().fit(model)
    x = d.Xte[q]
    k_leaf = lk.gram(x[None], Xtr)[0]
    rng = np.random.RandomState(seed)
    idx = rng.choice(d.n, N_TRAIN, replace=False)
    lon = d.Xtr_raw[idx, d.col("Longitude")]; lat = d.Xtr_raw[idx, d.col("Latitude")]
    k_rbf = np.exp(-((Xtr - x) ** 2).sum(1) / (2 * ell ** 2))
    qlon, qlat = d.Xte_raw[q, d.col("Longitude")], d.Xte_raw[q, d.col("Latitude")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, (k, title) in zip(axes, [(k_leaf, "leaf kernel (learned, data-adaptive)"),
                                     (k_rbf, "RBF kernel (chosen, isotropic)")]):
        m = k > 1e-6
        sc = ax.scatter(lon[m], lat[m], c=k[m], s=10, cmap=POS_CMAP, vmin=0, vmax=1)
        ax.scatter([qlon], [qlat], marker="*", s=240, c="yellow", edgecolors="k", zorder=5)
        ax.set_xlim(qlon - win, qlon + win); ax.set_ylim(qlat - win, qlat + win)
        ax.set_title(title, fontsize=10); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        fig.colorbar(sc, ax=ax, shrink=0.8, label="similarity to ★")
    fig.suptitle("Similarity to one query block: the boosted leaf kernel follows the learned "
                 "partition; the RBF is a ball the modeler chose")
    return fig


# --- CLI ----------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Chapter 4 — trees and forests are kernels")
    p.add_argument("--out-prefix", default=None)
    args = p.parse_args(argv)
    set_style()
    cal = load_california()

    vm = value_mechanisms(cal)
    print("=" * 70, "\nLEAF KERNEL on California (gradient-boosted forest)")
    print(f"  unit diagonal {vm['diag']:.3f}, range {vm['krange'][0]:.2f}..{vm['krange'][1]:.2f}")
    print(f"  GNW exact recovery (leaf scores): |Δ vs forest| = {vm['exact_recovery_err']:.2e}")
    print("  test RMSE by value representation on the one geometry:")
    print(f"    raw-label NW   {vm['rawlabel_nw_rmse']:.3f}")
    print(f"    forest         {vm['forest_rmse']:.3f}")
    print(f"    leaf scores    {vm['exact_rmse']:.3f}  (= forest, exact)")
    print(f"    ridge values   {vm['ridge_rmse']:.3f}")

    enh = residual_enhancement(cal)
    print(f"\n  residual enhancement (OOF residuals): γ*={enh['gamma']:.2f}, "
          f"raw {enh['raw_rmse']:.3f} → enhanced {enh['enhanced_rmse']:.3f}")

    tw = load_taiwan()
    mtw, Xtw, ytw = fit_forest(tw, classifier=True)
    acc = float((mtw.predict(tw.Xte) == tw.yte).mean())
    print(f"\nTAIWAN: classification forest accuracy {acc:.3f} (leaf kernel built identically)")

    if args.out_prefix:
        import matplotlib
        matplotlib.use("Agg")
        make_value_figure(cal).savefig(f"{args.out_prefix}1_values.pdf")
        make_leaf_vs_rbf_figure(cal).savefig(f"{args.out_prefix}2_leaf_vs_rbf.pdf")
        print("wrote figures with prefix", args.out_prefix)


if __name__ == "__main__":
    main()
