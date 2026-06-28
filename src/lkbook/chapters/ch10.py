"""Chapter 10 — fusing geometries.

Chapter 9 left us with a fact: the spectral kernel and the tree leaf kernel own *complementary*
regimes — smooth/periodic/high-order for the spectral, sharp axis-aligned thresholds for the
tree — and a real dataset usually carries some of each. Bike-sharing demand is the clean case:
ridership is *periodic* in the hour of day and the month (a spectral/Fourier job) and *sharp* in
the working-day and weather regimes (a tree job). The move is not to pick one geometry but to
**combine** the two as a convex mixture in one ridge head,

    K_alpha(x,x') = sum_c alpha_c K_c(x,x'),   alpha on the simplex,

and to choose the weights leakage-free on a held-out query fold (the Chapter 7 discipline). Each
channel is a *symmetric PSD, unit-diagonal* kernel, so the mixture is one too — and unit-diagonal
means the ridge `lambda` stays identifiable (Ch. 3).

Fusion appears at **two levels**. Stage 1 is *within* a geometry: the spectral channel (Ch. 8) is
itself a convex fusion of H Laplace banks at different bandwidths, k = sum_h w_h exp(-||..||/T_h),
whose bank weights are learned to span the data's scales. Stage 2 is *across* geometries: fuse the
resulting spectral kernel with the **tree** kernel — a hyperparameter-tuned CatBoost (the
library-neutral GBDT) read as the Ch. 4 leaf kernel. Each channel reads the raw features its own
way: the tree takes them as-is; the spectral kernel takes a representation where the cyclical
features (hour, month, weekday) are **Fourier-encoded** (sin/cos at their period) so a periodic
geometry can see the periodicity. The fitted fusion splits into an **exact additive decomposition**
— one Gaussian-posterior-mean component per channel, summing to the prediction — and the earned
weights and component shares read as a diagnostic of which geometry the data used. Fusion is
**not** output-averaging: one ridge on the summed kernel dominates an average of separate ridge
smoothers in the Loewner order.

Mirrors `fusekernel` (`kernels.mix_n`, `interpret.channel_contributions`, the support/query
selection, the soft-tree gate). The tree channel is a hyperparameter-tuned CatBoost read as the
Ch. 4 leaf kernel; the spectral channel reuses `ch08`'s learned spectral-Laplace kernel.

    python -m lkbook.chapters.ch10 --out-prefix fig10
"""
from __future__ import annotations

import argparse
import functools

import numpy as np
import matplotlib.pyplot as plt

from lkbook import load_bikeshare, load_taiwan, set_style
from lkbook.chapters import ch08

SEED = 0
LAM_GRID = np.logspace(-3.5, 0.5, 14)


def _single_thread(fn):
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


def _r2(p, y):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(1.0 - np.sum((p - y) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12))


# =============================================================================
# Fourier encoding of cyclical features (for the spectral and RBF channels)
# =============================================================================

class FourierEncoder:
    """Replace each cyclical integer feature (period P) by [sin(2 pi x / P), cos(2 pi x / P)] —
    so an hour at 23 sits next to hour 0 — and standardize the whole result on the train fold.
    The tree channel does not use this; it reads the raw integers directly."""

    def fit(self, X, names, cyclical):
        from sklearn.preprocessing import StandardScaler
        self.cyc = {names.index(c): float(P) for c, P in cyclical.items()}
        self.scaler = StandardScaler().fit(self._enc(np.atleast_2d(np.asarray(X, float))))
        return self

    def _enc(self, X):
        cols = []
        for j in range(X.shape[1]):
            if j in self.cyc:
                a = 2 * np.pi * X[:, j] / self.cyc[j]
                cols += [np.sin(a), np.cos(a)]
            else:
                cols.append(X[:, j])
        return np.column_stack(cols)

    def transform(self, X):
        return self.scaler.transform(self._enc(np.atleast_2d(np.asarray(X, float))))


# =============================================================================
# The channels — each a symmetric PSD, unit-diagonal kernel on its own representation
# =============================================================================

class TreeChannel:
    """The Chapter-4 leaf co-membership kernel of a *hyperparameter-tuned* gradient-boosted
    forest --- here CatBoost (the library-neutral GBDT; XGBoost/LightGBM give the same leaf
    kernel). A 3-fold-CV grid search over depth / iterations / learning-rate makes it a
    fair-strength channel that encodes sharp axis-aligned regimes and discontinuities. The kernel
    is the fraction of trees in which two points land in the same leaf,
    k(x,x') = (1/T) sum_t 1[leaf_t(x) = leaf_t(x')] --- exactly the Ch. 4 leaf kernel, read from
    CatBoost's per-tree leaf indices (calc_leaf_indexes) rather than sklearn's `apply`."""
    name = "tree"
    GRID = {"depth": [4, 6, 8], "iterations": [200, 400], "learning_rate": [0.03, 0.1]}

    def __init__(self, seed=SEED):
        self.seed = seed

    def fit(self, Xs, ys):
        from catboost import CatBoostRegressor
        from sklearn.model_selection import GridSearchCV
        base = CatBoostRegressor(loss_function="RMSE", random_seed=self.seed,
                                 verbose=0, allow_writing_files=False, thread_count=-1)
        gs = GridSearchCV(base, self.GRID, cv=3, n_jobs=1).fit(np.asarray(Xs, float), ys)
        self.model = gs.best_estimator_
        self.best_params_ = gs.best_params_
        return self

    def _leaves(self, X):
        from catboost import Pool
        return self.model.calc_leaf_indexes(Pool(np.atleast_2d(np.asarray(X, float))))

    def block(self, A, B):
        La, Lb = self._leaves(A), self._leaves(B)
        T = La.shape[1]
        K = np.zeros((La.shape[0], Lb.shape[0]))
        for t in range(T):
            K += (La[:, t][:, None] == Lb[:, t][None, :])
        return K / T


class SpectralChannel:
    """The Chapter-8 learned spectral-Laplace kernel (learned mode), itself a *convex fusion of H
    Laplace banks* --- k_spectral = sum_h w_h exp(-||phi_h(x)-phi_h(x')||/T_h), w on the H-simplex.
    This is stage 1 of the chapter's two-level fusion. The banks are pinned to a *fixed log-spaced
    grid* of bandwidths spanning fine to coarse (`T_init`, `learn_T=False`), so the multi-bank
    fusion is a genuine, reproducible spread of *scales*; the convex weights w_h (and the
    frequencies, amplitudes and ARD relevance) are learned, so the data chooses how much mass each
    scale carries. Fit on the already-encoded / standardized representation (standardize=False)."""
    name = "spectral"

    def __init__(self, seed=SEED, steps=400, H=4, T_lo=0.5, T_hi=12.0):
        self.seed, self.steps, self.H = seed, steps, H
        self.T_init = np.geomspace(T_lo, T_hi, H)              # fixed multi-scale bank grid

    def fit(self, Xs, ys):
        self.kernel, _ = ch08.fit_spectral(Xs, ys, mode="learned", objective="nlml", H=self.H,
                                           standardize=False, steps=self.steps, seed=self.seed,
                                           T_init=self.T_init, learn_T=False)
        return self

    def block(self, A, B):
        return self.kernel.gram(np.atleast_2d(A), np.atleast_2d(B))

    def bank_weights(self):
        """The learned stage-1 bank fusion: (w_h, T_h) over the H Laplace banks (numpy)."""
        import torch
        from torch.nn.functional import softplus
        with torch.no_grad():
            w = torch.softmax(self.kernel.w_logit, 0).cpu().numpy()
            T = softplus(self.kernel.log_T).clamp_min(1e-4).cpu().numpy()
        order = np.argsort(T)
        return w[order], T[order]


CHANNELS = {"tree": TreeChannel, "spectral": SpectralChannel}


# =============================================================================
# The convex fusion and its leakage-free selection
# =============================================================================

def mix_n(blocks, weights):
    """Convex mixture sum_c w_c blocks[c] over unit-diagonal kernel blocks (fusekernel.mix_n).
    Unit-diagonal in, unit-diagonal out, so the ridge lambda stays identifiable."""
    K = weights[0] * blocks[0]
    for c in range(1, len(blocks)):
        K = K + weights[c] * blocks[c]
    return K


def _compositions(total, parts):
    if parts == 1:
        yield (total,); return
    for i in range(total + 1):
        for rest in _compositions(total - i, parts - 1):
            yield (i,) + rest


def _simplex_grid(C, res):
    """All weight vectors on the C-simplex with coordinates in {0, 1/res, ..., 1}, summing to 1."""
    for comp in _compositions(res, C):
        yield np.array(comp, float) / res


def _support_query_split(n, seed):
    rng = np.random.RandomState(seed); perm = rng.permutation(n); ns = n // 2
    return perm[:ns], perm[ns:]


def same_reps(X, names):
    """Build a per-channel representation dict that gives every channel the same matrix X
    (used for datasets without cyclical features)."""
    X = np.atleast_2d(np.asarray(X, float))
    return {nm: X for nm in names}


class FusedModel:
    """A fitted convex fusion: channels (fit on the support fold), the earned simplex weights,
    the ridge, the dual on the support fold, and each channel's support representation."""

    def __init__(self, channels, names, weights, lam, alpha, Xs_by, ys, ybar):
        self.channels, self.names = channels, names
        self.w, self.lam, self.alpha, self.Xs_by, self._ys, self.ybar = (
            weights, lam, alpha, Xs_by, ys, ybar)

    def _blocks(self, reps):
        return [ch.block(reps[ch.name], self.Xs_by[ch.name]) for ch in self.channels]

    def predict(self, reps):
        return mix_n(self._blocks(reps), self.w) @ self.alpha + self.ybar

    def channel_contributions(self, reps):
        """Exact additive decomposition (fusekernel.interpret.channel_contributions):
        contrib_c = w_c K_c(X,S) alpha; sum_c contrib_c + intercept == predict(reps)."""
        contribs = {ch.name: (self.w[i] * (ch.block(reps[ch.name], self.Xs_by[ch.name]) @ self.alpha))
                    for i, ch in enumerate(self.channels)}
        return contribs, self.ybar

    def shares(self, reps):
        """Component shares rho_c = ||g_c|| / sum_l ||g_l|| over a reference sample."""
        contribs, _ = self.channel_contributions(reps)
        norms = {c: float(np.linalg.norm(np.atleast_1d(v))) for c, v in contribs.items()}
        tot = sum(norms.values()) + 1e-12
        return {c: norms[c] / tot for c in norms}


@_single_thread
def fit_fused(reps_tr, ytr, seed=SEED, res=10, n_fit=1500, score="rmse"):
    """Fit the fused kernel: build each channel on the support fold S of its own representation,
    select (alpha, lambda) on the held-out query fold Q (simplex grid x log-ridge grid), decode
    the fused KRR on S. Leakage-free. `reps_tr` maps channel name -> train matrix. `score` is the
    query criterion: "rmse" for regression, or a classification criterion on the {0,1} ridge decode
    --- "auc" (threshold-free, the stable default for classification) or "accuracy" (0/1, too flat a
    criterion and over-credits a channel by chance). Both grids contain every vertex, so the
    selected fusion is never worse on the query fold than the best single channel."""
    names = list(reps_tr)
    ytr = np.asarray(ytr, float).ravel(); n = len(ytr)
    idx = np.arange(n)
    if n > n_fit:
        idx = np.random.RandomState(seed).choice(n, n_fit, replace=False)
    reps_tr = {nm: np.atleast_2d(np.asarray(reps_tr[nm], float))[idx] for nm in names}
    ytr = ytr[idx]
    s_idx, q_idx = _support_query_split(len(ytr), seed)
    ys, yq = ytr[s_idx], ytr[q_idx]; ybar = ys.mean()

    channels, Xs_by = [], {}
    for nm in names:
        ch = CHANNELS[nm](seed=seed)
        Xs = reps_tr[nm][s_idx]
        ch.fit(Xs, ys)
        channels.append(ch); Xs_by[nm] = Xs
    Bss = [ch.block(Xs_by[ch.name], Xs_by[ch.name]) for ch in channels]
    Bqs = [ch.block(reps_tr[ch.name][q_idx], Xs_by[ch.name]) for ch in channels]
    Iss = np.eye(len(ys))

    def _loss(pred):
        if score == "auc":
            from sklearn.metrics import roc_auc_score
            return -float(roc_auc_score((yq > 0.5).astype(int), pred))
        if score == "accuracy":
            return -float(np.mean((pred > 0.5).astype(int) == (yq > 0.5).astype(int)))
        return _rmse(pred, yq)

    best = None
    for w in _simplex_grid(len(channels), res):
        Kss = mix_n(Bss, w); Kqs = mix_n(Bqs, w)
        for lam in LAM_GRID:
            alpha = np.linalg.solve(Kss + lam * Iss, ys - ybar)
            loss = _loss(Kqs @ alpha + ybar)
            if best is None or loss < best[0]:
                best = (loss, w.copy(), lam, alpha)
    _, w, lam, alpha = best
    return FusedModel(channels, list(names), w, lam, alpha, Xs_by, ys, ybar)


def fit_single_channel(reps_tr, ytr, name, seed=SEED, n_fit=1500, score="rmse"):
    """A pure-channel KRR baseline (the simplex vertex), selected and decoded the same way."""
    return fit_fused({name: reps_tr[name]}, ytr, seed=seed, n_fit=n_fit, res=1, score=score)


# =============================================================================
# Fusion is not output-averaging: the Loewner domination
# =============================================================================

def fusion_vs_averaging(reps_tr, ytr, reps_te, yte, seed=SEED, n_fit=1200, w=None):
    """Fusion is not output-averaging. The fused smoother applies one ridge to the summed kernel,
    S_fuse = K_w(K_w+lam I)^{-1}; averaging applies separate ridges, S_avg = sum_c w_c K_c(K_c+lam
    I)^{-1}. Operator concavity of g(t)=t/(t+lam) gives the Loewner domination S_fuse >= S_avg for
    any *interior* mixture. We evaluate at a fixed interior `w` (default uniform) --- at a vertex
    the two smoothers coincide and the inequality is vacuous --- and report the minimum eigenvalue
    of S_fuse - S_avg (>= 0 is the domination) beside the two test RMSEs and the selected weights.

    The ridge `lam` and the selected weights come from a leakage-free fit; the operator comparison
    holds the same lam for both sides (the inequality is about the kernels, not the ridge)."""
    fm = fit_fused(reps_tr, ytr, seed=seed, n_fit=n_fit)
    lam, ybar, ys_s = fm.lam, fm.ybar, fm._ys
    if w is None:
        w = np.full(len(fm.channels), 1.0 / len(fm.channels))   # interior point: equal weights
    w = np.asarray(w, float)
    Bss = [ch.block(fm.Xs_by[ch.name], fm.Xs_by[ch.name]) for ch in fm.channels]
    n = len(ys_s); I = np.eye(n)
    Kw = mix_n(Bss, w)
    S_fuse = Kw @ np.linalg.inv(Kw + lam * I)
    S_avg = sum(w[c] * (Bss[c] @ np.linalg.inv(Bss[c] + lam * I)) for c in range(len(Bss)))
    diff = 0.5 * (S_fuse - S_avg) + 0.5 * (S_fuse - S_avg).T
    eig_min = float(np.linalg.eigvalsh(diff).min())

    a_fuse = np.linalg.solve(Kw + lam * I, ys_s - ybar)
    a_each = [np.linalg.solve(Bss[c] + lam * I, ys_s - ybar) for c in range(len(Bss))]

    def fuse_predict(reps):
        Kx = mix_n([ch.block(reps[ch.name], fm.Xs_by[ch.name]) for ch in fm.channels], w)
        return Kx @ a_fuse + ybar

    def avg_predict(reps):
        out = np.zeros(len(np.atleast_2d(reps[fm.channels[0].name])))
        for c, ch in enumerate(fm.channels):
            out += w[c] * (ch.block(reps[ch.name], fm.Xs_by[ch.name]) @ a_each[c] + ybar)
        return out
    return {"eig_min": eig_min, "w_eval": dict(zip(fm.names, w)),
            "fused_rmse": _rmse(fuse_predict(reps_te), yte),
            "avg_rmse": _rmse(avg_predict(reps_te), yte),
            "selected_weights": dict(zip(fm.names, fm.w))}


# =============================================================================
# Soft tree gate: tau -> infinity recovers the hard leaf kernel
# =============================================================================

class SoftTree:
    """A differentiable soft tree parsed from one sklearn decision tree: each split's left-branch
    probability is sigmoid(tau (c - a^T x)), a leaf's membership is the path-conjunction product,
    and the soft co-membership is Z_a Z_b^T. As tau -> inf the gate sharpens to the hard indicator
    and the soft kernel recovers the exact leaf kernel (fusekernel.trees._SoftTree)."""

    def __init__(self, tree):
        t = tree.tree_
        self.feat, self.thr, self.cl, self.cr = t.feature, t.threshold, t.children_left, t.children_right
        self.leaves = [i for i in range(t.node_count) if t.children_left[i] == -1]
        self.leaf_idx = {nid: k for k, nid in enumerate(self.leaves)}
        self.paths = {}
        self._walk(0, [])

    def _walk(self, node, path):
        if self.cl[node] == -1:
            self.paths[node] = list(path); return
        f, th = self.feat[node], self.thr[node]
        self._walk(self.cl[node], path + [(f, th, True)])
        self._walk(self.cr[node], path + [(f, th, False)])

    def code(self, X, tau):
        from scipy.special import expit
        X = np.atleast_2d(np.asarray(X, float)); n = len(X)
        Z = np.zeros((n, len(self.leaves)))
        for nid, path in self.paths.items():
            g = np.ones(n)
            for (f, th, go_left) in path:
                p_left = expit(tau * (th - X[:, f]))          # sigmoid(tau (th - x_f)), stable
                g *= p_left if go_left else (1.0 - p_left)
            Z[:, self.leaf_idx[nid]] = g
        return Z

    def gram(self, A, B, tau):
        return self.code(A, tau) @ self.code(B, tau).T


def soft_tree_fidelity(n=300, d=4, depth=4, taus=(1, 5, 20, 100, 500), seed=SEED):
    """||K_soft(tau) - K_hard||_F / ||K_hard||_F as tau grows: the hard leaf kernel is the
    tau -> inf limit, and the soft kernel converges to it."""
    from sklearn.tree import DecisionTreeRegressor
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, (n, d)); y = np.sin(2 * X[:, 0]) + (X[:, 1] > 0).astype(float)
    tree = DecisionTreeRegressor(max_depth=depth, random_state=seed).fit(X, y)
    st = SoftTree(tree)
    leaves = tree.apply(X)
    Khard = (leaves[:, None] == leaves[None, :]).astype(float)
    return [(tau, float(np.linalg.norm(st.gram(X, X, tau) - Khard) / (np.linalg.norm(Khard) + 1e-12)))
            for tau in taus]


# =============================================================================
# Running example: Bike Sharing (regression, cyclical) and Taiwan (classification)
# =============================================================================

def _bike_reps(d, enc=None):
    """Per-channel representations of bike-sharing: the tree reads raw features; the spectral
    channel reads the Fourier-encoded, standardized representation (so a periodic kernel can see
    the hour/month/weekday cycles)."""
    if enc is None:
        enc = FourierEncoder().fit(d.Xtr_raw, d.names, d.cyclical)
    tr = {"tree": d.Xtr_raw, "spectral": enc.transform(d.Xtr_raw)}
    te = {"tree": d.Xte_raw, "spectral": enc.transform(d.Xte_raw)}
    return tr, te, enc


def run_bikeshare(seed=SEED, n_train=2000):
    """Two-level fusion on Bike Sharing. Stage 1: the spectral channel is a convex fusion of H
    Laplace banks (its learned bank weights span the data's scales). Stage 2: fuse that spectral
    kernel with the tuned CatBoost tree on the cross-geometry simplex. Report fused vs pure-channel
    R^2, the stage-2 weights, the stage-1 spectral bank weights, the shares, and an exact
    decomposition of one held-out prediction. Cyclical hour/month/weekday are Fourier-encoded for
    the spectral channel."""
    d = load_bikeshare()
    rng = np.random.RandomState(seed)
    idx = rng.choice(d.n, min(n_train, d.n), replace=False)
    dsub = type(d)(d.Xtr_raw[idx], d.Xte_raw, d.ytr[idx], d.yte, d.names, d.cyclical,
                   d.task, d.target_unit)
    reps_tr, reps_te, enc = _bike_reps(dsub)
    fm = fit_fused(reps_tr, dsub.ytr, seed=seed, n_fit=n_train)
    fused_r2 = _r2(fm.predict(reps_te), dsub.yte)
    singles = {nm: _r2(fit_single_channel(reps_tr, dsub.ytr, nm, seed=seed, n_fit=n_train)
                       .predict({nm: reps_te[nm]}), dsub.yte) for nm in fm.names}
    sp = [ch for ch in fm.channels if ch.name == "spectral"][0]
    bw, bT = sp.bank_weights()
    reps_te1 = {nm: reps_te[nm][:1] for nm in fm.names}
    contribs, intercept = fm.channel_contributions(reps_te1)
    pred0 = float(fm.predict(reps_te1)[0])
    recon = intercept + sum(float(np.atleast_1d(v)[0]) for v in contribs.values())
    reps_te500 = {nm: reps_te[nm][:500] for nm in fm.names}
    return {"weights": dict(zip(fm.names, fm.w)), "fused_r2": fused_r2, "single_r2": singles,
            "shares": fm.shares(reps_te500), "decomp": {c: float(np.atleast_1d(v)[0])
                                                        for c, v in contribs.items()},
            "bank_weights": [float(x) for x in bw], "bank_T": [float(x) for x in bT],
            "intercept": intercept, "pred0": pred0, "recon0": recon, "model": fm}


def run_taiwan(seed=SEED, n_train=1500):
    """Classification path: fuse the tree and spectral kernels on a ridge decode of the binary
    target, then threshold at 0.5. Classification is *not* selected on RMSE --- we select the
    cross-geometry weights and ridge on the held-out query fold's AUC (threshold-free, stable).
    Report fused vs pure-channel test accuracy and AUC, and the earned weights. No cyclical
    features here, so both channels share the standardized representation."""
    from sklearn.metrics import roc_auc_score
    d = load_taiwan()
    rng = np.random.RandomState(seed)
    idx = rng.choice(d.n, min(n_train, d.n), replace=False)
    names = ["tree", "spectral"]
    reps_tr = same_reps(d.Xtr[idx], names); reps_te = same_reps(d.Xte, names)
    ytr = d.ytr[idx].astype(float); yte = d.yte.astype(int)
    fm = fit_fused(reps_tr, ytr, seed=seed, n_fit=n_train, score="auc")
    acc = float(np.mean((fm.predict(reps_te) > 0.5).astype(int) == yte))
    auc = float(roc_auc_score(yte, fm.predict(reps_te)))
    singles, singles_auc = {}, {}
    for nm in fm.names:
        p = fit_single_channel(reps_tr, ytr, nm, seed=seed, n_fit=n_train, score="auc").predict(
            {nm: reps_te[nm]})
        singles[nm] = float(np.mean((p > 0.5).astype(int) == yte))
        singles_auc[nm] = float(roc_auc_score(yte, p))
    return {"weights": dict(zip(fm.names, fm.w)), "fused_acc": acc, "fused_auc": auc,
            "single_acc": singles, "single_auc": singles_auc}


# =============================================================================
# Synthetic smooth-to-sharp sweep: the selected weight tracks the target's geometry
# =============================================================================

def _unit(v):
    v = np.asarray(v, float); return (v - v.mean()) / (v.std() + 1e-12)


def smooth_to_sharp(t, n=1600, d=6, depth=6, noise=0.05, seed=SEED):
    """A target morphing from smooth/periodic (t=0) to a deep axis-aligned partition (t=1, the
    ch09 S10 geometry the tree owns): y = (1-t) sin(2 pi 1.3 x0) + t * jumps, each component
    standardized before mixing."""
    X = np.random.RandomState(seed).uniform(-1, 1, (n, d))
    smooth = _unit(np.sin(2 * np.pi * 1.3 * X[:, 0]))
    rng = np.random.RandomState(seed + 3)
    feats = rng.randint(0, 4, size=depth); thr = rng.uniform(-0.6, 0.6, size=depth)
    codes = np.zeros(n, dtype=int)
    for k in range(depth):
        codes = codes * 2 + (X[:, feats[k]] > thr[k]).astype(int)
    jumps = _unit(rng.randn(2 ** depth)[codes])
    y = _unit((1 - t) * smooth + t * jumps) + noise * np.random.RandomState(seed + 1).randn(n)
    return X, y


def vertex_sweep(ts=(0.0, 0.25, 0.5, 0.75, 1.0), seed=SEED, n_fit=1200):
    """Fit the spectral+tree fusion across the smooth-to-sharp morph; record the earned weights.
    No cyclical features, so both channels share the standardized matrix."""
    names = ["tree", "spectral"]; rows = []
    for t in ts:
        X, y = smooth_to_sharp(t, seed=seed)
        n = len(y); perm = np.random.RandomState(seed + 5).permutation(n); nte = n // 4
        tr = perm[nte:]
        fm = fit_fused(same_reps(X[tr], names), y[tr], seed=seed, n_fit=n_fit)
        w = dict(zip(fm.names, fm.w))
        rows.append({"t": t, "tree": w["tree"], "spectral": w["spectral"]})
    return rows


# =============================================================================
# Figures
# =============================================================================

SPECTRAL_C = "#c44e52"
TREE_C = "#555555"


def make_weights_figure(bike=None, tw=None, sweep=None, seed=SEED):
    """(10.1) The two-level fusion. Left: stage 1 --- the spectral channel's learned bank weights
    over its H Laplace banks (the convex fusion *inside* the spectral kernel), against bandwidth.
    Middle: stage 2 --- the cross-geometry weights (spectral vs tree) on Bike and Taiwan as stacked
    bars, with the Bike component shares. Right: the smooth-to-sharp vertex sweep."""
    if bike is None:
        bike = run_bikeshare(seed=seed)
    if tw is None:
        tw = run_taiwan(seed=seed)
    if sweep is None:
        sweep = vertex_sweep(seed=seed)
    order = ["spectral", "tree"]
    colors = {"spectral": SPECTRAL_C, "tree": TREE_C}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)

    # stage 1: spectral bank fusion (within the spectral kernel)
    ax0 = axes[0]
    bw, bT = bike["bank_weights"], bike["bank_T"]
    ax0.bar(range(len(bw)), bw, 0.6, color=SPECTRAL_C)
    for i, (w, T) in enumerate(zip(bw, bT)):
        ax0.text(i, w + 0.02, f"w={w:.2f}\nT={T:.2f}", ha="center", va="bottom", fontsize=8)
    ax0.set_xticks(range(len(bw))); ax0.set_xticklabels([f"bank {i+1}" for i in range(len(bw))],
                                                        fontsize=9)
    ax0.set_ylim(0, max(bw) * 1.35 + 0.05); ax0.set_ylabel("bank weight $w_h$")
    ax0.set_title("Stage 1: fusion within the spectral kernel\n(H Laplace banks, by bandwidth)",
                  fontsize=10)

    # stage 2: cross-geometry fusion (spectral vs tree)
    ax = axes[1]
    cols = [("Bike\nweights", bike["weights"]), ("Bike\nshares ρ", bike["shares"]),
            ("Taiwan\nweights", tw["weights"])]
    for j, (_, vals) in enumerate(cols):
        bottom = 0.0
        for ch in order:
            v = vals[ch]
            ax.bar(j, v, 0.7, bottom=bottom, color=colors[ch], label=ch if j == 0 else None)
            if v > 0.06:
                ax.text(j, bottom + v / 2, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white")
            bottom += v
    ax.set_xticks(range(3)); ax.set_xticklabels([c[0] for c in cols], fontsize=9)
    ax.set_ylim(0, 1.02); ax.set_ylabel("simplex weight / share")
    ax.set_title("Stage 2: cross-geometry fusion\n(spectral vs tree)", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")

    ax2 = axes[2]
    ts = [r["t"] for r in sweep]
    ax2.plot(ts, [r["spectral"] for r in sweep], "-o", color=SPECTRAL_C, label="spectral")
    ax2.plot(ts, [r["tree"] for r in sweep], "-o", color=TREE_C, label="tree")
    ax2.set_xlabel("target morph t  (0 = smooth/periodic → 1 = sharp partition)")
    ax2.set_ylabel("earned weight"); ax2.set_ylim(-0.02, 1.02)
    ax2.set_title("The selected vertex tracks\nthe target's geometry", fontsize=10)
    ax2.legend(fontsize=9)
    return fig


def make_decomposition_figure(bike=None, seed=SEED):
    """(10.2) One Bike-Sharing prediction decomposed into tree + spectral contributions (a waterfall
    summing to the prediction), beside the soft-tree → hard-leaf fidelity sweep."""
    if bike is None:
        bike = run_bikeshare(seed=seed)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    ax = axes[0]
    decomp, intercept, pred = bike["decomp"], bike["intercept"], bike["pred0"]
    colors = {"tree": TREE_C, "spectral": SPECTRAL_C}
    labels = ["intercept"] + list(decomp.keys()) + ["= prediction"]
    running = intercept
    ax.bar(0, intercept, 0.7, color="#999999")
    for i, (lab, v) in enumerate(decomp.items(), start=1):
        ax.bar(i, v, 0.7, bottom=running, color=colors.get(lab, "#999999"))
        running += v
    ax.bar(len(decomp) + 1, pred, 0.7, color="#222222")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("contribution (log rides/hr)")
    ax.set_title(f"One prediction = intercept + tree + spectral\n"
                 f"(reconstruction {bike['recon0']:.3f} vs predict {pred:.3f})", fontsize=10)

    ax2 = axes[1]
    fid = soft_tree_fidelity(seed=seed)
    ax2.semilogx([t for t, _ in fid], [e for _, e in fid], "-o", color=SPECTRAL_C)
    ax2.set_xlabel("gate sharpness τ"); ax2.set_ylabel("‖K_soft − K_hard‖_F / ‖K_hard‖_F")
    ax2.set_title("Soft tree gate → hard leaf kernel as τ → ∞", fontsize=10)
    return fig


# =============================================================================
# CLI
# =============================================================================

def main(argv=None):
    p = argparse.ArgumentParser(description="Chapter 10 — fusing geometries")
    p.add_argument("--out-prefix", default=None)
    args = p.parse_args(argv)
    set_style()

    bike = run_bikeshare()
    print("=" * 76)
    print("BIKE SHARING — two-level fusion: spectral (multi-bank) + tuned CatBoost tree")
    print("  stage 1 spectral banks: w =", [round(w, 3) for w in bike["bank_weights"]],
          " T =", [round(t, 3) for t in bike["bank_T"]])
    print("  stage 2 cross weights :", {k: round(v, 3) for k, v in bike["weights"].items()})
    print("  shares rho_c          :", {k: round(v, 3) for k, v in bike["shares"].items()})
    print(f"  fused R^2 {bike['fused_r2']:.3f}   pure channels "
          f"{ {k: round(v,3) for k,v in bike['single_r2'].items()} }")
    print(f"  one prediction: intercept {bike['intercept']:.3f} + "
          f"{ {k: round(v,3) for k,v in bike['decomp'].items()} }")
    print(f"    reconstruction {bike['recon0']:.4f}  vs  predict {bike['pred0']:.4f}")
    tw = run_taiwan()
    print("-" * 76)
    print("TAIWAN — classification fusion (query-AUC selection)")
    print("  earned weights:", {k: round(v, 3) for k, v in tw["weights"].items()})
    print(f"  fused acc {tw['fused_acc']:.3f} / auc {tw['fused_auc']:.3f}   pure-channel acc "
          f"{ {k: round(v,3) for k,v in tw['single_acc'].items()} }  auc "
          f"{ {k: round(v,3) for k,v in tw['single_auc'].items()} }")

    if args.out_prefix:
        import matplotlib
        matplotlib.use("Agg")
        make_weights_figure(bike=bike, tw=tw).savefig(f"{args.out_prefix}1_weights.pdf")
        make_decomposition_figure(bike=bike).savefig(f"{args.out_prefix}2_decomposition.pdf")
        print("wrote figures with prefix", args.out_prefix)


if __name__ == "__main__":
    main()
