"""Chapter 6 — similarity, smoothing, and case-based prediction.

A kernel machine predicts by weighting historical cases, so every prediction carries its own
evidence. Take the leaf kernel of Chapter 4 as the geometry and read its Nadaraya--Watson
smoother as a distribution over training rows:

    f(x) = Σ_i w_i(x) y_i,   w_i(x) = k(x,x_i)^ρ / Σ_r k(x,x_r)^ρ,   Σ_i w_i(x)=1, w_i ≥ 0.

The weights are nonnegative and sum to one, so the prediction is a convex average of training
labels --- it *is* its explanation. From the same weights fall a per-query evidence ledger:

  - **N_eff(x) = 1 / Σ_i w_i(x)²** --- effective number of cases the prediction leans on;
  - **Δ_y(x) = Σ_i w_i(x)(y_i − ȳ)²** --- whether the supporting cases agree on the outcome;
  - **G_q(x) = |q(x) − Σ_i w_i(x) q_i|** --- whether a teacher (the Chapter-4 forest) agrees
        with its own neighbors; bounded by the Cauchy--Schwarz local-fidelity radius;
  - **C_cal(x) = Σ_i w_i(x)(y_i − q_i)** --- local calibration of the teacher;
  - **Δ_K(x) = 2(1 − Σ_i w_i(x) k(x,x_i))** --- weighted kernel-distance radius;
  - the **witnesses** --- the top-weighted training rows the prediction cites.

Three prediction heads sit on the one geometry: empirical (average labels), teacher (average
the forest's scores), and blended z_i(ρ) = (1−ρ)y_i + ρq_i with ρ chosen on a validation
fold. The running-example finding is that ρ* → 0 on California (smooth labels reward direct
averaging) and ρ* → 1 on Taiwan Credit (noisy binary defaults reward the teacher's smoother
score) --- the same machine, opposite blends, dictated by label noise.

This is the NumPy/SciPy re-implementation of the TKLE methodology of Sudjianto et al.
(`kernel_xgb.tkle`), built on `lkbook.chapters.ch04.LeafKernel`. No torch, no kernel-xgb.

    python -m lkbook.chapters.ch06 --out-prefix fig6
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from lkbook import load_california, load_taiwan, set_style, POS_CMAP
from lkbook.chapters import ch04
from lkbook.chapters.ch04 import LeafKernel, fit_forest

EPS = 1e-12
N_TRAIN, N_VAL, SEED = ch04.N_TRAIN, ch04.N_VAL, ch04.SEED
TOPK, POWER = 200, 1.0


# --- weights ------------------------------------------------------------------

def nw_weights(K_qt: np.ndarray, topk: int | None = TOPK, power: float = POWER,
               eta: float = EPS) -> np.ndarray:
    """Row-stochastic Nadaraya--Watson weights from a kernel-row matrix K_qt (n_q, n_t).
    Optional top-k truncation (keep the k most similar cases) and power sharpening ρ. Rows
    with no positive similarity fall back to uniform, so a query with no leaf-mates predicts
    the global mean instead of NaN. Each output row is nonnegative and sums to one."""
    W = np.asarray(K_qt, dtype=np.float64).copy()
    if topk is not None and topk < W.shape[1]:
        drop = np.argpartition(-W, kth=topk - 1, axis=1)[:, topk:]
        rows = np.repeat(np.arange(W.shape[0]), drop.shape[1])
        W[rows, drop.ravel()] = 0.0
    if power != 1.0:
        W = np.power(np.maximum(W, 0.0), power)
    row_sum = W.sum(axis=1, keepdims=True)
    zero = (row_sum <= eta).ravel()
    W = W / np.clip(row_sum, eta, None)
    if zero.any():
        W[zero] = 1.0 / W.shape[1]
    return W


# --- the evidence ledger ------------------------------------------------------

def evidence(K_qt: np.ndarray, y_t: np.ndarray, q_t: np.ndarray, q_query: np.ndarray,
             topk: int | None = TOPK, power: float = POWER, top_k: int = 8) -> dict:
    """The per-query evidence ledger on the leaf kernel, mirroring `kernel_xgb.tkle`.

    Inputs are the query-vs-train kernel rows `K_qt` (n_q, n_t), training labels `y_t`,
    training teacher scores `q_t`, and the teacher score `q_query` at each query. Returns
    arrays aligned on the query axis plus the top-`top_k` witnesses (index, weight, y, q,
    kernel, distance) sorted by weight.
    """
    W = nw_weights(K_qt, topk=topk, power=power)
    label_local = W @ y_t                                  # ȳ(x) = Σ w_i y_i
    teacher_local = W @ q_t                                # q̄(x) = Σ w_i q_i
    # Var = E[Z²] − E[Z]², the E[y²]−E[y]² computation form
    delta_y = np.maximum(W @ (y_t ** 2) - label_local ** 2, 0.0)
    delta_q = np.maximum(W @ (q_t ** 2) - teacher_local ** 2, 0.0)
    neff = 1.0 / (np.sum(W ** 2, axis=1) + EPS)            # effective sample size
    g_q = np.abs(q_query - teacher_local)                  # teacher-fidelity gap
    c_cal = W @ (y_t - q_t)                                # local calibration residual
    delta_K = 2.0 * (1.0 - np.sum(W * K_qt, axis=1))       # weighted kernel radius, d_K²=2(1−k)

    k = min(top_k, W.shape[1])
    part = np.argpartition(-W, kth=k - 1, axis=1)[:, :k]
    gathered = np.take_along_axis(W, part, axis=1)
    order = np.argsort(-gathered, axis=1)
    top_idx = np.take_along_axis(part, order, axis=1)
    top_w = np.take_along_axis(gathered, order, axis=1)
    top_K = np.take_along_axis(K_qt, top_idx, axis=1)
    return {
        "weights": W,
        "label_local": label_local,
        "teacher_local": teacher_local,
        "neff": neff,
        "delta_y": delta_y,
        "delta_q": delta_q,
        "g_q": g_q,
        "c_cal": c_cal,
        "delta_K": delta_K,
        "top_idx": top_idx,
        "top_w": top_w,
        "top_y": y_t[top_idx],
        "top_q": q_t[top_idx],
        "top_K": top_K,
        "top_dist": np.sqrt(np.maximum(2.0 * (1.0 - top_K), 0.0)),
    }


def fidelity_radius(K_qt: np.ndarray, q_t: np.ndarray, q_query: np.ndarray,
                    topk: int | None = TOPK, power: float = POWER) -> np.ndarray:
    """The Cauchy--Schwarz local-fidelity radius (Σ_i w_i(q_i − q(x))²)^{1/2}, an upper bound
    on the teacher-fidelity gap G_q: the smoother matches the teacher when the teacher score
    is stable among similar cases."""
    W = nw_weights(K_qt, topk=topk, power=power)
    return np.sqrt(np.maximum(W @ (q_t ** 2) - 2.0 * q_query * (W @ q_t) + q_query ** 2, 0.0))


# --- the three prediction heads ----------------------------------------------

def _clip01(p, eps=1e-6):
    return np.clip(p, eps, 1.0 - eps)


def _logit(p, eps=1e-6):
    p = _clip01(p, eps)
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))),
                    np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))


def head(W: np.ndarray, y_t: np.ndarray, q_t: np.ndarray, which: str = "empirical",
         rho: float = 0.5, task: str = "regression") -> np.ndarray:
    """Predict from row-stochastic weights `W` on one geometry.

    'empirical' averages observed labels (Σ w_i y_i); 'teacher' averages the forest's scores
    (Σ w_i q_i); 'blended' mixes them with ρ∈[0,1] --- a convex z_i for regression, a
    logit-space blend for classification."""
    if which == "empirical":
        return W @ y_t
    if which == "teacher":
        return W @ q_t
    if which == "blended":
        if task == "regression":
            return W @ ((1.0 - rho) * y_t + rho * q_t)
        y_smooth = np.where(y_t > 0.5, 1.0 - 1e-6, 1e-6)
        z = (1.0 - rho) * _logit(y_smooth) + rho * _logit(_clip01(q_t))
        return _sigmoid(W @ z)
    raise ValueError(f"unknown head {which!r}")


def _loss(pred, y, task):
    if task == "regression":
        return float(np.sqrt(np.mean((pred - y) ** 2)))           # RMSE
    p = _clip01(pred)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))  # log-loss


def select_rho(Wv: np.ndarray, yv: np.ndarray, y_t: np.ndarray, q_t: np.ndarray,
               task: str, rhos: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Choose the blend ρ* = argmin_ρ L(validation) over a grid. Returns (ρ*, loss-per-ρ)."""
    if rhos is None:
        rhos = np.linspace(0.0, 1.0, 11)
    losses = np.array([_loss(head(Wv, y_t, q_t, "blended", rho=r, task=task), yv, task)
                       for r in rhos])
    return float(rhos[int(np.argmin(losses))]), losses


# --- the chapter assembly: fit forest, build kernel, teacher scores -----------

class CaseBasedModel:
    """Bundles the Chapter-4 leaf kernel with the Chapter-6 evidence ledger and heads.

    `fit` trains a gradient-boosted forest (the teacher), extracts its leaf kernel, and stores
    the training labels and teacher scores. Everything else is the NW smoother and ledger on
    that one geometry. A held-out validation slice selects the blend ρ*.
    """

    def __init__(self, topk: int = TOPK, power: float = POWER, seed: int = SEED):
        self.topk, self.power, self.seed = topk, power, seed

    def fit(self, d):
        self.task = d.task
        classifier = d.task == "classification"
        self.model, self.Xtr, self.ytr = fit_forest(d, classifier=classifier, seed=self.seed)
        self.lk = LeafKernel().fit(self.model)
        self.q_tr = self._teacher(self.Xtr)
        # a held-out validation slice for selecting (β, ρ), drawn with a different seed
        rng = np.random.RandomState(self.seed + 7)
        vi = rng.choice(d.n, min(N_VAL, d.n), replace=False)
        self.Xv, self.yv = d.Xtr[vi], d.ytr[vi]
        return self

    def _teacher(self, X):
        if self.task == "regression":
            return self.model.predict(X).astype(np.float64)
        return self.model.predict_proba(X)[:, 1].astype(np.float64)

    def kernel_rows(self, X):
        return self.lk.gram(X, self.Xtr)

    def ledger(self, X, top_k: int = 8) -> dict:
        K = self.kernel_rows(X)
        return evidence(K, self.ytr, self.q_tr, self._teacher(X),
                        topk=self.topk, power=self.power, top_k=top_k)

    def select_rho(self, rhos: np.ndarray | None = None) -> tuple[float, np.ndarray]:
        Wv = nw_weights(self.kernel_rows(self.Xv), topk=self.topk, power=self.power)
        return select_rho(Wv, self.yv, self.ytr, self.q_tr, self.task, rhos=rhos)

    def select_hparams(self, powers=(1, 2, 4, 8),
                       rhos: np.ndarray | None = None) -> tuple[int, float]:
        """Joint validation selection of the sharpening power β and the blend ρ over the
        paper's grid, mirroring TKLE. Sets `self.power` to the selected β and returns
        (β*, ρ*)."""
        Kv = self.kernel_rows(self.Xv)
        best, choice = np.inf, (self.power, 0.0)
        for pw in powers:
            Wv = nw_weights(Kv, topk=self.topk, power=pw)
            r, losses = select_rho(Wv, self.yv, self.ytr, self.q_tr, self.task, rhos=rhos)
            if losses.min() < best:
                best, choice = losses.min(), (int(pw), r)
        self.power = choice[0]
        return choice

    def predict(self, X, which: str = "blended", rho: float = 0.5) -> np.ndarray:
        W = nw_weights(self.kernel_rows(X), topk=self.topk, power=self.power)
        return head(W, self.ytr, self.q_tr, which=which, rho=rho, task=self.task)


def run_dataset(d, seed: int = SEED, select_power: bool = True) -> dict:
    """Fit the case-based model, select (β*, ρ*) on validation, and score the three heads on
    the test set. `select_power=False` pins β=1 (the plain leaf-kernel smoother)."""
    m = CaseBasedModel(seed=seed).fit(d)
    if select_power:
        power_star, rho_star = m.select_hparams()
    else:
        power_star = m.power
        rho_star, _ = m.select_rho()
    _, losses = m.select_rho()
    K_te = m.kernel_rows(d.Xte)
    W_te = nw_weights(K_te, topk=m.topk, power=m.power)
    preds = {h: head(W_te, m.ytr, m.q_tr, which=h, rho=rho_star, task=d.task)
             for h in ("empirical", "teacher", "blended")}
    teacher_score = m._teacher(d.Xte)
    forest_loss = _loss(teacher_score, d.yte, d.task)
    ev = m.ledger(d.Xte, top_k=8)
    return {
        "model": m,
        "rho_star": rho_star,
        "power_star": power_star,
        "rho_losses": losses,
        "head_loss": {h: _loss(preds[h], d.yte, d.task) for h in preds},
        "forest_loss": forest_loss,
        "neff_median": float(np.median(ev["neff"])),
        "neff": ev["neff"],
        "delta_y_median": float(np.median(ev["delta_y"])),
        "g_q_median": float(np.median(ev["g_q"])),
        "preds": preds,
        "teacher_score": teacher_score,
        "evidence": ev,
        "task": d.task,
    }


def format_ledger(d, ev: dict, i: int) -> str:
    """Render the evidence ledger for query `i` as the audit artifact a reviewer reads."""
    ycol = "label" if d.task == "classification" else "y"
    L = ["=" * 70, "TREE-KERNEL LOCAL EVIDENCE LEDGER", "=" * 70,
         f"  Empirical local mean:      {ev['label_local'][i]:+.4f}",
         f"  Teacher local mean:        {ev['teacher_local'][i]:+.4f}",
         "",
         "  Diagnostics:",
         f"    N_eff (effective cases):   {ev['neff'][i]:.2f}",
         f"    Delta_y (local label var): {ev['delta_y'][i]:.4f}",
         f"    Delta_q (local teacher v): {ev['delta_q'][i]:.4f}",
         f"    G_q   (teacher fidelity):  {ev['g_q'][i]:.4f}",
         f"    Delta_K (kernel radius):   {ev['delta_K'][i]:.4f}",
         f"    C_cal (local calibration): {ev['c_cal'][i]:+.4f}",
         "",
         f"  Top 8 weighted witnesses (cum. weight {ev['top_w'][i].sum():.4f}):",
         "  " + "-" * 66,
         f"  {'rank':>4} {'idx':>7} {'weight':>9} {ycol:>9} {'q':>9} {'k':>8} {'d_K':>8}",
         "  " + "-" * 66]
    for r in range(ev["top_idx"].shape[1]):
        L.append(f"  {r+1:>4} {ev['top_idx'][i, r]:>7} {ev['top_w'][i, r]:>9.4f} "
                 f"{ev['top_y'][i, r]:>9.4f} {ev['top_q'][i, r]:>9.4f} "
                 f"{ev['top_K'][i, r]:>8.4f} {ev['top_dist'][i, r]:>8.4f}")
    L.append("  " + "-" * 66)
    return "\n".join(L)


# --- figures ------------------------------------------------------------------

def well_supported_query(ev: dict) -> int:
    """A typical, well-supported headline query: above-median N_eff with the smallest
    teacher-fidelity gap G_q --- broad evidence the teacher agrees with."""
    hi = np.where(ev["neff"] >= np.median(ev["neff"]))[0]
    return int(hi[np.argmin(ev["g_q"][hi])])


def make_witness_figure(cal_run: dict, d, q: int | None = None, seed: int = SEED):
    """Figure 6.1 (California): the sorted weight distribution for one query (left) and its
    top-8 witnesses on the map colored by label (right). The prediction is a citation to
    specific, nearby, weighted cases; N_eff reads off the concentration of the weight tail."""
    m = cal_run["model"]
    ev = cal_run["evidence"]
    if q is None:
        q = well_supported_query(ev)
    W = ev["weights"][q]
    order = np.argsort(-W)
    w_sorted = W[order]
    neff, dy = ev["neff"][q], ev["delta_y"][q]

    rng = np.random.RandomState(seed)
    idx = rng.choice(d.n, N_TRAIN, replace=False)
    lon = d.Xtr_raw[idx, d.col("Longitude")]
    lat = d.Xtr_raw[idx, d.col("Latitude")]
    qlon = d.Xte_raw[q, d.col("Longitude")]
    qlat = d.Xte_raw[q, d.col("Latitude")]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    nshow = 60
    axL.bar(np.arange(nshow), w_sorted[:nshow], color="#3b6ea5")
    axL.set_xlabel("training case (sorted by weight)")
    axL.set_ylabel("weight $w_i(x)$")
    axL.set_title(f"Weight distribution for one query\n"
                  f"$N_{{\\mathrm{{eff}}}}={neff:.0f}$, $\\Delta_y={dy:.3f}$, "
                  f"top-8 mass $={w_sorted[:8].sum():.3f}$", fontsize=10)

    wit = ev["top_idx"][q]
    sc = axR.scatter(lon, lat, c="#cccccc", s=6, alpha=0.5)
    wlon = d.Xtr_raw[wit, d.col("Longitude")]
    wlat = d.Xtr_raw[wit, d.col("Latitude")]
    wy = m.ytr[wit]
    sc = axR.scatter(wlon, wlat, c=wy, s=130, cmap=POS_CMAP, edgecolors="k",
                     vmin=float(wy.min()), vmax=float(wy.max()), zorder=4)
    axR.scatter([qlon], [qlat], marker="*", s=320, c="yellow", edgecolors="k", zorder=5)
    axR.set_xlabel("Longitude"); axR.set_ylabel("Latitude")
    axR.set_title("Top-8 witnesses (★ = query), colored by label", fontsize=10)
    fig.colorbar(sc, ax=axR, shrink=0.8, label="house value ($100k)")
    fig.suptitle("A California prediction is a weighted vote of specific historical cases")
    return fig


def make_reliability_figure(tw_run: dict, d, n_bins: int = 10):
    """Figure 6.2 (Taiwan Credit): reliability curve (predicted P(default) vs empirical rate)
    for the blended head, with low-N_eff queries highlighted. The smoother is calibrated where
    evidence is dense; N_eff flags where it is not."""
    p = tw_run["preds"]["blended"]
    y = d.yte
    neff = tw_run["evidence"]["neff"]
    lowq = neff < np.quantile(neff, 0.10)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    edges = np.linspace(0, 1, n_bins + 1)
    binid = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        msk = binid == b
        if msk.sum() >= 5:
            xs.append(p[msk].mean()); ys.append(y[msk].mean()); ns.append(msk.sum())
    axL.plot([0, 1], [0, 1], "--", color="#888", lw=1, label="perfect calibration")
    axL.plot(xs, ys, "o-", color="#3b6ea5", label="blended head")
    axL.set_xlabel("predicted P(default)"); axL.set_ylabel("empirical default rate")
    axL.set_xlim(0, 1); axL.set_ylim(0, 1)
    axL.set_title(f"Reliability, blended head (ρ*={tw_run['rho_star']:.1f})", fontsize=10)
    axL.legend(fontsize=9)

    axR.hist(neff, bins=40, color="#c8c8c8")
    axR.axvline(np.quantile(neff, 0.10), ls="--", color="#c44e52", lw=1.5,
                label="10th pctile (flag)")
    axR.set_xlabel("$N_{\\mathrm{eff}}$ (effective cases)"); axR.set_ylabel("queries")
    axR.set_title(f"Evidence concentration; {lowq.sum()} flagged low-$N_{{\\mathrm{{eff}}}}$",
                  fontsize=10)
    axR.legend(fontsize=9)
    fig.suptitle("Taiwan Credit: calibrated where evidence is dense; $N_{\\mathrm{eff}}$ "
                 "flags where it is not")
    return fig


# --- CLI ----------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Chapter 6 — similarity, smoothing, evidence")
    p.add_argument("--out-prefix", default=None)
    args = p.parse_args(argv)
    set_style()

    cal = load_california()
    cal_run = run_dataset(cal)
    print("=" * 70, "\nCALIFORNIA (regression) — case-based prediction on the leaf kernel")
    print(f"  selected sharpening β* = {cal_run['power_star']}")
    print(f"  ρ* (validation-selected blend): {cal_run['rho_star']:.2f}   "
          f"(→ label averaging wins on smooth labels)")
    print(f"  test RMSE   empirical {cal_run['head_loss']['empirical']:.3f}   "
          f"teacher {cal_run['head_loss']['teacher']:.3f}   "
          f"blended {cal_run['head_loss']['blended']:.3f}   "
          f"forest {cal_run['forest_loss']:.3f}")
    print(f"  median N_eff {cal_run['neff_median']:.1f}, median Δ_y "
          f"{cal_run['delta_y_median']:.3f}, median G_q {cal_run['g_q_median']:.3f}")
    print(format_ledger(cal, cal_run["evidence"], well_supported_query(cal_run["evidence"])))

    tw = load_taiwan()
    tw_run = run_dataset(tw)
    print("\n", "=" * 70, "\nTAIWAN CREDIT (classification) — same machine, opposite blend")
    print(f"  selected sharpening β* = {tw_run['power_star']}")
    print(f"  ρ* (validation-selected blend): {tw_run['rho_star']:.2f}   "
          f"(→ teacher averaging wins on noisy binary defaults)")
    print(f"  test log-loss empirical {tw_run['head_loss']['empirical']:.4f}   "
          f"teacher {tw_run['head_loss']['teacher']:.4f}   "
          f"blended {tw_run['head_loss']['blended']:.4f}   "
          f"forest {tw_run['forest_loss']:.4f}")
    print(f"  median N_eff {tw_run['neff_median']:.1f}, median Δ_y "
          f"{tw_run['delta_y_median']:.3f}, median G_q {tw_run['g_q_median']:.3f}")
    # the boundary case: blended probability closest to 0.5, where evidence is most informative
    qb = int(np.argmin(np.abs(tw_run["preds"]["blended"] - 0.5)))
    print(format_ledger(tw, tw_run["evidence"], qb))

    print(f"\nρ* California {cal_run['rho_star']:.2f}  <  ρ* Taiwan {tw_run['rho_star']:.2f}"
          "  — same kernel machine, opposite blends, dictated by label noise.")

    if args.out_prefix:
        import matplotlib
        matplotlib.use("Agg")
        make_witness_figure(cal_run, cal).savefig(f"{args.out_prefix}1_witnesses.pdf")
        make_reliability_figure(tw_run, tw).savefig(f"{args.out_prefix}2_reliability.pdf")
        print("wrote figures with prefix", args.out_prefix)


if __name__ == "__main__":
    main()
