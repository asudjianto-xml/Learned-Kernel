"""Chapter 7 — what makes a kernel learnable.

Once a kernel carries many parameters its *geometry* can be overfit, and ordinary
in-sample criteria credit the most flexible kernel dishonestly: a near-interpolating
kernel looks best on the very data it was fit on. The cure is to score the kernel on a
held-out **query** fold it never touched. This module makes that operational on the
running data with three demonstrations, reusing the ARD kernel of Chapter 3 and the leaf
kernel of Chapter 4 (NumPy / SciPy / scikit-learn only — no torch):

  (a) **In-sample vs query-fold kernel selection.** Score three candidate kernels — a
      chosen RBF, the learned ARD kernel (Ch. 3), the supervised leaf/tree kernel (Ch. 4)
      — by in-sample fit and by held-out query risk. The leaf kernel nearly interpolates
      its support (support R2 ~ 1) so an in-sample criterion (SURE / GCV on the support)
      hands it all the fusion weight; query-fold selection collapses that weight to its
      honest value and the leakage-free mixture generalizes better.

  (b) **SURE.** Stein's unbiased risk estimate for a *fixed* KRR smoother is exactly
      unbiased for the denoising risk under only second-moment noise. On a controlled
      problem with a known clean signal we show SURE tracks the true denoising risk across
      a ridge sweep and selects near the oracle, while the in-sample residual drives the
      ridge to zero (interpolation) and falls below the noise floor.

  (c) **The two-term bound (Thm D) intuition.** The excess risk splits into a fixed
      kernel-ridge term plus a *selection* term of size sqrt(c(Theta)/n). Empirically the
      train-test gap grows with the free-atom capacity (tree depth K, Cor. D.2) but stays
      flat as convex banks are added (the H axis, Cor. D.3 — sqrt(log H), not sqrt(H)).

    python -m lkbook.chapters.ch07 --out-prefix fig7
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
from sklearn.ensemble import GradientBoostingRegressor

from lkbook import load_california, load_taiwan, set_style
from lkbook.chapters import ch03, ch04

N_SUP, N_QRY, LAM, SEED = 800, 800, 1e-2, 0
T_TREES, DEPTH, LR = 200, 5, 0.1          # the near-interpolating leaf kernel of Ch. 4
LAM_GRID = np.logspace(-4, 1, 16)
N_THREADS = 4                              # this box oversubscribes BLAS; cap it


def _rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def _r2(pred, y, ybar):
    return float(1.0 - np.sum((pred - y) ** 2) / np.sum((y - ybar) ** 2))


def _split(d, n_sup=N_SUP, n_qry=N_QRY, seed=SEED):
    rng = np.random.RandomState(seed)
    idx = rng.choice(d.n, n_sup + n_qry, replace=False)
    s, q = idx[:n_sup], idx[n_sup:]
    return d.Xtr[s], d.ytr[s], d.Xtr[q], d.ytr[q]


# --- candidate kernels, each fit on the SUPPORT only --------------------------

def _fit_ard_ell(Xs, ys, Xq, yq, lam=LAM, maxiter=20):
    """Per-feature ARD length scales selected by held-out query MSE (Ch. 3's leakage-free
    rule, fit here on the support/query split). Reuses ch03.ard_gram."""
    ybar = ys.mean()
    n = len(ys)

    def objective(log_ell):
        K = ch03.ard_gram(Xs, Xs, np.exp(log_ell))
        a = np.linalg.solve(K + lam * np.eye(n), ys - ybar)
        return float(np.mean((ch03.ard_gram(Xq, Xs, np.exp(log_ell)) @ a + ybar - yq) ** 2))

    from scipy.optimize import minimize
    res = minimize(objective, np.zeros(Xs.shape[1]), method="L-BFGS-B",
                   options={"maxiter": maxiter})
    return np.exp(res.x)


def build_candidates(d, seed=SEED, rbf_ell=2.0):
    """Three unit-diagonal PSD candidate kernels, each built from the SUPPORT fold only:
    a chosen isotropic RBF, the learned ARD kernel (Ch. 3), the supervised leaf kernel
    (Ch. 4). Returns names, the support Gram blocks, query blocks, test blocks, and the
    support/query targets — the raw material both selectors consume."""
    Xs, ys, Xq, yq = _split(d, seed=seed)
    ard_ell = _fit_ard_ell(Xs, ys, Xq, yq)
    model = GradientBoostingRegressor(n_estimators=T_TREES, max_depth=DEPTH,
                                      learning_rate=LR, random_state=seed).fit(Xs, ys)
    lk = ch04.LeafKernel().fit(model)

    kfns = [
        ("RBF (chosen)", lambda A, B: ch03.iso_gram(A, B, rbf_ell)),
        ("ARD (learned, Ch. 3)", lambda A, B: ch03.ard_gram(A, B, ard_ell)),
        ("leaf / tree (Ch. 4)", lambda A, B: lk.gram(A, B)),
    ]
    names = [n for n, _ in kfns]
    Bss = [k(Xs, Xs) for _, k in kfns]
    Bqs = [k(Xq, Xs) for _, k in kfns]
    Bts = [k(d.Xte, Xs) for _, k in kfns]
    return dict(names=names, Bss=Bss, Bqs=Bqs, Bts=Bts,
                ys=ys, yq=yq, yte=d.yte, ybar=float(ys.mean()))


# --- per-channel support-vs-query credit (the over-credit problem made visible) ---

def per_channel_credit(d, seed=SEED, lam=LAM):
    """For each candidate kernel alone: support R2 (in-sample fit) vs query R2 (held out)
    vs test RMSE. The leaf kernel's support R2 is near 1 and its query R2 is not — that
    gap is the over-credit problem."""
    with threadpool_limits(N_THREADS):
        c = build_candidates(d, seed=seed)
        n = len(c["ys"]); ybar = c["ybar"]; yc = c["ys"] - ybar
        rows = []
        for name, Kss, Kqs, Kts in zip(c["names"], c["Bss"], c["Bqs"], c["Bts"]):
            a = np.linalg.solve(Kss + lam * np.eye(n), yc)
            sup_r2 = _r2(Kss @ a + ybar, c["ys"], ybar)
            qry_r2 = _r2(Kqs @ a + ybar, c["yq"], ybar)
            test_rmse = _rmse(Kts @ a + ybar, c["yte"])
            rows.append(dict(name=name, support_r2=sup_r2, query_r2=qry_r2,
                             gap=sup_r2 - qry_r2, test_rmse=test_rmse))
    return rows


# --- the two selectors over the fusion simplex --------------------------------

def _simplex_grid(C, res):
    def comps(total, parts):
        if parts == 1:
            yield (total,); return
        for i in range(total + 1):
            for r in comps(total - i, parts - 1):
                yield (i,) + r
    for comp in comps(res, C):
        yield np.asarray(comp, dtype=float) / res


def _mix(blocks, w):
    return sum(wi * b for wi, b in zip(w, blocks))


def select_in_sample_vs_query(d, seed=SEED, grid_res=6):
    """Fit the three-channel fusion two ways. QUERY selection grids the simplex x ridge by
    held-out query R2 (leakage-free). IN-SAMPLE selection grids the same simplex x ridge by
    SURE on the support — a smoother that saw the labels through the supervised tree kernel,
    so SURE undercounts its degrees of freedom and over-credits it. Returns both selected
    weight vectors, ridges and test RMSEs side by side."""
    with threadpool_limits(N_THREADS):
        c = build_candidates(d, seed=seed)
        names, Bss, Bqs, Bts = c["names"], c["Bss"], c["Bqs"], c["Bts"]
        ys, yq, yte, ybar = c["ys"], c["yq"], c["yte"], c["ybar"]
        n = len(ys); yc = ys - ybar; C = len(names)
        cands = list(_simplex_grid(C, grid_res))

        # one eigendecomposition of each fused support Gram, reused for query + SURE + GCV
        best_q = (-np.inf, None, None)
        best_gcv = (np.inf, None)             # to estimate sigma^2 for SURE
        cache = []
        for w in cands:
            Kss = _mix(Bss, w); Kqs = _mix(Bqs, w)
            th, V = np.linalg.eigh(Kss); th = np.clip(th, 0.0, None)
            Vtyc = V.T @ yc; c2 = Vtyc ** 2; M = Kqs @ V
            cache.append((w, th, c2))
            for lam in LAM_GRID:
                coef = Vtyc / (th + lam)
                r2q = _r2(M @ coef + ybar, yq, ybar)
                if r2q > best_q[0]:
                    best_q = (r2q, w, float(lam))
                df = float(np.sum(th / (th + lam)))
                rss = float(np.sum((lam / (th + lam)) ** 2 * c2))
                gcv = (rss / n) / ((1.0 - df / n) ** 2 + 1e-12)
                if gcv < best_gcv[0]:
                    best_gcv = (gcv, (rss, df))

        rss_g, df_g = best_gcv[1]
        sigma2 = max(rss_g / max(n - df_g, 1.0), 1e-6)   # noise estimate from the GCV optimum
        best_s = (np.inf, None, None)
        for w, th, c2 in cache:
            for lam in LAM_GRID:
                df = float(np.sum(th / (th + lam)))
                rss = float(np.sum((lam / (th + lam)) ** 2 * c2))
                sure = rss / n + 2.0 * sigma2 * df / n - sigma2
                if sure < best_s[0]:
                    best_s = (sure, w, float(lam))

        def test_rmse(w, lam):
            Kss = _mix(Bss, w)
            th, V = np.linalg.eigh(Kss); th = np.clip(th, 0.0, None)
            coef = (V.T @ yc) / (th + lam)
            return _rmse(_mix(Bts, w) @ (V @ coef) + ybar, yte)

        wq, lamq = best_q[1], best_q[2]
        ws, lams = best_s[1], best_s[2]
    return dict(
        names=names, sigma2=sigma2,
        query={"weights": dict(zip(names, wq)), "lam": lamq,
               "test_rmse": test_rmse(wq, lamq)},
        in_sample={"weights": dict(zip(names, ws)), "lam": lams,
                   "test_rmse": test_rmse(ws, lams)},
    )


# --- (b) SURE tracks the true denoising risk (controlled, known signal) -------

def sure_tracks_risk(p=600, dim=6, sigma=1.0, ell=1.0, seed=SEED):
    """Controlled denoising problem with a KNOWN clean signal f, so the true denoising risk
    R(S)=||Sx-f||^2/p is computable. Fix an RBF smoother and sweep the ridge. SURE (which
    never sees f) tracks R across the sweep and selects near the oracle; the in-sample
    residual ||x-Sx||^2/p drives the ridge to zero (interpolation) and falls below sigma^2.
    Returns the sweep and the selected risks."""
    with threadpool_limits(N_THREADS):
        rng = np.random.RandomState(seed)
        X = rng.randn(p, dim)
        f = (np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 - X[:, 2] * X[:, 3]
             + 0.3 * np.cos(2 * X[:, 4]))
        f = f - f.mean()
        x = f + rng.randn(p) * sigma
        sigma2 = sigma ** 2                                   # known noise floor
        K = ch03.iso_gram(X, X, ell)
        th, V = np.linalg.eigh(K); th = np.clip(th, 0.0, None)
        xc = x - x.mean(); Vtx = V.T @ xc; c2 = Vtx ** 2
        fc = f - x.mean()
        lams = np.logspace(-3, 2, 60)
        true, sure, insample, dof = [], [], [], []
        for lam in lams:
            g = th / (th + lam)                              # smoother eigenvalues in [0,1)
            Sxc = V @ (g * Vtx)
            true.append(float(np.mean((Sxc - fc) ** 2)))     # true denoising risk (uses f)
            res = float(np.mean((xc - Sxc) ** 2))
            insample.append(res)
            df = float(np.sum(g))
            sure.append(res + 2.0 * sigma2 * df / p - sigma2)
            dof.append(df)
    lams = np.asarray(lams)
    true = np.asarray(true); sure = np.asarray(sure); insample = np.asarray(insample)
    return dict(
        lams=lams, true=true, sure=sure, insample=insample, dof=np.asarray(dof),
        sigma2=sigma2,
        lam_true=float(lams[np.argmin(true)]),
        lam_sure=float(lams[np.argmin(sure)]),
        lam_insample=float(lams[np.argmin(insample)]),
        true_min=float(true.min()),
        true_at_sure=float(true[np.argmin(sure)]),
        true_at_insample=float(true[np.argmin(insample)]),
        max_abs_err=float(np.max(np.abs(sure - true))),
        corr_sure=float(np.corrcoef(sure, true)[0, 1]),
        corr_insample=float(np.corrcoef(insample, true)[0, 1]),
    )


# --- (c) the capacity map: free atoms grow, banks are flat --------------------

def capacity_map(d, depths=(2, 3, 4, 6, 8, 10), Hs=(1, 2, 4, 8, 16, 32),
                 n=800, lam=LAM, seed=SEED):
    """Two capacity axes on California. FREE ATOMS: the leaf kernel's capacity grows with
    tree depth K (more leaves), and the train-test gap grows with it (Cor. D.2, sqrt(K)).
    BANKS: a convex (uniform-simplex) fusion of H RBF banks; the gap is flat in H once a
    useful scale is in the bank (Cor. D.3, sqrt(log H), not sqrt(H))."""
    with threadpool_limits(N_THREADS):
        rng = np.random.RandomState(seed)
        idx = rng.choice(d.n, n, replace=False)
        X, y = d.Xtr[idx], d.ytr[idx]; ybar = y.mean(); yc = y - ybar
        I = np.eye(n)

        free = []
        for depth in depths:
            m = GradientBoostingRegressor(n_estimators=150, max_depth=depth,
                                          learning_rate=LR, random_state=seed).fit(X, y)
            lk = ch04.LeafKernel().fit(m)
            K = lk.gram(X, X)
            a = np.linalg.solve(K + lam * I, yc)
            tr = _rmse(K @ a + ybar, y)
            te = _rmse(lk.gram(d.Xte, X) @ a + ybar, d.yte)
            free.append(dict(K=int(lk.n_cols), depth=int(depth),
                             train=tr, test=te, gap=te - tr))

        banks = []
        for H in Hs:
            ells = np.logspace(-0.5, 1.0, H)
            Bss = [ch03.iso_gram(X, X, e) for e in ells]
            Bts = [ch03.iso_gram(d.Xte, X, e) for e in ells]
            w = np.ones(H) / H                            # convex simplex (uniform)
            K = _mix(Bss, w)
            a = np.linalg.solve(K + lam * I, yc)
            tr = _rmse(K @ a + ybar, y)
            te = _rmse(_mix(Bts, w) @ a + ybar, d.yte)
            banks.append(dict(H=int(H), train=tr, test=te, gap=te - tr))
    return dict(free=free, banks=banks)


# --- figures ------------------------------------------------------------------

def make_credit_figure(d, seed=SEED):
    """Fig 7.1 — honest vs dishonest credit. Left: per-channel support R2 vs query R2 (the
    leaf kernel's gap is the over-credit). Right: fusion weights under in-sample vs query
    selection, with the resulting test RMSE."""
    rows = per_channel_credit(d, seed=seed)
    sel = select_in_sample_vs_query(d, seed=seed)
    names = [r["name"] for r in rows]
    short = [n.split(" (")[0].split(" /")[0] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), constrained_layout=True)

    ax = axes[0]
    xpos = np.arange(len(names)); width = 0.38
    sup = [r["support_r2"] for r in rows]; qry = [r["query_r2"] for r in rows]
    ax.bar(xpos - width / 2, sup, width, label="support $R^2$ (in-sample)", color="#c44e52")
    ax.bar(xpos + width / 2, qry, width, label="query $R^2$ (held out)", color="#3b6ea5")
    for i, r in enumerate(rows):
        ax.text(i - width / 2, r["support_r2"] + 0.01, f"{r['support_r2']:.2f}",
                ha="center", fontsize=8)
        ax.text(i + width / 2, r["query_r2"] + 0.01, f"{r['query_r2']:.2f}",
                ha="center", fontsize=8)
    ax.set_xticks(xpos); ax.set_xticklabels(short, fontsize=9)
    ax.set_ylabel(r"$R^2$"); ax.set_ylim(0, 1.08); ax.legend(fontsize=8.5, loc="lower left")
    ax.set_title("Each candidate alone: the leaf kernel nearly interpolates its support\n"
                 r"(support $R^2\!\approx\!1$) but its query $R^2$ is honest — that gap is "
                 "the over-credit", fontsize=9.5)

    ax = axes[1]
    wq = [sel["query"]["weights"][n] for n in names]
    ws = [sel["in_sample"]["weights"][n] for n in names]
    ax.bar(xpos - width / 2, ws, width, color="#c44e52",
           label=f"in-sample (SURE) — test RMSE {sel['in_sample']['test_rmse']:.3f}")
    ax.bar(xpos + width / 2, wq, width, color="#2ca02c",
           label=f"query fold — test RMSE {sel['query']['test_rmse']:.3f}")
    ax.set_xticks(xpos); ax.set_xticklabels(short, fontsize=9)
    ax.set_ylabel("selected fusion weight"); ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("In-sample selection routes all weight to the near-interpolating tree;\n"
                 "query selection collapses it to its honest value and generalizes better",
                 fontsize=9.5)
    return fig


def make_capacity_figure(d, seed=SEED):
    """Fig 7.2 — the capacity map. Left: train-test gap vs free-atom capacity (tree leaf
    count K) grows (Cor. D.2). Right: gap vs convex bank count H is flat (Cor. D.3)."""
    sure = sure_tracks_risk(seed=seed)
    cap = capacity_map(d, seed=seed)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), constrained_layout=True)

    # left: SURE tracks the true denoising risk; in-sample residual does not
    ax = axes[0]
    ax.semilogx(sure["lams"], sure["true"], "-", color="#2ca02c", lw=2,
                label="true denoising risk (knows $f$)")
    ax.semilogx(sure["lams"], sure["sure"], "--o", color="#3b6ea5", ms=3,
                label=f"SURE (corr {sure['corr_sure']:.2f})")
    ax.semilogx(sure["lams"], sure["insample"], ":", color="#c44e52", lw=2,
                label="in-sample residual")
    ax.axhline(sure["sigma2"], color="0.5", lw=0.8, ls="-")
    ax.text(sure["lams"][0], sure["sigma2"] * 1.04, r"noise floor $\sigma^2$",
            fontsize=8, color="0.4")
    ax.set_xlabel(r"ridge $\lambda$ (log scale)"); ax.set_ylabel("risk")
    ax.set_ylim(0, max(sure["insample"].max(), sure["true"].max()) * 1.05)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("SURE tracks the true denoising risk (near-oracle choice);\n"
                 "the in-sample residual collapses below "
                 r"$\sigma^2$ and picks interpolation", fontsize=9.5)

    # right: the capacity map — free atoms grow, banks flat. Both axes share a common
    # "widen the family" step index so the contrast is a fair side-by-side; the actual
    # K (leaf count) and H (bank count) are annotated. Drop the degenerate single-bank
    # interpolator (H=1) so the bank line shows the multi-scale regime.
    ax = axes[1]
    gfree = [r["gap"] for r in cap["free"]]; Kfree = [r["K"] for r in cap["free"]]
    xf = np.arange(1, len(gfree) + 1)
    ax.plot(xf, gfree, "-o", color="#c44e52", ms=5,
            label=r"free atoms (leaf count $K$): grows  (Cor. D.2, $\sqrt{K}$)")
    banks = [r for r in cap["banks"] if r["H"] >= 2]
    gb = [r["gap"] for r in banks]; Hb = [r["H"] for r in banks]
    xb = np.arange(1, len(gb) + 1)
    ax.plot(xb, gb, "-s", color="#3b6ea5", ms=5,
            label=r"convex banks (count $H$): flat  (Cor. D.3, $\sqrt{\log H}$)")
    for x, K in zip(xf, Kfree):
        ax.annotate(f"K={K}", (x, gfree[x - 1]), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=6.5, color="#c44e52")
    for x, H in zip(xb, Hb):
        ax.annotate(f"H={H}", (x, gb[x - 1]), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=6.5, color="#3b6ea5")
    ax.set_xlabel("steps widening the kernel family  $\\longrightarrow$")
    ax.set_ylabel("train$-$test RMSE gap")
    ax.set_xticks(xf)
    ax.legend(fontsize=8.0, loc="lower right")
    ax.set_title("The selection term is a map: raising the per-feature atom count $K$\n"
                 "widens the family fastest; adding convex banks is near-free", fontsize=9.5)
    return fig


# --- CLI ----------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Chapter 7 — what makes a kernel learnable")
    p.add_argument("--out-prefix", default=None)
    args = p.parse_args(argv)
    set_style()
    cal = load_california()

    print("=" * 72, "\nPER-CHANNEL CREDIT on California (support fit vs held-out query)")
    rows = per_channel_credit(cal)
    print(f"  {'kernel':22s} {'support R2':>11s} {'query R2':>9s} {'gap':>7s} {'test RMSE':>10s}")
    for r in rows:
        print(f"  {r['name']:22s} {r['support_r2']:11.4f} {r['query_r2']:9.4f} "
              f"{r['gap']:7.4f} {r['test_rmse']:10.4f}")

    sel = select_in_sample_vs_query(cal)
    print("\nFUSION SELECTION (3 channels: RBF / ARD / leaf), sigma^2 est = "
          f"{sel['sigma2']:.4f}")
    for tag in ("in_sample", "query"):
        s = sel[tag]
        ws = ", ".join(f"{n.split(' (')[0]}={w:.2f}" for n, w in s["weights"].items())
        label = "IN-SAMPLE (SURE on support)" if tag == "in_sample" else "QUERY-FOLD (held out)"
        print(f"  {label:30s} w=[{ws}] lam={s['lam']:.4g} -> test RMSE {s['test_rmse']:.4f}")

    print("\nSURE TRACKS THE TRUE DENOISING RISK (controlled, known signal)")
    sr = sure_tracks_risk()
    print(f"  argmin lambda: true={sr['lam_true']:.3g}  SURE={sr['lam_sure']:.3g}  "
          f"in-sample={sr['lam_insample']:.3g}")
    print(f"  oracle true risk {sr['true_min']:.4f}; SURE-selected {sr['true_at_sure']:.4f}; "
          f"in-sample-selected {sr['true_at_insample']:.4f}")
    print(f"  corr(SURE,true)={sr['corr_sure']:.3f}  corr(in-sample,true)={sr['corr_insample']:.3f}"
          f"  max|SURE-true|={sr['max_abs_err']:.4f}")

    print("\nCAPACITY MAP (train-test gap)")
    cap = capacity_map(cal)
    print("  free atoms (leaf count K):  " +
          ", ".join(f"K={r['K']}:{r['gap']:.2f}" for r in cap["free"]))
    print("  convex banks (count H):     " +
          ", ".join(f"H={r['H']}:{r['gap']:.2f}" for r in cap["banks"]))

    tw = load_taiwan()
    rows_tw = per_channel_credit(tw)
    lk_tw = next(r for r in rows_tw if r["name"].startswith("leaf"))
    print(f"\nTAIWAN: leaf-kernel support R2 {lk_tw['support_r2']:.3f} vs query R2 "
          f"{lk_tw['query_r2']:.3f} (gap {lk_tw['gap']:.3f}) — same over-credit pattern")

    if args.out_prefix:
        import matplotlib
        matplotlib.use("Agg")
        make_credit_figure(cal).savefig(f"{args.out_prefix}1_credit.pdf")
        make_capacity_figure(cal).savefig(f"{args.out_prefix}2_capacity.pdf")
        print("\nwrote figures with prefix", args.out_prefix)


if __name__ == "__main__":
    main()
