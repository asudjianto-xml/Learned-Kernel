"""Chapter 1 — every prediction is a weighted vote, and the weights are a geometry.

Fit ridge, a regression tree and k-NN on California Housing; for one held-out query
block compute the implied weight w_i(x) each model places on every training case, and
verify on the data that yhat(x) = sum_i w_i(x) y_i to numerical tolerance. Then read the
geometry off the weights: the top-weighted neighbors, and three influence fields over
the lat/long map. The weight functions and the field helper are importable, so the
Chapter 1 notebook reuses exactly this code, and the book figure regenerates from it.

    python -m lkbook.chapters.ch01 --out fig11_implied_weights.pdf
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt   # backend left as-is, so notebooks render inline
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from lkbook import load_california, load_taiwan, set_style, SIGNED_CMAP, POS_CMAP

LAM, DEPTH, K = 1.0, 6, 30


# --- the three weight functions w_i(x): ridge / tree / k-NN -------------------

def ridge_weights(Xtr, x, lam=LAM):
    """w_i(x) = x^T (X^T X + lam I)^{-1} x_i, with a bias column folded into X."""
    Xb = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    xb = np.append(x, 1.0)
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    return xb @ np.linalg.solve(A, Xb.T)          # length-n vector of weights


def tree_weights(tree, Xtr, x):
    """w_i(x) = 1{same leaf as x} / (leaf size): a flat, axis-aligned box."""
    leaf_x = tree.apply(x[None])[0]
    same = (tree.apply(Xtr) == leaf_x).astype(float)
    return same / same.sum()


def knn_weights(knn, Xtr, x, k=K):
    """w_i(x) = 1/k for the k nearest, 0 otherwise: an isotropic ball."""
    idx = knn.kneighbors(x[None], n_neighbors=k, return_distance=False)[0]
    w = np.zeros(len(Xtr))
    w[idx] = 1.0 / k
    return w


def assert_weighted_vote(w, ytr, pred_model, tol=1e-6):
    """The pedagogical payload: prediction == sum_i w_i(x) y_i on the data."""
    pred_vote = float(w @ ytr)
    assert abs(pred_vote - pred_model) < tol, (pred_vote, pred_model)
    return pred_vote


def top_neighbors(w, ytr, names, X_raw, k=5):
    """The k training cases with the largest |w_i|: the cases that mattered."""
    idx = np.argsort(-np.abs(w))[:k]
    rows = []
    for i in idx:
        snippet = {n: float(X_raw[i, j]) for j, n in enumerate(names[:3])}
        rows.append({"w": float(w[i]), "y": float(ytr[i]), **snippet})
    return rows


def fit_models(d):
    """Fit ridge / tree / k-NN on a RunningData split (shared by script + tests)."""
    ridge = Ridge(alpha=LAM, fit_intercept=False).fit(
        np.hstack([d.Xtr, np.ones((d.n, 1))]), d.ytr)
    tree = DecisionTreeRegressor(max_depth=DEPTH, random_state=0).fit(d.Xtr, d.ytr)
    knn = KNeighborsRegressor(n_neighbors=K).fit(d.Xtr, d.ytr)
    return ridge, tree, knn


# --- shared field helper: influence of a hypothetical case at each location ---

def geo_influence_fields(Gs, ytr, x2, grid_std, depth=DEPTH, k=K, lam=LAM):
    """For a query x2, the influence each geometry assigns to a hypothetical training
    case at every grid location, using geographic-only (lon, lat) models. Returns
    (ridge_field, tree_field, knn_field), each flat over grid_std. Shared by the book
    figure and the notebook widget so they never drift."""
    treeg = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(Gs, ytr)
    knng = KNeighborsRegressor(n_neighbors=k).fit(Gs, ytr)
    Ab = np.hstack([Gs, np.ones((len(Gs), 1))])
    A = Ab.T @ Ab + lam * np.eye(3)

    xb = np.append(x2, 1.0)
    ridge_f = np.hstack([grid_std, np.ones((len(grid_std), 1))]) @ np.linalg.solve(A, xb)
    tree_f = (treeg.apply(grid_std) == treeg.apply(x2[None])[0]).astype(float)
    rk = knng.kneighbors(x2[None], return_distance=True)[0][0, -1]
    knn_f = (np.linalg.norm(grid_std - x2, axis=1) <= rk).astype(float)
    return ridge_f, tree_f, knn_f


def pick_demo_block(Gs, scaler_geo, lon, lat, k=K, target_radius=0.5):
    """A representative moderate-density block (k-NN radius ~ target_radius degrees,
    away from the dense edges) so the box and disk are visible at one scale."""
    knng = KNeighborsRegressor(n_neighbors=k).fit(Gs, np.zeros(len(Gs)))
    deg = scaler_geo.scale_.mean()
    rad = knng.kneighbors(Gs, return_distance=True)[0][:, -1] * deg
    central = (lat > 33.5) & (lat < 38.5)
    qi = np.where(central)[0][np.argmin(np.abs(rad[central] - target_radius))]
    return qi


def make_influence_figure(d, win=2.2, res=300):
    """Figure 1.1: three influence fields over the map for one representative block.
    Returns the matplotlib Figure (the caller decides whether to show or save it)."""
    j_lon, j_lat = d.col("Longitude"), d.col("Latitude")
    G = d.Xtr_raw[:, [j_lon, j_lat]]
    scg = StandardScaler().fit(G)
    Gs = scg.transform(G)
    qi = pick_demo_block(Gs, scg, G[:, 0], G[:, 1])
    qlon, qlat, x2 = G[qi, 0], G[qi, 1], Gs[qi]

    lon_g = np.linspace(qlon - win, qlon + win, res)
    lat_g = np.linspace(qlat - win, qlat + win, res)
    LON, LAT = np.meshgrid(lon_g, lat_g)
    grid_std = scg.transform(np.c_[LON.ravel(), LAT.ravel()])
    rf, tf, kf = geo_influence_fields(Gs, d.ytr, x2, grid_std)

    fields = [(rf.reshape(LON.shape), "ridge: elongated, signed", SIGNED_CMAP, True),
              (tf.reshape(LON.shape), "tree: a hard box", POS_CMAP, False),
              (kf.reshape(LON.shape), "k-NN: a round patch", POS_CMAP, False)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), constrained_layout=True)
    for ax, (F, title, cmap, signed) in zip(axes, fields):
        kw = dict(extent=[lon_g[0], lon_g[-1], lat_g[0], lat_g[-1]],
                  origin="lower", aspect="auto", cmap=cmap)
        if signed:
            v = np.abs(F).max(); kw.update(vmin=-v, vmax=v)
        im = ax.imshow(F, **kw)
        inwin = ((G[:, 0] >= lon_g[0]) & (G[:, 0] <= lon_g[-1]) &
                 (G[:, 1] >= lat_g[0]) & (G[:, 1] <= lat_g[-1]))
        ax.scatter(G[inwin, 0], G[inwin, 1], s=2, c="k", alpha=0.15)
        ax.scatter([qlon], [qlat], marker="*", s=260, c="yellow",
                   edgecolors="k", zorder=5)
        ax.set_xlim(lon_g[0], lon_g[-1]); ax.set_ylim(lat_g[0], lat_g[-1])
        ax.set_title(title); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Influence each geometry assigns over the map, for one query block "
                 "(★) — same data, three geometries")
    return fig


# --- command-line: print the verification table and (optionally) save Figure 1.1 -

def _report(name, d, q):
    ridge, tree, knn = fit_models(d)
    x = d.Xte[q]
    weights = {"ridge": ridge_weights(d.Xtr, x), "tree": tree_weights(tree, d.Xtr, x),
               "k-NN": knn_weights(knn, d.Xtr, x)}
    preds = {"ridge": float(ridge.predict(np.append(x, 1.0)[None])[0]),
             "tree": float(tree.predict(x[None])[0]),
             "k-NN": float(knn.predict(x[None])[0])}
    print(f"\n[{name}]")
    for m in ("ridge", "tree", "k-NN"):
        w, p = weights[m], preds[m]
        pv = assert_weighted_vote(w, d.ytr, p)
        print(f"  {m:5s} model={p:.4f}  sum_i w_i y_i={pv:.4f}  "
              f"sum_i w_i={w.sum():.3f}  #nonzero={int((w != 0).sum())}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Chapter 1 verification + Figure 1.1")
    p.add_argument("--out", default=None,
                   help="path to save Figure 1.1 (PDF/PNG); omit to skip saving")
    args = p.parse_args(argv)

    set_style()
    cal = load_california()
    print("=" * 70, "\nCALIFORNIA HOUSING — query block:",
          ", ".join(f"{n}={cal.Xte_raw[7, j]:.2f}" for j, n in enumerate(cal.names)))
    _report("california", cal, 7)

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        make_influence_figure(cal).savefig(args.out)
        print("\nwrote", args.out)

    tw = load_taiwan()
    print("\n" + "=" * 70, "\nTAIWAN CREDIT — applicant:",
          ", ".join(f"{n}={tw.Xte_raw[3, j]:.0f}" for j, n in enumerate(tw.names[:6])))
    _report("taiwan", tw, 3)


if __name__ == "__main__":
    main()
