"""Chapter 2 — the kernel as the normal form.

Five familiar methods collapse to one machine: fix a positive-semidefinite kernel, form
the Gram matrix, solve one linear system. This module computes the same query's prediction
under each formalism on the running data and shows the identities that are exact:

  - kernel ridge regression (KRR):   α = (K + λI)⁻¹ y,  ŷ(x) = k(x,·)ᵀα
  - Gaussian-process posterior mean:  identical to KRR with λ = σ²  → exactly equal
  - Nadaraya–Watson (NW):             ŷ(x) = Σ k(x,xᵢ) yᵢ / Σ k(x,xᵢ)
  - single-head attention (Gaussian score): exactly the NW estimator → equal to NW
  - support vector regression (SVM):  same kernel, hinge/ε-tube loss, sparse α

The dense Gram is formed on a fixed sub-sample of the training set (Chapter 2 is about the
identity, not scale; scale is Part V). KRR mirrors `fuse-kernel/fusekernel/heads.py:krr_solve`
in NumPy so the package carries no torch dependency.

    python -m lkbook.chapters.ch02 --out-prefix fig2
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR, SVC

from lkbook import load_california, load_taiwan, set_style

N_TRAIN, LAM, SEED = 4000, 1e-2, 0


# --- kernel and solves --------------------------------------------------------

def rbf_gram(A, B, ell):
    """RBF/Gaussian Gram matrix k(a,b) = exp(-‖a-b‖² / 2ℓ²)."""
    a2 = (A * A).sum(1)[:, None]
    b2 = (B * B).sum(1)[None, :]
    d2 = np.maximum(a2 + b2 - 2.0 * A @ B.T, 0.0)
    return np.exp(-d2 / (2.0 * ell * ell))


def krr_alpha(K, y, lam):
    """Dual coefficients α = (K + λI)⁻¹ y — the solve Chapter 2 derives.
    Mirrors fuse-kernel/fusekernel/heads.py:krr_solve (NumPy, no torch)."""
    return np.linalg.solve(K + lam * np.eye(len(y)), y)


def median_lengthscale(X, sample=2000, seed=SEED):
    """Median pairwise-distance heuristic for the RBF bandwidth ℓ."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    Xs = X[idx]
    d2 = np.maximum((Xs * Xs).sum(1)[:, None] + (Xs * Xs).sum(1)[None, :]
                    - 2.0 * Xs @ Xs.T, 0.0)
    iu = np.triu_indices(len(Xs), k=1)
    return float(np.sqrt(np.median(d2[iu])))


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# --- the one-machine demonstration -------------------------------------------

def one_machine(d, q=7, ell=None, lam=LAM, n_train=N_TRAIN, seed=SEED):
    """Predict the query d.Xte[q] five ways under one shared RBF kernel.
    Returns (predictions dict, info dict)."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(d.n, min(n_train, d.n), replace=False)
    Xtr, ytr = d.Xtr[idx], d.ytr[idx]
    x = d.Xte[q]
    if ell is None:
        ell = median_lengthscale(d.Xtr)

    K = rbf_gram(Xtr, Xtr, ell)
    kq = rbf_gram(x[None], Xtr, ell)[0]

    alpha = krr_alpha(K, ytr, lam)
    krr = float(kq @ alpha)
    # GP posterior mean with noise variance σ² = λ: the SAME linear system as KRR
    gp = float(kq @ np.linalg.solve(K + lam * np.eye(len(ytr)), ytr))
    # Nadaraya–Watson: row-normalized kernel smoother
    nw = float((kq @ ytr) / kq.sum())
    # single-head attention with a Gaussian score is exactly NW (values = labels)
    attn = float(softmax(np.log(kq + 1e-300)) @ ytr)   # softmax(log k) = k/Σk = NW weights
    # SVM (ε-tube) on the same RBF kernel: a sparse kernel machine
    svr = SVR(kernel="rbf", gamma=1.0 / (2.0 * ell * ell), C=10.0, epsilon=0.1).fit(Xtr, ytr)
    svm = float(svr.predict(x[None])[0])

    preds = {"KRR": krr, "GP mean": gp, "NW": nw, "attention": attn, "SVM": svm}
    info = {"ell": ell, "lam": lam, "n_train": len(ytr), "n_support": int(svr.n_support_.sum()),
            "query_idx": q}
    return preds, info


def exact_identities(preds, tol=1e-9):
    """The equalities that hold exactly: KRR == GP mean, and attention == NW."""
    assert abs(preds["KRR"] - preds["GP mean"]) < tol, ("KRR vs GP", preds)
    assert abs(preds["attention"] - preds["NW"]) < 1e-9, ("attn vs NW", preds)
    return True


# --- figures ------------------------------------------------------------------

def make_predictions_figure(d, q=7, ell=None, lam=LAM):
    """Figure 2.2 — the same query predicted five ways under one shared kernel."""
    preds, info = one_machine(d, q=q, ell=ell, lam=lam)
    order = ["KRR", "GP mean", "NW", "attention", "SVM"]
    vals = [preds[m] * 100 for m in order]            # $100k -> $k
    anchor = preds["KRR"] * 100
    fig, ax = plt.subplots(figsize=(8, 4.4), constrained_layout=True)
    # color by family: solve K+λI (blue), normalized smoother (red), sparse solve (purple)
    colors = ["#3b6ea5", "#3b6ea5", "#c44e52", "#c44e52", "#8172b3"]
    bars = ax.bar(order, vals, color=colors)
    # brackets marking the two exact identities
    for (a, b), lab in [((0, 1), "KRR = GP mean\n(exact)"), ((2, 3), "attention = NW\n(exact)")]:
        ytop = max(vals[a], vals[b]) + 14
        ax.plot([a, a, b, b], [vals[a] + 3, ytop, ytop, vals[b] + 3], color="k", lw=1)
        ax.text((a + b) / 2, ytop + 2, lab, ha="center", va="bottom", fontsize=8.5)
    for bbar, v in zip(bars, vals):
        ax.text(bbar.get_x() + bbar.get_width() / 2, v / 2, f"${v:.0f}k",
                ha="center", color="white", fontsize=10, weight="bold")
    ax.set_ylabel("prediction ($k)")
    ax.set_title("One query, one shared RBF kernel. The algebra forces two exact identities —\n"
                 "KRR = GP mean and attention = NW — while the loss and the normalization set the rest.",
                 fontsize=9.5)
    ax.set_ylim(0, max(vals) * 1.30)
    return fig


def make_reproducing_figure():
    """Figure 2.1 — the reproducing property f(x) = ⟨f, k(x,·)⟩ as a projection."""
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    f = np.array([1.0, 1.7]); kx = np.array([1.5, 0.4])
    proj = (f @ kx) / (kx @ kx) * kx
    for v, c, lab, dx in [(f, "#c44e52", r"$f$", (0.05, 0.08)),
                          (kx, "#3b6ea5", r"$k(x,\cdot)$", (0.06, -0.12))]:
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2))
        ax.text(v[0] + dx[0], v[1] + dx[1], lab, color=c, fontsize=15)
    ax.annotate("", xy=proj, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#3b6ea5", lw=1.4, alpha=0.6))
    ax.plot([f[0], proj[0]], [f[1], proj[1]], ":", color="gray", lw=1.2)
    ax.text(proj[0] + 0.04, proj[1] - 0.16, r"$f(x)=\langle f,\,k(x,\cdot)\rangle$",
            color="#222", fontsize=12)
    ax.scatter([0], [0], c="k", s=18); ax.text(-0.13, -0.05, "0", fontsize=11)
    ax.set_xlim(-0.3, 2.1); ax.set_ylim(-0.3, 2.1); ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Evaluation is an inner product: the kernel is the geometry,\n"
                 "the RKHS norm is roughness", fontsize=10)
    return fig


# --- Taiwan beat: KRR score vs SVM decision -----------------------------------

def taiwan_decision(d, q=3, ell=None, lam=LAM, n_train=N_TRAIN, seed=SEED):
    """KRR on the 0/1 label gives a class score; the SVM gives a decision. Show they
    agree in sign on a held-out applicant — same kernel, same separating geometry."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(d.n, min(n_train, d.n), replace=False)
    Xtr, ytr = d.Xtr[idx], d.ytr[idx]
    x = d.Xte[q]
    if ell is None:
        ell = median_lengthscale(d.Xtr)
    K = rbf_gram(Xtr, Xtr, ell); kq = rbf_gram(x[None], Xtr, ell)[0]
    score = float(kq @ krr_alpha(K, ytr - ytr.mean(), lam)) + ytr.mean()
    svc = SVC(kernel="rbf", gamma=1.0 / (2.0 * ell * ell), C=1.0).fit(Xtr, ytr)
    decision = int(svc.predict(x[None])[0])
    return {"krr_score": score, "krr_class": int(score > 0.5), "svm_class": decision,
            "n_support": int(svc.n_support_.sum())}


# --- CLI ----------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Chapter 2 — one machine, five names")
    p.add_argument("--out-prefix", default=None, help="save figures as <prefix>1/<prefix>2 .pdf")
    args = p.parse_args(argv)
    set_style()

    cal = load_california()
    preds, info = one_machine(cal)
    exact_identities(preds)
    print("=" * 70, f"\nCALIFORNIA — query block {info['query_idx']}, "
          f"ℓ={info['ell']:.3f}, λ={info['lam']}, n_train={info['n_train']}")
    for m in ("KRR", "GP mean", "NW", "attention", "SVM"):
        print(f"  {m:10s} ${preds[m]*100:.1f}k")
    print(f"  KRR == GP mean (|Δ| < 1e-9): {abs(preds['KRR']-preds['GP mean']):.2e}")
    print(f"  attention == NW (|Δ| < 1e-9): {abs(preds['attention']-preds['NW']):.2e}")
    print(f"  SVM support vectors: {info['n_support']} / {info['n_train']}")

    tw = load_taiwan()
    td = taiwan_decision(tw)
    print("\nTAIWAN — applicant 3:", td)

    if args.out_prefix:
        import matplotlib
        matplotlib.use("Agg")
        make_reproducing_figure().savefig(f"{args.out_prefix}1_reproducing.pdf")
        make_predictions_figure(cal).savefig(f"{args.out_prefix}2_one_machine.pdf")
        print("wrote figures with prefix", args.out_prefix)


if __name__ == "__main__":
    main()
