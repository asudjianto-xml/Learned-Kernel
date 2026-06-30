"""Chapter 14 extension --- designing the prior.

The Chapter 14 probe converts "the frozen emitter trails on California" into the measurable
statement "the synthetic prior is too smooth and additive for California's interactions." This
module acts on that diagnosis. It builds the prior as a *generative simulator* fit to the data
(arfpy adversarial random forests; a discriminator-guided importance sampler; and a discriminator-
guided MCMC sampler in Gaussian-copula space), meta-trains the same width-8 emitter on the
synthetic tables, and evaluates zero-shot on the real California test split. It then decomposes the
residual with the architecture levers the emitted geometry implicates --- context size and the
shared interaction geometry W.

Findings (one axis: same emitter, same eval; only the meta-training distribution / one architecture
knob differs):

  * a tree/MCMC-generative prior closes most of the floor-to-ceiling gap (GP/bandwidth floor to the
    real-subtask ceiling), and across three generators of increasing fidelity the transfer is FLAT
    --- zero-shot is fidelity-saturated, so better joint modeling does not help past a point;
  * the residual to the real-subtask ceiling is a constant synthetic-vs-real gap (~0.04);
  * the gap to the fitted models is architectural: context (512->2048) and training W lift the
    real-subtask ceiling past gradient boosting, while head count does not; training W helps only a
    *real* prior --- on a synthetic prior it overfits the simulator's interactions and slightly
    hurts.

Reuses ``ch12.MetaMSSKM`` / ``ch12.meta_train`` / ``ch12.ceiling_incontext_real`` and
``ch08.fit_spectral``; nothing is re-implemented from the spectral machinery.
"""
import numpy as np

from . import ch12
from .ch12 import MetaMSSKM, meta_train, _device


# =============================================================================
# California split (native width 8, no padding --- mirrors the probe exactly)
# =============================================================================

def load_ca8(seed=0, device=None):
    import torch
    from lkbook import load_california
    device = _device(device)
    cal = load_california(seed=seed)
    Xtr = cal.Xtr[:, :8].astype(np.float32)
    Xte = cal.Xte[:, :8].astype(np.float32)
    ytr = np.asarray(cal.ytr, np.float32)
    yte = np.asarray(cal.yte, float)
    return Xtr, ytr, Xte, yte


# =============================================================================
# Generative priors fit to the data
# =============================================================================

def make_arf_sampler(Xtr, ytr, *, num_trees=30, seed=0, verbose=False):
    """arfpy adversarial random forest on the joint [X, y]; returns sample_fn(n) -> (X, y)."""
    import pandas as pd
    from arfpy import arf
    cols = [f"x{j}" for j in range(Xtr.shape[1])] + ["y"]
    Z = pd.DataFrame(np.column_stack([Xtr, ytr]), columns=cols)
    m = arf.arf(x=Z, num_trees=num_trees, random_state=seed, verbose=verbose)
    m.forde()

    def sample_fn(n):
        S = m.forge(n=n)[cols].to_numpy(np.float32)
        return S[:, :-1], S[:, -1]
    return sample_fn


def _softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


class CopulaAdvGenerator:
    """Discriminator-guided importance sampling with a Gaussian-copula proposal. The proposal
    preserves each marginal exactly and the linear correlation, so a CatBoost discriminator
    real(1) vs proposal(0) only has to correct the interaction / tail residual; importance-
    resampling the proposal pool by the accumulated log odds draws from the real joint."""

    def __init__(self, pool_mult=40, rounds=10, seed=0, depth=6, iters=300, auc_stop=0.03):
        self.pool_mult, self.rounds, self.seed = pool_mult, rounds, seed
        self.depth, self.iters, self.auc_stop = depth, iters, auc_stop

    def _copula_pool(self, Z, rng, P):
        from scipy.stats import norm
        n, d = Z.shape
        ranks = np.argsort(np.argsort(Z, axis=0), axis=0)
        zc = norm.ppf((ranks + 1.0) / (n + 1.0))
        cov = np.corrcoef(zc, rowvar=False)
        L = np.linalg.cholesky(cov + 1e-6 * np.eye(d))
        g = rng.standard_normal((P, d)) @ L.T
        pos = np.clip((norm.cdf(g) * n).astype(int), 0, n - 1)
        srt = np.sort(Z, axis=0)
        pool = np.empty((P, d), np.float32)
        for j in range(d):
            pool[:, j] = srt[pos[:, j], j]
        return pool

    def fit(self, Xtr, ytr, verbose=False):
        from catboost import CatBoostClassifier
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(self.seed)
        Z = np.column_stack([Xtr, ytr]).astype(np.float32)
        n = Z.shape[0]; P = n * self.pool_mult
        pool = self._copula_pool(Z, rng, P)
        logw = np.zeros(P); self.aucs = []
        for r in range(self.rounds):
            neg = pool[rng.choice(P, n, p=_softmax(logw), replace=True)]
            Xd = np.vstack([Z, neg]); yd = np.r_[np.ones(n), np.zeros(n)]
            clf = CatBoostClassifier(iterations=self.iters, depth=self.depth, learning_rate=0.05,
                                     verbose=0, random_seed=self.seed + r)
            clf.fit(Xd, yd)
            self.aucs.append(roc_auc_score(yd, clf.predict_proba(Xd)[:, 1]))
            p = np.clip(clf.predict_proba(pool)[:, 1], 1e-4, 1 - 1e-4)
            logw += np.log(p / (1 - p))
            if verbose:
                print(f"      [copula] round {r}: AUC={self.aucs[-1]:.3f}")
            if abs(self.aucs[-1] - 0.5) < self.auc_stop:
                break
        self.pool, self.logw = pool, logw
        self._rng = np.random.default_rng(self.seed + 777)
        return self

    def sample_fn(self):
        w = _softmax(self.logw)
        def fn(n):
            S = self.pool[self._rng.choice(len(self.pool), n, p=w, replace=True)]
            return S[:, :-1], S[:, -1]
        return fn


class MCMCAdvGenerator:
    """Adversarial MCMC in Gaussian-copula space z = Phi^{-1}(rank), where the base is exactly
    N(0, Sigma). Random-walk Metropolis-Hastings targets pi(z) prop. N(z;0,Sigma) * prod_r odds_r,
    with odds_r from a CatBoost discriminator (round r, real vs the current MCMC sample) and the
    inverse map x(z) through empirical quantiles (marginals stay exact, so the discriminator must
    work on the joint). MH avoids the resample degeneracy of sequential importance resampling, so
    iterating the discriminator drives the sample toward the real joint without blowing up the
    discriminator AUC. Energy U(z) = 0.5 z^T Sigma^{-1} z - sum_r logit_r(x(z))."""

    def __init__(self, walkers=8192, rounds=8, mh_steps=220, eps=0.25, seed=0, depth=6, iters=300,
                 disc_n=8192, auc_stop=0.02):
        self.walkers, self.rounds, self.mh_steps, self.eps = walkers, rounds, mh_steps, eps
        self.seed, self.depth, self.iters, self.disc_n, self.auc_stop = seed, depth, iters, disc_n, auc_stop

    def _gaussianize(self, Z):
        from scipy.stats import norm
        n = Z.shape[0]
        ranks = np.argsort(np.argsort(Z, axis=0), axis=0)
        return norm.ppf((ranks + 1.0) / (n + 1.0))

    def _inv_map(self, zz):
        from scipy.stats import norm
        pos = np.clip((norm.cdf(zz) * self.n).astype(np.int64), 0, self.n - 1)
        x = np.empty_like(zz, dtype=np.float32)
        for j in range(self.d):
            x[:, j] = self.srt[pos[:, j], j]
        return x

    def _logit(self, x):
        s = np.zeros(len(x))
        for clf in self.discs:
            p = np.clip(clf.predict_proba(x)[:, 1], 1e-4, 1 - 1e-4)
            s += np.log(p / (1 - p))
        return s

    def _energy(self, zz):
        quad = 0.5 * np.einsum("ij,jk,ik->i", zz, self.Sinv, zz)
        return quad - self._logit(self._inv_map(zz))

    def _mh(self, zz, rng):
        U = self._energy(zz)
        for _ in range(self.mh_steps):
            prop = zz + self.eps * (rng.standard_normal(zz.shape) @ self.L.T)
            Up = self._energy(prop)
            a = rng.random(len(zz)) < np.exp(np.clip(U - Up, -50, 50))
            zz[a] = prop[a]; U[a] = Up[a]
        return zz

    def fit(self, Xtr, ytr, verbose=False):
        from catboost import CatBoostClassifier
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(self.seed)
        Z = np.column_stack([Xtr, ytr]).astype(np.float32)
        self.n, self.d = Z.shape
        self.srt = np.sort(Z, axis=0)
        zc = self._gaussianize(Z)
        Sigma = np.corrcoef(zc, rowvar=False)
        self.L = np.linalg.cholesky(Sigma + 1e-6 * np.eye(self.d))
        self.Sinv = np.linalg.inv(Sigma + 1e-6 * np.eye(self.d))
        self.discs = []; self.aucs = []
        zz = (rng.standard_normal((self.walkers, self.d)) @ self.L.T).astype(np.float32)
        for r in range(self.rounds):
            x_cur = self._inv_map(zz)
            m = min(self.disc_n, self.n, self.walkers)
            real = Z[rng.choice(self.n, m, replace=False)]
            fake = x_cur[rng.choice(self.walkers, m, replace=False)]
            Xd = np.vstack([real, fake]); yd = np.r_[np.ones(m), np.zeros(m)]
            clf = CatBoostClassifier(iterations=self.iters, depth=self.depth, learning_rate=0.05,
                                     verbose=0, random_seed=self.seed + r)
            clf.fit(Xd, yd)
            self.aucs.append(roc_auc_score(yd, clf.predict_proba(Xd)[:, 1]))
            self.discs.append(clf)
            zz = self._mh(zz, rng)
            if verbose:
                print(f"      [MCMC] round {r}: AUC={self.aucs[-1]:.3f}")
            if abs(self.aucs[-1] - 0.5) < self.auc_stop:
                break
        self.pool = self._inv_map(zz)
        self._rng = np.random.default_rng(self.seed + 777)
        return self

    def sample_fn(self):
        def fn(n):
            S = self.pool[self._rng.integers(0, len(self.pool), n)]
            return S[:, :-1], S[:, -1]
        return fn


# =============================================================================
# Emitters / evaluation (one axis: only the meta-training distribution / W differs)
# =============================================================================

def _new_net(seed, device, *, H=4, train_w=False):
    import torch
    torch.manual_seed(seed)
    net = MetaMSSKM(max_features=8, H=H, Q=3, n_quad=6, d_phi=64, decode="krr",
                    pool="pma", seed=seed).to(device)
    net.W.requires_grad_(train_w)
    return net


def train_gp_prior(*, steps=3000, B=32, seed=0, device=None):
    """The floor: the width-8 emitter on the self-consistent GP (bandwidth-continuum) prior."""
    device = _device(device)
    net = _new_net(seed, device)
    meta_train(net, 8, 4, 3, steps=steps, B=B, seed=seed, device=device, log_every=0)
    net.eval()
    return net


def train_on_generator(sample_fn, *, n_syn=8192, steps=3000, B=16, q=64, seed=0, device=None,
                       H=4, train_w=False):
    """Meta-train the width-8 emitter on context/query splits of a synthetic pool from the
    generator (mirrors ``ch12.ceiling_incontext_real`` with a synthetic pool in place of real rows)."""
    import torch
    import torch.nn.functional as F
    device = _device(device)
    Xs, ys = sample_fn(n_syn)
    Xs = torch.tensor(Xs, device=device); ys = torch.tensor(ys, dtype=torch.float32, device=device)
    Ns = Xs.shape[0]
    net = _new_net(seed, device, H=H, train_w=train_w); net.train()
    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.Adam((p for p in net.parameters() if p.requires_grad), lr=2e-3)
    choices = torch.as_tensor((64, 128, 256, 512), device=device)
    for _ in range(steps):
        k = int(choices[torch.randint(len(choices), (1,), generator=gen, device=device)].item())
        Xc = torch.empty(B, k, 8, device=device); yc = torch.empty(B, k, 1, device=device)
        Xq = torch.empty(B, q, 8, device=device); yq = torch.empty(B, q, 1, device=device)
        for b in range(B):
            idx = torch.randperm(Ns, generator=gen, device=device)[:k + q]
            ci, qi = idx[:k], idx[k:]
            Xc[b], Xq[b] = Xs[ci], Xs[qi]
            mn, sd = ys[ci].mean(), ys[ci].std().clamp_min(1e-6)
            yc[b, :, 0] = (ys[ci] - mn) / sd
            yq[b, :, 0] = (ys[qi] - mn) / sd
        opt.zero_grad()
        loss = F.mse_loss(net(Xq, Xc, yc), yq)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in net.parameters() if p.requires_grad), 5.0)
        opt.step()
    net.eval()
    return net


def eval_ca_zeroshot(net, Xtr, ytr, Xte, yte, *, cap=512, reps=5, seed=0, device=None):
    """Zero-shot on real California: real TRAIN rows as context, real TEST as query (mirrors the
    eval of ``ch12.ceiling_incontext_real``)."""
    import torch
    from sklearn.metrics import r2_score
    device = _device(device)
    Xtr_t = torch.tensor(Xtr, device=device); ytr_t = torch.tensor(ytr, device=device)
    Xte_t = torch.tensor(Xte, device=device)
    Ntr = Xtr_t.shape[0]
    egen = torch.Generator(device=device).manual_seed(seed + 123)
    r2s = []
    with torch.no_grad():
        for _ in range(reps):
            ci = torch.randperm(Ntr, generator=egen, device=device)[:cap]
            m, sd = ytr_t[ci].mean(), ytr_t[ci].std().clamp_min(1e-6)
            yc = ((ytr_t[ci] - m) / sd)[None, :, None]
            pred = net(Xte_t[None], Xtr_t[ci][None], yc)[0, :, 0].cpu().numpy() * sd.item() + m.item()
            r2s.append(r2_score(yte, pred))
    return float(np.mean(r2s)), float(np.std(r2s))


def learnability(sample_fn, seed=0):
    """Are the synthetic tasks learnable? GBDT R^2 on a synthetic holdout (a coarse fidelity proxy)."""
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    from sklearn.metrics import r2_score
    X, y = sample_fn(4000); n = 3000
    g = HGB(random_state=seed).fit(X[:n], y[:n])
    return float(r2_score(y[n:], g.predict(X[n:])))


def ceiling_lift(*, train_w=False, H=4, steps=3000, B=16, q=64, caps=(512, 2048), seed=0, device=None):
    """The real-subtask ceiling with architecture knobs relaxed (frozen-W/H=4 reproduces
    ``ch12.ceiling_incontext_real``). Returns {cap: (mean_R2, std_R2)}."""
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import r2_score
    from lkbook import load_california
    device = _device(device)
    cal = load_california(seed=seed)
    Xtr = torch.tensor(cal.Xtr[:, :8].astype(np.float32), device=device)
    ytr = torch.tensor(np.asarray(cal.ytr, np.float32), device=device)
    Xte = torch.tensor(cal.Xte[:, :8].astype(np.float32), device=device)
    yte = np.asarray(cal.yte, float)
    Ntr = Xtr.shape[0]
    net = _new_net(seed, device, H=H, train_w=train_w); net.train()
    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.Adam((p for p in net.parameters() if p.requires_grad), lr=2e-3)
    choices = torch.as_tensor((64, 128, 256, 512), device=device)
    for _ in range(steps):
        k = int(choices[torch.randint(len(choices), (1,), generator=gen, device=device)].item())
        Xc = torch.empty(B, k, 8, device=device); yc = torch.empty(B, k, 1, device=device)
        Xq = torch.empty(B, q, 8, device=device); yq = torch.empty(B, q, 1, device=device)
        for b in range(B):
            idx = torch.randperm(Ntr, generator=gen, device=device)[:k + q]
            ci, qi = idx[:k], idx[k:]
            Xc[b], Xq[b] = Xtr[ci], Xtr[qi]
            m, sd = ytr[ci].mean(), ytr[ci].std().clamp_min(1e-6)
            yc[b, :, 0] = (ytr[ci] - m) / sd
            yq[b, :, 0] = (ytr[qi] - m) / sd
        opt.zero_grad()
        loss = F.mse_loss(net(Xq, Xc, yc), yq)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in net.parameters() if p.requires_grad), 5.0)
        opt.step()
    net.eval()
    egen = torch.Generator(device=device).manual_seed(seed + 123)
    res = {}
    with torch.no_grad():
        for cap in caps:
            r2s = []
            for _ in range(5):
                ci = torch.randperm(Ntr, generator=egen, device=device)[:cap]
                m, sd = ytr[ci].mean(), ytr[ci].std().clamp_min(1e-6)
                yc = ((ytr[ci] - m) / sd)[None, :, None]
                pred = net(Xte[None], Xtr[ci][None], yc)[0, :, 0].cpu().numpy() * sd.item() + m.item()
                r2s.append(r2_score(yte, pred))
            res[cap] = (float(np.mean(r2s)), float(np.std(r2s)))
    return res


# =============================================================================
# Aggregate driver (one call regenerates every number the chapter quotes)
# =============================================================================

def run_all(*, seed=1, steps=3000, device=None, verbose=True):
    """Designing-the-prior experiment. Returns a dict with:
      floor      : GP/bandwidth-prior zero-shot (cap512)
      gens       : {name: {auc, learn, zs512}} for indep/copula/MCMC generative priors
      mcmc2048   : best generative prior (MCMC) at cap2048, frozen vs trained W
      ceiling    : real-subtask ceiling {('frozenW'|'trainW'|'H8'): {cap: (mean,std)}}
      catboost, ch8 : fitted references
    """
    from sklearn.metrics import r2_score
    from catboost import CatBoostRegressor
    from lkbook import load_california
    from lkbook.chapters import ch08
    device = _device(device)
    Xtr, ytr, Xte, yte = load_ca8(seed=seed, device=device)
    cal = load_california(seed=seed)
    out = {}

    if verbose:
        print("floor: GP/bandwidth prior ...")
    net = train_gp_prior(steps=steps, seed=seed, device=device)
    out["floor"] = eval_ca_zeroshot(net, Xtr, ytr, Xte, yte, seed=seed, device=device)

    gens = {}
    fitted_gens = {}
    for name in ("arf", "copula", "mcmc"):
        if verbose:
            print(f"generative prior: {name} ...")
        if name == "copula":
            g = CopulaAdvGenerator(seed=seed).fit(Xtr, ytr)
            fn = g.sample_fn(); auc = g.aucs[-1]
        elif name == "mcmc":
            g = MCMCAdvGenerator(seed=seed).fit(Xtr, ytr)
            fn = g.sample_fn(); auc = g.aucs[-1]; fitted_gens["mcmc"] = g
        else:
            fn = make_arf_sampler(Xtr, ytr, seed=seed); auc = None
        net = train_on_generator(fn, steps=steps, seed=seed, device=device)
        zs = eval_ca_zeroshot(net, Xtr, ytr, Xte, yte, seed=seed, device=device)
        gens[name] = {"auc": auc, "learn": learnability(fn, seed=seed), "zs512": zs}
        if verbose:
            print(f"   {name}: auc={auc} learn={gens[name]['learn']:.3f} zs512={zs[0]:.3f}")
    out["gens"] = gens

    # best generative prior at larger context, frozen vs trained W
    if verbose:
        print("combiner: MCMC prior at cap2048, frozen vs trained W ...")
    fn = fitted_gens["mcmc"].sample_fn()
    out["mcmc2048"] = {}
    for tw in (False, True):
        net = train_on_generator(fn, steps=steps, seed=seed, device=device, train_w=tw)
        out["mcmc2048"]["trainW" if tw else "frozenW"] = \
            eval_ca_zeroshot(net, Xtr, ytr, Xte, yte, cap=2048, seed=seed, device=device)

    # real-subtask ceiling, architecture levers
    if verbose:
        print("ceiling-lift: frozen/train W, H=8 ...")
    out["ceiling"] = {
        "frozenW": ceiling_lift(train_w=False, H=4, steps=steps, seed=seed, device=device),
        "trainW": ceiling_lift(train_w=True, H=4, steps=steps, seed=seed, device=device),
        "H8": ceiling_lift(train_w=False, H=8, steps=steps, seed=seed, device=device),
    }

    cb = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05, verbose=0, random_seed=seed)
    cb.fit(Xtr, ytr)
    out["catboost"] = float(r2_score(yte, cb.predict(Xte)))
    _, pred8 = ch08.fit_spectral(cal.Xtr, np.asarray(cal.ytr, float), mode="learned", H=2, K=8,
                                 steps=500, seed=seed)
    out["ch8"] = float(r2_score(yte, pred8(cal.Xte)))
    return out


# =============================================================================
# Figure
# =============================================================================

def make_designing_figure(res):
    """Figure --- designing the prior. (left) zero-shot transfer is flat across three generators of
    rising fidelity: better joint modeling does not move transfer (fidelity-saturated), and all sit
    between the GP floor and the real-subtask ceiling. (right) decomposition: the generative prior
    closes the prior gap, then context and training W lift the real-subtask ceiling past the fitted
    models, while head count does not."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.7))

    # left: fidelity (learnability) vs transfer, with floor / ceiling bands
    order = ["arf", "copula", "mcmc"]
    labels = {"arf": "ARF", "copula": "copula", "mcmc": "MCMC"}
    learn = [res["gens"][g]["learn"] for g in order]
    zs = [res["gens"][g]["zs512"][0] for g in order]
    floor = res["floor"][0]
    ceil = res["ceiling"]["frozenW"][512][0]
    ax[0].axhline(floor, color="#c0392b", ls="--", lw=1.2, label=f"GP floor ({floor:.2f})")
    ax[0].axhline(ceil, color="#2e7d32", ls="--", lw=1.2, label=f"real-subtask ceiling ({ceil:.2f})")
    ax[0].plot(learn, zs, "o-", color="#3b6fb6", ms=8, lw=1.6)
    for g, lx, zy in zip(order, learn, zs):
        ax[0].annotate(labels[g], (lx, zy), textcoords="offset points", xytext=(6, 7), fontsize=9)
    ax[0].set_xlabel("generator fidelity (synthetic-task learnability $R^2$)")
    ax[0].set_ylabel("California zero-shot $R^2$ (cap 512)")
    ax[0].set_title("Transfer is flat in generator fidelity\n(zero-shot is fidelity-saturated)", fontsize=9.5)
    ax[0].legend(fontsize=8.5, loc="center right")
    ax[0].set_ylim(floor - 0.06, ceil + 0.04)

    # right: decomposition bars
    bars = [
        ("GP\nfloor", floor, "#c0392b"),
        ("generative\nprior (512)", res["gens"]["mcmc"]["zs512"][0], "#c98a3b"),
        ("generative\nprior (2048)", res["mcmc2048"]["frozenW"][0], "#e0a458"),
        ("real ceiling\nfrozen-W (512)", res["ceiling"]["frozenW"][512][0], "#7aa6c2"),
        ("real ceiling\nfrozen-W (2048)", res["ceiling"]["frozenW"][2048][0], "#3b6fb6"),
        ("CatBoost", res["catboost"], "#2e7d32"),
    ]
    xs = np.arange(len(bars))
    ax[1].bar(xs, [b[1] for b in bars], width=0.62, color=[b[2] for b in bars], edgecolor="white")
    ax[1].axhline(res["catboost"], color="#2e7d32", ls=":", lw=1.0)
    ax[1].set_xticks(xs); ax[1].set_xticklabels([b[0] for b in bars], fontsize=8)
    ax[1].set_ylabel("California held-out $R^2$")
    ax[1].set_ylim(0, max(b[1] for b in bars) * 1.16)
    ax[1].set_title("Decomposing the residual\n(prior closes the gap; context lifts the ceiling past CatBoost)",
                    fontsize=9.5)
    for x, b in zip(xs, bars):
        ax[1].text(x, b[1], f"{b[1]:.2f}", ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout()
    return fig
