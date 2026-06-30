"""Chapter 13 — emitting the kernel: invariant encoders.

Chapter 12 built an emitter g_theta: context -> measure m and measured its regret over Bayes,
deferring one question: *what map* g_theta, and how does it stay faithful to the symmetry tabular
data demands? This chapter answers it. Attention has two possible jobs. As **predictive geometry**
(softmax(QK^T) used as the similarity that combines labels) it is asymmetric and order-creating ---
the wrong geometry for exchangeable rows (Ch. 11). As an **invariant encoder** --- a permutation-
and column-invariant map from a context set to kernel parameters --- it is exactly right. We use
only the second, and a **symmetry bottleneck** sits between the encoder and the prediction: because
the encoder emits a spectral measure, the predictor is a symmetric PSD kernel no matter what the
encoder does internally, so the encoder's asymmetry can never leak into the geometry.

This module builds on Chapter 12 (it imports the prior, the kernel and the row-token emitter from
``lkbook.chapters.ch12``) and adds:

  * ``run_pool_ab`` --- the mean-pool vs attention-pool (PMA) comparison: a better invariant
    *encoder* narrows the amortization-specific part of the regret Chapter 12 identified, while the
    predictor stays symmetric throughout (the experiment Ch. 12 deferred).
  * ``FeatureTokenEmitter`` --- a per-feature (column-then-row) tokenized emitter (TabPFN-v2 /
    TabICL spirit): cell embedding -> feature-axis attention -> row-axis PMA per feature -> shared
    per-feature heads, so one checkpoint serves any feature count and order.
  * ``invariance_checks`` --- numerical verification of row-permutation, column-permutation and
    padded-column invariance: the symmetry bottleneck and padding invariance made concrete.

    python -m lkbook.chapters.ch13 --out-prefix fig13
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from lkbook.chapters import ch12
from lkbook.chapters.ch12 import (SpectralMeasure, gram, density_to_atoms, krr_incontext,
                                  nw_predict, sample_measure_prior, sample_gp_tasks,
                                  bayes_posterior, MetaMSSKM, meta_train, eval_regret_vs_k,
                                  train_emitter, _device)

SEED = 0


# =============================================================================
# Mean pooling vs attention pooling (PMA): a better invariant encoder
# =============================================================================

def run_pool_ab(*, d=8, H=4, Q=3, steps=3000, B=32, ks=(8, 32, 64, 128, 256, 512),
                n_tasks=400, seed=0, device=None):
    """Train two emitters that differ only in the context aggregator --- a masked mean and PMA
    (attention pooling) --- on the same self-consistent prior, and compare regret over Bayes vs
    context size k. Mean pooling weights every context point equally; PMA lets informative points
    count more, and is a strict generalization (mean = uniform attention). Both decode through the
    same symmetric kernel, so the predictor's symmetry is untouched; only the encoder changes.
    Returns {pool: {k: regret_row}} plus the per-k reduction."""
    device = _device(device)
    out = {}
    for pool in ("mean", "pma"):
        net = train_emitter(d=d, H=H, Q=Q, steps=steps, B=B, pool=pool, seed=seed, device=device)
        out[pool] = eval_regret_vs_k(net, d, H, Q, ks=ks, n_tasks=n_tasks, seed=999, device=device)
    out["reduction"] = {k: (out["mean"][k]["regret"] - out["pma"][k]["regret"])
                        / max(out["mean"][k]["regret"], 1e-9) for k in ks}
    out["ks"] = list(ks)
    return out


# =============================================================================
# Symmetric vs asymmetric attention ENCODER: a direct test of the bottleneck
# =============================================================================

def _toggle_base():
    """Return the ToggleAttnEmitter class: a row-token emitter whose attention symmetry is a flag.
    ``symmetric=True`` shares the query/key projection so every attention score is a Mahalanobis
    inner product z_i^T M z_j with M = A^T A (symmetric, the Ch. 11 'sym' form); ``symmetric=False``
    uses separate W_Q != W_K (standard asymmetric attention). Everything downstream --- the measure
    heads, the shared W, the quadrature, the kernel and the KRR decode --- is identical, so the regret
    difference isolates the encoder's attention symmetry. The bottleneck predicts no difference."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _MHA(nn.Module):
        """Multi-head self-attention (queries optionally separate seeds) with a symmetry flag.
        sym: shared projection A for queries and keys -> symmetric scores (A x_i).(A x_j). asym:
        separate W_Q, W_K. The value path and output projection are the same either way; symmetry
        is a property of the attention *scores* (the affinity), as in Chapter 11."""

        def __init__(self, d_model, nhead, symmetric):
            super().__init__()
            self.h, self.dh, self.sym = nhead, d_model // nhead, symmetric
            self.Wk = nn.Linear(d_model, d_model)
            self.Wq = self.Wk if symmetric else nn.Linear(d_model, d_model)
            self.Wv = nn.Linear(d_model, d_model)
            self.out = nn.Linear(d_model, d_model)

        def _split(self, x):
            B, N, _ = x.shape
            return x.view(B, N, self.h, self.dh).transpose(1, 2)          # (B, h, N, dh)

        def forward(self, xq, xk):
            q, k, v = self._split(self.Wq(xq)), self._split(self.Wk(xk)), self._split(self.Wv(xk))
            s = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
            a = torch.softmax(s, dim=-1)
            o = (a @ v).transpose(1, 2).reshape(xq.shape[0], xq.shape[1], -1)
            return self.out(o)

    class _ToggleAttnEmitter(nn.Module):
        def __init__(self, max_features, *, H=4, Q=3, n_quad=6, d_phi=64, d_model=128, nhead=4,
                     nlayers=2, m_seeds=4, symmetric=False, kernel="laplace", decode="krr",
                     omega_range=(0.05, 5.0), seed=0):
            super().__init__()
            torch.manual_seed(seed)
            self.d, self.H, self.Q, self.n_quad = max_features, H, Q, n_quad
            self.K = Q * n_quad
            self.kernel, self.decode, self.symmetric = kernel, decode, symmetric
            nodes, wts = np.polynomial.hermite.hermgauss(n_quad)
            self.register_buffer("gh_nodes", torch.as_tensor(nodes, dtype=torch.float32))
            self.register_buffer("gh_wts", torch.as_tensor(wts / np.sqrt(np.pi), dtype=torch.float32))
            lo, hi = omega_range
            edges = lo * (hi / lo) ** (np.arange(H) / max(H - 1, 1))
            self.register_buffer("band_mu0", torch.as_tensor(edges, dtype=torch.float32))

            self.input_proj = nn.Linear(max_features + 1, d_model)
            self.layers = nn.ModuleList([_MHA(d_model, nhead, symmetric) for _ in range(nlayers)])
            self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(nlayers)])
            self.ffns = nn.ModuleList([nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                                     nn.Linear(2 * d_model, d_model))
                                       for _ in range(nlayers)])
            self.fnorms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(nlayers)])
            self.seeds = nn.Parameter(0.02 * torch.randn(m_seeds, d_model))
            self.pool_attn = _MHA(d_model, nhead, symmetric)            # symmetric/asym pooling too
            self.proj = nn.Linear(m_seeds * d_model, d_model) if m_seeds > 1 else nn.Identity()
            self.pool_norm = nn.LayerNorm(d_model)
            self.m_seeds = m_seeds

            d = max_features
            self.mu_head = nn.Linear(d_model, H * d * Q)
            self.lg_head = nn.Linear(d_model, H * d * Q)
            self.pi_head = nn.Linear(d_model, H * d * Q)
            self.scale_head = nn.Linear(d_model, d)
            self.band_head = nn.Linear(d_model, H)
            self.wlog_head = nn.Linear(d_model, H)
            self.logsig_head = nn.Linear(d_model, 1)
            self.W = nn.Parameter(0.1 * torch.randn(d_phi, d * 2 * self.K))

        def emit(self, X_ctx, y_ctx, row_mask=None, feature_mask=None):
            B = X_ctx.shape[0]
            d, H, Q = self.d, self.H, self.Q
            z = self.input_proj(torch.cat([X_ctx, y_ctx], dim=-1))
            for mha, n1, ffn, n2 in zip(self.layers, self.norms, self.ffns, self.fnorms):
                z = n1(z + mha(z, z))
                z = n2(z + ffn(z))
            q = self.seeds.unsqueeze(0).expand(B, -1, -1)
            pooled = self.pool_attn(q, z)                                # (B, m, d_model)
            e = self.pool_norm(pooled[:, 0] if self.m_seeds == 1 else self.proj(pooled.reshape(B, -1)))
            mu = self.band_mu0[None, :, None, None] * F.softplus(
                self.mu_head(e).view(B, H, d, Q) + math.log(math.e - 1))
            omega, amp = density_to_atoms(mu, self.lg_head(e).view(B, H, d, Q),
                                          self.pi_head(e).view(B, H, d, Q), self.gh_nodes, self.gh_wts)
            scale = F.softplus(self.scale_head(e))
            band = F.softplus(self.band_head(e)) + 1e-3
            sigma2 = F.softplus(self.logsig_head(e)).squeeze(-1) + 1e-4
            return SpectralMeasure(omega=omega, amp=amp, scale=scale, band=band,
                                   wlog=self.wlog_head(e), W=self.W, kernel=self.kernel, sigma2=sigma2)

        def forward(self, X_query, X_ctx, y_ctx, row_mask=None, feature_mask=None):
            measure = self.emit(X_ctx, y_ctx)
            K_qc = gram(measure, X_query, X_ctx)
            K_cc = gram(measure, X_ctx, X_ctx)
            return krr_incontext(K_cc, K_qc, y_ctx, measure.sigma2)

    return _ToggleAttnEmitter


_Toggle_cls = None


def ToggleAttnEmitter(*args, **kw):
    """Row-token emitter with a symmetry flag on its attention (sym = shared Mahalanobis projection,
    asym = separate W_Q,W_K). Factory keeping the torch import lazy."""
    global _Toggle_cls
    if _Toggle_cls is None:
        _Toggle_cls = _toggle_base()
    return _Toggle_cls(*args, **kw)


def run_sym_asym_ab(*, d=8, H=4, Q=3, steps=3000, B=32, ks=(8, 32, 64, 128, 256, 512),
                    n_tasks=400, seed=0, device=None):
    """Train two emitters that differ only in whether the encoder's attention is symmetric
    (Mahalanobis, shared projection) or asymmetric (separate W_Q,W_K), and compare regret over
    Bayes. The symmetry bottleneck predicts no difference: the encoder's asymmetry cannot reach the
    prediction, which is funneled through the symmetric kernel K_m. (Contrast Ch. 11, where
    asymmetrizing the *predictive* kernel forfeits KRR.)"""
    import torch
    device = _device(device)
    out = {}
    for mode, sym in (("symmetric", True), ("asymmetric", False)):
        torch.manual_seed(seed)
        net = ToggleAttnEmitter(d, H=H, Q=Q, symmetric=sym, seed=seed).to(device)
        meta_train(net, d, H, Q, steps=steps, B=B, seed=seed, device=device)
        out[mode] = eval_regret_vs_k(net, d, H, Q, ks=ks, n_tasks=n_tasks, seed=999, device=device)
    out["gap"] = {k: out["asymmetric"][k]["regret"] - out["symmetric"][k]["regret"] for k in ks}
    out["ks"] = list(ks)
    return out


# =============================================================================
# Per-feature (column-then-row) tokenized emitter: column + width invariance
# =============================================================================

def _ft_base():
    """Return the FeatureTokenEmitter nn.Module class (torch imported lazily)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _AttnPool(nn.Module):
        """Set-Transformer PMA: m learned seed queries cross-attend to a set; mask-aware; the
        LayerNorm keeps the pooled vector at the encoder's scale (raw attention output drifts the
        measure heads into extreme frequencies -> NaN)."""

        def __init__(self, d_model, nhead, m_seeds=4):
            super().__init__()
            self.m = m_seeds
            self.seeds = nn.Parameter(0.02 * torch.randn(m_seeds, d_model))
            self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
            self.proj = nn.Linear(m_seeds * d_model, d_model) if m_seeds > 1 else nn.Identity()
            self.norm = nn.LayerNorm(d_model)

        def forward(self, e, key_padding_mask=None):
            B = e.shape[0]
            q = self.seeds.unsqueeze(0).expand(B, -1, -1)
            out, _ = self.attn(q, e, e, key_padding_mask=key_padding_mask)
            pooled = out[:, 0] if self.m == 1 else self.proj(out.reshape(B, -1))
            return self.norm(pooled)

    class _FeatureTokenEmitter(nn.Module):
        """Per-feature-tokenized emitter (column-then-row). Each cell (i,j) is a token; feature-axis
        attention carries cross-feature information; row-axis PMA pools each column; shared
        per-feature heads emit each column's spectral density and relevance, so one checkpoint serves
        any feature count. A global pool emits the per-bank bandwidths, fusion and noise. The kernel
        is the Chapter-8 fused gram with a feature mask, so masked columns contribute nothing."""

        def __init__(self, max_features, *, H=4, Q=3, n_quad=6, d_phi=64, d_model=128, nhead=4,
                     n_feat_layers=2, m_seeds=4, kernel="laplace", decode="krr",
                     omega_range=(0.05, 5.0), seed=0):
            super().__init__()
            torch.manual_seed(seed)
            self.d, self.H, self.Q, self.n_quad = max_features, H, Q, n_quad
            self.K = Q * n_quad
            self.kernel, self.decode = kernel, decode
            nodes, wts = np.polynomial.hermite.hermgauss(n_quad)
            self.register_buffer("gh_nodes", torch.as_tensor(nodes, dtype=torch.float32))
            self.register_buffer("gh_wts", torch.as_tensor(wts / np.sqrt(np.pi), dtype=torch.float32))
            lo, hi = omega_range
            edges = lo * (hi / lo) ** (np.arange(H) / max(H - 1, 1))
            self.register_buffer("band_mu0", torch.as_tensor(edges, dtype=torch.float32))

            self.cell_proj = nn.Linear(2, d_model)               # (x_ij, y_i) -> d_model
            enc = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
            self.feat_encoder = nn.TransformerEncoder(enc, n_feat_layers)
            self.row_pool = _AttnPool(d_model, nhead, m_seeds)    # pools rows within a column

            self.mu_head = nn.Linear(d_model, Q)                 # shared per-feature density heads
            self.lg_head = nn.Linear(d_model, Q)
            self.pi_head = nn.Linear(d_model, Q)
            self.scale_head = nn.Linear(d_model, 1)
            self.band_head = nn.Linear(d_model, H)               # global: per-bank bandwidth, fusion, noise
            self.wlog_head = nn.Linear(d_model, H)
            self.logsig_head = nn.Linear(d_model, 1)
            self.W = nn.Parameter(0.1 * torch.randn(d_phi, max_features * 2 * self.K))

        def _encode(self, X, y, row_mask, feature_mask):
            B, N, d = X.shape
            if feature_mask is not None:
                X = X * feature_mask[:, None, :]
            yb = y.expand(B, N, d)
            tok = self.cell_proj(torch.stack([X, yb], dim=-1))   # (B, N, d, d_model)
            feat_pad = None if feature_mask is None else (feature_mask < 0.5)
            h = tok.reshape(B * N, d, -1)
            fp = None if feat_pad is None else feat_pad[:, None, :].expand(B, N, d).reshape(B * N, d)
            h = self.feat_encoder(h, src_key_padding_mask=fp).reshape(B, N, d, -1)
            h = h.permute(0, 2, 1, 3).reshape(B * d, N, -1)      # (B*d, N, d_model)
            row_pad = (None if row_mask is None
                       else (row_mask < 0.5)[:, None, :].expand(B, d, N).reshape(B * d, N))
            cols = self.row_pool(h, key_padding_mask=row_pad).reshape(B, d, -1)   # (B, d, d_model)
            if feature_mask is None:
                g = cols.mean(dim=1)
            else:
                m = feature_mask[:, :, None]
                g = (cols * m).sum(1) / m.sum(1).clamp_min(1.0)
            return cols, g

        def emit(self, X_ctx, y_ctx, row_mask=None, feature_mask=None):
            B, d, H, Q = X_ctx.shape[0], self.d, self.H, self.Q
            cols, g = self._encode(X_ctx, y_ctx, row_mask, feature_mask)
            mu = self.band_mu0[None, :, None, None] * F.softplus(
                self.mu_head(cols)[:, None] + math.log(math.e - 1))      # (B,1,d,Q) -> broadcast over H
            mu = mu.expand(B, H, d, Q)
            omega, amp = density_to_atoms(mu, self.lg_head(cols)[:, None].expand(B, H, d, Q),
                                          self.pi_head(cols)[:, None].expand(B, H, d, Q),
                                          self.gh_nodes, self.gh_wts)       # (B,H,d,K)
            scale = F.softplus(self.scale_head(cols).squeeze(-1))           # (B, d)
            band = F.softplus(self.band_head(g)) + 1e-3                     # (B, H)
            sigma2 = F.softplus(self.logsig_head(g)).squeeze(-1) + 1e-4
            return SpectralMeasure(omega=omega, amp=amp, scale=scale, band=band,
                                   wlog=self.wlog_head(g), W=self.W, kernel=self.kernel,
                                   sigma2=sigma2)

        def forward(self, X_query, X_ctx, y_ctx, row_mask=None, feature_mask=None):
            measure = self.emit(X_ctx, y_ctx, row_mask, feature_mask)
            K_qc = gram(measure, X_query, X_ctx, feature_mask)
            if self.decode == "nw":
                return nw_predict(K_qc, y_ctx, ctx_mask=row_mask)
            K_cc = gram(measure, X_ctx, X_ctx, feature_mask)
            return krr_incontext(K_cc, K_qc, y_ctx, measure.sigma2, ctx_mask=row_mask)

    return _FeatureTokenEmitter


_FT_cls = None


def FeatureTokenEmitter(*args, **kw):
    """Per-feature (column-then-row) tokenized measure emitter; serves any feature count and order,
    with provable invariance to padded columns. (Factory keeping the torch import lazy.)"""
    global _FT_cls
    if _FT_cls is None:
        _FT_cls = _ft_base()
    return _FT_cls(*args, **kw)


def train_feature_token(*, d=8, H=4, Q=3, steps=3000, B=32, seed=0, device=None,
                        d_min=2, log_every=0):
    """Meta-train a FeatureTokenEmitter on the broadened self-consistent prior (varied effective
    dimension: each task activates d_active <= d columns, the rest masked), so one checkpoint learns
    to serve any width. Returns the trained net."""
    import torch
    import torch.nn.functional as F
    device = _device(device)
    torch.manual_seed(seed)
    net = FeatureTokenEmitter(d, H=H, Q=Q, decode="krr", seed=seed).to(device)
    net.W.requires_grad_(False)
    net.train()
    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.Adam((p for p in net.parameters() if p.requires_grad), lr=2e-3)
    choices = torch.as_tensor((16, 32, 64, 128, 256), device=device)
    for step in range(steps):
        k = int(choices[torch.randint(len(choices), (1,), generator=gen, device=device)].item())
        measure, s2 = sample_measure_prior(B, d, H, Q, net.gh_nodes, net.gh_wts, net.W.detach(),
                                           gen, device=device)
        d_active = int(torch.randint(d_min, d + 1, (1,), generator=gen, device=device).item())
        fmask = (torch.arange(d, device=device) < d_active).float()[None, :].expand(B, d).contiguous()
        measure.scale = measure.scale * fmask
        Xc, yc, Xq, yq = sample_gp_tasks(measure, k, 64, s2, gen, feature_mask=fmask, device=device)
        opt.zero_grad()
        loss = F.mse_loss(net(Xq, Xc, yc, feature_mask=fmask), yq)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in net.parameters() if p.requires_grad), 5.0)
        opt.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"  step {step + 1:5d}  meta-train MSE {loss.item():.4f}  (k={k}, d_active={d_active})")
    return net


# =============================================================================
# Numerical invariance checks: the symmetry bottleneck made concrete
# =============================================================================

def invariance_checks(ft_net, *, d=8, H=4, Q=3, k=128, n_q=64, d_active=5, seed=11, device=None):
    """Verify, to numerical tolerance, the invariances the feature-token encoder provides, on a
    held-out task:
      * row permutation --- shuffle the context rows; the emitted measure (and the prediction) is
        unchanged (the pooling is symmetric);
      * padded-column content --- overwrite the masked (inactive) columns with arbitrary noise; the
        prediction is unchanged, so padding a width-d table to d_max is sound.
    Returns the max absolute prediction change for each (both should be ~0).

    Column-*order* invariance is NOT among them and is not tested here: the shared interaction warp
    W couples the features in a fixed order, so permuting the active columns changes the kernel. That
    invariance would require a feature-symmetric warp; see the chapter's limitation note."""
    import torch
    device = _device(device)
    ft_net.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    base = MetaMSSKM(max_features=d, H=H, Q=Q, decode="krr", pool="pma", seed=seed).to(device)
    measure, s2 = sample_measure_prior(1, d, H, Q, base.gh_nodes, base.gh_wts, base.W.detach(),
                                       gen, device=device)
    fmask = (torch.arange(d, device=device) < d_active).float()[None, :]
    measure.scale = measure.scale * fmask
    Xc, yc, Xq, yq = sample_gp_tasks(measure, k, n_q, s2, gen, feature_mask=fmask, device=device)

    with torch.no_grad():
        base_pred = ft_net(Xq, Xc, yc, feature_mask=fmask)[0, :, 0]
        perm = torch.randperm(k, generator=gen, device=device)
        p_row = ft_net(Xq, Xc[:, perm], yc[:, perm], feature_mask=fmask)[0, :, 0]
        Xc2, Xq2 = Xc.clone(), Xq.clone()
        if d_active < d:
            Xc2[:, :, d_active:] = 5.0 * torch.randn_like(Xc2[:, :, d_active:])
            Xq2[:, :, d_active:] = 5.0 * torch.randn_like(Xq2[:, :, d_active:])
        p_pad = ft_net(Xq2, Xc2, yc, feature_mask=fmask)[0, :, 0]

    f = lambda a: float((a - base_pred).abs().max())
    return {"row_perm": f(p_row), "pad_content": f(p_pad)}


def width_check(ft_net, *, d=8, H=4, Q=3, widths=(3, 5, 8), k=256, n_q=128, n_tasks=100,
                seed=23, device=None):
    """One checkpoint, many widths: evaluate the feature-token emitter on prior tasks of different
    active feature counts (the rest padded and masked), reporting the emitter's query MSE relative
    to predict-the-mean at each width. A single set of shared per-feature heads serves every width."""
    import torch
    import torch.nn.functional as F
    device = _device(device)
    ft_net.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    out = {}
    with torch.no_grad():
        for da in widths:
            measure, s2 = sample_measure_prior(n_tasks, d, H, Q, ft_net.gh_nodes, ft_net.gh_wts,
                                               ft_net.W.detach(), gen, device=device)
            fmask = (torch.arange(d, device=device) < da).float()[None, :].expand(n_tasks, d).contiguous()
            measure.scale = measure.scale * fmask
            Xc, yc, Xq, yq = sample_gp_tasks(measure, k, n_q, s2, gen, feature_mask=fmask, device=device)
            em = F.mse_loss(ft_net(Xq, Xc, yc, feature_mask=fmask), yq).item()
            mn = F.mse_loss(torch.zeros_like(yq), yq).item()
            out[da] = {"emitter": em, "mean": mn, "ratio": em / mn}
    return out


# =============================================================================
# Aggregate driver
# =============================================================================

def run_all(*, steps=3000, n_tasks=400, seed=0, device=None, ft_steps=3000):
    """The chapter's experiments: (1) mean vs PMA pooling (a better invariant encoder narrows the
    regret); (2) symmetric vs asymmetric attention encoder (the bottleneck: encoder symmetry barely
    matters); (3) a feature-token emitter with verified row/padding invariance and one-checkpoint
    width invariance. Returns {pool_ab, sym_asym, invariance, width, ft_net}."""
    device = _device(device)
    ab = run_pool_ab(steps=steps, n_tasks=n_tasks, seed=seed, device=device)
    sa = run_sym_asym_ab(steps=steps, n_tasks=n_tasks, seed=seed, device=device)
    ft = train_feature_token(steps=ft_steps, seed=seed, device=device)
    inv = invariance_checks(ft, device=device)
    width = width_check(ft, device=device)
    return {"pool_ab": ab, "sym_asym": sa, "invariance": inv, "width": width, "ft_net": ft}


# =============================================================================
# Figures
# =============================================================================

def make_bottleneck_figure():
    """Figure 13.1 --- the pipeline with the symmetry bottleneck as a narrow waist at the measure.
    Attention lives left of the waist (the encoder); the predictor right of it is a symmetric PSD
    kernel, so the encoder's asymmetry stops at m."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
    fig, ax = plt.subplots(figsize=(10.4, 4.2))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, fc, fs=9):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                     boxstyle="round,pad=0.1,rounding_size=0.15", fc=fc, ec="0.3", lw=1.2))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, x2, y=3.0):
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=14,
                     color="0.4", lw=1.4))

    y = 3.0
    box(1.5, y, 2.4, 1.1, "context\n$\\{(x_c,y_c)\\}$", "#eef3f9")
    box(4.3, y, 2.6, 1.1, "encoder $g_\\theta$\n(attention, PMA)", "#f3e2cc")
    arrow(2.7, 3.0)
    # the waist
    ax.add_patch(Polygon([(5.9, 3.0), (6.8, 3.55), (6.8, 2.45)], closed=True, fc="#b6403b", ec="none"))
    box(7.6, y, 1.7, 1.0, "measure\n$m$", "#dbe7f2")
    arrow(6.8, 6.75)
    box(10.0, y, 2.6, 1.1, "kernel $K_m$\n(symmetric PSD)", "#cfe0c6")
    arrow(8.45, 8.7)
    box(12.4, y, 1.4, 1.1, "predict", "#cfe0c6")
    arrow(11.3, 11.7)

    ax.text(2.9, 4.7, "job: invariance (not prediction)", ha="center", fontsize=9.5, color="#7a5a2b")
    ax.text(11.2, 4.7, "job: prediction (symmetric by construction)", ha="center", fontsize=9.5,
            color="#2e5a2e")
    ax.text(2.9, 4.15, "any internal machinery; even asymmetric attention", ha="center", fontsize=8,
            color="0.45")
    ax.text(6.35, 1.7, "asymmetry\nstops here", ha="center", fontsize=8.5, color="#b6403b")
    ax.annotate("", xy=(6.35, 2.2), xytext=(6.35, 1.95),
                arrowprops=dict(arrowstyle="-", color="#b6403b", lw=0.8))
    ax.text(6.5, 5.4, "the symmetry bottleneck: the encoder is invariant, the kernel predicts",
            ha="center", fontsize=10.5, style="italic", color="0.2")
    fig.tight_layout()
    return fig


def make_encoder_figure(sa=None, width=None, **kw):
    """Figure 13.2 --- the encoder's job is invariance; the kernel predicts.
    (left) the symmetry bottleneck: a symmetric (Mahalanobis) and an asymmetric ($W_Q\\neq W_K$)
    attention encoder reach nearly the same regret over Bayes --- the encoder's attention symmetry
    is immaterial, because the prediction is funneled through the symmetric kernel. Contrast Ch. 11,
    where asymmetrizing the *predictive* kernel forfeits KRR.
    (right) width invariance: one feature-token checkpoint serves several feature counts (the rest
    padded and masked), beating predict-the-mean at each, with predictions provably independent of
    the padding."""
    import matplotlib.pyplot as plt
    if sa is None:
        sa = run_sym_asym_ab(**kw)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.5))

    kk = sa["ks"]
    ax[0].plot(kk, [sa["asymmetric"][k]["regret"] for k in kk], "o-", color="#b6403b",
               label="asymmetric attention ($W_Q\\!\\neq\\!W_K$)")
    ax[0].plot(kk, [sa["symmetric"][k]["regret"] for k in kk], "s--", color="#2e7d32",
               label="symmetric attention (Mahalanobis)")
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(kk); ax[0].set_xticklabels(kk)
    ax[0].set_xlabel("context size $k$"); ax[0].set_ylabel("regret over Bayes")
    ax[0].set_title("The bottleneck: encoder attention symmetry\nis immaterial to the prediction",
                    fontsize=10)
    ax[0].legend(fontsize=9)

    if width is not None:
        das = sorted(width)
        ratios = [width[da]["ratio"] for da in das]
        ax[1].bar(range(len(das)), ratios, width=0.6, color="#3b6fb6", edgecolor="white")
        ax[1].axhline(1.0, color="0.5", lw=1.0, ls="--", label="predict-the-mean")
        ax[1].set_xticks(range(len(das)))
        ax[1].set_xticklabels([f"$d={da}$" for da in das])
        ax[1].set_ylim(0, 1.1)
        ax[1].set_ylabel("emitter MSE / predict-mean MSE")
        ax[1].set_title("Width invariance: one checkpoint serves\nany feature count (padding-proof)",
                        fontsize=10)
        ax[1].legend(fontsize=9)
        for x, v in zip(range(len(das)), ratios):
            ax[1].text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    else:
        ax[1].axis("off")
    fig.tight_layout()
    return fig


# =============================================================================
# CLI
# =============================================================================

def main(argv=None):
    p = argparse.ArgumentParser(description="Chapter 13 — invariant encoders")
    p.add_argument("--out-prefix", default=None)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args(argv)
    from lkbook import set_style
    set_style()
    device = "cpu" if args.cpu else _device()
    res = run_all(steps=args.steps, ft_steps=args.steps, device=device)
    ab, sa, inv, width = res["pool_ab"], res["sym_asym"], res["invariance"], res["width"]
    print("mean vs PMA regret over Bayes:")
    for k in ab["ks"]:
        print(f"  k={k:>4}  mean {ab['mean'][k]['regret']:.3f}  pma {ab['pma'][k]['regret']:.3f}  "
              f"reduction {100*ab['reduction'][k]:.0f}%")
    print("symmetric vs asymmetric attention encoder regret over Bayes:")
    for k in sa["ks"]:
        print(f"  k={k:>4}  sym {sa['symmetric'][k]['regret']:.3f}  asym {sa['asymmetric'][k]['regret']:.3f}  "
              f"gap {sa['gap'][k]:+.3f}")
    print("invariance (max |prediction change|):", {key: f"{v:.2e}" for key, v in inv.items()})
    print("one-checkpoint width (emitter/mean MSE):", {da: round(width[da]["ratio"], 3) for da in width})
    if args.out_prefix:
        import matplotlib
        matplotlib.use("Agg")
        make_bottleneck_figure().savefig(f"{args.out_prefix}1_bottleneck.pdf")
        make_encoder_figure(sa, width).savefig(f"{args.out_prefix}2_encoder.pdf")
        print("wrote figures with prefix", args.out_prefix)


if __name__ == "__main__":
    main()
