"""Chapter 12 — meta-learning a prior over kernels.

Up to now learning the kernel meant fitting a spectral measure ``m`` to one dataset by
gradient descent on its evidence (Ch. 8). Amortization asks a different question: can a
single map, learned once across many tasks, *infer* the measure from a context set of
(x, y) pairs in one forward pass, with nothing fit per dataset? The move that makes the
question answerable is to draw the meta-training tasks from the model's *own* kernel prior.
Then the generator and the inferrer share one geometry, the problem is well specified, the
Bayes-optimal predictor is the GP posterior mean at the true ``m``, and the cost of
inference is measurable exactly — the regret over Bayes.

This module is a self-contained, faithful port of the research code
(``kernel-machine/icl/metaskm`` + ``skm``):

  * The spectral measure and its fused Laplace/Gauss Gram (``SpectralMeasure``, ``gram``,
    ``density_to_atoms``) — the kernel of Ch. 8, batched over tasks.
  * The self-consistent prior (``sample_measure_prior``, ``sample_gp_tasks`` with its
    bandwidth calibration, ``bayes_posterior``, ``binarize``).
  * The in-context emitter (``MetaMSSKM``): a permutation-invariant Transformer that reads
    the context set and emits a measure, decoded by in-context KRR.
  * Meta-training on the prior stream and the regret-over-Bayes evaluation
    (``meta_train``, ``eval_regret_vs_k``).
  * The zero-shot transfer teaser to real data (``zeroshot_transfer`` on California),
    setting up Ch. 14.

The pedagogical payload: the *same* ``gram`` + KRR code decodes both the emitter and Bayes,
so the regret is an apples-to-apples one-line subtraction, and it is a property of the
context, not of misspecification.

    python -m lkbook.chapters.ch12 --out-prefix fig12
"""
from __future__ import annotations

import argparse
import math

import numpy as np

SEED = 0


def _device(device=None):
    import torch
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# The spectral measure and its fused Gram  (the kernel of Ch. 8, batched)
# =============================================================================

class SpectralMeasure:
    """A batched spectral measure consumed by :func:`gram` (Ch. 8, made a first-class value).

    Shapes (B tasks, H banks, d features, K freqs/feature):
      omega (B,H,d,K) frequencies, amp (B,H,d,K) amplitudes (the spectral density),
      scale (B,d) per-coordinate ARD relevance, band (B,H) per-bank bandwidth,
      wlog (B,H) fusion logits (softmax -> simplex), W (d_phi, 2dK) shared interaction
      geometry, kernel in {"laplace","gauss"}, sigma2 optional per-task noise.
    """

    def __init__(self, omega, amp, scale, band, wlog, W, kernel="laplace", sigma2=None):
        self.omega, self.amp, self.scale = omega, amp, scale
        self.band, self.wlog, self.W = band, wlog, W
        self.kernel, self.sigma2 = kernel, sigma2

    @property
    def w(self):
        import torch
        return torch.softmax(self.wlog, dim=-1)


def feature_map(measure, X, feature_mask=None):
    """Per-coordinate spectral features of X (B,N,d) -> psi (B,H,N,2dK).

    Per feature the 2K block is [a*cos(K), a*sin(K)] with argument
    2*pi*scale_j*x_j*omega_{j,k}."""
    import torch
    sx = X * measure.scale[:, None, :]
    arg = 2.0 * math.pi * sx[:, None, :, :, None] * measure.omega[:, :, None, :, :]
    a = measure.amp[:, :, None, :, :]
    cos, sin = a * torch.cos(arg), a * torch.sin(arg)
    if feature_mask is not None:
        m = feature_mask[:, None, None, :, None]
        cos, sin = cos * m, sin * m
    B, H, N, d, K = cos.shape
    return torch.cat([cos, sin], dim=-1).reshape(B, H, N, d * 2 * K)


def density_to_atoms(mu, log_gamma, pi_logits, gh_nodes, gh_wts):
    """Discretize a per-coordinate Q-component Gaussian spectral density into atoms by
    Gauss-Hermite quadrature. mu, log_gamma, pi_logits are (...,Q); gh_nodes/gh_wts are
    (G,) (weights pre-divided by sqrt(pi)). Returns (omega, amp) of shape (...,Q*G) with
    atom omega = |mu_q + sqrt(2) gamma_q x_g| and amplitude a^2 = pi_q * w_g (unit mass)."""
    import torch
    import torch.nn.functional as F
    gamma = F.softplus(log_gamma) + 1e-4
    pi = torch.softmax(pi_logits, dim=-1)
    om = (mu[..., None] + math.sqrt(2.0) * gamma[..., None] * gh_nodes).abs()
    a2 = pi[..., None] * gh_wts
    om = om.reshape(*om.shape[:-2], -1)
    a = a2.clamp_min(0.0).sqrt().reshape(*a2.shape[:-2], -1)
    return om, a


def _warp(psi, W):
    """psi (B,H,N,2dK) -> phi (B,H,N,d_phi) under shared or per-task W."""
    import torch
    if W.dim() == 2:
        return torch.einsum("bhnf,pf->bhnp", psi, W)
    return torch.einsum("bhnf,bpf->bhnp", psi, W)


def gram(measure, X1, X2, feature_mask=None):
    """Batched convex-fused Laplace/Gauss kernel K(X1,X2) -> (B,N1,N2), unit diagonal when
    X1 is X2. The kernel acts on ||W psi(x) - W psi(x')||, fused over banks."""
    import torch
    phi1 = _warp(feature_map(measure, X1, feature_mask), measure.W)
    phi2 = _warp(feature_map(measure, X2, feature_mask), measure.W)
    dist = torch.cdist(phi1, phi2)
    if measure.kernel == "gauss":
        dist = dist * dist
    band = measure.band[:, :, None, None].clamp_min(1e-6)
    kh = torch.exp(-dist / band)
    return (measure.w[:, :, None, None] * kh).sum(dim=1)


# =============================================================================
# Decoders: NW smoother and the in-context KRR (GP posterior mean)
# =============================================================================

def nw_predict(K, y_ctx, ctx_mask=None, eps=1e-8):
    """Mask-aware, NaN-safe Nadaraya-Watson smoother."""
    import torch
    if ctx_mask is not None:
        K = K * ctx_mask.unsqueeze(-2)
    denom = K.sum(dim=-1, keepdim=True).clamp_min(eps)
    return torch.matmul(K / denom, y_ctx)


def robust_cholesky(A, base_jitter=1e-6, tries=8):
    """Symmetrize and add escalating jitter until Cholesky succeeds (GP practice)."""
    import torch
    A = 0.5 * (A + A.transpose(-1, -2))
    eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    jit = base_jitter
    for _ in range(tries):
        try:
            return torch.linalg.cholesky(A + jit * eye)
        except Exception:
            jit *= 10.0
    return torch.linalg.cholesky(A + jit * eye)


def krr_incontext(K_cc, K_qc, y_ctx, sigma2, ctx_mask=None):
    """Batched, mask-aware in-context KRR readout: the GP posterior mean on the context,
    pred = K_qc (K_cc + sigma2 I)^{-1} y_c, solved per task by a robust Cholesky."""
    import torch
    B, Nc, _ = K_cc.shape
    eye = torch.eye(Nc, device=K_cc.device, dtype=K_cc.dtype)
    s2 = sigma2.reshape(-1, 1, 1) if torch.is_tensor(sigma2) and sigma2.ndim else sigma2
    A = K_cc + s2 * eye
    if ctx_mask is not None:
        m = ctx_mask
        mm = m[:, :, None] * m[:, None, :]
        A = A * mm + torch.diag_embed(1.0 - m)
        y_ctx = y_ctx * m[..., None]
        K_qc = K_qc * m[:, None, :]
    L = robust_cholesky(A)
    alpha = torch.cholesky_solve(y_ctx, L)
    return torch.bmm(K_qc, alpha)


# =============================================================================
# The self-consistent prior: data generation for meta-training
# =============================================================================

def sample_measure_prior(B, d, H, Q, gh_nodes, gh_wts, W, gen, *, kernel="laplace",
                         band_mult_range=(1.0, 6.0), freq_range=(0.1, 2.5), p_active=0.6,
                         sigma2_range=(1e-3, 3e-2), device=None, dtype=None):
    """Draw a batch of B measures from the hyperprior -> a :class:`SpectralMeasure`.

    The atoms are a Gauss-Hermite quadrature of a per-coordinate Q-component Gaussian
    spectral density, so the measure is well specified against the emitter. W (shared
    interaction geometry) and gh_nodes/gh_wts (shared quadrature) are passed in — they are
    the emitter's own, not resampled. The per-bank band is sampled as a *multiplier*, made
    absolute by :func:`sample_gp_tasks` (calibrated to the embedding-distance scale).
    The Bernoulli mask on the ARD scale switches coordinates off at random, so the prior
    contains feature-selection structure. Returns (measure, sigma2)."""
    import torch
    import torch.nn.functional as F
    if device is None:
        device = W.device
    if dtype is None:
        dtype = W.dtype

    def U(*shape, lo=0.0, hi=1.0):
        return lo + (hi - lo) * torch.rand(*shape, generator=gen, device=device, dtype=dtype)

    def Nrm(*shape, mu=0.0, sd=1.0):
        return mu + sd * torch.randn(*shape, generator=gen, device=device, dtype=dtype)

    lo_f, hi_f = freq_range
    mu = U(B, H, d, Q, lo=lo_f, hi=hi_f)
    log_gamma = Nrm(B, H, d, Q, mu=-1.0, sd=0.5)
    pi_logits = Nrm(B, H, d, Q, sd=1.0)
    omega, amp = density_to_atoms(mu, log_gamma, pi_logits, gh_nodes, gh_wts)

    active = (torch.rand(B, d, generator=gen, device=device, dtype=dtype) < p_active).to(dtype)
    active[:, 0] = 1.0
    scale = active * F.softplus(Nrm(B, d, mu=0.5, sd=0.5))

    lo_b, hi_b = band_mult_range
    band = torch.exp(U(B, H, lo=math.log(lo_b), hi=math.log(hi_b)))
    wlog = Nrm(B, H, sd=1.0)
    lo_s, hi_s = sigma2_range
    sigma2 = torch.exp(U(B, lo=math.log(lo_s), hi=math.log(hi_s)))

    measure = SpectralMeasure(omega=omega, amp=amp, scale=scale, band=band, wlog=wlog,
                              W=W, kernel=kernel, sigma2=sigma2)
    return measure, sigma2


def sample_inputs(B, N, d, gen, *, mode="uniform", x_range=(-1.0, 1.0), device="cpu", dtype=None):
    """Task inputs X (B,N,d). uniform = independent U(x_range) (the base prior);
    correlated/clustered broaden the input distribution for transfer."""
    import torch
    if dtype is None:
        dtype = torch.float32
    lo, hi = x_range
    if mode == "uniform":
        return lo + (hi - lo) * torch.rand(B, N, d, generator=gen, device=device, dtype=dtype)

    def _stdz(X):
        mu = X.mean(dim=1, keepdim=True)
        sd = X.std(dim=1, keepdim=True).clamp_min(1e-6)
        return (X - mu) / sd

    if mode == "correlated":
        r = max(2, d // 2)
        A = torch.randn(B, d, r, generator=gen, device=device, dtype=dtype) / math.sqrt(r)
        z = torch.randn(B, N, r, generator=gen, device=device, dtype=dtype)
        eps = 0.3 * torch.randn(B, N, d, generator=gen, device=device, dtype=dtype)
        return _stdz(torch.einsum("bnr,bdr->bnd", z, A) + eps)
    raise ValueError(f"unknown input mode {mode!r}")


def sample_gp_tasks(measure, n_ctx, n_q, sigma2, gen, *, x_range=(-1.0, 1.0),
                    standardize=True, feature_mask=None, input_mode="uniform",
                    device=None, dtype=None):
    """Draw y ~ GP(0, K + sigma^2 I) on random inputs for each task in the batch.

    Builds K = gram(measure, X, X), factors K + sigma^2 I by a robust Cholesky, draws
    y = L z. First calibrates measure.band *in place* from a multiplier to an absolute
    bandwidth: each bank's correlation length is set to multiplier x the mean pairwise
    embedding distance over X. Without this the fixed random W and high frequencies push
    embedding distances far past any fixed bandwidth, the Gram collapses to the identity,
    and the draw is white noise. Returns (Xc, yc, Xq, yq)."""
    import torch
    if dtype is None:
        dtype = measure.W.dtype
    B = measure.omega.shape[0]
    d = measure.omega.shape[2]
    if device is None:
        device = measure.W.device
    N = n_ctx + n_q
    X = sample_inputs(B, N, d, gen, mode=input_mode, x_range=x_range, device=device, dtype=dtype)

    phi = _warp(feature_map(measure, X, feature_mask), measure.W)
    dist = torch.cdist(phi, phi)
    if measure.kernel == "gauss":
        dist = dist * dist
    charscale = dist.sum(dim=(-1, -2)) / (N * (N - 1))
    measure.band = (measure.band * charscale).clamp_min(1e-6)

    K = gram(measure, X, X, feature_mask)
    s2 = sigma2.reshape(B, 1, 1)
    eye = torch.eye(N, device=device, dtype=dtype)
    L = robust_cholesky(K + s2 * eye)
    z = torch.randn(B, N, 1, generator=gen, device=device, dtype=dtype)
    y = torch.bmm(L, z)

    Xc, Xq = X[:, :n_ctx], X[:, n_ctx:]
    yc, yq = y[:, :n_ctx], y[:, n_ctx:]
    if standardize:
        mu = yc.mean(dim=1, keepdim=True)
        sd = yc.std(dim=1, keepdim=True).clamp_min(1e-6)
        yc, yq = (yc - mu) / sd, (yq - mu) / sd
    return Xc, yc, Xq, yq


def bayes_posterior(measure, Xc, yc, Xq, sigma2, feature_mask=None):
    """The exact GP posterior mean under the true measure — the Bayes-optimal predictor.
    Reuses the in-context KRR solve with the *true* generating measure, so this is the
    optimum the emitter is trained to imitate. Returns (B, n_q, 1)."""
    K_cc = gram(measure, Xc, Xc, feature_mask)
    K_qc = gram(measure, Xq, Xc, feature_mask)
    return krr_incontext(K_cc, K_qc, yc, sigma2)


def binarize(yc, yq):
    """Threshold the latent GP at the per-task context median -> balanced binary labels,
    so one prior serves both regression and classification tasks."""
    thr = yc.median(dim=1, keepdim=True).values
    return (yc > thr).to(yc.dtype), (yq > thr).to(yq.dtype)


# =============================================================================
# The in-context emitter: a permutation-invariant Transformer that emits a measure
# =============================================================================

def _make_emitter(max_features=8, H=4, Q=3, n_quad=6, d_phi=64, d_model=128, nhead=4,
                  nlayers=2, kernel="laplace", decode="krr", pool="pma", m_seeds=4,
                  omega_range=(0.05, 5.0), seed=0):
    """Construct the emitter network. Separated from the class so the heavy torch import
    stays lazy. See :class:`MetaMSSKM`."""
    return MetaMSSKM(max_features, H=H, Q=Q, n_quad=n_quad, d_phi=d_phi, d_model=d_model,
                     nhead=nhead, nlayers=nlayers, kernel=kernel, decode=decode, pool=pool,
                     m_seeds=m_seeds, omega_range=omega_range, seed=seed)


def _emitter_base():
    """Return the (nn.Module) base class, importing torch lazily."""
    import torch.nn as nn

    class _AttnPool(nn.Module):
        """Pooling by multihead attention (Set Transformer PMA): m learned seed queries
        cross-attend to the encoded context. Mean pooling is the uniform-attention special
        case, so PMA is a strict generalization. The LayerNorm keeps the pooled summary at
        the encoder's scale (raw attention output drifts the measure heads into extreme
        frequencies -> NaN)."""

        def __init__(self, d_model, nhead, m_seeds=1):
            super().__init__()
            import torch
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

    class _MetaMSSKM(nn.Module):
        def __init__(self, max_features, H=4, Q=3, n_quad=6, d_phi=64, d_model=128, nhead=4,
                     nlayers=2, kernel="laplace", decode="krr", pool="pma", m_seeds=4,
                     omega_range=(0.05, 5.0), seed=0):
            super().__init__()
            import torch
            if decode not in ("nw", "krr"):
                raise ValueError(f"decode must be 'nw' or 'krr', got {decode!r}")
            if pool not in ("mean", "pma"):
                raise ValueError(f"pool must be 'mean' or 'pma', got {pool!r}")
            torch.manual_seed(seed)
            self.d, self.H, self.Q, self.n_quad = max_features, H, Q, n_quad
            self.K = Q * n_quad
            self.kernel, self.decode, self.pool_mode = kernel, decode, pool
            self.omega_range = omega_range

            nodes, wts = np.polynomial.hermite.hermgauss(n_quad)
            self.register_buffer("gh_nodes", torch.as_tensor(nodes, dtype=torch.float32))
            self.register_buffer("gh_wts", torch.as_tensor(wts / np.sqrt(np.pi), dtype=torch.float32))
            lo, hi = omega_range
            edges = lo * (hi / lo) ** (np.arange(H) / max(H - 1, 1))
            self.register_buffer("band_mu0", torch.as_tensor(edges, dtype=torch.float32))

            self.input_proj = nn.Linear(max_features + 1, d_model)
            enc = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc, nlayers)
            if pool == "pma":
                self.pma = _AttnPool(d_model, nhead, m_seeds)

            d = max_features
            self.mu_head = nn.Linear(d_model, H * d * Q)
            self.lg_head = nn.Linear(d_model, H * d * Q)
            self.pi_head = nn.Linear(d_model, H * d * Q)
            self.scale_head = nn.Linear(d_model, d)
            self.band_head = nn.Linear(d_model, H)
            self.wlog_head = nn.Linear(d_model, H)
            self.logsig_head = nn.Linear(d_model, 1)
            self.W = nn.Parameter(0.1 * torch.randn(d_phi, d * 2 * self.K))

        def _pool(self, X_ctx, y_ctx, row_mask, feature_mask):
            import torch
            if feature_mask is not None:
                X_ctx = X_ctx * feature_mask[:, None, :]
            tokens = self.input_proj(torch.cat([X_ctx, y_ctx], dim=-1))
            pad = None if row_mask is None else (row_mask < 0.5)
            e = self.encoder(tokens, src_key_padding_mask=pad)
            if self.pool_mode == "pma":
                return self.pma(e, key_padding_mask=pad)
            if row_mask is None:
                return e.mean(dim=1)
            m = row_mask[:, :, None]
            return (e * m).sum(1) / m.sum(1).clamp_min(1.0)

        def emit(self, X_ctx, y_ctx, row_mask=None, feature_mask=None):
            import torch
            import torch.nn.functional as F
            B = X_ctx.shape[0]
            d, H, Q = self.d, self.H, self.Q
            e = self._pool(X_ctx, y_ctx, row_mask, feature_mask)
            mu = self.band_mu0[None, :, None, None] * F.softplus(
                self.mu_head(e).view(B, H, d, Q) + math.log(math.e - 1))
            omega, amp = density_to_atoms(mu, self.lg_head(e).view(B, H, d, Q),
                                          self.pi_head(e).view(B, H, d, Q),
                                          self.gh_nodes, self.gh_wts)
            scale = F.softplus(self.scale_head(e))
            band = F.softplus(self.band_head(e)) + 1e-3
            sigma2 = F.softplus(self.logsig_head(e)).squeeze(-1) + 1e-4
            return SpectralMeasure(omega=omega, amp=amp, scale=scale, band=band,
                                   wlog=self.wlog_head(e), W=self.W, kernel=self.kernel,
                                   sigma2=sigma2)

        def forward(self, X_query, X_ctx, y_ctx, row_mask=None, feature_mask=None):
            measure = self.emit(X_ctx, y_ctx, row_mask, feature_mask)
            K_qc = gram(measure, X_query, X_ctx, feature_mask)
            if self.decode == "nw":
                return nw_predict(K_qc, y_ctx, ctx_mask=row_mask)
            K_cc = gram(measure, X_ctx, X_ctx, feature_mask)
            return krr_incontext(K_cc, K_qc, y_ctx, measure.sigma2, ctx_mask=row_mask)

    return _MetaMSSKM


_MetaMSSKM_cls = None


def MetaMSSKM(*args, **kw):
    """The in-context emitter: a permutation-invariant Transformer reading a context set
    and emitting a :class:`SpectralMeasure`, decoded by NW or in-context KRR. (Thin factory
    so the torch import stays lazy; behaves like the class.)"""
    global _MetaMSSKM_cls
    if _MetaMSSKM_cls is None:
        _MetaMSSKM_cls = _emitter_base()
    return _MetaMSSKM_cls(*args, **kw)


# =============================================================================
# Meta-training on the prior stream; regret over Bayes
# =============================================================================

def meta_train(net, d, H, Q, *, steps=3000, B=32, n_q=64,
               n_ctx_choices=(16, 32, 64, 128, 256), lr=2e-3, seed=0, device="cpu",
               log_every=0):
    """Meta-train over the prior stream. W is frozen (the shared base geometry), so the
    task distribution stays fixed and well specified while the emitter learns to invert it.
    Context size is resampled per step so the emitter is robust across k."""
    import torch
    import torch.nn.functional as F
    net.W.requires_grad_(False)
    net.train()
    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.Adam((p for p in net.parameters() if p.requires_grad), lr=lr)
    choices = torch.as_tensor(n_ctx_choices, device=device)
    for step in range(steps):
        n_ctx = int(choices[torch.randint(len(choices), (1,), generator=gen, device=device)].item())
        measure, sigma2 = sample_measure_prior(B, d, H, Q, net.gh_nodes, net.gh_wts,
                                               net.W.detach(), gen, device=device)
        Xc, yc, Xq, yq = sample_gp_tasks(measure, n_ctx, n_q, sigma2, gen, device=device)
        opt.zero_grad()
        loss = F.mse_loss(net(Xq, Xc, yc), yq)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in net.parameters() if p.requires_grad), 5.0)
        opt.step()
        if log_every and (step + 1) % log_every == 0:
            print(f"  step {step + 1:5d}  meta-train MSE {loss.item():.4f}  (n_ctx={n_ctx})")
    return net


def eval_regret_vs_k(net, d, H, Q, *, ks=(8, 32, 64, 128, 256, 512), n_q=128,
                     n_tasks=400, B=50, seed=999, device="cpu"):
    """Held-out-task regret over Bayes vs context size k. For each k draw fresh tasks from
    the prior (a new measure per task) and report emitter MSE, exact Bayes MSE under the
    true measure, their gap (the regret), and the predict-the-mean baseline."""
    import torch
    import torch.nn.functional as F
    net.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    rows = {}
    with torch.no_grad():
        for k in ks:
            em, ba, mn = [], [], []
            done = 0
            while done < n_tasks:
                b = min(B, n_tasks - done)
                measure, sigma2 = sample_measure_prior(b, d, H, Q, net.gh_nodes, net.gh_wts,
                                                       net.W.detach(), gen, device=device)
                Xc, yc, Xq, yq = sample_gp_tasks(measure, k, n_q, sigma2, gen, device=device)
                em.append(F.mse_loss(net(Xq, Xc, yc), yq).item())
                ba.append(F.mse_loss(bayes_posterior(measure, Xc, yc, Xq, sigma2), yq).item())
                mn.append(F.mse_loss(torch.zeros_like(yq), yq).item())
                done += b
            e, a, m = float(np.mean(em)), float(np.mean(ba)), float(np.mean(mn))
            rows[k] = dict(emitter=e, bayes=a, regret=e - a, mean=m)
    return rows


# =============================================================================
# Zero-shot transfer to real data — the teaser that sets up Ch. 14
# =============================================================================

def _pad(X, d_max):
    """Pad feature columns to d_max with zeros; return (X_pad, feature_mask)."""
    n, dd = X.shape
    if dd > d_max:
        X = X[:, :d_max]
        dd = d_max
    Xp = np.zeros((n, d_max), dtype=np.float32)
    Xp[:, :dd] = X
    m = np.zeros(d_max, dtype=np.float32)
    m[:dd] = 1.0
    return Xp, m


def zeroshot_transfer(net, *, d_max=8, ctx_cap=512, n_test=2000, seed=0, device="cpu"):
    """Freeze the emitter and predict a held-out California task from its context alone — no
    per-dataset gradient steps. The emitter reads a context subsample of the training rows,
    emits a measure, and decodes the test queries. The number is now governed by how well
    the real data matches the synthetic prior (the prior<->reality gap of Ch. 14), not by
    the regret of the controlled world. Returns the zero-shot R^2."""
    import torch
    from sklearn.metrics import r2_score
    from lkbook import load_california
    net.eval()
    cal = load_california(seed=seed)
    Xtr, ytr = cal.Xtr.astype(np.float32), np.asarray(cal.ytr, float)
    Xte, yte = cal.Xte.astype(np.float32), np.asarray(cal.yte, float)
    rng = np.random.default_rng(seed)
    if len(Xtr) > ctx_cap:
        idx = rng.choice(len(Xtr), ctx_cap, replace=False)
        Xc_np, yc_np = Xtr[idx], ytr[idx]
    else:
        Xc_np, yc_np = Xtr, ytr
    if len(Xte) > n_test:
        jdx = rng.choice(len(Xte), n_test, replace=False)
        Xte, yte = Xte[jdx], yte[jdx]

    Xc_p, fmask = _pad(Xc_np, d_max)
    Xq_p, _ = _pad(Xte, d_max)
    fm = torch.as_tensor(fmask, device=device)[None]
    Xc = torch.as_tensor(Xc_p, device=device)[None]
    Xq = torch.as_tensor(Xq_p, device=device)[None]
    yc = torch.as_tensor(yc_np, dtype=torch.float32, device=device)
    mu, sd = yc.mean(), yc.std().clamp_min(1e-6)
    with torch.no_grad():
        pred = net(Xq, Xc, ((yc - mu) / sd)[None, :, None], feature_mask=fm)[0, :, 0]
        pred = pred.cpu().numpy() * sd.item() + mu.item()
    return dict(metric="R2", emitter=float(r2_score(yte, pred)))


# =============================================================================
# Does the one-pass emitter recover a Chapter-8 measure on a real table?
# =============================================================================

def recover_vs_chapter8(net, *, d_max=8, ctx_cap=512, n_test=2000, seed=0, device=None,
                        ch8_mode="learned", ch8_H=2, ch8_K=8, ch8_steps=500):
    """Compare amortized inference (one forward pass of the emitter) against a Chapter-8
    per-dataset fit on a single real table (California Housing), on two axes:

      * **prediction** — zero-shot emitter R^2 vs the per-dataset learned spectral-Laplace R^2;
      * **geometry** — the per-feature ARD relevances. Chapter 8 fits relevances s_j by gradient
        descent on the table's evidence; the emitter emits relevances from the context in one pass.
        We report both vectors (normalized to sum 1) and their rank correlation, so we can see
        whether the emitter recovers the *same geometry* even where its accuracy trails.

    The emitter was meta-trained on the synthetic prior (uniform inputs), so on a real table it is
    out of distribution: the gap here is the prior<->reality gap of Ch. 14, not the regret of the
    controlled world. Returns a dict with both R^2s, both relevance vectors, the feature names and
    the Spearman correlation of the relevances.
    """
    import torch
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score
    from lkbook import load_california
    from lkbook.chapters import ch08

    cal = load_california(seed=seed)
    names = list(cal.names)
    Xtr, ytr = cal.Xtr.astype(np.float32), np.asarray(cal.ytr, float)
    Xte, yte = cal.Xte.astype(np.float32), np.asarray(cal.yte, float)
    rng = np.random.default_rng(seed)
    if len(Xte) > n_test:
        jdx = rng.choice(len(Xte), n_test, replace=False)
        Xte, yte = Xte[jdx], yte[jdx]

    # --- Chapter 8: per-dataset fit by gradient descent on the evidence ---
    ker8, pred8 = ch08.fit_spectral(Xtr, ytr, mode=ch8_mode, H=ch8_H, K=ch8_K, steps=ch8_steps,
                                    seed=seed)
    p8 = pred8(Xte)
    r2_ch8 = float(r2_score(yte, p8))
    rel8 = torch.nn.functional.softplus(ker8.log_s).detach().cpu().numpy().ravel()
    rel8 = rel8[:d_max]
    rel8 = rel8 / (rel8.sum() + 1e-12)

    # --- Chapter 12: one forward pass of the frozen emitter ---
    net.eval()
    if len(Xtr) > ctx_cap:
        idx = rng.choice(len(Xtr), ctx_cap, replace=False)
        Xc_np, yc_np = Xtr[idx], ytr[idx]
    else:
        Xc_np, yc_np = Xtr, ytr
    Xc_p, fmask = _pad(Xc_np, d_max)
    Xq_p, _ = _pad(Xte, d_max)
    fm = torch.as_tensor(fmask, device=device)[None]
    Xc = torch.as_tensor(Xc_p, device=device)[None]
    Xq = torch.as_tensor(Xq_p, device=device)[None]
    yc = torch.as_tensor(yc_np, dtype=torch.float32, device=device)
    mu, sd = yc.mean(), yc.std().clamp_min(1e-6)
    with torch.no_grad():
        measure = net.emit(Xc, ((yc - mu) / sd)[None, :, None], feature_mask=fm)
        pred = net(Xq, Xc, ((yc - mu) / sd)[None, :, None], feature_mask=fm)[0, :, 0]
        pred = pred.cpu().numpy() * sd.item() + mu.item()
    r2_emit = float(r2_score(yte, pred))
    rel_emit = measure.scale[0].detach().cpu().numpy().ravel()[:d_max]
    rel_emit = rel_emit / (rel_emit.sum() + 1e-12)

    rho = float(spearmanr(rel8, rel_emit).correlation)           # parametric (non-identified)
    pred_corr = float(np.corrcoef(pred, p8)[0, 1])               # functional agreement on the table
    return {"names": names[:d_max], "r2_ch8": r2_ch8, "r2_emitter": r2_emit,
            "rel_ch8": rel8, "rel_emitter": rel_emit, "spearman": rho, "pred_corr": pred_corr}


def _per_dataset_fit_r2(Xc, yc, Xq, yq, steps=200):
    """Fit a Chapter-8 learned spectral-Laplace kernel on ONE task's context by gradient descent
    (the per-dataset baseline), evaluate R^2 on that task's queries. Same kernel family the emitter
    emits, fit the slow way."""
    from sklearn.metrics import r2_score
    from lkbook.chapters import ch08
    Xc = Xc.cpu().numpy(); yc = yc.cpu().numpy().ravel()
    Xq = Xq.cpu().numpy(); yq = yq.cpu().numpy().ravel()
    try:
        _, pred = ch08.fit_spectral(Xc, yc, mode="learned", H=2, K=8, steps=steps,
                                    n_fit=len(yc), standardize=True)
        return float(r2_score(yq, pred(Xq)))
    except Exception:
        return float("nan")


def decompose_amortization(net, *, d=8, H=4, Q=3, ks=(64, 256), n_q=128, n_tasks=16,
                           pd_steps=200, seed=777, device=None):
    """In-distribution decomposition: on tasks drawn from the prior (well specified, no OOD), compare
    three predictors by held-out query R^2 --- the Bayes predictor at the true measure (the oracle),
    a per-dataset gradient fit on the same context (Chapter 8, the slow way), and the one-pass
    emitter. The amortization cost is per-dataset minus emitter; the finite-context cost is Bayes
    minus per-dataset. Returns {k: {bayes, per_dataset, emitter}} mean R^2."""
    import torch
    from sklearn.metrics import r2_score
    device = _device(device)
    net.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    out = {}
    for k in ks:
        with torch.no_grad():
            measure, s2 = sample_measure_prior(n_tasks, d, H, Q, net.gh_nodes, net.gh_wts,
                                               net.W.detach(), gen, device=device)
            Xc, yc, Xq, yq = sample_gp_tasks(measure, k, n_q, s2, gen, device=device)
            emit = net(Xq, Xc, yc)[:, :, 0].cpu().numpy()
            bayes = bayes_posterior(measure, Xc, yc, Xq, s2)[:, :, 0].cpu().numpy()
        yqn = yq[:, :, 0].cpu().numpy()
        rb, re, rp = [], [], []
        for i in range(n_tasks):
            rb.append(r2_score(yqn[i], bayes[i]))
            re.append(r2_score(yqn[i], emit[i]))
            rp.append(_per_dataset_fit_r2(Xc[i], yc[i], Xq[i], yq[i], steps=pd_steps))
        out[k] = {"bayes": float(np.nanmean(rb)), "per_dataset": float(np.nanmean(rp)),
                  "emitter": float(np.nanmean(re))}
    return out


def ceiling_incontext_real(*, steps=3000, B=16, q=64, ctx_caps=(256, 512, 1024), seed=0,
                           device=None):
    """The ceiling experiment: meta-train the SAME emitter in-context on REAL California sub-tasks
    (random context/query row splits of the TRAIN pool, real labels), then evaluate zero-shot on the
    held-out TEST rows. If R^2 approaches the per-dataset Chapter-8 fit, the synthetic-prior gap was
    prior-misspecification, not amortization: amortized inference matches per-dataset training once
    the meta-distribution matches reality. Returns {cap: (mean_R2, std_R2)} plus the Ch8 reference."""
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import r2_score
    from lkbook import load_california
    from lkbook.chapters import ch08
    device = _device(device)
    cal = load_california(seed=seed)
    Xtr = torch.tensor(cal.Xtr[:, :8].astype(np.float32), device=device)
    ytr = torch.tensor(np.asarray(cal.ytr, np.float32), device=device)
    Xte = torch.tensor(cal.Xte[:, :8].astype(np.float32), device=device)
    yte = np.asarray(cal.yte, float)
    Ntr = Xtr.shape[0]
    torch.manual_seed(seed)
    net = MetaMSSKM(max_features=8, H=4, Q=3, decode="krr", pool="pma", seed=seed).to(device)
    net.W.requires_grad_(False)
    net.train()
    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.Adam((p for p in net.parameters() if p.requires_grad), lr=2e-3)
    choices = torch.as_tensor((64, 128, 256, 512), device=device)
    for _ in range(steps):
        k = int(choices[torch.randint(len(choices), (1,), generator=gen, device=device)].item())
        Xc = torch.empty(B, k, 8, device=device); yc = torch.empty(B, k, 1, device=device)
        Xq = torch.empty(B, q, 8, device=device); yq = torch.empty(B, q, 1, device=device)
        for b in range(B):                                       # B random context/query splits of train rows
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
        for cap in ctx_caps:
            r2s = []
            for _ in range(5):
                ci = torch.randperm(Ntr, generator=egen, device=device)[:cap]
                m, sd = ytr[ci].mean(), ytr[ci].std().clamp_min(1e-6)
                yc = ((ytr[ci] - m) / sd)[None, :, None]
                pred = net(Xte[None], Xtr[ci][None], yc)[0, :, 0].cpu().numpy() * sd.item() + m.item()
                r2s.append(r2_score(yte, pred))
            res[cap] = (float(np.mean(r2s)), float(np.std(r2s)))
    _, pred8 = ch08.fit_spectral(cal.Xtr, np.asarray(cal.ytr, float), mode="learned", H=2, K=8,
                                 steps=500, seed=seed)
    res["ch8"] = float(r2_score(yte, pred8(cal.Xte)))
    return res


def make_recovery_figure(decomp, calif):
    """Figure --- amortization is nearly free; the prior is the binding constraint.
    (left) in-distribution held-out R^2 by context size: Bayes (knows m), a per-dataset gradient
    fit, and the one-pass emitter. The emitter ties or beats per-dataset fitting (amortization is
    nearly free); both trail Bayes by the finite-context cost. (right) real California held-out R^2:
    the emitter under the synthetic prior, the same architecture meta-trained in-context on real
    California sub-tasks (the ceiling), and the per-dataset Chapter-8 fit. Matching the
    meta-distribution to reality closes the gap to per-dataset training.

    ``decomp`` = decompose_amortization(...); ``calif`` = {"synthetic": R2, "ceiling": R2,
    "ch8": R2}."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ks = sorted(decomp)
    xs = np.arange(len(ks))
    w = 0.26
    ax[0].bar(xs - w, [decomp[k]["bayes"] for k in ks], width=w, color="#2e7d32",
              label="Bayes (knows $m$)", edgecolor="white")
    ax[0].bar(xs, [decomp[k]["per_dataset"] for k in ks], width=w, color="#7aa6c2",
              label="per-dataset fit (GD)", edgecolor="white")
    ax[0].bar(xs + w, [decomp[k]["emitter"] for k in ks], width=w, color="#3b6fb6",
              label="emitter (one pass)", edgecolor="white")
    ax[0].axhline(0, color="0.6", lw=0.8)
    ax[0].set_xticks(xs); ax[0].set_xticklabels([f"$k={k}$" for k in ks])
    ax[0].set_ylabel("held-out $R^2$ (prior tasks)")
    ax[0].set_title("In-distribution: amortization is nearly free\n(emitter $\\approx$ per-dataset; gap to Bayes is finite-context)",
                    fontsize=9.5)
    ax[0].legend(fontsize=8.5, loc="upper left")

    labels = ["emitter\n(synthetic prior)", "emitter\n(in-context on real)", "Ch. 8 fit\n(per-dataset)"]
    vals = [calif["synthetic"], calif["ceiling"], calif["ch8"]]
    ax[1].bar([0, 1, 2], vals, width=0.6, color=["#c98a3b", "#3b6fb6", "#2e7d32"], edgecolor="white")
    ax[1].set_xticks([0, 1, 2]); ax[1].set_xticklabels(labels, fontsize=9)
    ax[1].set_ylabel("held-out $R^2$ (California)")
    ax[1].set_ylim(0, max(vals) * 1.18)
    ax[1].set_title("Real California: the prior is the binding constraint\n(match the meta-distribution and the gap closes)",
                    fontsize=9.5)
    for x, v in zip([0, 1, 2], vals):
        ax[1].text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


# =============================================================================
# Aggregate driver
# =============================================================================

def train_emitter(*, d=8, H=4, Q=3, steps=3000, B=32, pool="pma", decode="krr",
                   seed=0, device=None, log_every=0):
    """Build and meta-train one emitter on the model's own prior. Returns the trained net."""
    import torch
    device = _device(device)
    torch.manual_seed(seed)
    net = MetaMSSKM(max_features=d, H=H, Q=Q, n_quad=6, d_phi=64, decode=decode,
                    pool=pool, seed=seed).to(device)
    meta_train(net, d, H, Q, steps=steps, B=B, seed=seed, device=device, log_every=log_every)
    return net


def run_all(*, d=8, H=4, Q=3, steps=3000, B=32, pool="pma", n_tasks=400, seed=0,
            device=None, do_transfer=True, log_every=0):
    """Train the emitter on the self-consistent prior, evaluate regret over Bayes vs context
    size, and (optionally) run the zero-shot California transfer teaser. Returns a dict with
    ``regret`` (the per-k table), ``ks``, ``net``, and ``transfer``."""
    device = _device(device)
    net = train_emitter(d=d, H=H, Q=Q, steps=steps, B=B, pool=pool, seed=seed,
                         device=device, log_every=log_every)
    rows = eval_regret_vs_k(net, d, H, Q, n_tasks=n_tasks, seed=999, device=device)
    out = {"regret": rows, "ks": sorted(rows), "net": net, "d": d, "H": H, "Q": Q,
           "device": device}
    if do_transfer:
        out["transfer"] = zeroshot_transfer(net, d_max=d, seed=seed, device=device)
    return out


# =============================================================================
# An explorer: one task, the emitter prediction against Bayes
# =============================================================================

def explore_task(net, k=64, n_q=128, seed=7, d=8, H=4, Q=3, device=None):
    """Draw one held-out task from the prior at context size k; return the emitter and Bayes
    predictions on the queries together with their MSEs and the predict-mean baseline. For
    the interactive explorer: as k grows, both predictors improve and the regret (the
    emitter's shortfall) settles to a small bounded gap."""
    import torch
    import torch.nn.functional as F
    device = _device(device)
    net.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    measure, sigma2 = sample_measure_prior(1, d, H, Q, net.gh_nodes, net.gh_wts,
                                           net.W.detach(), gen, device=device)
    Xc, yc, Xq, yq = sample_gp_tasks(measure, k, n_q, sigma2, gen, device=device)
    with torch.no_grad():
        ye = net(Xq, Xc, yc)
        yb = bayes_posterior(measure, Xc, yc, Xq, sigma2)
    return {
        "yq": yq[0, :, 0].cpu().numpy(),
        "emitter": ye[0, :, 0].cpu().numpy(),
        "bayes": yb[0, :, 0].cpu().numpy(),
        "mse_emitter": float(F.mse_loss(ye, yq)),
        "mse_bayes": float(F.mse_loss(yb, yq)),
        "mse_mean": float(F.mse_loss(torch.zeros_like(yq), yq)),
        "k": k,
    }


# =============================================================================
# Figures
# =============================================================================

def make_pipeline_figure():
    """Figure 12.1 — the self-consistent generative pipeline as a flow: a hyperprior draws a
    measure m, m builds the kernel K_m, K_m defines a GP, the GP emits a task (X, y); the
    *same* m feeds the Bayes box. Generator and inferrer share one geometry, so the problem
    is well specified by design and the optimum is known."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, text, fc, fs=9):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                     boxstyle="round,pad=0.1,rounding_size=0.15", fc=fc, ec="0.3", lw=1.2))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, label=None, color="0.4", dy=0.35):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=15, color=color, lw=1.4))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, ha="center", fontsize=8.5,
                    color="0.3")

    y0 = 5.2
    box(1.5, y0, 2.4, 1.0, "hyperprior\n$p(m)$", "#eef3f9")
    box(4.4, y0, 2.0, 1.0, "measure\n$m$", "#dbe7f2")
    box(7.1, y0, 2.0, 1.0, "kernel\n$K_m$", "#cfe0c6")
    box(9.9, y0, 2.4, 1.0, r"$\mathcal{GP}(0,K_m{+}\sigma^2 I)$", "#cfe0c6")
    arrow(2.7, y0, 3.4, y0, "draw")
    arrow(5.4, y0, 6.1, y0, "build")
    arrow(8.1, y0, 8.7, y0, "")

    box(9.9, 2.0, 2.4, 1.0, "task $(X, y)$\ncontext + query", "#f3e2cc")
    arrow(9.9, y0 - 0.5, 9.9, 2.5, r"$y=Lz$")

    box(4.4, 2.0, 3.0, 1.1, "Bayes predictor\n$K_m(\\cdot,X_c)(K_m{+}\\sigma^2I)^{-1}y_c$", "#dbe7f2")
    arrow(8.7, 2.0, 5.9, 2.0, "context", color="0.45")
    # the same m feeds Bayes — one red labeled arrow
    arrow(4.4, y0 - 0.5, 4.4, 2.55, color="#b6403b")
    ax.text(4.7, 3.5, "same $m$", color="#b6403b", fontsize=9, rotation=90, va="center")

    ax.text(6.0, 0.7, "generator and inferrer share one geometry — the optimum is known",
            ha="center", fontsize=9.5, style="italic", color="0.25")
    fig.tight_layout()
    return fig


def make_regret_figure(res=None, **kw):
    """Figure 12.2 — regret over Bayes vs context size k.
    (left) MSE curves: emitter, exact Bayes, predict-the-mean baseline.
    (right) the regret = emitter MSE - Bayes MSE: a small bounded gap that levels off — an
    amortization cost, not a misspecification (Bayes is the zero line)."""
    import matplotlib.pyplot as plt
    if res is None:
        res = run_all(**kw)
    rows = res["regret"]
    ks = res["ks"]
    em = [rows[k]["emitter"] for k in ks]
    ba = [rows[k]["bayes"] for k in ks]
    mn = [rows[k]["mean"] for k in ks]
    rg = [rows[k]["regret"] for k in ks]

    fig, ax = plt.subplots(1, 2, figsize=(12.0, 4.4))
    ax[0].plot(ks, mn, "o--", color="0.6", label="predict-the-mean")
    ax[0].plot(ks, em, "o-", color="#3b6fb6", label="emitter (one forward pass)")
    ax[0].plot(ks, ba, "o-", color="#2e7d32", label="exact Bayes (knows $m$)")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xticks(ks)
    ax[0].set_xticklabels(ks)
    ax[0].set_xlabel("context size $k$")
    ax[0].set_ylabel("query MSE")
    ax[0].set_title("The emitter beats the mean and tracks Bayes\nwithin a small gap", fontsize=10)
    ax[0].legend(fontsize=8.5)

    ax[1].axhline(0, color="#2e7d32", lw=1.2, label="Bayes (optimum)")
    ax[1].plot(ks, rg, "o-", color="#b6403b", lw=1.8, label="regret = emitter $-$ Bayes")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xticks(ks)
    ax[1].set_xticklabels(ks)
    ax[1].set_xlabel("context size $k$")
    ax[1].set_ylabel("regret over Bayes")
    ax[1].set_ylim(bottom=min(0, min(rg) - 0.02))
    ax[1].set_title("The amortization gap: bounded, non-vanishing,\nlevels off as $k$ grows",
                    fontsize=10)
    ax[1].legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    return fig


# =============================================================================
# CLI
# =============================================================================

def main(argv=None):
    p = argparse.ArgumentParser(description="Chapter 12 — meta-learning a prior over kernels")
    p.add_argument("--out-prefix", default=None)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--n-tasks", type=int, default=400)
    p.add_argument("--pool", choices=["mean", "pma"], default="pma")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args(argv)

    from lkbook import set_style
    set_style()
    device = "cpu" if args.cpu else _device()
    print(f"meta-training the emitter on the model's own kernel prior [device={device}]")
    res = run_all(steps=args.steps, n_tasks=args.n_tasks, pool=args.pool, device=device,
                  log_every=max(args.steps // 6, 1))
    print("\nHeld-out-task regret over Bayes vs context size k:")
    print(f"  {'k':>5}  {'emitter':>9}  {'bayes':>9}  {'regret':>9}  {'mean':>9}")
    for k in res["ks"]:
        r = res["regret"][k]
        print(f"  {k:>5}  {r['emitter']:>9.4f}  {r['bayes']:>9.4f}  {r['regret']:>9.4f}  {r['mean']:>9.4f}")
    if "transfer" in res:
        print(f"\nzero-shot California transfer: R2 = {res['transfer']['emitter']:.3f}")

    if args.out_prefix:
        import matplotlib
        matplotlib.use("Agg")
        make_pipeline_figure().savefig(f"{args.out_prefix}1_pipeline.pdf")
        make_regret_figure(res).savefig(f"{args.out_prefix}2_regret.pdf")
        print("wrote figures with prefix", args.out_prefix)


if __name__ == "__main__":
    main()
