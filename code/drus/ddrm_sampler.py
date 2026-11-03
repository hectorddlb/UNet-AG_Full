"""DDRM sampler for DRUS — Phase D.

Port of ``functions/denoising.py::efficient_generalized_steps`` and
``functions/svd_replacement.py::ultrasound0`` from the DRUS-v1 repo, adapted to
our Phase B artifacts:

- SVD in the numpy C-order convention: ``BH = U @ diag(S) @ Vt`` with the
  row-major image vector ``p = iz*Nx + ix`` (without the permutations the
  original needed for MATLAB column-major).
- ``Uty = Uᵀ·By`` comes precomputed from the Phase B job (the sampler never
  needs the full U: y is fixed per phantom).

Vector layout (identical to the original to replicate the arithmetic):
- image space: flat channel-major ``[c0 p0..pN, c1 p0..pN, c2 ...]``
  (= ``x.view(B, -1)`` of a (B,3,S,S) tensor).
- spectral space: flat pixel-major with triplets ``[m0c0 m0c1 m0c2, m1c0...]``
  (spectral mode × channel), with σ_k replicated 3 consecutive times.

The model (553M UNet, RGB input) runs on GPU; the spectral matmuls with
Vt (17 GB) run on CPU — as in Zhang's original code. γ is multiplied by √3 as
in the original (y replicated to 3 channels).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

CHANNELS = 3


# ---------------------------------------------------------------------------
# Spectral operator (replacement of ultrasound0)
# ---------------------------------------------------------------------------

class SpectralOps:
    """Multiplications by V/Vᵀ replicated to 3 channels.

    ``vt``: (N, N) float32 — rows = right singular vectors (numpy Vt).
    The products run on ``spectral_device`` (CPU because of the size).
    """

    def __init__(self, s: np.ndarray, vt: np.ndarray, spectral_device: str = "cpu"):
        self.dev = torch.device(spectral_device)
        self.vt = torch.from_numpy(vt).to(self.dev)            # (N, N)
        self.n = self.vt.shape[0]
        s_t = torch.from_numpy(np.ascontiguousarray(s)).float()
        self.singulars = s_t.repeat_interleave(CHANNELS).to(self.dev)   # (3N,) triplets

    # --- image (channel-major flat) → spectral (pixel-major flat) ---
    def Vt(self, vec: torch.Tensor) -> torch.Tensor:
        b = vec.shape[0]
        x = vec.reshape(b, CHANNELS, self.n).permute(0, 2, 1).to(self.dev)  # (B,N,3)
        out = torch.matmul(self.vt, x)                                       # (B,N,3)
        return out.reshape(b, -1)                                            # pixel-major

    # --- spectral (pixel-major flat) → image (channel-major flat) ---
    def V(self, vec: torch.Tensor) -> torch.Tensor:
        b = vec.shape[0]
        x = vec.reshape(b, self.n, CHANNELS).to(self.dev)                    # (B,N,3)
        out = torch.matmul(self.vt.transpose(0, 1), x)                       # (B,N,3)
        return out.permute(0, 2, 1).reshape(b, -1)                           # channel-major


def expand_uty(uty: np.ndarray) -> torch.Tensor:
    """Uᵀ·By (N,) → (1, 3N) pixel-major (equivalent to Ut(y repeated over 3 channels))."""
    t = torch.from_numpy(np.ascontiguousarray(uty)).float()
    return t.repeat_interleave(CHANNELS)[None, :]


# ---------------------------------------------------------------------------
# DDRM (efficient_generalized_steps)
# ---------------------------------------------------------------------------

@dataclass
class DDRMResult:
    x_final: np.ndarray     # (3, S, S) — final x_t (≡ xs[-1] of the original: what the paper evaluates)
    x_mean: np.ndarray      # (S, S) — channel mean of x_final
    x0_pred: np.ndarray     # (3, S, S) — last x0 prediction of the model (diagnostic)
    seconds: float
    gamma_effective: float


def _compute_alpha(beta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    beta = torch.cat([torch.zeros(1, device=beta.device), beta], dim=0)
    return (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)


@torch.no_grad()
def ddrm_sample(
    model: torch.nn.Module,
    betas: np.ndarray,           # (1000,) linear 1e-4..0.02 — same as the FT
    ops: SpectralOps,
    uty: torch.Tensor,           # (1, 3N) de expand_uty
    gamma: float,                # observation-noise σ (HP, paper grid)
    image_size: int = 256,
    timesteps: int = 50,
    eta: float = 0.85,           # etaA = etaC (defaults del repo DRUS-v1)
    eta_b: float = 1.0,
    seed: int = 0,
    model_device: str = "cuda",
    verbose: bool = True,
) -> DDRMResult:
    t_start = time.time()
    dev_m = torch.device(model_device)
    dev_s = ops.dev
    b = torch.from_numpy(betas.astype(np.float32)).to(dev_s)
    num_t = b.shape[0]

    sigma_0 = float(gamma) * np.sqrt(CHANNELS)     # like the original (in_channels==3)
    singulars = ops.singulars                       # (3N,)
    u_t_y = uty.to(dev_s)                           # (1, 3N)
    sig_inv_u_t_y = u_t_y / singulars.clamp_min(1e-30)[None, :]

    g = torch.Generator(device="cpu").manual_seed(seed)
    shape = (1, CHANNELS, image_size, image_size)
    x = torch.randn(*shape, generator=g)

    skip = num_t // timesteps
    seq = list(range(0, num_t, skip))
    seq_next = [-1] + seq[:-1]

    # ---- init x_T | y (p(x_T | x0, y) of the DDRM paper) ----
    largest_alpha = _compute_alpha(b, torch.tensor([seq[-1]], device=dev_s).long())
    largest_sigma = ((1 - largest_alpha).sqrt() / largest_alpha.sqrt()).item()
    large_idx = singulars * largest_sigma > sigma_0
    inv_sing_zero = torch.zeros_like(singulars)
    inv_sing_zero[large_idx] = sigma_0 / singulars[large_idx]

    init_y = torch.zeros(1, CHANNELS * ops.n, device=dev_s)
    init_y[0, large_idx] = u_t_y[0, large_idx] / singulars[large_idx]
    remaining = (largest_sigma ** 2 - inv_sing_zero ** 2).clamp_min(0.0).sqrt()
    init_y = init_y + remaining[None, :] * x.view(1, -1).to(dev_s)
    init_y = init_y / largest_sigma
    xt = ops.V(init_y).view(*shape)                # image channel-major

    x0_t = None
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = torch.tensor([i], device=dev_s).long()
        t_next = torch.tensor([j], device=dev_s).long()
        at = _compute_alpha(b, t)
        at_next = _compute_alpha(b, t_next)
        sigma = ((1 - at).sqrt() / at.sqrt()).item()
        sigma_next = ((1 - at_next).sqrt() / at_next.sqrt()).item()

        # --- UNet en GPU ---
        et = model(xt.to(dev_m), torch.tensor([i], device=dev_m).float())
        if et.size(1) == 2 * CHANNELS:
            et = et[:, :CHANNELS]                   # learn_sigma: descarta varianza
        et = et.to(dev_s).float()

        x0_t = (xt.to(dev_s) - et * (1 - at).sqrt().item()) / at.sqrt().item()

        # --- spectral update (3 DDRM cases) ---
        v_t_x0 = ops.Vt(x0_t.view(1, -1))                       # (1, 3N)
        sv_t_x0 = v_t_x0 * singulars[None, :]

        cond_before = singulars * sigma_next > sigma_0
        cond_after = singulars * sigma_next < sigma_0

        std_c = sigma_next * eta
        sig_tilde_c = float(np.sqrt(max(sigma_next ** 2 - std_c ** 2, 0.0)))
        std_a = sigma_next * eta
        sig_tilde_a = float(np.sqrt(max(sigma_next ** 2 - std_a ** 2, 0.0)))

        vt_et = ops.Vt(et.view(1, -1))
        noise = torch.randn(v_t_x0.shape, generator=g).to(dev_s)

        # "missing" case (σ = 0 does not occur here, but keeps the shape of the original)
        vt_next = v_t_x0 + sig_tilde_c * vt_et + std_c * noise

        # "after" case: σ·σ_next < σ_0 — pulls towards y with a relative weight
        vt_next[:, cond_after] = (
            v_t_x0[:, cond_after]
            + sig_tilde_a * ((u_t_y - sv_t_x0) / sigma_0)[:, cond_after]
            + std_a * noise[:, cond_after]
        )

        # "before" case: σ·σ_next > σ_0 — mix with S⁻¹Uᵀy
        if cond_before.any():
            diff_sig = (sigma_next ** 2
                        - (sigma_0 / singulars[cond_before]) ** 2 * eta_b ** 2).clamp_min(0.0).sqrt()
            noise_b = torch.randn(u_t_y.shape, generator=g).to(dev_s)
            vt_next[:, cond_before] = (
                sig_inv_u_t_y[:, cond_before] * eta_b
                + (1 - eta_b) * v_t_x0[:, cond_before]
                + diff_sig[None, :] * noise_b[:, cond_before]
            )

        xt = (at_next.sqrt().item() * ops.V(vt_next)).view(*shape)
        if verbose:
            print(f"  [ddrm] t={i:4d} σ={sigma:8.3f} | {time.time()-t_start:6.1f}s", flush=True)

    x_final = xt.view(CHANNELS, image_size, image_size).cpu().numpy().astype(np.float32)
    x0_pred = x0_t.view(CHANNELS, image_size, image_size).cpu().numpy().astype(np.float32)
    return DDRMResult(
        x_final=x_final,
        x_mean=x_final.mean(axis=0),
        x0_pred=x0_pred,
        seconds=time.time() - t_start,
        gamma_effective=sigma_0,
    )
