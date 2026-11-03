"""Fourier-weighted axial loss for proposal A2.

The Schiffner F-number defines, per acoustic frequency, what fraction of the
aperture participates in beamforming; smaller F → sharper PSF → image content
at the corresponding spatial frequency is more diagnostic. We use this as a
per-frequency weight on the axial-rfft residual between prediction and target:

    L_fourier = mean_(k_z, x) [ W(f(k_z)) · |FFT_z(pred) − FFT_z(target)|² ]

with ``W(f) = 1 / F(f)²`` (modes ``flat``, ``inv_F``, ``inv_F_squared``
delegated to ``physics.f_number_map.compute_spectral_weight``).

Mapping spatial → acoustic frequency:
- For raw RF-like targets centred at 0 Hz: ``f = k_z · c / 2`` (round-trip).
- For envelope+log targets (modulation envelope around baseband): ``f = fc +
  k_z · c / 2`` — A2 trains on env+log targets, so use ``fc_offset=True``.

Contract:
- Loss is gradient-friendly w.r.t. ``pred`` (real-valued FFT, real residual).
- The weight tensor is a *fixed* numpy/torch array per phantom (depends only
  on grid and geometry); compute it once outside the training loop.
- No autograd through ``H_Fdf`` (this loss does not use H — A2 is the
  "spectral fidelity in image domain" alternative to A1's "spatial fidelity
  via H").
"""

from __future__ import annotations

import numpy as np
import torch
import torch.fft
import torch.nn.functional as F

from ..physics.f_number_map import compute_spectral_weight


def axial_freq_to_acoustic(nz: int, dz: float, c: float) -> np.ndarray:
    """Axial rfft bins (cycles/m) → acoustic frequency (Hz) via ``f = k_z·c/2``.

    Returns array of shape ``(nz//2 + 1,)``.
    """
    k_z = np.fft.rfftfreq(nz, d=dz).astype(np.float64)   # cycles/m
    return k_z * c / 2.0                                  # Hz


def build_axial_spectral_weight(
    nz: int,
    dz: float,
    geometry: dict,
    mode: str = "inv_F_squared",
    fc_offset: bool = True,
    f_floor_hz: float = 1.0e5,
) -> torch.Tensor:
    """1-D weight ``W[k_z]`` for the axial-rfft loss.

    Parameters
    ----------
    nz : axial samples of the image
    dz : axial sampling step (m)
    geometry : phantom geometry (``pitch_m``, ``speed_of_sound``,
        ``sampling_freq_hz``, optional ``fc``)
    mode : forwarded to ``compute_spectral_weight``
    fc_offset : if True, treat ``k_z`` as a baseband modulation offset and
        map ``f = fc + k_z·c/2``. For env+log targets (A2 default).
    f_floor_hz : floor applied before computing F (avoid DC blowup when
        ``mode="inv_F_squared"`` and F→0 at f→0).

    Returns Tensor shape ``(nz//2 + 1,)`` float32, normalised to mean 1.
    """
    c = float(geometry["speed_of_sound"])
    f_acoustic = axial_freq_to_acoustic(nz, dz, c)
    if fc_offset:
        fc = float(geometry.get("fc", geometry["sampling_freq_hz"] / 4.0))
        f_eff = fc + f_acoustic
    else:
        f_eff = f_acoustic
    f_eff = np.maximum(f_eff, f_floor_hz)
    W = compute_spectral_weight(f_eff, geometry, mode=mode)
    W = W / float(W.mean() + 1e-12)                       # mean-1 → loss scale stable
    return torch.from_numpy(W.astype(np.float32))


def build_axial_spectral_weight_full(
    nz: int,
    dz: float,
    geometry: dict,
    mode: str = "inv_F_squared",
    fc_offset: bool = True,
    f_floor_hz: float = 1.0e5,
) -> torch.Tensor:
    """Full-Nz axial weight (signed ``k_z`` via ``np.fft.fftfreq``).

    Needed for ``rfft2`` output where the axial axis carries the full Nz bins
    (Hermitian-symmetric for real input) and only the lateral axis is folded.
    Uses ``|k_z|`` to remain symmetric.

    Returns Tensor shape ``(nz,)`` float32, mean-1.
    """
    c = float(geometry["speed_of_sound"])
    k_z = np.fft.fftfreq(nz, d=dz).astype(np.float64)     # signed cycles/m
    f_acoustic = np.abs(k_z) * c / 2.0
    if fc_offset:
        fc = float(geometry.get("fc", geometry["sampling_freq_hz"] / 4.0))
        f_eff = fc + f_acoustic
    else:
        f_eff = f_acoustic
    f_eff = np.maximum(f_eff, f_floor_hz)
    W = compute_spectral_weight(f_eff, geometry, mode=mode)
    W = W / float(W.mean() + 1e-12)
    return torch.from_numpy(W.astype(np.float32))


def build_lateral_spectral_weight(
    nx: int,
    mode: str = "highpass_linear",
    floor: float = 0.1,
) -> torch.Tensor:
    """1-D weight ``W[k_x]`` for the lateral-rfft loss (mean-1).

    Geometry-free: PWUS does NOT define a clean per-frequency F-number in the
    lateral direction (it depends on the synthetic aperture angle and depth),
    so we use a simple high-pass shape parameterised by the floor:

        W(k) = floor + (1 − floor) · (|k| / k_max)^α

    where α = 1 (``highpass_linear``) or 2 (``highpass_quad``).

    The floor keeps low-frequency lateral content non-trivially weighted —
    needed for cysts (contrast lives at low spatial frequencies). High-pass
    push targets the missing PSF sharpness in points.

    Returns Tensor shape ``(nx // 2 + 1,)`` float32.
    """
    n_bins = nx // 2 + 1
    if n_bins <= 1:
        return torch.ones(n_bins, dtype=torch.float32)
    k_norm = np.arange(n_bins, dtype=np.float64) / float(n_bins - 1)  # 0..1
    if mode == "flat":
        W = np.ones(n_bins, dtype=np.float64)
    elif mode == "highpass_linear":
        W = floor + (1.0 - floor) * k_norm
    elif mode == "highpass_quad":
        W = floor + (1.0 - floor) * (k_norm ** 2)
    else:
        raise ValueError(f"unknown lateral mode {mode!r}")
    W = W / float(W.mean() + 1e-12)
    return torch.from_numpy(W.astype(np.float32))


def build_2d_spectral_weight(
    nz: int,
    dz: float,
    nx: int,
    geometry: dict,
    axial_mode: str = "inv_F_squared",
    lateral_mode: str = "highpass_linear",
    fc_offset: bool = True,
    lateral_floor: float = 0.1,
    f_floor_hz: float = 1.0e5,
) -> torch.Tensor:
    """Separable 2-D weight ``W[k_z, k_x] = W_z[k_z] · W_x[k_x]`` (mean-1).

    Shape matches ``torch.fft.rfft2`` output along the last two axes:
    ``(nz, nx // 2 + 1)``. Axial uses full-Nz bins (FFT, Hermitian), lateral
    uses Nx//2+1 bins (real-to-complex). A2-v2 uses this to penalise residual
    not just per acoustic frequency (axial) but also per spatial lateral
    frequency — the missing knob to force PSF sharpness laterally.
    """
    Wz = build_axial_spectral_weight_full(
        nz, dz, geometry, mode=axial_mode,
        fc_offset=fc_offset, f_floor_hz=f_floor_hz,
    )                                                       # (Nz,)
    Wx = build_lateral_spectral_weight(nx, mode=lateral_mode, floor=lateral_floor)
    W2 = Wz.view(-1, 1) * Wx.view(1, -1)                    # (Nz, Nx//2+1)
    W2 = W2 / (W2.mean() + 1e-12)
    return W2.contiguous()


def fourier_weighted_axial_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Weighted L2 of the axial rfft residual.

    Parameters
    ----------
    pred, target : ``(B, 1, Nz, Nx)`` real tensors
    weight : ``(Nz//2 + 1,)`` real tensor (any device, will be moved to pred)
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    Fp = torch.fft.rfft(pred, dim=-2)
    Ft = torch.fft.rfft(target, dim=-2)
    res = Fp - Ft                                          # complex (B,1,Nz//2+1,Nx)
    mag2 = res.real ** 2 + res.imag ** 2
    w = weight.to(device=pred.device, dtype=pred.dtype).view(1, 1, -1, 1)
    return (w * mag2).mean()


def fourier_weighted_2d_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight_2d: torch.Tensor,
) -> torch.Tensor:
    """Weighted L2 of the 2-D rfft2 residual (axial + lateral spectral weights).

    Parameters
    ----------
    pred, target : ``(B, 1, Nz, Nx)`` real tensors
    weight_2d : ``(Nz, Nx//2 + 1)`` real tensor — matches ``rfft2`` output
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    Fp = torch.fft.rfft2(pred, dim=(-2, -1))               # (B,1,Nz,Nx//2+1) complex
    Ft = torch.fft.rfft2(target, dim=(-2, -1))
    res = Fp - Ft
    mag2 = res.real ** 2 + res.imag ** 2
    w = weight_2d.to(device=pred.device, dtype=pred.dtype)
    w = w.view(1, 1, w.shape[0], w.shape[1])
    return (w * mag2).mean()


def target_weighted_mse_l1_fourier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    mse_weight: float = 0.7,
    l1_weight: float = 0.2,
    fourier_weight: float = 0.15,
    alpha: float = 9.0,
) -> tuple[torch.Tensor, dict]:
    """A2-v6 loss: foreground-emphasized MSE+L1 + Fourier-W.

    Per-pixel weighting ``w(t) = 1 + α·t`` rebalances the gradient against the
    background-pixel domination diagnosed in v1-v5 (target env+log is ~99%
    low-value background and ~1% bright wires; mean-reduced MSE/L1 collapses
    to the background mean → predict-zero on point targets).

    With ``α=9`` brightness-1 pixels weigh 10× the background; combined with a
    ~1% bright-pixel ratio this puts ~50% of the gradient on the wires (which
    is the diagnostic content) instead of <5%.

    Mean-normalisation by ``w.mean()`` keeps the loss scale comparable to the
    standard ``mse_l1_fourier_loss`` (drop-in replacement at α=0).

    The Fourier-W term is unweighted — it already operates in the spectral
    domain where the sparse-bright vs dense-background imbalance does not
    apply.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    w = 1.0 + alpha * target
    w_mean = w.mean() + 1e-12
    err = pred - target
    mse = (w * err.pow(2)).mean() / w_mean
    l1 = (w * err.abs()).mean() / w_mean
    fou = fourier_weighted_2d_loss(pred, target, weight)
    total = mse_weight * mse + l1_weight * l1 + fourier_weight * fou
    return total, {
        "mse_w": float(mse.item()),
        "l1_w": float(l1.item()),
        "fourier": float(fou.item()),
        "total": float(total.item()),
    }


def mse_l1_fourier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    mse_weight: float = 0.7,
    l1_weight: float = 0.2,
    fourier_weight: float = 0.1,
    fourier_dim: int = 1,
) -> tuple[torch.Tensor, dict]:
    """A2 combined loss: ``α·MSE + β·L1 + γ·Fourier_W``.

    Defaults match B1's mse_l1_ssim_loss (0.7 / 0.2 / 0.1) — keep parity so
    the only difference between B1 and A2 is the third term (SSIM vs Fourier-W)
    and the input modality (adjoint image vs RF).

    ``fourier_dim`` selects the spectral domain:
    - 1 (default): axial-only rfft loss, weight shape ``(Nz//2+1,)``.
    - 2 (A2-v2): 2-D rfft2 loss, weight shape ``(Nz, Nx//2+1)``.
    """
    mse = F.mse_loss(pred, target)
    l1 = F.l1_loss(pred, target)
    if fourier_dim == 1:
        fou = fourier_weighted_axial_loss(pred, target, weight)
    elif fourier_dim == 2:
        fou = fourier_weighted_2d_loss(pred, target, weight)
    else:
        raise ValueError(f"fourier_dim must be 1 or 2, got {fourier_dim}")
    total = mse_weight * mse + l1_weight * l1 + fourier_weight * fou
    return total, {
        "mse": float(mse.item()),
        "l1": float(l1.item()),
        "fourier": float(fou.item()),
        "total": float(total.item()),
    }
