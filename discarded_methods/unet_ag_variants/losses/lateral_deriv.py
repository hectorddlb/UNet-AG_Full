"""Edge-sharpness losses for proposal A2-v7 — penalize residual on the
**lateral derivative** (and optionally axial derivative) of the image.

Motivation. The B1/A2 family failure root cause is that
MSE/L1/SSIM on env+log images penalize differences in *integrated wire
intensity* but not in *wire edge sharpness*. A smoothed and a sharp wire
have the same mean intensity over the wire body, so a UNet that minimises
mean-reduced MSE has a degenerate optimum at "isotropic smoothing".

This module supplies the missing knob: a per-pixel L1/L2 on the lateral
finite-difference of the residual,

    L_lat = mean( | ∂_x pred − ∂_x target | )

implemented via 1-D finite-difference conv kernel ``[−1/2, 0, +1/2]``
along the lateral axis. The optimum *requires* ``pred`` to have the same
lateral gradient as ``target``, which on point-target wires means the same
PSF width. Background pixels contribute ~0 gradient because their lateral
derivative is ~0 in both pred and target.

The axial counterpart ``∂_z`` is included for completeness but should be
weighted lower or off because the Schiffner_75PW target is *axially wider*
than the DAS_1PW input — pushing pred to match target's ∂_z would actually
*worsen* the axial PSF on points. The default exposes a knob ``axial_weight``
that defaults to 0.05 (light regularisation) or can be set to 0 to disable.

Contract
--------
- Both losses are gradient-friendly w.r.t. ``pred`` (real residual, fixed
  conv kernel — no learnable params).
- Combined-loss API matches ``fourier_weighted.target_weighted_mse_l1_fourier_loss``
  so it slots into the same trainer scaffolding (``a2_v4_adjoint_fourier._iter_split``).
- No autograd through ``H_Fdf`` (this module never touches H — Α2-v7 is the
  "edge-sharpness fidelity in image domain" rescue).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


_KERNEL_X = torch.tensor([[-0.5, 0.0, 0.5]], dtype=torch.float32).view(1, 1, 1, 3)
_KERNEL_Z = torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.float32).view(1, 1, 3, 1)


def _lateral_diff(x: torch.Tensor) -> torch.Tensor:
    """∂_x via 1×3 central finite difference (padding=(0,1) → same Nx)."""
    w = _KERNEL_X.to(device=x.device, dtype=x.dtype)
    return F.conv2d(x, w, padding=(0, 1))


def _axial_diff(x: torch.Tensor) -> torch.Tensor:
    """∂_z via 3×1 central finite difference (padding=(1,0) → same Nz)."""
    w = _KERNEL_Z.to(device=x.device, dtype=x.dtype)
    return F.conv2d(x, w, padding=(1, 0))


def lateral_derivative_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 of the lateral-derivative residual ``∂_x pred − ∂_x target``.

    Inputs are ``(B, 1, Nz, Nx)`` real tensors. Returns a scalar tensor.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    return (_lateral_diff(pred) - _lateral_diff(target)).abs().mean()


def axial_derivative_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 of the axial-derivative residual ``∂_z pred − ∂_z target``."""
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    return (_axial_diff(pred) - _axial_diff(target)).abs().mean()


def target_weighted_mse_l1_lateral_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float = 0.5,
    l1_weight: float = 0.15,
    lateral_weight: float = 0.3,
    axial_weight: float = 0.05,
    alpha: float = 9.0,
) -> Tuple[torch.Tensor, dict]:
    """A2-v7 loss: target-weighted MSE+L1 plus lateral (and optional axial)
    derivative residual.

        L = mse_w·MSE_w + l1_w·L1_w + lat_w·|∂_x e| + ax_w·|∂_z e|

    where ``e = pred − target`` and the body terms use the v6 target-weighting
    ``w(t) = 1 + α·t`` to keep ~50% of the body gradient on bright wires.

    ``lateral_weight=0.3`` puts a third of the total gradient mass on the
    edge — empirically this is where the Pareto sweet spot should live based
    on the B1/A2 diagnosis.

    Default ``axial_weight=0.05`` is a *light* regulariser: setting it ≫
    ``lateral_weight`` would force pred to track Schiffner_75PW's *broader*
    axial PSF and would worsen FWHM_ax on points. Set to 0 to disable
    axial regularisation entirely.
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    w = 1.0 + alpha * target
    w_mean = w.mean() + 1e-12
    err = pred - target
    mse = (w * err.pow(2)).mean() / w_mean
    l1 = (w * err.abs()).mean() / w_mean

    dx_pred = _lateral_diff(pred)
    dx_target = _lateral_diff(target)
    lat = (dx_pred - dx_target).abs().mean()

    total = mse_weight * mse + l1_weight * l1 + lateral_weight * lat
    br = {
        "mse_w": float(mse.item()),
        "l1_w": float(l1.item()),
        "lateral": float(lat.item()),
    }

    if axial_weight > 0.0:
        dz_pred = _axial_diff(pred)
        dz_target = _axial_diff(target)
        ax = (dz_pred - dz_target).abs().mean()
        total = total + axial_weight * ax
        br["axial"] = float(ax.item())

    br["total"] = float(total.item())
    return total, br
