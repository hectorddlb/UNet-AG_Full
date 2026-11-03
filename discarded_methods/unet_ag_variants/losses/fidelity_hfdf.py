"""H_Fdf fidelity loss for propuestas A1 and A2.

Thin alias over ``unet_ag.physics.h_fdf_wrapper.residual_fidelity_loss``,
plus a multi-angle variant that averages ``‖y_a − H_Fdf_a x'‖²`` across
selected angles.

Per the α contract: the operator is fixed (no learnable params). Gradient
flows only to ``x_pred``.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor

from ..physics.h_fdf_wrapper import residual_fidelity_loss


def fidelity_loss_1pw(x_pred: Tensor, y_obs: Tensor, op_a) -> Tensor:
    return residual_fidelity_loss(x_pred, y_obs, op_a, reduction="mean")


def fidelity_loss_multiangle(
    x_pred: Tensor,
    y_list: Iterable[Tensor],            # [(D, K), ...] per angle
    op_list: Iterable,                   # [OpHFdf, ...] same order
    reduction: str = "mean",
) -> Tensor:
    """``(1/K) Σ_a ‖y_a − H_Fdf_a x_pred‖²``."""
    y_list = list(y_list)
    op_list = list(op_list)
    if len(y_list) != len(op_list):
        raise ValueError(f"y_list and op_list length mismatch ({len(y_list)} vs {len(op_list)})")
    n = len(y_list)
    accum = None
    for y_a, op_a in zip(y_list, op_list):
        l = residual_fidelity_loss(x_pred, y_a, op_a, reduction=reduction)
        accum = l if accum is None else accum + l
    return accum / n
