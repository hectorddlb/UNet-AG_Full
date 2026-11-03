"""
MSE losses for Stage 1 and Stage 2 (Eq. 7-8 of Zhang et al. 2021).

Stage 1 (Eq. 7):  L = (1/2) * ||x_hat - x_tilde||^2
Stage 2 (Eq. 8):  L = (1/2) * ||y - H @ x'||^2
"""

from __future__ import annotations

import torch
from torch import Tensor


def mse_stage1(x_hat: Tensor, x_tilde: Tensor) -> Tensor:
    """Autoencoder MSE loss (Eq. 7).

    Parameters
    ----------
    x_hat : (batch, 1, Nz, Nx) — initial solution (target).
    x_tilde : (batch, 1, Nz, Nx) — network output.
    """
    return 0.5 * torch.mean((x_hat - x_tilde) ** 2)


def mse_stage2(y_measured: Tensor, y_predicted: Tensor) -> Tensor:
    """Self-supervised inverse-problem MSE loss (Eq. 8).

    Parameters
    ----------
    y_measured : (batch, D*K) — measured RF data (vectorised).
    y_predicted : (batch, D*K) — H @ x' (forward projection of network output).
    """
    return 0.5 * torch.mean((y_measured - y_predicted) ** 2)
