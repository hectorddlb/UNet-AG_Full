"""
Combined loss for Stage 2 training (Eq. 13, Zhang et al. 2021).

    Loss = L_MSE_Stage2 + lambda1 * L_SpW + lambda2 * L_Sm + lambda3 * L_SpE

Optimal weights from the paper:
    lambda* = {0.2, 0.2, 0.1}
"""

from __future__ import annotations

from torch import Tensor

from zhang_dnn.losses.mse_loss import mse_stage2
from zhang_dnn.losses.wavelet_sparsity import WaveletSparsityLoss
from zhang_dnn.losses.smoothness import smoothness_loss
from zhang_dnn.losses.envelope_sparsity import envelope_sparsity_loss


def combined_loss(
    y_measured: Tensor,
    y_predicted: Tensor,
    x_prime: Tensor,
    wavelet_loss_fn: WaveletSparsityLoss,
    lambda1: float = 0.2,
    lambda2: float = 0.2,
    lambda3: float = 0.1,
) -> tuple[Tensor, dict[str, float]]:
    """Compute the complete Stage 2 loss (Eq. 13).

    Parameters
    ----------
    y_measured : (batch, D*K) — measured RF data.
    y_predicted : (batch, D*K) — H @ x' forward projection.
    x_prime : (batch, 1, Nz, Nx) — network output.
    wavelet_loss_fn : pre-built WaveletSparsityLoss module.
    lambda1, lambda2, lambda3 : regularisation weights.

    Returns
    -------
    total_loss : scalar tensor with grad.
    breakdown : dict with individual loss values (detached, for logging).
    """
    l_mse = mse_stage2(y_measured, y_predicted)
    l_wav = wavelet_loss_fn(x_prime)
    l_sm = smoothness_loss(x_prime)
    l_env = envelope_sparsity_loss(x_prime)

    total = l_mse + lambda1 * l_wav + lambda2 * l_sm + lambda3 * l_env

    breakdown = {
        "mse": l_mse.item(),
        "wavelet": l_wav.item(),
        "smoothness": l_sm.item(),
        "envelope": l_env.item(),
        "total": total.item(),
    }

    return total, breakdown
