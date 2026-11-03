"""Loss functions for the two-stage training (Sec. 7 of Zhang et al. 2021)."""

from zhang_dnn.losses.mse_loss import mse_stage1, mse_stage2
from zhang_dnn.losses.wavelet_sparsity import wavelet_sparsity_loss
from zhang_dnn.losses.smoothness import smoothness_loss
from zhang_dnn.losses.envelope_sparsity import envelope_sparsity_loss
from zhang_dnn.losses.combined_loss import combined_loss

__all__ = [
    "mse_stage1",
    "mse_stage2",
    "wavelet_sparsity_loss",
    "smoothness_loss",
    "envelope_sparsity_loss",
    "combined_loss",
]
