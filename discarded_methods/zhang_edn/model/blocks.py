"""
Reusable building blocks for the 12-layer Encoder-Decoder Network.

All convolution blocks use **Tanh** activation (not ReLU) because RF
signals are bipolar.
"""

from __future__ import annotations

import torch.nn as nn


class ConvTanhBN(nn.Module):
    """Conv2d -> Tanh -> Norm. Norm switchable via ``norm_type``.

    Phase 2v3 hygiene fix: Stage 2 trains with batch_size=1 (per-sample H);
    BatchNorm2d collapses to InstanceNorm with corrupted running stats and
    untrained gamma/beta in that regime. GroupNorm(8) is independent of
    batch size and stable for all our channel widths (32, 64, 128 → groups
    of 4, 8, 16). ``norm_type='batchnorm'`` exists for ablation only.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 norm_type: str = "groupnorm") -> None:
        super().__init__()
        padding = kernel_size // 2  # same-padding
        if norm_type == "groupnorm":
            norm_layer = nn.GroupNorm(num_groups=8, num_channels=out_ch)
        elif norm_type == "batchnorm":
            norm_layer = nn.BatchNorm2d(out_ch)
        else:
            raise ValueError(
                f"Unknown norm_type: {norm_type!r}. Use 'groupnorm' or 'batchnorm'."
            )
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
            nn.Tanh(),
            norm_layer,
        )

    def forward(self, x):
        return self.block(x)


class DeConv(nn.Module):
    """Upsample(nearest, 2x) + Conv2d(3x3) for artefact-free 2x upsampling.

    Replaces ``ConvTranspose2d(kernel=2, stride=2)`` which produces
    checkerboard artefacts.  The paper says "Deconvolution" but
    upsample+conv is the standard fix and produces equivalent results.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.up(x)
