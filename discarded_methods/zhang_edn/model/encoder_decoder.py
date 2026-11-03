"""
12-layer Encoder-Decoder Network (EDN) from Table 2 of Zhang et al. 2021.

Architecture
------------
Encoder 1:  Conv(1->32,  15x15) + Tanh + GN(8)  ->  Conv(32->64,  15x15) + Tanh + GN(8)  ->  MaxPool(2)
Encoder 2:  Conv(64->128, 7x7)  + Tanh + GN(8)  ->  Conv(128->128, 7x7)  + Tanh + GN(8)  ->  MaxPool(2)
Decoder 1:  DeConv(128->128, 2x2, s2)  ->  Conv(128->128, 3x3) + Tanh + GN(8)  ->  Conv(128->128, 3x3) + Tanh + GN(8)
Decoder 2:  DeConv(128->64, 2x2, s2)   ->  Conv(64->64, 3x3)   + Tanh + GN(8)  ->  Conv(64->32, 3x3)   + Tanh + GN(8)
Output:     Conv(32->1, 1x1)

Key design notes:
  - Tanh everywhere (bipolar RF signals).
  - Large encoder kernels (15x15, 7x7) for wide receptive field.
  - NO skip connections — this is NOT a U-Net.
  - Input dims must be divisible by 4 (two MaxPool layers). The network
    pads internally if needed and crops the output to match input size.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from zhang_dnn.model.blocks import ConvTanhBN, DeConv


class EncoderDecoderNet(nn.Module):
    """12-layer EDN for ultrasound image reconstruction."""

    def __init__(self, norm_type: str = "groupnorm") -> None:
        super().__init__()

        # ── Encoder 1 ──
        self.enc1 = nn.Sequential(
            ConvTanhBN(1, 32, kernel_size=15, norm_type=norm_type),
            ConvTanhBN(32, 64, kernel_size=15, norm_type=norm_type),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── Encoder 2 ──
        self.enc2 = nn.Sequential(
            ConvTanhBN(64, 128, kernel_size=7, norm_type=norm_type),
            ConvTanhBN(128, 128, kernel_size=7, norm_type=norm_type),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── Decoder 1 ──
        self.dec1 = nn.Sequential(
            DeConv(128, 128),
            ConvTanhBN(128, 128, kernel_size=3, norm_type=norm_type),
            ConvTanhBN(128, 128, kernel_size=3, norm_type=norm_type),
        )

        # ── Decoder 2 ──
        self.dec2 = nn.Sequential(
            DeConv(128, 64),
            ConvTanhBN(64, 64, kernel_size=3, norm_type=norm_type),
            ConvTanhBN(64, 32, kernel_size=3, norm_type=norm_type),
        )

        # ── Output projection ──
        self.output_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (batch, 1, Nz, Nx) float tensor.

        Returns
        -------
        (batch, 1, Nz, Nx) — same spatial size as input.
        """
        _, _, H_in, W_in = x.shape

        # Pad to multiples of 4 if needed
        pad_h = (4 - H_in % 4) % 4
        pad_w = (4 - W_in % 4) % 4
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)

        x = self.enc1(x)
        x = self.enc2(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.output_conv(x)

        # Crop back to original size
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H_in, :W_in]

        return x
