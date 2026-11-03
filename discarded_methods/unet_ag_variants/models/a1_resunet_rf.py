"""Proposal A1 — RF encoder + ResU-Net (no Attention Gate).

**Branch**: with H (α). H_Fdf lives only in the loss; the model does not
include it in its forward graph.

**Declared input**: raw 1 PW RF, ``(B, 1, D, K)``, z-score normalized +
``/max(|.|)``.

**Output**: linear ``(B, 1, Nz, Nx)`` image (no sigmoid — the target is raw
signed, normalized by ``/max(|.|)`` per phantom; the fidelity term
``‖y − H_Fdf x'‖²`` protects against the "predict zero" collapse).

**Differences vs B1**:
- Input: direct RF instead of the adjoint image.
- Backbone: plain ResU-Net (no Attention Gate or F-number conditioning).
- Loss: adds ``λ·‖y − H_Fdf x'‖²`` (α contract — H without learnable parameters).
- Target: raw signed (no envelope+log compress).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .shared_blocks import (
    ConvBlock,
    DownBlock,
    ResBlock,
    RFEncoder1DTime,
    UpBlock,
)


class A1ResUNetRF(nn.Module):
    """RF encoder + 3-level ResU-Net image decoder."""

    def __init__(
        self,
        n_samples: int,
        n_elements: int,
        nz: int,
        nx: int,
        base_ch: int = 32,
        norm: str = "bn",
    ):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
        self.nz, self.nx = int(nz), int(nx)

        # (B, 1, D, K) → (B, c1, Nz, Nx)
        self.rf_encoder = RFEncoder1DTime(
            n_samples=n_samples,
            n_elements=n_elements,
            nz=nz,
            nx=nx,
            out_ch=c1,
        )

        # 3-level U-Net (Down × 3 + bottleneck + Up × 3)
        self.d1 = DownBlock(c1, c2, norm=norm)
        self.d2 = DownBlock(c2, c3, norm=norm)
        self.d3 = DownBlock(c3, c4, norm=norm)

        self.bot = nn.Sequential(
            ResBlock(c4, norm=norm),
            ResBlock(c4, norm=norm),
        )

        self.u3 = UpBlock(c4, c4, c3, norm=norm)
        self.u2 = UpBlock(c3, c3, c2, norm=norm)
        self.u1 = UpBlock(c2, c2, c1, norm=norm)

        self.head = nn.Sequential(
            ConvBlock(c1, c1, 3, norm=norm),
            nn.Conv2d(c1, 1, 1),
        )

    def forward(self, rf: torch.Tensor) -> torch.Tensor:
        """rf : (B, 1, D, K)  →  image : (B, 1, Nz, Nx) linear (signed)."""
        feat = self.rf_encoder(rf)                   # (B, c1, Nz, Nx)
        s1, x = self.d1(feat)                        # s1 @ c2 spatial/1, x @ c2 spatial/2
        s2, x = self.d2(x)                           # s2 @ c3 /2,        x @ c3 /4
        s3, x = self.d3(x)                           # s3 @ c4 /4,        x @ c4 /8
        x = self.bot(x)
        x = self.u3(x, s3)                           # → c3 /4
        x = self.u2(x, s2)                           # → c2 /2
        x = self.u1(x, s1)                           # → c1 /1
        return self.head(x)                          # (B, 1, Nz, Nx)
