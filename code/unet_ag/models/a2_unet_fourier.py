"""Proposal A2 — RF encoder + ResU-Net with sigmoid head (env+log target).

**Branch**: with H (α, *optional*). H_Fdf does not appear in the autograd
graph; it can only be used as an extra loss term if hard spatial fidelity is
required. The default A2 variant **does not use H**: the "fidelity" comes from
the Fourier-weighted loss in the image domain, which is the contribution
physically motivated by the Schiffner F-number.

**Declared input**: raw 1 PW RF, ``(B, 1, D, K)``, z-score + ``/max(|.|)``.

**Output**: ``(B, 1, Nz, Nx)`` with a final ``sigmoid`` → ``[0, 1]``. The
target is the **envelope+log compress (60 dB)** of Schiffner Fd-F 75 PW, as
in B1. Reason: A1 collapsed into ``predict-zero`` with a signed linear target
(run 185174, 2026-05-21); env+log destroys that basin and B1 already
validated the recipe.

**Differences vs A1**:
- A1 linear output (signed); A2 sigmoid output [0,1].
- A1 supervises against the raw signed Schiffner target; A2 against env+log [0,1].
- A1 loss = MSE+L1 + ``λ·‖y − H_Fdf x'‖²``; A2 loss = MSE+L1 + ``γ·Fourier_W``
  where W(f) = 1/F²_Schiffner(f). See
  ``unet_ag.losses.fourier_weighted.mse_l1_fourier_loss``.

**Differences vs B1**:
- B1 input = ``H_Fdf^T y`` (adjoint image, outside the graph) + ``f_map`` for
  Attention Gate conditioning. A2 input = raw RF, no AG, no spatial
  conditioning.
- B1 third term = ``1 − SSIM`` (local structure). A2 third term =
  Fourier-weighted (spectral structure with a physical F-number prior).
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


class A2UNetFourier(nn.Module):
    """RF encoder + 3-level ResU-Net; sigmoid output for env+log target."""

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

        self.rf_encoder = RFEncoder1DTime(
            n_samples=n_samples,
            n_elements=n_elements,
            nz=nz,
            nx=nx,
            out_ch=c1,
        )

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
        """rf : (B, 1, D, K)  →  sigmoid output (B, 1, Nz, Nx) in [0, 1]."""
        feat = self.rf_encoder(rf)
        s1, x = self.d1(feat)
        s2, x = self.d2(x)
        s3, x = self.d3(x)
        x = self.bot(x)
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)
        return torch.sigmoid(self.head(x))
