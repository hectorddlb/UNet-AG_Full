"""
Wavelet sparsity loss via Sparsity Averaging (Eq. 9-10, Zhang et al. 2021).

    L_SpW = (1/sqrt(8)) * sum_i ||psi_i(x')||_1     for i = db1..db8

The 2D DWT is implemented as separable convolutions with fixed (frozen)
Daubechies filter weights so that gradients flow through the loss.
"""

from __future__ import annotations

import math

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _build_filters(wavelet_name: str) -> tuple[Tensor, Tensor]:
    """Return (lo, hi) 1-D filter tensors for the given wavelet."""
    w = pywt.Wavelet(wavelet_name)
    lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)
    hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)
    return lo, hi


def _dwt2d_one_level(
    x: Tensor,
    lo: Tensor,
    hi: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """One level of separable 2-D DWT.

    Parameters
    ----------
    x : (B, 1, H, W)
    lo, hi : 1-D filter tensors (already reversed for convolution).

    Returns
    -------
    (LL, LH, HL, HH) — each (B, 1, H//2, W//2).
    """
    flen = lo.shape[0]
    pad = flen - 1  # full "same" padding before stride-2 downsampling

    # Row filtering -> (B, 1, H, W_padded) -> stride-2 on W
    # Build 2-D row kernels: (1, 1, 1, flen)
    lo_row = lo.view(1, 1, 1, flen).to(x.device)
    hi_row = hi.view(1, 1, 1, flen).to(x.device)

    x_pad_w = F.pad(x, (pad // 2, pad - pad // 2, 0, 0), mode="reflect")
    L = F.conv2d(x_pad_w, lo_row, stride=(1, 2))
    H = F.conv2d(x_pad_w, hi_row, stride=(1, 2))

    # Column filtering -> stride-2 on H
    lo_col = lo.view(1, 1, flen, 1).to(x.device)
    hi_col = hi.view(1, 1, flen, 1).to(x.device)

    L_pad = F.pad(L, (0, 0, pad // 2, pad - pad // 2), mode="reflect")
    H_pad = F.pad(H, (0, 0, pad // 2, pad - pad // 2), mode="reflect")

    LL = F.conv2d(L_pad, lo_col, stride=(2, 1))
    LH = F.conv2d(L_pad, hi_col, stride=(2, 1))
    HL = F.conv2d(H_pad, lo_col, stride=(2, 1))
    HH = F.conv2d(H_pad, hi_col, stride=(2, 1))

    return LL, LH, HL, HH


class WaveletSparsityLoss(nn.Module):
    """Sparsity Averaging loss with 8 Daubechies wavelets (db1-db8).

    Pre-builds the filter banks at construction time.  All filters are
    registered as buffers (not parameters) so they move with ``.to(device)``
    but are never trained.
    """

    def __init__(self, n_wavelets: int = 8, n_levels: int = 1) -> None:
        super().__init__()
        self.n_wavelets = n_wavelets
        self.n_levels = n_levels
        self.scale = 1.0 / math.sqrt(n_wavelets)

        for i in range(1, n_wavelets + 1):
            lo, hi = _build_filters(f"db{i}")
            self.register_buffer(f"lo_{i}", lo)
            self.register_buffer(f"hi_{i}", hi)

    def forward(self, x: Tensor) -> Tensor:
        """Compute wavelet sparsity loss.

        Parameters
        ----------
        x : (batch, 1, Nz, Nx) — network output x'.
        """
        total = torch.tensor(0.0, device=x.device)

        for i in range(1, self.n_wavelets + 1):
            lo = getattr(self, f"lo_{i}")
            hi = getattr(self, f"hi_{i}")

            current = x
            for _ in range(self.n_levels):
                LL, LH, HL, HH = _dwt2d_one_level(current, lo, hi)
                # L1 of detail coefficients only (LH, HL, HH)
                total = total + LH.abs().mean() + HL.abs().mean() + HH.abs().mean()
                current = LL  # for multi-level decomposition

        return self.scale * total


# Module-level convenience function
_default_loss = None


def wavelet_sparsity_loss(x: Tensor) -> Tensor:
    """Functional interface (lazy-initialises a shared module)."""
    global _default_loss
    if _default_loss is None or next(_default_loss.buffers()).device != x.device:
        _default_loss = WaveletSparsityLoss().to(x.device)
    return _default_loss(x)
