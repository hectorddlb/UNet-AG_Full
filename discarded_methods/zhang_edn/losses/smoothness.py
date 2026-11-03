"""
Smoothness loss in the spatial domain (Eq. 11, Zhang et al. 2021).

    L_Sm = ||D1 * Env(x')||_1 + ||D2 * Env(x')||_1

D1, D2 are finite difference operators (lateral, axial) and Env() is the
Hilbert envelope.
"""

from __future__ import annotations

from torch import Tensor

from zhang_dnn.utils import hilbert_envelope


def smoothness_loss(x: Tensor) -> Tensor:
    """Total-variation of the Hilbert envelope.

    Parameters
    ----------
    x : (batch, 1, Nz, Nx) — network output x'.
    """
    env = hilbert_envelope(x, dim=-2)  # (batch, 1, Nz, Nx)

    # Finite differences
    diff_axial = env[:, :, 1:, :] - env[:, :, :-1, :]   # D2: along Nz
    diff_lateral = env[:, :, :, 1:] - env[:, :, :, :-1]  # D1: along Nx

    return diff_axial.abs().mean() + diff_lateral.abs().mean()
