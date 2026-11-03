"""
Envelope sparsity loss (Eq. 12, Zhang et al. 2021).

    L_SpE = ||Env(x')||_1

Encourages anechoic regions (cysts, vessels) to be truly dark.
"""

from __future__ import annotations

from torch import Tensor

from zhang_dnn.utils import hilbert_envelope


def envelope_sparsity_loss(x: Tensor) -> Tensor:
    """L1 norm of the Hilbert envelope.

    Parameters
    ----------
    x : (batch, 1, Nz, Nx) — network output x'.
    """
    env = hilbert_envelope(x, dim=-2)
    return env.abs().mean()
