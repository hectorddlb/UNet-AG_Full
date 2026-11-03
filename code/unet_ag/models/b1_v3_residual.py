"""B1-v3 — Residual skip wrapper over B1 UNet+AG with AG-Schiffner-Fd-F.

A "safe" reformulation after 8 negative B1/A2 results. It keeps the
advisor's constraint (Conditional Attention Gates conditioned on the Schiffner
Fd-F F-number) and adds a **mathematical guarantee of initial identity**:

    pred = clamp( x_DAS  +  γ_gain · UNet_AG(x_DAS, f_map),  0, 1 )

with the UNet's last 1×1 conv **zero-initialized** (weight and bias = 0). At
epoch 0 the residual is identically zero → ``pred ≡ x_DAS``. Since
``x_DAS = env_log(H_Fdf^T y)`` already reaches ``n_lat = 484/600 @
FWHM_lat = 2.59 mm`` on its own in the Field II eval (``das_1pw`` column), the
rescue criterion ``FWHM_lat<4.9 AND n_lat≥150/600`` is **guaranteed from
initialization**. Training can only improve (refine towards the
Schiffner_75PW target) or, at worst, be held back by the ``best.pt`` selected
by val_loss at an epoch close to 0.

Why the B1 baseline failed with the same input:
    pred = sigmoid( UNet(x_DAS, f_map) )      # B1 baseline
- σ output activation with a random-init UNet → pred ≈ 0.5 per pixel at ep 0.
- The UNet learned a "smooth envelope" mapping towards env+log Schiffner_75PW
  that **lost** the lateral information of the input. FWHM_lat = 38 mm.

B1-v3 forces the model to start from x_DAS and refine it. If the lateral
geometry of the input is preserved (which happens at the start by
construction), the UNet can only degrade it actively.

Schiffner Fd-F enters at 2 points:
1. **AG conditioning**: ``f_map = compute_f_number_map_band_avg`` →
   `FNumberEmbedding` → `ConditionalAttentionGate` at each decoder level.
2. **Supervision target**: ``Schiffner_75PW`` compounded env+log.

It does NOT enter as an apodization of the input (the input is
``env_log(H^T y)`` as in the B1 baseline, already laterally resolved in eval).

γ-params: **1 scalar** (``log_gain``) → respects the project's α/γ contract.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from .b1_unet_attention_fnum import B1UNetAttentionFNum


class B1V3Residual(nn.Module):
    """B1 UNet+AG with a residual skip and zero-init of the output conv.

    Parameters
    ----------
    base_ch, embed_ch : int
        UNet capacity (same default as the B1 baseline: ``base_ch=32``).
    init_gain : float, default 1.0
        Initial value of the learnable ``gain`` that multiplies the residual.
        Kept in log-space for positivity. At the start the residual is 0
        (zero-init out conv), so the value of ``gain`` is irrelevant for the
        initial output; it starts to matter after the first steps.
    train_gain : bool, default True
        If ``True`` ``log_gain`` takes part in autograd; if ``False`` it stays
        fixed at ``init_gain``.

    Forward
    -------
    ``forward(x0, f_map) -> pred``

    - ``x0``     : (B, 1, Nz, Nx) — ``env_log(H_Fdf^T y)``, already in [0,1].
    - ``f_map``  : (B, 1, Nz, Nx) — Schiffner Fd-F band-averaged depth cap.
    - ``pred``   : (B, 1, Nz, Nx) — ``clamp(x0 + γ_gain · residual, 0, 1)``.
    """

    def __init__(
        self,
        base_ch: int = 32,
        embed_ch: int = 16,
        init_gain: float = 1.0,
        train_gain: bool = True,
    ):
        super().__init__()

        # UNet without a final sigmoid — the out conv must produce ≈ 0 at the
        # start (zero-init) without sigmoid(0)=0.5 contaminating it.
        self.unet = B1UNetAttentionFNum(
            in_ch=1, base_ch=base_ch, embed_ch=embed_ch,
            output_activation="none",
        )

        # Zero-init of the final 1×1 conv → residual ≡ 0 at ep 0.
        nn.init.zeros_(self.unet.out.weight)
        if self.unet.out.bias is not None:
            nn.init.zeros_(self.unet.out.bias)

        # γ_gain learnable en log-space (positividad + gradientes suaves).
        self.log_gain = nn.Parameter(
            torch.tensor(math.log(init_gain), dtype=torch.float32),
            requires_grad=train_gain,
        )

    @property
    def gain(self) -> Tensor:
        return torch.exp(self.log_gain)

    @property
    def gamma(self) -> dict[str, float]:
        with torch.no_grad():
            return {"gain": float(self.gain)}

    def forward(self, x0: Tensor, f_map: Tensor) -> Tensor:
        residual = self.unet(x0, f_map)                  # (B, 1, Nz, Nx), ≈0 en ep 0
        return torch.clamp(x0 + self.gain * residual, 0.0, 1.0)

    def extra_repr(self) -> str:
        with torch.no_grad():
            return f"gain={float(self.gain):.3f}"
