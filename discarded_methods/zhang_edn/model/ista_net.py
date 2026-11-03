"""ISTA-Net deep unfolding for self-supervised PWUS reconstruction.

N unfolded ISTA iterations:

    x_{k+1} = soft_threshold(x_k - eta_k * H^T (H x_k - y), theta_k)

Stability:
- eta_k and theta_k are stored as raw parameters and exposed via softplus
  so they are always strictly positive (no negative eta sign-flips).
- eta_k is additionally upper-clamped at eta_max = 2/L where
  L = ||H^T H||_op, enforcing the ISTA contractivity bound eta * L < 2.

Init:
- eta_init = 1/L  ⇒  eta_raw_init = inv_softplus(1/L)  ≈ -log(L) (for small 1/L)
- theta_init = 1e-3  ⇒  theta_raw_init = inv_softplus(1e-3)  ≈ -6.9
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from zhang_dnn.training.stage2_inverse import _SparseMatMul


def soft_threshold(x: Tensor, theta: Tensor) -> Tensor:
    return torch.sign(x) * torch.relu(torch.abs(x) - theta)


def _inv_softplus(y: float) -> float:
    """Return x such that softplus(x) = y, for y > 0. Uses log(expm1(y)) for stability."""
    if y <= 0:
        raise ValueError(f"inv_softplus requires y > 0, got {y}")
    return math.log(math.expm1(y))


def power_iteration_L(H: Tensor, H_T: Tensor, n_iter: int = 50,
                      verbose: bool = True) -> float:
    """Estimate L = ||H^T H||_op via power iteration. Returns largest eigenvalue."""
    n_cols = H.shape[1]
    device = H.device
    x = torch.randn(n_cols, device=device, dtype=torch.float32)
    x = x / x.norm()
    L = 0.0
    with torch.no_grad():
        for i in range(n_iter):
            Hx = _SparseMatMul.apply(H, x)
            HTHx = _SparseMatMul.apply(H_T, Hx)
            L = float(HTHx.norm())
            x = HTHx / (L + 1e-30)
            if verbose and (i % 10 == 0 or i == n_iter - 1):
                print(f"    iter {i:3d}: L = {L:.4f}")
    return L


class ISTALayer(nn.Module):
    """One unfolded ISTA iteration with softplus-reparametrized eta and theta."""

    def __init__(self, eta_init: float, theta_init: float, eta_max: float) -> None:
        super().__init__()
        self.eta_raw = nn.Parameter(
            torch.tensor(_inv_softplus(eta_init), dtype=torch.float32))
        self.theta_raw = nn.Parameter(
            torch.tensor(_inv_softplus(theta_init), dtype=torch.float32))
        self._eta_max = float(eta_max)

    @property
    def eta(self) -> Tensor:
        return torch.clamp(F.softplus(self.eta_raw), max=self._eta_max)

    @property
    def theta(self) -> Tensor:
        return F.softplus(self.theta_raw)

    def forward(self, x_flat: Tensor, y_flat: Tensor,
                H: Tensor, H_T: Tensor) -> Tensor:
        Hx = _SparseMatMul.apply(H, x_flat)
        residual = Hx - y_flat
        grad = _SparseMatMul.apply(H_T, residual)
        x_step = x_flat - self.eta * grad
        return soft_threshold(x_step, self.theta)


class ISTANet(nn.Module):
    def __init__(self, n_layers: int = 10, eta_init: float | None = None,
                 theta_init: float = 1e-3,
                 eta_max: float | None = None,
                 H_for_init: Tensor | None = None,
                 H_T_for_init: Tensor | None = None) -> None:
        """Build N ISTA layers (softplus reparam + upper clamp on eta).

        Parameters
        ----------
        n_layers : number of unfolded iterations.
        eta_init : per-layer step-size init. If None, computed as 1/L where
            L = ||H^T H||_op (50 power iterations). Requires H_for_init.
        theta_init : per-layer threshold init (must be > 0).
        eta_max : hard upper clamp on eta. If None, set to 2/L.
        H_for_init, H_T_for_init : sparse torch tensors for power iteration.
            Only consulted when eta_init or eta_max is None.
        """
        super().__init__()
        if eta_init is None or eta_max is None:
            if H_for_init is None or H_T_for_init is None:
                raise ValueError(
                    "eta_init=None or eta_max=None requires H_for_init and "
                    "H_T_for_init for power-iteration estimation of L")
            print("  Estimating L = ||H^T H||_op via power iteration:")
            L = power_iteration_L(H_for_init, H_T_for_init, n_iter=50, verbose=True)
            if eta_init is None:
                eta_init = 1.0 / L
            if eta_max is None:
                eta_max = 2.0 / L
            print(f"  L = {L:.4f}  →  eta_init = 1/L = {eta_init:.6e}  "
                  f"eta_max = 2/L = {eta_max:.6e}")

        self.eta_init = float(eta_init)
        self.theta_init = float(theta_init)
        self.eta_max = float(eta_max)
        self.n_layers = n_layers
        self.layers = nn.ModuleList([
            ISTALayer(eta_init, theta_init, eta_max) for _ in range(n_layers)
        ])

    def forward(self, x_init_flat: Tensor, y_flat: Tensor,
                H: Tensor, H_T: Tensor,
                return_intermediates: bool = False):
        x = x_init_flat
        if return_intermediates:
            intermediates = [x.detach().clone()]
        for layer in self.layers:
            x = layer(x, y_flat, H, H_T)
            if return_intermediates:
                intermediates.append(x.detach().clone())
        if return_intermediates:
            return x, intermediates
        return x
