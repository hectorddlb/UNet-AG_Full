"""
Stage 2: Self-supervised inverse-problem training (Eq. 8+13, Zhang et al. 2021).

Loads Stage 1 weights, then trains with the complete loss that includes
the forward projection  y' = H @ x'  to compare against measured RF data.

The sparse matrix-vector multiply is wrapped in a custom autograd Function
so that gradients flow back through x'.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from zhang_dnn.losses.wavelet_sparsity import WaveletSparsityLoss
from zhang_dnn.losses.combined_loss import combined_loss


# ───────────────────────────────────────────────────────────────────
# Differentiable sparse matrix multiply
# ───────────────────────────────────────────────────────────────────

def scipy_to_torch_sparse(H_scipy: sp.spmatrix) -> torch.Tensor:
    """Convert a scipy sparse matrix to a torch sparse COO tensor (float32)."""
    H_coo = sp.coo_matrix(H_scipy, dtype=np.float32)
    indices = torch.from_numpy(
        np.vstack([H_coo.row, H_coo.col]).astype(np.int64)
    )
    values = torch.from_numpy(H_coo.data)
    return torch.sparse_coo_tensor(indices, values, H_coo.shape).coalesce()


class _SparseMatMul(torch.autograd.Function):
    """y = H @ x  with gradient  dx = H^T @ dy."""

    @staticmethod
    def forward(ctx, H_sparse: Tensor, x_flat: Tensor) -> Tensor:
        ctx.save_for_backward(H_sparse)
        # x_flat: (l,)  ->  result: (r,)
        return torch.sparse.mm(H_sparse, x_flat.unsqueeze(1)).squeeze(1)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (H_sparse,) = ctx.saved_tensors
        grad_x = torch.sparse.mm(H_sparse.t(), grad_output.unsqueeze(1)).squeeze(1)
        return None, grad_x


def sparse_forward_project(H_torch: Tensor, x_image: Tensor, n_z: int, n_x: int) -> Tensor:
    """Compute y' = H @ x' for a single sample.

    Parameters
    ----------
    H_torch : sparse (D*K, Nz*Nx) tensor.
    x_image : (1, Nz, Nx) — single image with channel dim.
    n_z, n_x : spatial dimensions.

    Returns
    -------
    y_pred : (D*K,) tensor with gradients.
    """
    x_flat = x_image.view(-1)  # (Nz*Nx,)
    return _SparseMatMul.apply(H_torch, x_flat)


# ───────────────────────────────────────────────────────────────────
# Stage 2 training loop
# ───────────────────────────────────────────────────────────────────

def train_stage2(
    model: nn.Module,
    x_hats: Tensor,
    y_measured_list: list[Tensor],
    H_torch: Tensor,
    n_z: int,
    n_x: int,
    epochs: int = 160,
    lr: float = 1e-4,
    lambda1: float = 0.2,
    lambda2: float = 0.2,
    lambda3: float = 0.1,
    checkpoint_dir: Optional[str | Path] = None,
    device: str = "cpu",
) -> list[dict[str, float]]:
    """Run Stage 2 self-supervised training.

    Parameters
    ----------
    model : EncoderDecoderNet (should have Stage 1 weights loaded).
    x_hats : (N, 1, Nz, Nx) — initial solutions.
    y_measured_list : list of N tensors, each (D*K,) — measured RF vectors.
    H_torch : sparse torch tensor (D*K, Nz*Nx).
    n_z, n_x : image spatial dimensions.
    epochs : training epochs.
    lr : learning rate for Adam.
    lambda1..3 : regularisation weights (Eq. 13).
    checkpoint_dir : save checkpoint after training if given.
    device : 'cpu' or 'cuda'.

    Returns
    -------
    epoch_logs : list of dicts with loss breakdown per epoch.
    """
    model = model.to(device)
    model.train()

    H_dev = H_torch.to(device)
    wav_loss_fn = WaveletSparsityLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    N = x_hats.shape[0]
    epoch_logs: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        running = {"mse": 0.0, "wavelet": 0.0, "smoothness": 0.0,
                   "envelope": 0.0, "total": 0.0}

        # Stage 2 uses batch_size=1 because H @ x' is per-sample
        indices = torch.randperm(N)

        for idx in indices:
            x_hat_i = x_hats[idx : idx + 1].to(device)   # (1, 1, Nz, Nx)
            y_meas_i = y_measured_list[idx].to(device)     # (D*K,)

            # Forward: x_hat -> EDN -> x' -> H @ x' = y'
            x_prime = model(x_hat_i)                       # (1, 1, Nz, Nx)
            y_pred = sparse_forward_project(H_dev, x_prime[0], n_z, n_x)

            loss, breakdown = combined_loss(
                y_meas_i.unsqueeze(0),
                y_pred.unsqueeze(0),
                x_prime,
                wav_loss_fn,
                lambda1, lambda2, lambda3,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k in running:
                running[k] += breakdown[k]

        # Average over samples
        for k in running:
            running[k] /= N
        epoch_logs.append(running)

        print(
            f"  Stage 2 epoch {epoch:3d}/{epochs}  "
            f"total={running['total']:.6f}  "
            f"mse={running['mse']:.4f}  wav={running['wavelet']:.4f}  "
            f"sm={running['smoothness']:.4f}  env={running['envelope']:.4f}"
        )

    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        save_path = ckpt_path / "stage2_checkpoint.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epochs,
            "loss": epoch_logs[-1]["total"],
        }, save_path)
        print(f"  Stage 2 checkpoint saved: {save_path}")

    return epoch_logs
