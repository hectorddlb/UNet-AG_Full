"""
Stage 1: Autoencoder pretraining (Eq. 7, Zhang et al. 2021).

The network learns to reconstruct x_hat from itself — a denoising
autoencoder that initialises weights before the harder Stage 2 loss.

Usage::

    losses = train_stage1(model, x_hats, epochs=40, lr=1e-4)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from zhang_dnn.losses.mse_loss import mse_stage1


def train_stage1(
    model: torch.nn.Module,
    x_hats: Tensor,
    epochs: int = 40,
    lr: float = 1e-4,
    batch_size: int = 1,
    checkpoint_dir: Optional[str | Path] = None,
    device: str = "cpu",
    x_targets: Optional[Tensor] = None,
) -> list[float]:
    """Run Stage 1 training.

    Parameters
    ----------
    model : EncoderDecoderNet instance.
    x_hats : (N, 1, Nz, Nx) — initial solutions (inputs).
    epochs : number of training epochs.
    lr : learning rate for Adam.
    batch_size : samples per batch.
    checkpoint_dir : if given, save model checkpoint after training.
    device : 'cpu' or 'cuda'.
    x_targets : (N, 1, Nz, Nx) — supervised targets. If None, uses x_hats
        (original autoencoder mode). If provided, trains x_hat → x_target
        (supervised mode for LSQR-guided pretraining).

    Returns
    -------
    epoch_losses : list of mean loss per epoch.
    """
    model = model.to(device)
    model.train()

    supervised = x_targets is not None
    mode_str = "supervised (x_hat→x_lsqr)" if supervised else "autoencoder"
    print(f"  Stage 1 mode: {mode_str}")

    if supervised:
        dataset = TensorDataset(x_hats, x_targets)
    else:
        dataset = TensorDataset(x_hats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch_losses: list[float] = []

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        n_batches = 0

        for batch in loader:
            if supervised:
                x_hat_batch, x_tgt_batch = batch
                x_hat_batch = x_hat_batch.to(device)
                x_tgt_batch = x_tgt_batch.to(device)
            else:
                (x_hat_batch,) = batch
                x_hat_batch = x_hat_batch.to(device)
                x_tgt_batch = x_hat_batch

            x_tilde = model(x_hat_batch)
            loss = mse_stage1(x_tgt_batch, x_tilde)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        mean_loss = running_loss / n_batches
        epoch_losses.append(mean_loss)
        print(f"  Stage 1 epoch {epoch:3d}/{epochs}  loss={mean_loss:.6f}")

    if checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        save_path = ckpt_path / "stage1_checkpoint.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epochs,
            "loss": epoch_losses[-1],
        }, save_path)
        print(f"  Stage 1 checkpoint saved: {save_path}")

    return epoch_losses
