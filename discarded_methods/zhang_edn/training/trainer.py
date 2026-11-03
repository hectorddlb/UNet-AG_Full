"""
End-to-end trainer: data preparation, two-stage training, evaluation.

Usage::

    from zhang_dnn.training.trainer import run_pipeline
    results = run_pipeline(data_root="datasets/Field II/field_ii_data", ...)
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from zhang_dnn.config import GridConfig, TrainingConfig
from zhang_dnn.data import FieldIIDataset
from zhang_dnn.measurement_matrix import build_measurement_matrix
from zhang_dnn.initialization import compute_initial_solution
from zhang_dnn.model import EncoderDecoderNet
from zhang_dnn.training.stage1_autoencoder import train_stage1
from zhang_dnn.training.stage2_inverse import train_stage2, scipy_to_torch_sparse


@dataclass
class PipelineResult:
    """Container for pipeline outputs."""

    stage1_losses: list[float] = field(default_factory=list)
    stage2_logs: list[dict[str, float]] = field(default_factory=list)
    model: Optional[EncoderDecoderNet] = None
    x_hats: Optional[Tensor] = None
    x_outputs: Optional[Tensor] = None
    grid_z: Optional[np.ndarray] = None
    grid_x: Optional[np.ndarray] = None
    elapsed_s: float = 0.0


def run_pipeline(
    data_root: str | Path,
    grid: Optional[GridConfig] = None,
    training: Optional[TrainingConfig] = None,
    max_samples: int = 5,
    phantom_types: Optional[list[str]] = None,
    output_dir: Optional[str | Path] = None,
    device: str = "cpu",
) -> PipelineResult:
    """Run the complete two-stage training pipeline.

    Parameters
    ----------
    data_root : path to Field II data (with points/ and/or cysts/ subdirs).
    grid : reconstruction grid config (default: MiniConfig grid).
    training : training hyperparameters (default: MiniConfig training).
    max_samples : limit number of phantoms loaded.
    phantom_types : which phantom types to load.
    output_dir : save checkpoints and loss log here.
    device : 'cpu' or 'cuda'.
    """
    t0 = time.perf_counter()

    if grid is None:
        grid = GridConfig(n_z=64, n_x=32, z_range_m=(5e-3, 45e-3), x_range_m=(-20e-3, 20e-3))
    if training is None:
        training = TrainingConfig(batch_size=1, stage1_epochs=5, stage2_epochs=10)
    if phantom_types is None:
        phantom_types = ["points", "cysts"]

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────
    print("=" * 60)
    print("PHASE 1: Data loading")
    print("=" * 60)

    ds = FieldIIDataset(data_root, phantom_types=phantom_types, max_samples=max_samples)
    N = len(ds)
    print(f"  Loaded {N} samples from {data_root}")

    if N == 0:
        raise RuntimeError(f"No data found in {data_root}")

    # Collect RF data and find max D for padding
    all_samples = [ds[i] for i in range(N)]
    all_rf = [s["rf_data"] for s in all_samples]
    K = all_rf[0].shape[0]
    D_max = max(rf.shape[1] for rf in all_rf)

    # Pad to common D
    for i in range(len(all_rf)):
        D = all_rf[i].shape[1]
        if D < D_max:
            all_rf[i] = torch.cat([all_rf[i], torch.zeros(K, D_max - D)], dim=1)

    print(f"  K={K}, D_max={D_max}, grid={grid.n_z}x{grid.n_x}")

    # ── 2. Build measurement matrix ──────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: Measurement matrix H")
    print("=" * 60)

    grid_z = np.linspace(grid.z_range_m[0], grid.z_range_m[1], grid.n_z)
    grid_x = np.linspace(grid.x_range_m[0], grid.x_range_m[1], grid.n_x)

    s0 = all_samples[0]
    H = build_measurement_matrix(
        grid_z=grid_z, grid_x=grid_x,
        n_elements=K, pitch_m=s0["pitch_m"],
        sampling_freq_hz=s0["fs"], speed_of_sound=s0["c"],
        n_samples=D_max, angle_rad=0.0,
    )
    print(f"  H shape: {H.shape}, nnz: {H.nnz}")

    # ── 3. Compute initial solutions ─────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3: Initial solutions x_hat = H^T @ y")
    print("=" * 60)

    x_hats = []
    y_measured_list = []
    for i in range(N):
        x_hat = compute_initial_solution(H, all_rf[i], n_z=grid.n_z, n_x=grid.n_x)
        x_hats.append(x_hat.unsqueeze(0))  # (1, Nz, Nx)
        y_vec = all_rf[i].numpy().astype(np.float64).T.ravel()
        y_measured_list.append(torch.from_numpy(y_vec.astype(np.float32)))

    x_hats = torch.stack(x_hats)  # (N, 1, Nz, Nx)
    print(f"  x_hats shape: {x_hats.shape}")

    # ── 4. Stage 1 ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"PHASE 4: Stage 1 — Autoencoder ({training.stage1_epochs} epochs)")
    print("=" * 60)

    model = EncoderDecoderNet()
    s1_losses = train_stage1(
        model, x_hats,
        epochs=training.stage1_epochs,
        lr=training.learning_rate,
        batch_size=training.batch_size,
        checkpoint_dir=output_dir,
        device=device,
    )

    # ── 5. Stage 2 ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"PHASE 5: Stage 2 — Inverse Problem ({training.stage2_epochs} epochs)")
    print("=" * 60)

    H_torch = scipy_to_torch_sparse(H)
    s2_logs = train_stage2(
        model, x_hats, y_measured_list, H_torch,
        n_z=grid.n_z, n_x=grid.n_x,
        epochs=training.stage2_epochs,
        lr=training.learning_rate,
        lambda1=training.lambda_wavelet,
        lambda2=training.lambda_smoothness,
        lambda3=training.lambda_envelope,
        checkpoint_dir=output_dir,
        device=device,
    )

    # ── 6. Generate outputs ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 6: Generating output images")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        x_outputs = model(x_hats.to(device)).cpu()
    print(f"  x_outputs shape: {x_outputs.shape}")

    # ── 7. Save loss log ──────────────────────────────────────────
    if output_dir is not None:
        _save_loss_log(output_dir, s1_losses, s2_logs)

    elapsed = time.perf_counter() - t0
    print(f"\nPipeline complete in {elapsed:.1f}s")

    return PipelineResult(
        stage1_losses=s1_losses,
        stage2_logs=s2_logs,
        model=model,
        x_hats=x_hats,
        x_outputs=x_outputs,
        grid_z=grid_z,
        grid_x=grid_x,
        elapsed_s=elapsed,
    )


def _save_loss_log(
    output_dir: Path,
    s1_losses: list[float],
    s2_logs: list[dict[str, float]],
) -> None:
    """Save training losses to a CSV file."""
    csv_path = output_dir / "training_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "epoch", "total", "mse", "wavelet", "smoothness", "envelope"])

        for i, loss in enumerate(s1_losses, 1):
            writer.writerow([1, i, f"{loss:.6f}", f"{loss:.6f}", "", "", ""])

        for i, log in enumerate(s2_logs, 1):
            writer.writerow([
                2, i,
                f"{log['total']:.6f}",
                f"{log['mse']:.6f}",
                f"{log['wavelet']:.6f}",
                f"{log['smoothness']:.6f}",
                f"{log['envelope']:.6f}",
            ])

    print(f"  Loss log saved: {csv_path}")
