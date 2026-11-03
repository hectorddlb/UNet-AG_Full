"""Train proposal A2 — RF → image + envelope+log target + Fourier-weighted loss.

Input pipeline (per phantom):
  1. ``rf_1pw`` (D, K) ← FieldIIMultiAngleLoader (z-scored at load).
  2. Pad/truncate to ``--rf_n_samples`` (default 3500); per-phantom
     ``/max(|.|)`` on top of z-score so RF lives in [-1, 1] numerically.
  3. Schiffner Fd-F 75 PW target → **Hilbert envelope + log compress 60 dB**
     → ``[0, 1]`` (same recipe as B1 F1 fix).

Loss (A2 contract — see ``losses/fourier_weighted.py``):
  L = α·MSE(pred, target) + β·L1(pred, target) + γ·Fourier_W(pred, target)
  with weights default 0.7 / 0.2 / 0.1 (matching B1's MSE/L1/SSIM split).

Fourier weight ``W(f) = 1 / F²_Schiffner(f)`` is built once per phantom from
``compute_spectral_weight`` over the axial-rfft bins, mapped via
``f = fc + k_z·c/2`` (env+log image lives in the modulation domain).

Why A2 over A1 (post-2026-05-21):
  A1 collapsed (run 185174) on signed linear target — predict-zero is the
  MSE minimum on sparse Field II points even with l1_w=0.5, fid_w=5.0. A2
  reuses B1's env+log target recipe (breaks the zero attractor) and replaces
  the spatial H_Fdf fidelity with a *spectral* fidelity weighted by the
  physical F-number.

CLI smoke:
  python -m unet_ag.train.a2_fourier_loss \\
      --data_root datasets/Field\\ II/field_ii_data_multiangle \\
      --target_root results_schiffner_targets_fieldii \\
      --output_dir results_v2/a2 \\
      --max_phantoms_train 5 --epochs 2 --smoke_steps 6 --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from scipy.signal import hilbert

from ..data.augmentations import AugConfig, apply_augmentations
from ..data.manifest import field_ii_split
from ..data.rf_loader import FieldIIMultiAngleLoader, zscore_rf
from ..losses.fourier_weighted import (
    build_axial_spectral_weight,
    build_2d_spectral_weight,
    mse_l1_fourier_loss,
)
from ..models.a2_unet_fourier import A2UNetFourier
from .base_trainer import parse_common_args, run_training


DYN_RANGE_DB = 60.0


def _envelope_log_compress(img: Tensor, dyn_range_db: float = DYN_RANGE_DB) -> Tensor:
    """Hilbert envelope along axial → log compress to [0, 1] over ``dyn_range_db``.

    Same routine used by B1 (F1 fix). Eliminates the predict-zero attractor on
    sparse Field II targets.
    """
    arr = img.detach().cpu().numpy()
    env = np.abs(hilbert(arr, axis=0))
    env_max = float(env.max()) + 1e-12
    db = 20.0 * np.log10(env / env_max + 1e-30)
    db = np.clip(db, -dyn_range_db, 0.0)
    norm = (db + dyn_range_db) / dyn_range_db
    return torch.from_numpy(norm.astype(np.float32))


def _build_grid(geometry: dict, nz: int, nx: int) -> tuple[Tensor, Tensor]:
    c = float(geometry["speed_of_sound"])
    fs = float(geometry["sampling_freq_hz"])
    ns = int(geometry["n_samples"])
    K = int(geometry["n_elements"])
    pitch = float(geometry["pitch_m"])
    z_max = (ns - 1) * c / (2.0 * fs)
    grid_z = torch.linspace(5e-3, z_max, nz, dtype=torch.float32)
    grid_x = torch.linspace(-K * pitch / 2.0, +K * pitch / 2.0, nx, dtype=torch.float32)
    return grid_z, grid_x


def _load_target(target_root: Path, phantom_id: str, ptype: str, nz: int, nx: int) -> Optional[Tensor]:
    candidates = [
        target_root / ptype / phantom_id / "compounded.npy",
        target_root / phantom_id / "compounded.npy",
    ]
    for p in candidates:
        if p.exists():
            arr = np.load(p).astype(np.float32)
            t = torch.from_numpy(arr)
            if t.shape != (nz, nx):
                t = F.interpolate(
                    t.unsqueeze(0).unsqueeze(0), size=(nz, nx), mode="bilinear", align_corners=False
                )[0, 0]
            return t
    return None


def _pad_or_truncate_rf(rf: Tensor, n_samples: int) -> Tensor:
    D = rf.shape[0]
    if D == n_samples:
        return rf
    if D > n_samples:
        return rf[:n_samples].contiguous()
    pad = torch.zeros(n_samples - D, rf.shape[1], dtype=rf.dtype)
    return torch.cat([rf, pad], dim=0).contiguous()


def _iter_split(
    loader: FieldIIMultiAngleLoader,
    indices: list[int],
    target_root: Path,
    grid_z: Tensor,
    grid_x: Tensor,
    device: str,
    rf_n_samples: int,
    aug_cfg: Optional[AugConfig],
    seed_base: int,
    fourier_mode: str,
    fc_offset: bool,
    fourier_dim: int,
    lateral_mode: str,
    lateral_floor: float,
) -> Iterator[tuple[dict, Tensor]]:
    Nz, Nx = grid_z.shape[0], grid_x.shape[0]
    dz = float((grid_z[-1] - grid_z[0]).item() / max(Nz - 1, 1))
    for step, idx in enumerate(indices):
        sample = loader[idx]
        if aug_cfg is not None:
            sample = apply_augmentations(sample, aug_cfg, seed=seed_base + step)
            rf, _ = zscore_rf(sample.rf_1pw)
            sample.rf_1pw = rf

        ptype = "points" if sample.phantom_id.startswith("points") else "cysts"
        target_raw = _load_target(target_root, sample.phantom_id, ptype, Nz, Nx)
        if target_raw is None:
            continue

        # Env+log target (B1 F1 recipe). Linear signed target was abandoned
        # after A1 v3 collapse (run 185174, 2026-05-21).
        target_envlog = _envelope_log_compress(target_raw)

        rf_padded = _pad_or_truncate_rf(sample.rf_1pw, rf_n_samples)
        rf_scale = float(rf_padded.abs().max()) + 1e-12
        rf_norm = rf_padded / rf_scale

        # F-number Fourier weight (mean-1 normalised). Geometry-dependent;
        # recompute per-phantom is fine — sample.geometry is a small dict.
        if fourier_dim == 2:
            weight = build_2d_spectral_weight(
                Nz, dz, Nx, sample.geometry,
                axial_mode=fourier_mode,
                lateral_mode=lateral_mode,
                fc_offset=fc_offset,
                lateral_floor=lateral_floor,
            ).to(device)                                           # (Nz, Nx//2+1)
        else:
            weight = build_axial_spectral_weight(
                Nz, dz, sample.geometry, mode=fourier_mode, fc_offset=fc_offset,
            ).to(device)                                           # (Nz//2 + 1,)

        inputs = {
            "rf": rf_norm.to(device).unsqueeze(0).unsqueeze(0),    # (1, 1, D, K)
            "fourier_weight": weight,
            "fourier_dim": fourier_dim,
        }
        yield inputs, target_envlog.to(device).unsqueeze(0).unsqueeze(0)


def _forward_and_loss(model, inputs, target, args):
    pred = model(inputs["rf"])                                    # (1, 1, Nz, Nx) ∈ [0,1]
    loss, br = mse_l1_fourier_loss(
        pred, target,
        weight=inputs["fourier_weight"],
        mse_weight=args.mse_weight,
        l1_weight=args.l1_weight,
        fourier_weight=args.fourier_weight,
        fourier_dim=inputs["fourier_dim"],
    )
    return loss, br, pred


def main() -> None:
    parser = argparse.ArgumentParser(description="A2 — RF→image + Fourier-weighted loss")
    parser.add_argument("--target_root", type=Path, required=True)
    parser.add_argument("--phantom_type", choices=["points", "cysts", "both"], default="points")
    parser.add_argument("--nz", type=int, default=256)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--rf_n_samples", type=int, default=3500)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--mse_weight", type=float, default=0.7)
    parser.add_argument("--l1_weight", type=float, default=0.2)
    parser.add_argument("--fourier_weight", type=float, default=0.1)
    parser.add_argument("--fourier_mode", choices=["flat", "inv_F", "inv_F_squared"],
                        default="inv_F_squared",
                        help="Spectral weight profile. 'flat' = Parseval (sanity baseline).")
    parser.add_argument("--no_fc_offset", action="store_true",
                        help="Disable fc-baseband offset; use f = k_z·c/2 directly (RF-domain mode).")
    parser.add_argument("--fourier_dim", type=int, choices=[1, 2], default=1,
                        help="1 = axial-only rfft loss (A2 default); 2 = rfft2 with "
                             "separable axial 1/F² × lateral high-pass (A2-v2).")
    parser.add_argument("--lateral_mode", choices=["flat", "highpass_linear", "highpass_quad"],
                        default="highpass_linear",
                        help="Shape of the lateral spectral weight when fourier_dim=2.")
    parser.add_argument("--lateral_floor", type=float, default=0.1,
                        help="Floor of the lateral high-pass; keeps low-freq lateral "
                             "content non-trivially weighted (preserves cyst contrast).")
    parser.add_argument("--no_aug", action="store_true")
    args = parse_common_args(parser)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    loader = FieldIIMultiAngleLoader(args.data_root, phantom_type=args.phantom_type, keep_all_angles=False)
    splits = field_ii_split(args.data_root, phantom_types=[args.phantom_type] if args.phantom_type != "both" else None)
    if args.phantom_type == "both":
        train_files = splits["points"].train + splits["cysts"].train
        val_files = splits["points"].val + splits["cysts"].val
    else:
        train_files = splits[args.phantom_type].train
        val_files = splits[args.phantom_type].val

    name_to_idx = {p.stem: i for i, p in enumerate(loader.file_list)}
    train_indices = [name_to_idx[p.stem] for p in train_files if p.stem in name_to_idx]
    val_indices = [name_to_idx[p.stem] for p in val_files if p.stem in name_to_idx]
    if args.max_phantoms_train:
        train_indices = train_indices[: args.max_phantoms_train]
    if args.max_phantoms_val:
        val_indices = val_indices[: args.max_phantoms_val]

    sample0 = loader[train_indices[0]] if train_indices else loader[0]
    grid_z, grid_x = _build_grid(sample0.geometry, args.nz, args.nx)
    geom0 = sample0.geometry

    model = A2UNetFourier(
        n_samples=args.rf_n_samples,
        n_elements=int(geom0["n_elements"]),
        nz=args.nz,
        nx=args.nx,
        base_ch=args.base_ch,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"A2 params: {n_params:,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    fc_offset = not args.no_fc_offset
    aug_cfg = None if args.no_aug else AugConfig()

    def train_iter():
        return _iter_split(loader, train_indices, args.target_root, grid_z, grid_x,
                           args.device, args.rf_n_samples, aug_cfg, seed_base=args.seed * 1000,
                           fourier_mode=args.fourier_mode, fc_offset=fc_offset,
                           fourier_dim=args.fourier_dim, lateral_mode=args.lateral_mode,
                           lateral_floor=args.lateral_floor)

    def val_iter():
        return _iter_split(loader, val_indices, args.target_root, grid_z, grid_x,
                           args.device, args.rf_n_samples, None, seed_base=0,
                           fourier_mode=args.fourier_mode, fc_offset=fc_offset,
                           fourier_dim=args.fourier_dim, lateral_mode=args.lateral_mode,
                           lateral_floor=args.lateral_floor)

    run_training(args, model, optimizer, train_iter,
                 val_iter if val_indices else None, _forward_and_loss)


if __name__ == "__main__":
    main()
