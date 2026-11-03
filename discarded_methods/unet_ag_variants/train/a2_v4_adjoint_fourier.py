"""Train proposal A2-v4 — adjoint-image input + 2-D Fourier-weighted loss.

Motivation. A2 / A2-v2 / A2-v3b all produced **0/20 wires resolved laterally**
on Field II points. Inspection of ``RFEncoder1DTime`` showed the RF→image
encoder convolves only along the *time* axis, then ``F.interpolate``s from
``K=128`` elements onto ``Nx=128`` lateral pixels — treating element index as
pixel index. This is geometrically wrong (DAS requires per-pixel delays across
elements), and no spectral-loss tuning could rescue lateral resolution.

A2-v4 isolates this by replacing the RF encoder with B1's already-beamformed
adjoint input ``x0 = envelope_log(H_Fdf^T · rf)`` while keeping the 2-D
Fourier-weighted loss (the A2 contract). The result: same backbone & input as
B1, only the third loss term differs (Fourier-W vs SSIM). Reads as a clean
ablation of "physical-spectral fidelity" vs "local-structural fidelity"
holding everything else fixed.

CLI smoke:
  python -m unet_ag.train.a2_v4_adjoint_fourier \\
      --data_root datasets/Field\\ II/field_ii_data_multiangle \\
      --target_root results_schiffner_targets_fieldii \\
      --output_dir results_v2/a2_v4 \\
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
    build_2d_spectral_weight,
    mse_l1_fourier_loss,
)
from ..models.b1_unet_attention_fnum import B1UNetAttentionFNum
from ..physics.f_number_map import compute_f_number_map_band_avg
from ..physics.h_fdf_wrapper import make_op_for_angle
from .base_trainer import parse_common_args, run_training


DYN_RANGE_DB = 60.0


def _envelope_log_compress(img: Tensor, dyn_range_db: float = DYN_RANGE_DB) -> Tensor:
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


def _adjoint_image(rf_1pw: Tensor, geometry: dict, grid_z: Tensor, grid_x: Tensor, device: str) -> Tensor:
    """``H_Fdf^T y`` for the 1 PW angle (α contract: no autograd through H)."""
    with torch.no_grad():
        op = make_op_for_angle(0.0, geometry, grid_z, grid_x, device=device, nf_chunk=None)
        y = rf_1pw.to(device).to(op.float_dtype)
        x0 = op.adjoint(y)
        if x0.is_complex():
            x0 = x0.real
    return x0.float().detach()


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


def _iter_split(
    loader: FieldIIMultiAngleLoader,
    indices: list[int],
    target_root: Path,
    grid_z: Tensor,
    grid_x: Tensor,
    device: str,
    aug_cfg: Optional[AugConfig],
    seed_base: int,
    fourier_mode: str,
    fc_offset: bool,
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

        x0_raw = _adjoint_image(sample.rf_1pw, sample.geometry, grid_z, grid_x, device)
        x0 = _envelope_log_compress(x0_raw)
        target_envlog = _envelope_log_compress(target_raw)

        f_map = compute_f_number_map_band_avg(
            grid_z.cpu().numpy(), grid_x.cpu().numpy(), sample.geometry
        )
        f_map_t = torch.from_numpy(f_map).to(device)

        weight = build_2d_spectral_weight(
            Nz, dz, Nx, sample.geometry,
            axial_mode=fourier_mode,
            lateral_mode=lateral_mode,
            fc_offset=fc_offset,
            lateral_floor=lateral_floor,
        ).to(device)                                              # (Nz, Nx//2+1)

        inputs = {
            "x0": x0.to(device).unsqueeze(0).unsqueeze(0),        # (1, 1, Nz, Nx)
            "f_map": f_map_t.unsqueeze(0).unsqueeze(0),           # (1, 1, Nz, Nx)
            "fourier_weight": weight,
        }
        yield inputs, target_envlog.to(device).unsqueeze(0).unsqueeze(0)


def _forward_and_loss(model, inputs, target, args):
    pred = model(inputs["x0"], inputs["f_map"])                   # (1, 1, Nz, Nx) ∈ [0,1]
    loss, br = mse_l1_fourier_loss(
        pred, target,
        weight=inputs["fourier_weight"],
        mse_weight=args.mse_weight,
        l1_weight=args.l1_weight,
        fourier_weight=args.fourier_weight,
        fourier_dim=2,
    )
    return loss, br, pred


def main() -> None:
    parser = argparse.ArgumentParser(description="A2-v4 — adjoint input + 2-D Fourier-weighted loss")
    parser.add_argument("--target_root", type=Path, required=True)
    parser.add_argument("--phantom_type", choices=["points", "cysts", "both"], default="points")
    parser.add_argument("--nz", type=int, default=256)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--embed_ch", type=int, default=16)
    parser.add_argument("--mse_weight", type=float, default=0.7)
    parser.add_argument("--l1_weight", type=float, default=0.2)
    parser.add_argument("--fourier_weight", type=float, default=0.15)
    parser.add_argument("--fourier_mode", choices=["flat", "inv_F", "inv_F_squared"],
                        default="inv_F_squared")
    parser.add_argument("--no_fc_offset", action="store_true")
    parser.add_argument("--lateral_mode", choices=["flat", "highpass_linear", "highpass_quad"],
                        default="highpass_linear")
    parser.add_argument("--lateral_floor", type=float, default=0.1)
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

    model = B1UNetAttentionFNum(in_ch=1, base_ch=args.base_ch, embed_ch=args.embed_ch).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"A2-v4 (B1 backbone + Fourier-W loss) params: {n_params:,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    fc_offset = not args.no_fc_offset
    aug_cfg = None if args.no_aug else AugConfig()

    def train_iter():
        return _iter_split(loader, train_indices, args.target_root, grid_z, grid_x,
                           args.device, aug_cfg, seed_base=args.seed * 1000,
                           fourier_mode=args.fourier_mode, fc_offset=fc_offset,
                           lateral_mode=args.lateral_mode, lateral_floor=args.lateral_floor)

    def val_iter():
        return _iter_split(loader, val_indices, args.target_root, grid_z, grid_x,
                           args.device, None, seed_base=0,
                           fourier_mode=args.fourier_mode, fc_offset=fc_offset,
                           lateral_mode=args.lateral_mode, lateral_floor=args.lateral_floor)

    run_training(args, model, optimizer, train_iter,
                 val_iter if val_indices else None, _forward_and_loss)


if __name__ == "__main__":
    main()
