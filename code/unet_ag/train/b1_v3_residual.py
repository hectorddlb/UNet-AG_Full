"""Train B1-v3 — Residual skip + zero-init out conv + AG-Schiffner-Fd-F.

Minimal trainer: reuses the B1 baseline input/target/grid pipeline
(``b1_attention_gate.py``) — the change is purely architectural
(``B1V3Residual`` model with initial identity). Loss, augmentations,
optimizer and split are identical to the B1 baseline to isolate the
contribution of the residual skip.

Guaranteed initial identity:
    ep 0 → pred ≡ x_DAS = env_log(H_Fdf^T y)
    eval ep 0 → metrics identical to DAS_1PW (n_lat≥484/600, FWHM_lat≈2.59mm)
    rescue criterion ``FWHM_lat<4.9 AND n_lat≥150/600`` already met at the start.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from torch import Tensor

from ..data.augmentations import AugConfig, apply_augmentations
from ..data.manifest import field_ii_split
from ..data.rf_loader import FieldIIMultiAngleLoader, zscore_rf
from ..losses.supervised import mse_l1_ssim_loss
from ..models.b1_v3_residual import B1V3Residual
from ..physics.f_number_map import compute_f_number_map_band_avg
from .b1_attention_gate import (
    _adjoint_image,
    _build_grid,
    _envelope_log_compress,
    _load_target,
)
from .base_trainer import parse_common_args, run_training


def _iter_split(
    loader: FieldIIMultiAngleLoader,
    indices: list[int],
    target_root: Path,
    grid_z: Tensor,
    grid_x: Tensor,
    device: str,
    aug_cfg: Optional[AugConfig],
    seed_base: int,
) -> Iterator[tuple[dict, Tensor]]:
    Nz, Nx = grid_z.shape[0], grid_x.shape[0]
    grid_z_np = grid_z.cpu().numpy()
    grid_x_np = grid_x.cpu().numpy()
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
        x0 = _envelope_log_compress(x0_raw)                        # (Nz, Nx)
        target = _envelope_log_compress(target_raw)                # (Nz, Nx)

        f_map = compute_f_number_map_band_avg(grid_z_np, grid_x_np, sample.geometry)
        f_map_t = torch.from_numpy(f_map).to(device)

        inputs = {
            "x0":    x0.to(device).unsqueeze(0).unsqueeze(0),       # (1, 1, Nz, Nx)
            "f_map": f_map_t.unsqueeze(0).unsqueeze(0),             # (1, 1, Nz, Nx)
        }
        yield inputs, target.to(device).unsqueeze(0).unsqueeze(0)   # (1, 1, Nz, Nx)


def _forward_and_loss(model, inputs, target, args):
    pred = model(inputs["x0"], inputs["f_map"])
    loss, br = mse_l1_ssim_loss(
        pred, target,
        mse_weight=args.mse_weight,
        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        data_range=1.0,
    )
    g = model.gamma if hasattr(model, "gamma") else {}
    for k, v in g.items():
        br[f"gamma_{k}"] = v
    return loss, br, pred


def main() -> None:
    parser = argparse.ArgumentParser(description="B1-v3 — Residual skip + AG-Fd-F")
    parser.add_argument("--target_root", type=Path, required=True)
    parser.add_argument("--phantom_type", choices=["points", "cysts", "both"], default="points")
    parser.add_argument("--nz", type=int, default=612)
    parser.add_argument("--nx", type=int, default=388)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--embed_ch", type=int, default=16)
    parser.add_argument("--mse_weight", type=float, default=0.7)
    parser.add_argument("--l1_weight", type=float, default=0.2)
    parser.add_argument("--ssim_weight", type=float, default=0.1)
    parser.add_argument("--init_gain", type=float, default=1.0)
    parser.add_argument("--freeze_gain", action="store_true")
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

    model = B1V3Residual(
        base_ch=args.base_ch, embed_ch=args.embed_ch,
        init_gain=args.init_gain, train_gain=not args.freeze_gain,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_unet = sum(p.numel() for p in model.unet.parameters() if p.requires_grad)
    print(f"B1-v3 (residual + AG-Fd-F base_ch={args.base_ch}) params: {n_params:,} "
          f"(unet={n_unet:,}, gain γ=1)")
    print(f"  init γ: gain={args.init_gain}  freeze={args.freeze_gain}")
    print(f"  identity init verified: out_conv zero-init → residual ≡ 0 → pred₀ ≡ x_DAS")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    aug_cfg = None if args.no_aug else AugConfig()
    def train_iter():
        return _iter_split(loader, train_indices, args.target_root, grid_z, grid_x, args.device, aug_cfg, seed_base=args.seed * 1000)
    def val_iter():
        return _iter_split(loader, val_indices, args.target_root, grid_z, grid_x, args.device, None, seed_base=0)

    run_training(args, model, optimizer, train_iter, val_iter if val_indices else None, _forward_and_loss)


if __name__ == "__main__":
    main()
