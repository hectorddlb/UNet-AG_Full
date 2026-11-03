"""Train proposal B1-v2 — Soft-DAS frontend + UNet+AG refine.

Last aggressive B1 attempt before closing the B1/A2 family. The structural
change vs B1 baseline (and the entire A2 family) is the input: instead of
``x0 = env_log(H_Fdf^T y)`` (envelope + log compress, phase destroyed) we
feed the UNet a *signed* soft-DAS image with learnable F-number aperture.
Phase is preserved → lateral position information that envelopes smooth out
is now available to the network.

Input pipeline (per phantom):
  1. ``rf_1pw`` (D, K) ← FieldIIMultiAngleLoader, z-score normalised.
  2. ``f_map`` (Nz, Nx) ← compute_f_number_map_band_avg.
  3. Soft-DAS front-end (inside the model) → I_signed (Nz, Nx).
  4. UNet+AG with F# conditioning → pred ∈ [0, 1] (env+log space).

Target: same as B1 — Schiffner Fd-F 75 PW, env+log compressed to [0, 1].
Loss:   MSE + 0.2·L1 + 0.1·SSIM (B1 baseline).
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
from ..models.b1_v2_softdas import B1V2SoftDAS
from ..physics.f_number_map import compute_f_number_map_band_avg
from .b1_attention_gate import _build_grid, _envelope_log_compress, _load_target
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
        target = _envelope_log_compress(target_raw)

        f_map = compute_f_number_map_band_avg(grid_z_np, grid_x_np, sample.geometry)
        f_map_t = torch.from_numpy(f_map).to(device)

        rf = sample.rf_1pw.to(device).float()                              # (D, K)
        inputs = {
            "rf": rf,                                                      # (D, K)
            "f_map": f_map_t.unsqueeze(0).unsqueeze(0),                    # (1, 1, Nz, Nx)
        }
        yield inputs, target.to(device).unsqueeze(0).unsqueeze(0)          # (1, 1, Nz, Nx)


def _forward_and_loss(model, inputs, target, args):
    pred = model(inputs["rf"], inputs["f_map"])
    loss, br = mse_l1_ssim_loss(
        pred, target,
        mse_weight=args.mse_weight,
        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        data_range=1.0,
    )
    # Add γ-params to the report so they show up in the train_log.
    g = model.gamma if hasattr(model, "gamma") else {}
    for k, v in g.items():
        br[f"gamma_{k}"] = v
    return loss, br, pred


def main() -> None:
    parser = argparse.ArgumentParser(description="B1-v2 — Soft-DAS frontend + UNet+AG")
    parser.add_argument("--target_root", type=Path, required=True)
    parser.add_argument("--phantom_type", choices=["points", "cysts", "both"], default="points")
    parser.add_argument("--nz", type=int, default=612)
    parser.add_argument("--nx", type=int, default=388)
    parser.add_argument("--base_ch", type=int, default=48)
    parser.add_argument("--embed_ch", type=int, default=16)
    parser.add_argument("--mse_weight", type=float, default=0.7)
    parser.add_argument("--l1_weight", type=float, default=0.2)
    parser.add_argument("--ssim_weight", type=float, default=0.1)
    parser.add_argument("--init_F", type=float, default=1.5)
    parser.add_argument("--init_edge", type=float, default=0.1)
    parser.add_argument("--init_gain", type=float, default=1.0)
    parser.add_argument("--freeze_F", action="store_true")
    parser.add_argument("--freeze_edge", action="store_true")
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
    grid_z_np = grid_z.cpu().numpy()
    grid_x_np = grid_x.cpu().numpy()

    model = B1V2SoftDAS(
        grid_z_np, grid_x_np, sample0.geometry,
        base_ch=args.base_ch, embed_ch=args.embed_ch,
        init_F=args.init_F, init_edge=args.init_edge, init_gain=args.init_gain,
        train_F=not args.freeze_F,
        train_edge=not args.freeze_edge,
        train_gain=not args.freeze_gain,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_unet = sum(p.numel() for p in model.unet.parameters() if p.requires_grad)
    n_front = sum(p.numel() for p in model.frontend.parameters() if p.requires_grad)
    print(f"B1-v2 (SoftDAS + UNet+AG base_ch={args.base_ch}) params: {n_params:,} "
          f"(unet={n_unet:,}, frontend_γ={n_front})")
    print(f"  init γ: F={args.init_F}  edge={args.init_edge}  gain={args.init_gain}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    aug_cfg = None if args.no_aug else AugConfig()
    def train_iter():
        return _iter_split(loader, train_indices, args.target_root, grid_z, grid_x, args.device, aug_cfg, seed_base=args.seed * 1000)
    def val_iter():
        return _iter_split(loader, val_indices, args.target_root, grid_z, grid_x, args.device, None, seed_base=0)

    run_training(args, model, optimizer, train_iter, val_iter if val_indices else None, _forward_and_loss)


if __name__ == "__main__":
    main()
