"""Train proposal A2-v6 — target-weighted MSE+L1 + 2-D Fourier-weighted loss.

Motivation. v1-v5 plateaued at ``n_resolved_lat ≈ 0/20`` despite improving
gCNR on cysts. Diagnosis: the
env+log target is ~99% background and ~1% bright wires, so mean-reduced MSE+L1
collapses to the background mean — the optimal "predict-zero on wires" predictor
sits at the loss minimum. v6 keeps everything from v5 (adjoint input + B1
backbone + native grid + 2-D Fourier-W loss) and changes ONLY the loss to
target-weighted: ``w(t) = 1 + α·t`` (α=9 → 10× weight on bright pixels) and
bumps backbone capacity (base_ch=48 ≈ 7M params) to exploit the better
gradient signal.

CLI smoke:
  python -m unet_ag.train.a2_v6_target_weighted \\
      --data_root datasets/Field\\ II/field_ii_data_multiangle \\
      --target_root results_schiffner_targets_fieldii \\
      --output_dir results_v2/a2_v6 \\
      --max_phantoms_train 5 --epochs 2 --smoke_steps 6 --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..data.augmentations import AugConfig
from ..data.manifest import field_ii_split
from ..data.rf_loader import FieldIIMultiAngleLoader
from ..losses.fourier_weighted import target_weighted_mse_l1_fourier_loss
from ..models.b1_unet_attention_fnum import B1UNetAttentionFNum
from .a2_v4_adjoint_fourier import _build_grid, _iter_split
from .base_trainer import parse_common_args, run_training


def _forward_and_loss(model, inputs, target, args):
    pred = model(inputs["x0"], inputs["f_map"])
    loss, br = target_weighted_mse_l1_fourier_loss(
        pred, target,
        weight=inputs["fourier_weight"],
        mse_weight=args.mse_weight,
        l1_weight=args.l1_weight,
        fourier_weight=args.fourier_weight,
        alpha=args.alpha,
    )
    return loss, br, pred


def main() -> None:
    parser = argparse.ArgumentParser(description="A2-v6 — target-weighted loss + 2-D Fourier-W")
    parser.add_argument("--target_root", type=Path, required=True)
    parser.add_argument("--phantom_type", choices=["points", "cysts", "both"], default="points")
    parser.add_argument("--nz", type=int, default=612)
    parser.add_argument("--nx", type=int, default=388)
    parser.add_argument("--base_ch", type=int, default=48)
    parser.add_argument("--embed_ch", type=int, default=16)
    parser.add_argument("--mse_weight", type=float, default=0.6)
    parser.add_argument("--l1_weight", type=float, default=0.25)
    parser.add_argument("--fourier_weight", type=float, default=0.15)
    parser.add_argument("--alpha", type=float, default=9.0,
                        help="target-weighting coefficient: w = 1 + alpha*t")
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
    print(f"A2-v6 (B1 backbone base_ch={args.base_ch} + target-weighted loss α={args.alpha}) "
          f"params: {n_params:,}")
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
