"""Train proposal A2-v7 — target-weighted MSE+L1 + **lateral-derivative L1**.

Rescue path for the B1/A2 family closure. The diagnosis
is that on env+log Schiffner_75PW targets the optimum of MSE/L1/SSIM is an
isotropic-smoothing predictor: a smoothed wire and a sharp wire share the
same integrated mass, so mean-reduced body losses cannot distinguish them.
A2-v6 added target-weighting (``w = 1 + 9·t``) to put ~50 % of the body
gradient on bright wires but the lateral PSF still collapsed because the
weighted body is still indifferent to *edge sharpness*.

A2-v7 keeps everything from v6 (B1 backbone base_ch=48, adjoint input,
env+log target, NATIVE 612×388 grid) and changes ONLY the loss:

    L = α·MSE_w(pred, t)  +  β·L1_w(pred, t)
      + γ·L1(∂_x pred − ∂_x t)        ← NEW: lateral edge fidelity
      + δ·L1(∂_z pred − ∂_z t)        ← optional, axial counterpart (default 0.05)

Mathematical argument. For a 1-D Gaussian wire profile of std σ_pred at the
same location as a ground-truth wire of std σ_target:
- Body MSE on the wire integral is *independent* of σ_pred (Gaussian mass).
- L1(∂_x · − ∂_x ·) reaches zero only when σ_pred = σ_target — the
  derivative pencil width is the FWHM-determining parameter.

Therefore the lateral-derivative term *defines* the FWHM_lat metric the
verdict requires. The Pareto front between gCNR (body fidelity) and FWHM_lat
(edge fidelity) is no longer degenerate.

Default sweep candidates (see ``slurm/train_a2_v7.sbatch``):
- ``LATERAL_WEIGHT ∈ {0.3, 1.0, 3.0}`` with ``ALPHA=9`` and ``AXIAL_WEIGHT=0.05``.

CLI smoke:
  python -m unet_ag.train.a2_v7_lateral_deriv \\
      --data_root datasets/Field\\ II/field_ii_data_multiangle \\
      --target_root results_schiffner_targets_fieldii \\
      --output_dir results_v2/a2_v7_smoke \\
      --max_phantoms_train 5 --max_phantoms_val 3 \\
      --epochs 2 --smoke_steps 6 --device cpu
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
from ..losses.lateral_deriv import target_weighted_mse_l1_lateral_loss
from ..models.b1_unet_attention_fnum import B1UNetAttentionFNum
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
        x0 = _envelope_log_compress(x0_raw)
        target = _envelope_log_compress(target_raw)

        f_map = compute_f_number_map_band_avg(grid_z_np, grid_x_np, sample.geometry)
        f_map_t = torch.from_numpy(f_map).to(device)

        inputs = {
            "x0":    x0.to(device).unsqueeze(0).unsqueeze(0),
            "f_map": f_map_t.unsqueeze(0).unsqueeze(0),
        }
        yield inputs, target.to(device).unsqueeze(0).unsqueeze(0)


def _forward_and_loss(model, inputs, target, args):
    pred = model(inputs["x0"], inputs["f_map"])
    loss, br = target_weighted_mse_l1_lateral_loss(
        pred, target,
        mse_weight=args.mse_weight,
        l1_weight=args.l1_weight,
        lateral_weight=args.lateral_weight,
        axial_weight=args.axial_weight,
        alpha=args.alpha,
    )
    return loss, br, pred


def main() -> None:
    parser = argparse.ArgumentParser(description="A2-v7 — target-weighted + lateral-derivative L1")
    parser.add_argument("--target_root", type=Path, required=True)
    parser.add_argument("--phantom_type", choices=["points", "cysts", "both"], default="points")
    parser.add_argument("--nz", type=int, default=612)
    parser.add_argument("--nx", type=int, default=388)
    parser.add_argument("--base_ch", type=int, default=48)
    parser.add_argument("--embed_ch", type=int, default=16)
    parser.add_argument("--mse_weight", type=float, default=0.5)
    parser.add_argument("--l1_weight", type=float, default=0.15)
    parser.add_argument("--lateral_weight", type=float, default=0.3,
                        help="γ in L = ... + γ·L1(∂_x pred − ∂_x target). Sweep {0.3, 1.0, 3.0}.")
    parser.add_argument("--axial_weight", type=float, default=0.05,
                        help="δ in L = ... + δ·L1(∂_z pred − ∂_z target). Set 0 to disable.")
    parser.add_argument("--alpha", type=float, default=9.0,
                        help="target-weighting coefficient: w(t) = 1 + α·t for body MSE+L1.")
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
    print(f"A2-v7 (B1 backbone base_ch={args.base_ch} + target-weighted + lateral-deriv L1) "
          f"params: {n_params:,}")
    print(f"  loss   : mse_w={args.mse_weight}  l1_w={args.l1_weight}  "
          f"lat={args.lateral_weight}  ax={args.axial_weight}  alpha={args.alpha}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    aug_cfg = None if args.no_aug else AugConfig()
    def train_iter():
        return _iter_split(loader, train_indices, args.target_root, grid_z, grid_x, args.device,
                           aug_cfg, seed_base=args.seed * 1000)
    def val_iter():
        return _iter_split(loader, val_indices, args.target_root, grid_z, grid_x, args.device,
                           None, seed_base=0)

    run_training(args, model, optimizer, train_iter, val_iter if val_indices else None, _forward_and_loss)


if __name__ == "__main__":
    main()
