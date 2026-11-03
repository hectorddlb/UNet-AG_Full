"""B1-v4 — B1-v3 architecture + L1-anchor regularizer to ``x_DAS``.

Rescue #1 (2026-05-26). B1-v3 showed that the residual architecture with
identity init is correct (pred₀ ≡ x_DAS, n_lat₀=484/600), but the optimizer
destroys 414 wires in 462 steps because MSE + L1 + (1-SSIM) towards
env_log(Schiffner_75PW) pushes pred away from x_DAS over the WHOLE field (the
gradient mass lives in the dense background, not in the sparse wires).

B1-v4 keeps the same architecture (model ``B1V3Residual``) and adds an L1
regularizer on the drift ``pred − x_DAS``:

    L = α·MSE(pred, t)  +  β_L1·L1(pred, t)  +  γ·(1 − SSIM)
      + anchor_w · L1(pred − x_DAS)              ← NEW "lateral anchor" term
      + gain_reg_w · log_gain²                   ← optional, L2 on log_gain

Physical-mathematical argument:

- Since ``pred = clamp(x_DAS + γ_gain·UNet(...), 0, 1)``, in non-saturated
  regions ``pred − x_DAS = γ_gain·UNet(...)``, so the anchor is **directly**
  an L1 prior (Lasso) on the UNet residual — it favours sparse residuals that
  only appear where the improvement exceeds the cost β.
- If ``anchor_w → ∞``: pred=x_DAS → eval=DAS_1PW (n_lat=484, gCNR=0.110).
  Guaranteed floor.
- If ``anchor_w = 0``: equivalent to B1-v3 (collapse to n_lat=70).
- There is a β* in the continuum where the network improves gCNR without
  sacrificing more than N_max lateral wires; that β* is searched empirically.

If NO anchor_w in the sweep passes the verdict (FWHM_lat<4.9 AND n_lat≥150)
then the Pareto front is degenerate → definitive closure of B1 with
mechanistic (not only numerical) evidence.

Recommended sweep (see `unet_ag/slurm/train_b1_v4.sbatch`):
    ANCHOR_WEIGHT ∈ {0.3, 1.0, 3.0}   ← 3 chained runs, ~3 cluster days.
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
    target_is_bmode: bool = False,
    anchor_root: Optional[Path] = None,
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
        if target_is_bmode:
            # F.3 distillation: the DRUS teacher output is ALREADY an
            # envelope-log-compressed B-mode image in the diffusion prior domain
            # [-1, 1]. Re-running hilbert/log-compress on it (as das75_fixed RF
            # targets require) would double-envelope and destroy it. Just remap
            # [-1, 1] → [0, 1] to match the trainer's target space (x0 and loss
            # live in [0, 1]).
            target = ((target_raw.clamp(-1.0, 1.0) + 1.0) * 0.5)   # (Nz, Nx)
        else:
            target = _envelope_log_compress(target_raw)            # (Nz, Nx)

        f_map = compute_f_number_map_band_avg(grid_z_np, grid_x_np, sample.geometry)
        f_map_t = torch.from_numpy(f_map).to(device)

        inputs = {
            "x0":    x0.to(device).unsqueeze(0).unsqueeze(0),       # (1, 1, Nz, Nx)
            "f_map": f_map_t.unsqueeze(0).unsqueeze(0),             # (1, 1, Nz, Nx)
        }

        # Alternative anchor: by default the anchor is x0 (the Fd-F adjoint,
        # which is also the input). With --anchor_root the anchor is the image
        # of the das_pw BEAMFORMER with the same Fd-F apodization: they
        # correlate at 0.931, they are not the same image, and das_pw has
        # better contrast and worse resolution. If the file is missing we fall
        # back to x0 instead of skipping the phantom, so that an incomplete
        # precomputation does not silently change the split size (the
        # `continue` on the targets already cost us one diagnosis).
        if anchor_root is not None:
            ap = Path(anchor_root) / ptype / sample.phantom_id / "anchor.npy"
            if ap.exists():
                a_raw = torch.from_numpy(np.load(ap)).float()
                inputs["x_anchor"] = _envelope_log_compress(a_raw).to(
                    device).unsqueeze(0).unsqueeze(0)
            else:
                print(f"WARN anchor missing, using x0: {ap}")
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

    # L1 anchor: penalizes the drift pred − x_anchor uniformly over the field,
    # while MSE+L1+SSIM refine towards the Schiffner 75 PW target.
    #
    # BEWARE of the historical name: by default the anchor is inputs["x0"],
    # which is NOT full-aperture DAS but the Fd-F ADJOINT (the OpHFdf defaults
    # are GratingAngleLB(60,1.5) + Tukey(0.2) + NormalizationOn). That is, the
    # anchor has always been a 1 plane wave Fd-F image.
    x_anchor = inputs.get("x_anchor", inputs["x0"])
    anchor = (pred - x_anchor).abs().mean()
    loss = loss + args.anchor_weight * anchor
    br["anchor"] = float(anchor.item())

    # Optional: L2 on log_gain (shrinks γ_gain towards 1.0; complements the
    # AdamW weight_decay=1e-4 already active on all params).
    if args.gain_reg_weight > 0.0 and hasattr(model, "log_gain"):
        gain_reg = model.log_gain.pow(2)
        loss = loss + args.gain_reg_weight * gain_reg
        br["gain_reg"] = float(gain_reg.item())

    br["total"] = float(loss.item())

    g = model.gamma if hasattr(model, "gamma") else {}
    for k, v in g.items():
        br[f"gamma_{k}"] = v

    return loss, br, pred


def main() -> None:
    parser = argparse.ArgumentParser(description="B1-v4 — Residual skip + L1-anchor to x_DAS")
    parser.add_argument("--target_root", type=Path, required=True)
    parser.add_argument("--phantom_type", choices=["points", "cysts", "both"], default="points")
    parser.add_argument("--nz", type=int, default=612)
    parser.add_argument("--nx", type=int, default=388)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--embed_ch", type=int, default=16)
    parser.add_argument("--mse_weight", type=float, default=0.7)
    parser.add_argument("--l1_weight", type=float, default=0.2)
    parser.add_argument("--ssim_weight", type=float, default=0.1)
    parser.add_argument("--anchor_weight", type=float, default=1.0,
                        help="β en L = ... + β·L1(pred − x_DAS). Sweep recomendado {0.3, 1.0, 3.0}.")
    parser.add_argument("--gain_reg_weight", type=float, default=0.0,
                        help="λ_g en L = ... + λ_g·log_gain². 0 desactiva.")
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
    print(f"B1-v4 (B1-v3 + L1-anchor, base_ch={args.base_ch}) params: {n_params:,} "
          f"(unet={n_unet:,}, gain γ=1)")
    print(f"  init γ: gain={args.init_gain}  freeze={args.freeze_gain}")
    print(f"  loss   : mse={args.mse_weight}  l1={args.l1_weight}  ssim={args.ssim_weight}"
          f"  anchor={args.anchor_weight}  gain_reg={args.gain_reg_weight}")
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
