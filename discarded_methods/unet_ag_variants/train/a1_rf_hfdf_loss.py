"""Train proposal A1 — RF encoder + ResU-Net + H_Fdf fidelity loss.

Input pipeline (per phantom):
  1. ``rf_1pw`` (D, K) ← FieldIIMultiAngleLoader (z-scored).
  2. Pad/truncate to ``--rf_n_samples`` (default 3500); per-phantom
     /max(|.|) on top of z-score so RF lives in [-1, 1].
  3. Schiffner Fd-F 75 PW target (raw signed), /max(|.|) into [-1, 1].

Loss:
  L = MSE(x_pred, target_norm) + l1_w · L1(x_pred, target_norm)
      + fidelity_w · MSE(H_Fdf · x_pred · rf_align, rf_obs_norm)

where ``rf_align = target_scale / rf_scale`` is an empirical scale matcher
in post-normalisation units; H_Fdf's absolute output units depend on its
internal apodisation/dft normalisations, so we sidestep that by aligning
in the same [-1,1] space the loss lives in.

Anti-zero discipline (v3 after run 185159 NaN-blowup):
  Run 185037 with l1_w=0.1, fid_w=0.1 collapsed to zero (gap from zero to
  correct ≈ 0.004). Run 185159 tried /std normalisation but std_raw of
  Field II RF is ~1e-23 (native units), making rf_align ~ 1e24 → float32
  overflow → NaN from ep 0. v3 reverts to /max(|.|) (numerically stable,
  rf_align ~ 1e3) and instead raises ``l1_w=0.5`` and ``fid_w=5.0`` so
  the predict-zero floor jumps ~25× without changing units.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..data.augmentations import AugConfig, apply_augmentations
from ..data.manifest import field_ii_split
from ..data.rf_loader import FieldIIMultiAngleLoader, zscore_rf
from ..losses.supervised import mse_l1_loss
from ..models.a1_resunet_rf import A1ResUNetRF
from ..physics.h_fdf_wrapper import hfdf_forward_checkpointed, make_op_for_angle
from .base_trainer import parse_common_args, run_training


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
    """Pad with zeros (later times have no echo) or truncate to ``n_samples``.

    Field II phantoms vary in n_samples (~1300–1700); we standardise to
    n_samples=3500 (PICMUS-style) so the model has a fixed input.
    """
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
) -> Iterator[tuple[dict, Tensor]]:
    Nz, Nx = grid_z.shape[0], grid_x.shape[0]
    for step, idx in enumerate(indices):
        sample = loader[idx]
        if aug_cfg is not None:
            sample = apply_augmentations(sample, aug_cfg, seed=seed_base + step)
            rf, _ = zscore_rf(sample.rf_1pw)
            sample.rf_1pw = rf

        ptype = "points" if sample.phantom_id.startswith("points") else "cysts"
        target = _load_target(target_root, sample.phantom_id, ptype, Nz, Nx)
        if target is None:
            continue                                  # skip phantoms without Schiffner target

        # Pad/truncate RF and renormalise by /max(|.|) on top of zscore.
        rf_padded = _pad_or_truncate_rf(sample.rf_1pw, rf_n_samples)
        rf_scale = float(rf_padded.abs().max()) + 1e-12
        rf_norm = rf_padded / rf_scale

        # Target /max(|.|) → [-1, 1]. Anti-zero pressure comes from raised
        # l1_w and fid_w (see header), not from a different normalisation.
        target_scale = float(target.abs().max()) + 1e-12
        target_norm = target / target_scale

        # Build OpHFdf for 1 PW (angle 0°). n_samples MUST match rf_n_samples
        # so H_Fdf @ x_pred returns the same shape as rf_norm.
        geom_for_op = dict(sample.geometry)
        geom_for_op["n_samples"] = rf_n_samples
        op = make_op_for_angle(0.0, geom_for_op, grid_z, grid_x, device=device, nf_chunk=None)

        inputs = {
            "rf": rf_norm.to(device).unsqueeze(0).unsqueeze(0),    # (1, 1, D, K)
            "op": op,
            "rf_obs": rf_norm.to(device),                          # (D, K)
            "rf_align": float(target_scale / rf_scale),
        }
        yield inputs, target_norm.to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, Nz, Nx)


def _forward_and_loss(model, inputs, target, args):
    pred = model(inputs["rf"])                                    # (1, 1, Nz, Nx) linear
    loss_img, br_img = mse_l1_loss(pred, target, l1_weight=args.l1_weight)

    # Fidelity: gradient-checkpointed H_Fdf forward (recomputes apod/phase/kernel
    # during backward; without it, OOM on 24GB with Nf~992 bins). The gradient
    # only flows to x_pred (α contract preserved).
    op = inputs["op"]
    y_pred = hfdf_forward_checkpointed(op, pred[0, 0])             # (D, K) real
    if y_pred.is_complex():
        y_pred = y_pred.real
    y_pred = y_pred.to(inputs["rf_obs"].dtype) * inputs["rf_align"]
    loss_fid = F.mse_loss(y_pred, inputs["rf_obs"])

    loss = loss_img + args.fidelity_weight * loss_fid
    br = {
        **br_img,
        "fidelity": float(loss_fid.item()),
        "total": float(loss.item()),
    }
    return loss, br, pred


def main() -> None:
    parser = argparse.ArgumentParser(description="A1 — RF→image + H_Fdf fidelity")
    parser.add_argument("--target_root", type=Path, required=True)
    parser.add_argument("--phantom_type", choices=["points", "cysts", "both"], default="points")
    parser.add_argument("--nz", type=int, default=612)
    parser.add_argument("--nx", type=int, default=388)
    parser.add_argument("--rf_n_samples", type=int, default=3500)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--l1_weight", type=float, default=0.5)
    parser.add_argument("--fidelity_weight", type=float, default=5.0)
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

    model = A1ResUNetRF(
        n_samples=args.rf_n_samples,
        n_elements=int(geom0["n_elements"]),
        nz=args.nz,
        nx=args.nx,
        base_ch=args.base_ch,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"A1 params: {n_params:,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    aug_cfg = None if args.no_aug else AugConfig()
    def train_iter():
        return _iter_split(loader, train_indices, args.target_root, grid_z, grid_x,
                           args.device, args.rf_n_samples, aug_cfg, seed_base=args.seed * 1000)
    def val_iter():
        return _iter_split(loader, val_indices, args.target_root, grid_z, grid_x,
                           args.device, args.rf_n_samples, None, seed_base=0)

    run_training(args, model, optimizer, train_iter,
                 val_iter if val_indices else None, _forward_and_loss)


if __name__ == "__main__":
    main()
