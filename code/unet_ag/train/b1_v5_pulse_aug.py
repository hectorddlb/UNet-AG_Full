"""B1-v5 — B1-v4 arch + L1-anchor + runtime pulse-shape augmentation.

Motivation (2026-05-29). The FT-PICMUS diagnosis (jobs 187577-582) came back
null: no short post-hoc mode rescues the OOD penalty on PICMUS. The spectral
measurement on RF (2026-05-29) revealed the gap the FT did not test: Field II
training fc_peak≈4.97 MHz BW(-6 dB)≈1.82 MHz vs PICMUS exp resolution
fc_peak≈5.04 MHz BW≈1.04 MHz. Diagnosis (B), a 5×5 (fc, BW) grid search on the
pre-adjoint PICMUS RF, confirmed that the gap is spectral (some pre-filters
improve val_total by >10% vs identity).

B1-v5 attacks the gap during FULL TRAINING instead of post-hoc: each batch of
the Field II dataloader receives, with probability ``pulse_apply_prob``, a
Gaussian bandpass with BW ∈ ``pulse_bw_range_hz`` and centre jittered by ±
``pulse_fc_shift_hz``. The model learns invariance to the pulse bandwidth
instead of memorizing the Field II-specific PSF.

Architecture identical to B1-v4: ``B1V3Residual`` (residual skip + zero-init
``out_conv`` + learnable ``log_gain``), same Schiffner_75PW env+log target,
same MSE+L1+SSIM+L1-anchor loss. Only change: AugConfig enables the
pulse-shape jitter by default.

Hypotheses to validate:
- (H1) In the Field II eval, B1-v5 keeps the B1-v4(0.3) metric
  (it does not degrade the in-distribution case; aug regularizes without
  killing).
- (H2) In the PICMUS exp resolution eval, B1-v5 reduces the FWHM_lat
  penalty (4.96 mm vs DAS 2.47 mm) towards the DAS range.
- (H3) In the PICMUS exp contrast eval, B1-v5 at least preserves the gCNR
  gain of 0.776 already reached by B1-v4(0.3).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..data.augmentations import AugConfig
from ..data.manifest import field_ii_split
from ..data.rf_loader import FieldIIMultiAngleLoader
from ..models.b1_v3_residual import B1V3Residual
from .b1_attention_gate import _build_grid
from .b1_v4_anchored import _iter_split, _forward_and_loss
from .base_trainer import parse_common_args, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="B1-v5 — B1-v4 + pulse-shape augmentation")
    parser.add_argument("--target_root", type=Path, required=True)
    parser.add_argument("--anchor_root", type=Path, default=None,
                        help="If given, the anchor term uses the "
                             "das_pw(Fd-F, 1 PW) image precomputed there instead "
                             "of x0 (the Fd-F adjoint).")
    parser.add_argument("--phantom_type", choices=["points", "cysts", "both"], default="points")
    parser.add_argument("--nz", type=int, default=612)
    parser.add_argument("--nx", type=int, default=388)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--embed_ch", type=int, default=16)
    parser.add_argument("--mse_weight", type=float, default=0.7)
    parser.add_argument("--l1_weight", type=float, default=0.2)
    parser.add_argument("--ssim_weight", type=float, default=0.1)
    parser.add_argument("--anchor_weight", type=float, default=0.3,
                        help="B1-v4 winning value (default 0.3).")
    parser.add_argument("--gain_reg_weight", type=float, default=0.0)
    parser.add_argument("--init_gain", type=float, default=1.0)
    parser.add_argument("--freeze_gain", action="store_true")
    parser.add_argument("--no_aug", action="store_true",
                        help="Disable ALL augmentations (incl. pulse jitter). For ablation.")
    parser.add_argument("--target_is_bmode", action="store_true",
                        help="F.3 distillation: targets are already envelope-log-compressed "
                             "B-mode images in [-1, 1] (DRUS teacher output). Skip the trainer's "
                             "hilbert/log-compress and just remap [-1, 1] -> [0, 1].")
    parser.add_argument("--pulse_apply_prob", type=float, default=0.7,
                        help="Per-step probability of applying the bandpass.")
    parser.add_argument("--pulse_bw_min_hz", type=float, default=1.0e6,
                        help="Lower bound on sampled -6 dB BW (Hz). Default covers PICMUS exp 1.04 MHz.")
    parser.add_argument("--pulse_bw_max_hz", type=float, default=2.5e6,
                        help="Upper bound on sampled -6 dB BW (Hz). Default >Field II native 1.82 MHz.")
    parser.add_argument("--pulse_fc_shift_hz", type=float, default=300e3,
                        help="± uniform shift on filter fc, around geometry['fc'].")
    args = parse_common_args(parser)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    loader = FieldIIMultiAngleLoader(args.data_root, phantom_type=args.phantom_type, keep_all_angles=False)
    splits = field_ii_split(args.data_root,
                            phantom_types=[args.phantom_type] if args.phantom_type != "both" else None)
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
    print(f"B1-v5 (B1-v4 + pulse-shape aug, base_ch={args.base_ch}) params: {n_params:,} (unet={n_unet:,})")
    print(f"  init γ : gain={args.init_gain}  freeze={args.freeze_gain}")
    if args.target_is_bmode:
        print(f"  target : B-mode [-1,1]->[0,1] (F.3 distillation, NO re-envelope)")
    print(f"  loss   : mse={args.mse_weight}  l1={args.l1_weight}  ssim={args.ssim_weight}"
          f"  anchor={args.anchor_weight}  gain_reg={args.gain_reg_weight}")
    print(f"  anchor : {'das_pw Fd-F 1PW <- ' + str(args.anchor_root) if args.anchor_root else 'x0 (Fd-F adjoint)'}")
    if args.no_aug:
        print(f"  aug    : DISABLED (--no_aug)")
    else:
        print(f"  aug    : pulse_apply_prob={args.pulse_apply_prob}"
              f"  bw=[{args.pulse_bw_min_hz/1e6:.2f}, {args.pulse_bw_max_hz/1e6:.2f}] MHz"
              f"  fc_shift=±{args.pulse_fc_shift_hz/1e3:.0f} kHz")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.no_aug:
        aug_cfg = None
    else:
        aug_cfg = AugConfig(
            pulse_apply_prob=args.pulse_apply_prob,
            pulse_bw_range_hz=(args.pulse_bw_min_hz, args.pulse_bw_max_hz),
            pulse_fc_shift_hz=args.pulse_fc_shift_hz,
        )

    def train_iter():
        return _iter_split(loader, train_indices, args.target_root, grid_z, grid_x,
                           args.device, aug_cfg, seed_base=args.seed * 1000,
                           target_is_bmode=args.target_is_bmode,
                           anchor_root=args.anchor_root)

    def val_iter():
        return _iter_split(loader, val_indices, args.target_root, grid_z, grid_x,
                           args.device, None, seed_base=0,
                           target_is_bmode=args.target_is_bmode,
                           anchor_root=args.anchor_root)

    run_training(args, model, optimizer, train_iter, val_iter if val_indices else None,
                 _forward_and_loss)


if __name__ == "__main__":
    main()
