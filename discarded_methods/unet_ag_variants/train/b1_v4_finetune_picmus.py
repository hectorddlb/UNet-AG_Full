"""B1-v4 fine-tuning on PICMUS — OOD penalty diagnosis.

Loads the B1-v4(β=0.3) ckpt trained on Field II and fine-tunes it on the 2
PICMUS sim phantoms (resolution + contrast), using as target the DAS-75PW
reconstructed image provided by PICMUS (slot ``number_plane_waves=75`` in
``reconstructed_image/.../*_img_from_rf.hdf5``).

It is a diagnosis, not production:
- 4 phantoms in total (2 train sim, 2 val exp). That is very little data — the
  goal is NOT to "train well", but to see whether the model CAN adapt to the
  PICMUS sidelobe pattern with few weights tuned.
- 3 modes:
  · ``--train_mode gain_only``: ``log_gain`` only (1 scalar parameter).
  · ``--train_mode decoder_only``: ``log_gain`` + UNet decoder (the U-Net
    `up*`/`out_conv` convs), encoder frozen.
  · ``--train_mode full``: the whole model.
- lr 1e-5 default, AdamW weight_decay 1e-3 (high), 30 epochs default.
- Loss = same as B1-v4 (MSE+L1+SSIM + L1-anchor) but with a stronger anchor
  (β=0.5) by default, to avoid destruction with so little data.

If gain_only moves the metrics → OOD is pure scaling.
If gain_only does not move them and decoder_only does → it is superficial,
feature-level.
If full moves nothing → deep manifold mismatch, it needs geometry
augmentation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterator, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ..data.rf_loader import PICMUSLoader
from ..losses.supervised import mse_l1_ssim_loss
from ..models.b1_v3_residual import B1V3Residual
from ..physics.f_number_map import compute_f_number_map_band_avg
from .b1_attention_gate import (
    _adjoint_image,
    _build_grid,
    _envelope_log_compress,
)


# ──────────────────────────────────────────────────────────────────
# PICMUS reconstructed_image (npw=75) → fine-tuning target
# ──────────────────────────────────────────────────────────────────


def _load_picmus_target(
    picmus_root: Path, mode: str, phantom: str, nz: int, nx: int
) -> Optional[np.ndarray]:
    """Return the PICMUS DAS-75PW reconstructed image, env+log compressed
    and resized to (nz, nx) — analogous to env+log(Schiffner_75PW) used in
    Field II training.
    """
    sub_mode = {"sim": "simulation", "exp": "experiments"}[mode]
    sub_ph = {"resolution_distortion": "resolution_distorsion",
              "contrast_speckle": "contrast_speckle"}[phantom]
    tag = "simu" if mode == "sim" else "expe"
    rpath = (picmus_root / "reconstructed_image" / sub_mode / sub_ph
             / f"{sub_ph}_{tag}_img_from_rf.hdf5")
    if not rpath.exists():
        return None
    with h5py.File(rpath, "r") as f:
        g = f["US/US_DATASET0000"]
        npw = np.asarray(g["number_plane_waves"]).reshape(-1)
        idx_75 = int(np.argmin(np.abs(npw - 75)))
        real = np.asarray(g["data/real"])[idx_75]   # (X=387, Z=609) — magnitude
    img = real.T.astype(np.float32)                  # (Z, X)
    # The PICMUS reconstructed image is magnitude, *not* RF — skip Hilbert.
    img_max = float(img.max()) + 1e-12
    db = 20.0 * np.log10(img / img_max + 1e-30)
    db = np.clip(db, -60.0, 0.0)
    env_log = ((db + 60.0) / 60.0).astype(np.float32)
    if env_log.shape != (nz, nx):
        t = torch.from_numpy(env_log).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(nz, nx), mode="bilinear", align_corners=False)
        env_log = t[0, 0].numpy()
    return env_log


# ──────────────────────────────────────────────────────────────────
# Phantom iterator (PICMUS)
# ──────────────────────────────────────────────────────────────────


_COMBOS = [
    ("sim", "resolution_distortion"),
    ("sim", "contrast_speckle"),
    ("exp", "resolution_distortion"),
    ("exp", "contrast_speckle"),
]


def _iter_picmus(
    picmus_root: Path,
    which: list[tuple[str, str]],
    nz: int, nx: int,
    device: str,
) -> Iterator[tuple[dict, Tensor, str]]:
    for mode, ph in which:
        try:
            ld = PICMUSLoader(picmus_root, mode=mode, phantom=ph, normalize=True)
        except FileNotFoundError:
            continue
        sample = ld[0]
        grid_z, grid_x = _build_grid(sample.geometry, nz, nx)
        target = _load_picmus_target(picmus_root, mode, ph, nz, nx)
        if target is None:
            continue
        target_t = torch.from_numpy(target).to(device).unsqueeze(0).unsqueeze(0)

        x0_raw = _adjoint_image(sample.rf_1pw, sample.geometry, grid_z, grid_x, device)
        x0 = _envelope_log_compress(x0_raw)
        f_map = compute_f_number_map_band_avg(grid_z.cpu().numpy(), grid_x.cpu().numpy(), sample.geometry)
        inputs = {
            "x0":    x0.to(device).unsqueeze(0).unsqueeze(0),
            "f_map": torch.from_numpy(f_map).to(device).unsqueeze(0).unsqueeze(0),
        }
        yield inputs, target_t, f"{mode}/{ph}"


# ──────────────────────────────────────────────────────────────────
# Param-group selection by train_mode
# ──────────────────────────────────────────────────────────────────


def _select_trainable(model: B1V3Residual, mode: str) -> int:
    """Freeze layers per ``mode``; return number of trainable params."""
    for p in model.parameters():
        p.requires_grad = False
    if mode == "gain_only":
        if hasattr(model, "log_gain"):
            model.log_gain.requires_grad = True
    elif mode == "decoder_only":
        if hasattr(model, "log_gain"):
            model.log_gain.requires_grad = True
        # B1V3Residual.unet has encoder (down*) and decoder (up*, out_conv).
        # Train the decoder + AGs.
        for name, p in model.named_parameters():
            if any(tok in name for tok in (".up", ".out_conv", ".ag", ".dec")):
                p.requires_grad = True
    elif mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"unknown train_mode={mode!r}")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="B1-v4 PICMUS fine-tuning diagnostic")
    p.add_argument("--ckpt",        type=Path, required=True)
    p.add_argument("--picmus_root", type=Path, required=True)
    p.add_argument("--output_dir",  type=Path, required=True)
    p.add_argument("--train_mode",  default="gain_only",
                   choices=["gain_only", "decoder_only", "full"])
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--lr",          type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--mse_weight",  type=float, default=0.7)
    p.add_argument("--l1_weight",   type=float, default=0.2)
    p.add_argument("--ssim_weight", type=float, default=0.1)
    p.add_argument("--anchor_weight", type=float, default=0.5,
                   help="Higher than training (0.3) to prevent destruction on tiny data.")
    p.add_argument("--nz", type=int, default=612)
    p.add_argument("--nx", type=int, default=388)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--embed_ch", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"device = {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load ckpt
    model = B1V3Residual(base_ch=args.base_ch, embed_ch=args.embed_ch).to(device)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd, strict=True)
    print(f"loaded ckpt: {args.ckpt}")
    print(f"  initial γ_gain: {float(model.gamma['gain']):.4f}")

    n_train = _select_trainable(model, args.train_mode)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"train_mode = {args.train_mode}  trainable={n_train:,}/{n_total:,} "
          f"({100*n_train/n_total:.2f}%)")

    # 2 sim phantoms = train; 2 exp phantoms = val.
    train_combos = [c for c in _COMBOS if c[0] == "sim"]
    val_combos   = [c for c in _COMBOS if c[0] == "exp"]

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    train_log_path = args.output_dir / "train_log.csv"
    val_log_path   = args.output_dir / "val_log.csv"
    train_log = open(train_log_path, "w", newline=""); tw = csv.writer(train_log)
    val_log   = open(val_log_path,   "w", newline=""); vw = csv.writer(val_log)
    tw.writerow(["epoch", "phantom", "mse", "l1", "ssim_loss", "anchor", "total", "gamma_gain"])
    vw.writerow(["epoch", "phantom", "mse", "l1", "ssim_loss", "anchor", "total", "gamma_gain"])

    best_val_total = float("inf")
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_totals = []
        for inputs, target, name in _iter_picmus(args.picmus_root, train_combos, args.nz, args.nx, device):
            pred = model(inputs["x0"], inputs["f_map"])
            loss, br = mse_l1_ssim_loss(
                pred, target,
                mse_weight=args.mse_weight, l1_weight=args.l1_weight,
                ssim_weight=args.ssim_weight, data_range=1.0,
            )
            anchor = (pred - inputs["x0"]).abs().mean()
            total = loss + args.anchor_weight * anchor
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            g = float(model.gamma["gain"])
            tw.writerow([epoch, name, br["mse"], br["l1"], br["ssim_loss"],
                         float(anchor), float(total), g])
            train_totals.append(float(total))

        # Val
        model.eval()
        val_totals = []
        with torch.no_grad():
            for inputs, target, name in _iter_picmus(args.picmus_root, val_combos, args.nz, args.nx, device):
                pred = model(inputs["x0"], inputs["f_map"])
                loss, br = mse_l1_ssim_loss(
                    pred, target,
                    mse_weight=args.mse_weight, l1_weight=args.l1_weight,
                    ssim_weight=args.ssim_weight, data_range=1.0,
                )
                anchor = (pred - inputs["x0"]).abs().mean()
                total = loss + args.anchor_weight * anchor
                g = float(model.gamma["gain"])
                vw.writerow([epoch, name, br["mse"], br["l1"], br["ssim_loss"],
                             float(anchor), float(total), g])
                val_totals.append(float(total))

        train_log.flush(); val_log.flush()
        mean_train = float(np.mean(train_totals)) if train_totals else float("nan")
        mean_val   = float(np.mean(val_totals))   if val_totals   else float("nan")
        print(f"ep {epoch:3d}  train_total={mean_train:.5f}  val_total={mean_val:.5f}  γ_gain={float(model.gamma['gain']):.4f}")

        # Save best by val_total
        ckpt_dir = args.output_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
        if mean_val < best_val_total:
            best_val_total = mean_val
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_total": mean_val},
                       ckpt_dir / "best.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "last.pt")

    train_log.close(); val_log.close()
    print(f"DONE. best_val_total = {best_val_total:.5f}  →  {args.output_dir / 'checkpoints' / 'best.pt'}")


if __name__ == "__main__":
    main()
