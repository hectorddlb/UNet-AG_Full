"""Grid search over (fc, BW) Gaussian bandpass pre-filter on PICMUS RF.

Diagnosis of the B1-v4 OOD penalty on PICMUS. Spectral measurement
(2026-05-29) on RF: Field II training fc_peak≈4.97 MHz BW≈1.82 MHz vs PICMUS
exp resolution fc_peak≈5.04 MHz BW≈1.04 MHz. If the OOD penalty originates in
the difference of the transducer spectral response, then there is some
pre-filter applied to the PICMUS RF that reduces val_total against the PICMUS
DAS-75 PW target.

Design:
- Gaussian bandpass in frequency: H(f) = exp(-0.5·((|f|−fc)/σ)²) with
  σ = BW / 2.355 (BW interpreted as the -6 dB full width).
- Applied to ``sample.rf_1pw`` along the time axis via rfft → mul → irfft,
  followed by a re-zscore (the filtering changes σ_rf).
- Frozen B1-v4(0.3); everything in ``torch.no_grad()``. NO autograd → avoids
  reintroducing the Phase 4.1.2 OOM.
- For each (fc, BW): adjoint (cached OpHFdf) → env+log → pred → loss and
  metrics. ``BW=identity`` uses a flat filter (σ = 100 MHz) that passes
  everything, equivalent to no filtering.

Output:
- CSV ``results_v2/b1_v4_a0p3_prefilter_grid/grid.csv`` with columns
  ``fc_mhz, bw_mhz, mode, phantom, val_total, mse, l1, ssim_loss,
   gcnr, fwhm_lat_mm, fwhm_n_lat``.
- Console: heatmap of the best val_total per (fc, BW) over the exp phantoms
  + delta vs the identity baseline.

Verdict:
- If some (fc, BW) ≠ identity gives val_total < identity on ≥1 exp phantom
  with a margin ≥3% → confirms a spectral component of the gap → defensible
  as an ablation and as motivation for the pulse-shape augmentation of B1-v5.
- If the whole grid converges to the same val_total → the gap is NOT spectral
  to the degree capturable by (fc, BW) → it needs an intervention during full
  training (justifies B1-v5 as the only route).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from ..data.rf_loader import PICMUSLoader, zscore_rf
from ..eval.evaluate_fieldii import (
    _build_grid,
    _envelope_log_compress,
    _load_b1_v3_ckpt,
)
from ..eval.masks_picmus import cyst_masks as picmus_cyst_masks
from ..eval.masks_picmus import points_wire_targets as picmus_wire_targets
from ..losses.supervised import mse_l1_ssim_loss
from ..metrics.fwhm import aggregate_fwhms_per_wire
from ..metrics.gcnr import compute_gcnr
from ..physics.f_number_map import compute_f_number_map_band_avg
from ..physics.h_fdf_wrapper import make_op_for_angle


# ──────────────────────────────────────────────────────────────────
# Pre-filter
# ──────────────────────────────────────────────────────────────────


_BW_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))      # σ = BW / 2.355


def gaussian_bandpass_rf(rf: torch.Tensor, fs: float, fc_hz: float, bw_hz: float) -> torch.Tensor:
    """Apply a Gaussian-magnitude bandpass to ``rf`` (D, K) along axis 0.

    Parameters
    ----------
    rf : (D, K) float tensor (z-scored).
    fs : sampling frequency Hz.
    fc_hz : passband center frequency.
    bw_hz : -6 dB full-width passband.
    """
    if rf.dim() != 2:
        raise ValueError(f"rf must be 2D (D, K), got {tuple(rf.shape)}")
    D = rf.shape[0]
    sigma = bw_hz * _BW_TO_SIGMA
    Y = torch.fft.rfft(rf, dim=0)                                # (D//2+1, K)
    freqs = torch.fft.rfftfreq(D, d=1.0 / fs).to(rf.device).to(rf.dtype)
    H = torch.exp(-0.5 * ((freqs - fc_hz) / sigma) ** 2)         # (D//2+1,)
    return torch.fft.irfft(Y * H.unsqueeze(-1), n=D, dim=0)


# ──────────────────────────────────────────────────────────────────
# PICMUS DAS-75 PW target (env+log) for val_total
# ──────────────────────────────────────────────────────────────────


def _load_picmus_target(picmus_root: Path, mode: str, phantom: str, nz: int, nx: int) -> Optional[np.ndarray]:
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
        real = np.asarray(g["data/real"])[idx_75]
    img = real.T.astype(np.float32)
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
# Per-phantom evaluation under one (fc, bw)
# ──────────────────────────────────────────────────────────────────


@torch.no_grad()
def _eval_one(
    model, sample, op, grid_z, grid_x, target_env_log: Optional[np.ndarray],
    fc_hz: float, bw_hz: float, mode: str, phantom: str, device: str,
) -> dict:
    # Pre-filter the (possibly already z-scored) RF, then re-zscore.
    rf = sample.rf_1pw.to(device).float()
    rf_f = gaussian_bandpass_rf(rf, sample.geometry["sampling_freq_hz"], fc_hz, bw_hz)
    rf_f, _ = zscore_rf(rf_f.cpu())                                          # (D, K)
    rf_f = rf_f.to(device).to(op.float_dtype)

    x0 = op.adjoint(rf_f)
    if x0.is_complex():
        x0 = x0.real
    x0_np = x0.float().detach().cpu().numpy()
    x0_env = _envelope_log_compress(x0_np)                                   # (Nz, Nx)
    x0_t = torch.from_numpy(x0_env).to(device).unsqueeze(0).unsqueeze(0)

    f_map = compute_f_number_map_band_avg(grid_z.cpu().numpy(), grid_x.cpu().numpy(), sample.geometry)
    f_map_t = torch.from_numpy(f_map).to(device).unsqueeze(0).unsqueeze(0)
    pred_t = model(x0_t, f_map_t)
    pred_env = pred_t[0, 0].cpu().numpy().astype(np.float32)

    row = {
        "fc_mhz":   round(fc_hz / 1e6, 4),
        "bw_mhz":   round(bw_hz / 1e6, 4),
        "mode":     mode,
        "phantom":  phantom,
    }
    if target_env_log is not None:
        target_t = torch.from_numpy(target_env_log).to(device).unsqueeze(0).unsqueeze(0)
        _, br = mse_l1_ssim_loss(pred_t, target_t,
                                 mse_weight=0.7, l1_weight=0.2, ssim_weight=0.1, data_range=1.0)
        row.update({
            "mse":         float(br["mse"]),
            "l1":          float(br["l1"]),
            "ssim_loss":   float(br["ssim_loss"]),
            "val_total":   float(br["total"]),
        })

    # Task metrics (phantom HDF5 path attached by caller in main()).
    phantom_hdf5 = sample._phantom_hdf5
    if phantom == "contrast_speckle":
        mi, mo, cysts = picmus_cyst_masks(phantom_hdf5,
                                          grid_z.cpu().numpy(), grid_x.cpu().numpy())
        if cysts and mo.sum() > 0:
            row["gcnr"] = float(compute_gcnr(pred_env, mi, mo, on_envelope=False))
        else:
            row["gcnr"] = float("nan")
    elif phantom == "resolution_distortion":
        wires = picmus_wire_targets(phantom_hdf5)
        gz_mm = grid_z.cpu().numpy() * 1e3
        gx_mm = grid_x.cpu().numpy() * 1e3
        dz_mm = float(gz_mm[1] - gz_mm[0])
        dx_mm = float(gx_mm[1] - gx_mm[0])
        res = aggregate_fwhms_per_wire(pred_env, wires, dx_mm=dx_mm, dz_mm=dz_mm,
                                       grid_z_mm=gz_mm, grid_x_mm=gx_mm,
                                       search_radius_mm=1.5)
        row["fwhm_lat_mm"] = float(res["lat_median"])
        row["fwhm_n_lat"]  = int(res["n_resolved_lat"])
    return row


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────


_COMBOS = [
    ("sim", "resolution_distortion"),
    ("sim", "contrast_speckle"),
    ("exp", "resolution_distortion"),
    ("exp", "contrast_speckle"),
]

# fc grid covers measured PICMUS peak range [4.64, 5.04] MHz + Field II 4.97
# BW grid covers measured PICMUS BW range [1.04, 1.57] MHz + Field II 1.82
# IDENTITY = wide sigma → passes everything (sanity baseline).
DEFAULT_FC_GRID_MHZ = [4.6, 4.8, 5.0, 5.2, 5.4]
DEFAULT_BW_GRID_MHZ = [0.8, 1.2, 1.6, 2.0, 3.0]
IDENTITY_BW_MHZ = 100.0          # Gaussian σ ≫ Nyquist → flat in [0, fs/2]


def main() -> None:
    p = argparse.ArgumentParser(description="PICMUS pre-filter (fc, BW) grid search")
    p.add_argument("--ckpt",        type=Path, required=True)
    p.add_argument("--picmus_root", type=Path, required=True)
    p.add_argument("--output_dir",  type=Path, required=True)
    p.add_argument("--nz", type=int, default=612)
    p.add_argument("--nx", type=int, default=388)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fc_mhz_grid", type=float, nargs="+", default=DEFAULT_FC_GRID_MHZ)
    p.add_argument("--bw_mhz_grid", type=float, nargs="+", default=DEFAULT_BW_GRID_MHZ)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"device = {device}")
    print(f"ckpt   = {args.ckpt}")
    print(f"fc grid (MHz) = {args.fc_mhz_grid}")
    print(f"bw grid (MHz) = {args.bw_mhz_grid}  (+ identity={IDENTITY_BW_MHZ})")

    model = _load_b1_v3_ckpt(args.ckpt, device, base_ch=args.base_ch)
    model.eval()

    rows: list[dict] = []

    for mode, phantom in _COMBOS:
        print(f"\n=== {mode}/{phantom} ===")
        try:
            loader = PICMUSLoader(args.picmus_root, mode=mode, phantom=phantom, normalize=True)
        except FileNotFoundError as e:
            print(f"  SKIP — {e}")
            continue
        sample = loader[0]
        sample._phantom_hdf5 = loader.phantom_path
        grid_z, grid_x = _build_grid(sample.geometry, args.nz, args.nx)
        target_env_log = _load_picmus_target(args.picmus_root, mode, phantom, args.nz, args.nx)
        print(f"  target loaded: {target_env_log is not None}")

        op = make_op_for_angle(0.0, sample.geometry, grid_z, grid_x, device=device, nf_chunk=None)
        fc_nominal = float(sample.geometry["fc"])

        # Identity baseline (Gaussian σ huge → unity passband).
        row_id = _eval_one(model, sample, op, grid_z, grid_x, target_env_log,
                           fc_nominal, IDENTITY_BW_MHZ * 1e6, mode, phantom, device)
        row_id["bw_mhz"] = -1.0          # sentinel: identity
        rows.append(row_id)
        print(f"  identity: val_total={row_id.get('val_total', float('nan')):.5f}"
              + (f"  gCNR={row_id.get('gcnr', float('nan')):.3f}" if "gcnr" in row_id else "")
              + (f"  FWHM_lat={row_id.get('fwhm_lat_mm', float('nan')):.2f} mm  n_lat={row_id.get('fwhm_n_lat', -1)}" if "fwhm_lat_mm" in row_id else ""))

        for fc_mhz in args.fc_mhz_grid:
            for bw_mhz in args.bw_mhz_grid:
                fc_hz = fc_mhz * 1e6
                bw_hz = bw_mhz * 1e6
                row = _eval_one(model, sample, op, grid_z, grid_x, target_env_log,
                                fc_hz, bw_hz, mode, phantom, device)
                rows.append(row)
                tag = ""
                if "val_total" in row:
                    delta = row["val_total"] - row_id["val_total"]
                    sign = "↓" if delta < 0 else ("↑" if delta > 0 else "=")
                    tag = f"  val_total={row['val_total']:.5f} ({sign}{abs(delta):.5f})"
                print(f"  fc={fc_mhz:.2f} bw={bw_mhz:.2f}{tag}")

    # Write CSV
    out_csv = args.output_dir / "grid.csv"
    all_keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=all_keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows → {out_csv}")

    # Console summary — best (fc, bw) per phantom by val_total, only EXP rows
    print("\n=== Summary — best (fc, bw) by val_total (exp only) ===")
    by_combo: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_combo.setdefault((r["mode"], r["phantom"]), []).append(r)
    for (m, ph), rs in by_combo.items():
        if m != "exp" or "val_total" not in rs[0]:
            continue
        identity = next((r for r in rs if r["bw_mhz"] == -1.0), None)
        best     = min((r for r in rs if r["bw_mhz"] != -1.0), key=lambda r: r["val_total"])
        d = (best["val_total"] - identity["val_total"]) if identity else float("nan")
        sign = "↓" if d < 0 else ("↑" if d > 0 else "=")
        print(f"  {m}/{ph}")
        if identity:
            print(f"    identity                     val_total = {identity['val_total']:.5f}")
        print(f"    best: fc={best['fc_mhz']} bw={best['bw_mhz']}  val_total = {best['val_total']:.5f}  Δ={sign}{abs(d):.5f}")


if __name__ == "__main__":
    main()
