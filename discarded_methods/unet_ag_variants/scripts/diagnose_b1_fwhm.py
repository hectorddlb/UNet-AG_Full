"""Diagnose the B1 ``FWHM_lat = 38.4 mm`` anomaly.

For 1-2 val phantoms, dump:
- Per-image stats (min/max/mean/std).
- Lateral profile through each wire's brightest pixel row (B1, DAS_1PW, Schiffner_75PW).
- Where the lateral profile drops below -6 dB of its peak (if anywhere).

Verdict:
- If B1 profile drops below -6 dB within a few mm of the peak → metric bug,
  recover by adding a fallback "no half-max crossing" → return NaN/inf.
- If B1 profile stays within -6 dB of peak across most of the image → B1
  really is blurring; retrain needed.

CLI:
  python -m unet_ag.scripts.diagnose_b1_fwhm \\
      --ckpt results_v2/b1/checkpoints/best.pt \\
      --data_root "datasets/Field II/field_ii_data_multiangle" \\
      --target_root results_schiffner_targets_fieldii \\
      --out_dir results_v2/b1/diagnose \\
      --max_phantoms 2 --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..data.manifest import field_ii_split
from ..data.rf_loader import FieldIIMultiAngleLoader
from ..eval.evaluate_fieldii import (
    _build_grid,
    _envelope_log_compress,
    _load_b1_ckpt,
    _load_target,
    _reconstruct_b1,
)
from ..eval.masks_fieldii import points_wire_targets


def _lateral_profile_db(img: np.ndarray, peak_idx: tuple[int, int]) -> tuple[np.ndarray, float]:
    z0, x0 = peak_idx
    profile = img[z0, :].astype(np.float64)
    peak = float(profile.max())
    if peak <= 0.0:
        return np.full_like(profile, -np.inf), peak
    return 20.0 * np.log10(profile / peak + 1e-30), peak


def _within_db_extent(profile_db: np.ndarray, peak_idx_local: int,
                      threshold_db: float = -6.02) -> tuple[int, int, bool, bool]:
    """Find the lateral extent where the profile stays above threshold.

    Returns (left_idx, right_idx, left_crossed, right_crossed).
    If left_crossed is False, the profile never dropped to threshold on the left side.
    """
    n = profile_db.size
    li = peak_idx_local
    while li > 0 and profile_db[li] >= threshold_db:
        li -= 1
    left_crossed = profile_db[li] < threshold_db

    ri = peak_idx_local
    while ri < n - 1 and profile_db[ri] >= threshold_db:
        ri += 1
    right_crossed = profile_db[ri] < threshold_db

    return li, ri, left_crossed, right_crossed


def _find_wire_peak(env: np.ndarray, wire_z_mm: float, wire_x_mm: float,
                    grid_z_mm: np.ndarray, grid_x_mm: np.ndarray,
                    search_radius_mm: float = 1.5) -> tuple[int, int]:
    dz_mm = float(grid_z_mm[1] - grid_z_mm[0])
    dx_mm = float(grid_x_mm[1] - grid_x_mm[0])
    rz = int(round(search_radius_mm / dz_mm))
    rx = int(round(search_radius_mm / dx_mm))
    iz0 = int(np.argmin(np.abs(grid_z_mm - wire_z_mm)))
    ix0 = int(np.argmin(np.abs(grid_x_mm - wire_x_mm)))
    z_lo, z_hi = max(0, iz0 - rz), min(env.shape[0], iz0 + rz + 1)
    x_lo, x_hi = max(0, ix0 - rx), min(env.shape[1], ix0 + rx + 1)
    sub = env[z_lo:z_hi, x_lo:x_hi]
    local = np.unravel_index(int(np.argmax(sub)), sub.shape)
    return (z_lo + local[0], x_lo + local[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--target_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--nz", type=int, default=612)
    parser.add_argument("--nx", type=int, default=388)
    parser.add_argument("--max_phantoms", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"device = {device}")
    print(f"ckpt   = {args.ckpt}")

    model = _load_b1_ckpt(args.ckpt, device)
    loader = FieldIIMultiAngleLoader(args.data_root, phantom_type="points",
                                     keep_all_angles=False)
    splits = field_ii_split(args.data_root, phantom_types=["points"])
    val_files = splits["points"].val[: args.max_phantoms]
    name_to_idx = {p.stem: i for i, p in enumerate(loader.file_list)}
    val_indices = [name_to_idx[p.stem] for p in val_files if p.stem in name_to_idx]
    print(f"phantoms = {len(val_indices)}")

    for idx in val_indices:
        sample = loader[idx]
        pid = sample.phantom_id
        mat_path = loader.file_list[idx]
        grid_z, grid_x = _build_grid(sample.geometry, args.nz, args.nx)
        grid_z_m = grid_z.numpy()
        grid_x_m = grid_x.numpy()
        dz_mm = float((grid_z_m[1] - grid_z_m[0]) * 1e3)
        dx_mm = float((grid_x_m[1] - grid_x_m[0]) * 1e3)

        pred_env, das_env = _reconstruct_b1(model, sample, grid_z, grid_x, device)
        target_raw = _load_target(args.target_root, pid, "points", args.nz, args.nx)
        schiffner_env = _envelope_log_compress(target_raw) if target_raw is not None else None

        # Save images as .npy (lighter than PNG and lets us re-plot later)
        out_phantom = args.out_dir / pid
        out_phantom.mkdir(parents=True, exist_ok=True)
        np.save(out_phantom / "b1_envlog.npy", pred_env)
        np.save(out_phantom / "das1pw_envlog.npy", das_env)
        if schiffner_env is not None:
            np.save(out_phantom / "schiffner75_envlog.npy", schiffner_env)

        wires = points_wire_targets(mat_path)               # returns (z_mm, x_mm)
        print(f"\n=== {pid} — {len(wires)} wires ===")
        print(f"grid: Nz={args.nz} Nx={args.nx}, dz={dz_mm:.3f} mm, dx={dx_mm:.3f} mm")
        print(f"image stats:")
        for name, im in [("B1", pred_env), ("DAS_1PW", das_env)]:
            print(f"  {name:10s} min={im.min():.3f} max={im.max():.3f} mean={im.mean():.3f} std={im.std():.3f}")
        if schiffner_env is not None:
            print(f"  Schiffner75  min={schiffner_env.min():.3f} max={schiffner_env.max():.3f} mean={schiffner_env.mean():.3f} std={schiffner_env.std():.3f}")

        # Pick 4 wires spread in depth (sort by z, sample evenly)
        wires_sorted = sorted(wires, key=lambda w: w[0])
        n = len(wires_sorted)
        sel = [wires_sorted[i * (n - 1) // 3] for i in range(4)] if n >= 4 else wires_sorted
        for wi, (zw_mm, xw_mm) in enumerate(sel):
            print(f"\nwire #{wi}: expected ({zw_mm:.1f}, {xw_mm:.1f}) mm")
            for name, im in [("B1", pred_env), ("DAS_1PW", das_env),
                             ("Schiffner75", schiffner_env)]:
                if im is None:
                    continue
                peak = _find_wire_peak(im, zw_mm, xw_mm,
                                       grid_z_m * 1e3, grid_x_m * 1e3,
                                       search_radius_mm=1.5)
                profile_db, peak_val = _lateral_profile_db(im, peak)
                li, ri, lc, rc = _within_db_extent(profile_db, int(peak[1]))
                lat_extent_mm = (ri - li) * dx_mm
                peak_z_mm = peak[0] * dz_mm + float(grid_z_m[0] * 1e3)
                peak_x_mm = peak[1] * dx_mm + float(grid_x_m[0] * 1e3)
                # Sample of profile values at offsets -5, -2, 0, +2, +5 mm from peak
                offsets_mm = [-5, -2, 0, 2, 5]
                samples = []
                for off in offsets_mm:
                    ix = peak[1] + int(round(off / dx_mm))
                    if 0 <= ix < profile_db.size:
                        samples.append(f"{off:+d}:{profile_db[ix]:+.1f}")
                    else:
                        samples.append(f"{off:+d}:oob")
                print(f"  {name:12s} peak_val={peak_val:.3f} at ({peak_z_mm:.1f},{peak_x_mm:.1f})mm "
                      f"crossed L={lc} R={rc}  -6dB-extent={lat_extent_mm:.1f}mm  "
                      f"profile(dB)[{','.join(samples)}]")

    print("\nDONE.  Inspect .npy in out_dir or load them for plotting.")


if __name__ == "__main__":
    main()
