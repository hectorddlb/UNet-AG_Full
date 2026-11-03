"""Precompute MV 1 PW baselines for the multi-angle Field II dataset.

Output:
    results_v2/mv/<ptype>/<phantom_id>/mv_1pw.npy   # (Nz, Nx) float32
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from unet_ag.data.rf_loader import FieldIIMultiAngleLoader, select_1pw_angle
from unet_ag.physics.classical.mv_beamformer import mv_1pw


def _build_grid(geometry: dict, nz: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    c = float(geometry["speed_of_sound"])
    fs = float(geometry["sampling_freq_hz"])
    ns = int(geometry["n_samples"])
    K = int(geometry["n_elements"])
    pitch = float(geometry["pitch_m"])
    z_max = (ns - 1) * c / (2.0 * fs)
    grid_z = np.linspace(5e-3, z_max, nz, dtype=np.float64)
    grid_x = np.linspace(-K * pitch / 2.0, +K * pitch / 2.0, nx, dtype=np.float64)
    return grid_z, grid_x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--phantom_type", choices=["points", "cysts", "both"], default="both")
    parser.add_argument("--nz", type=int, default=384)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--subarray_len", type=int, default=64)
    parser.add_argument("--diag_load", type=float, default=1e-3)
    parser.add_argument("--f_number", type=float, default=1.5)
    parser.add_argument("--array_idx", type=int, default=0)
    parser.add_argument("--array_total", type=int, default=1)
    args = parser.parse_args()

    ptypes = ["points", "cysts"] if args.phantom_type == "both" else [args.phantom_type]
    for pt in ptypes:
        loader = FieldIIMultiAngleLoader(args.data_root, phantom_type=pt, keep_all_angles=True, normalize=False)
        for idx in range(args.array_idx, len(loader), args.array_total):
            t0 = time.perf_counter()
            sample = loader[idx]
            grid_z, grid_x = _build_grid(sample.geometry, args.nz, args.nx)
            i_1pw = select_1pw_angle(sample.angles_deg)
            rf_1pw = sample.rf_all[:, :, i_1pw].cpu().numpy()
            angle_rad = float(np.deg2rad(sample.angles_deg[i_1pw]))
            img = mv_1pw(
                rf_1pw, sample.geometry, grid_z, grid_x,
                subarray_len=args.subarray_len, diag_load=args.diag_load,
                f_number=args.f_number, angle_rad=angle_rad,
            )
            out_dir = args.out_root / pt / sample.phantom_id
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "mv_1pw.npy", img)
            print(f"[{pt}/{sample.phantom_id}] mv done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
