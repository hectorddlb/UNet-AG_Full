"""Precompute DAS 75 PW compounded targets for the multi-angle Field II dataset.

Output structure mirrors ``results_schiffner_targets_fieldii/``:

    results_v2/das75/<ptype>/<phantom_id>/compounded.npy   # (Nz, Nx) float32

Designed for the cluster as a slurm array job — see
``unet_ag/slurm/precompute_das_75pw.sbatch``. CPU-only.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]          # repository root
sys.path.insert(0, str(_REPO / "code"))

from unet_ag.data.rf_loader import FieldIIMultiAngleLoader
from unet_ag.physics.classical.das_75pw import das_75pw


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
    parser.add_argument("--nz", type=int, default=612)
    parser.add_argument("--nx", type=int, default=388)
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
            angles_rad = np.deg2rad(sample.angles_deg).astype(np.float64)
            rf_all_np = sample.rf_all.cpu().numpy()
            img = das_75pw(rf_all_np, angles_rad, sample.geometry, grid_z, grid_x, f_number=args.f_number)
            out_dir = args.out_root / pt / sample.phantom_id
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "compounded.npy", img)
            print(f"[{pt}/{sample.phantom_id}] done in {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
