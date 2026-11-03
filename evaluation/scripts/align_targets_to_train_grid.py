"""Resample the Schiffner targets to the axial grid used by the trainer.

BUG IT FIXES
------------
``precompute_schiffner_targets_fieldii*.py`` beamforms on a FIXED grid
z in [5, 50] mm x 612 rows (dz = 0.0736 mm).  The trainer and the evaluator,
on the other hand, build theirs with ``evaluate_fieldii._build_grid``:

    z_max = (n_samples - 1) * c / (2 * fs)      # 54.22 mm with n_samples=1468
    grid_z = linspace(5e-3, z_max, nz)          # dz = 0.0806 mm

Both have 612 rows, so ``_load_target`` does NOT reinterpolate (it only does
so if the *shape* changes) and the target is overlaid on the input **row by
row**. The result is a 9.4 % axial scale error: an echo at 44 mm falls 45 rows
(3.7 mm) lower in the target than in the DAS-1PW input, far above the axial
FWHM (~1.6 mm).  With blurry targets the partial overlap hid it; with the
sharp targets (corrected TX convention) target and input no longer overlap and
the network minimizes the loss by **blurring**.

WHAT IT DOES
------------
For each ``compounded.npy`` (real, RF-like):
  1. analytic signal via Hilbert;
  2. demodulation to baseband with the carrier exp(-j 2pi fc 2z/c) -- the
     complex envelope is ~13x oversampled, so interpolating it is exact,
     whereas interpolating the RF (4.0 samples/cycle) would not be;
  3. linear interpolation to the trainer grid;
  4. remodulation, and the real part is saved (same format as the input).
Rows with z > z_max_target (50 mm) are filled with zeros: there is no target
information there (the deepest scatterers are at ~44 mm).

Usage
-----
    python evaluation/scripts/align_targets_to_train_grid.py \
        --src results/targets_schiffner_fixed \
        --dst results/targets_schiffner_gridfix \
        --data_root "datasets/Field II/field_ii_data_multiangle"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import hilbert


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--data_root", type=Path, required=True,
                   help="root of the Field II .mat files (for c, fs, fc, n_samples)")
    p.add_argument("--phantom_types", nargs="+", default=["points", "cysts"])
    p.add_argument("--src_z_min", type=float, default=5e-3)
    p.add_argument("--src_z_max", type=float, default=50e-3)
    p.add_argument("--train_z_min", type=float, default=5e-3)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def train_grid_z(geom_mat: dict, nz: int, z_min: float) -> np.ndarray:
    """Exact replica of evaluate_fieldii._build_grid (z axis)."""
    c = float(np.array(geom_mat["c"]).ravel()[0])
    fs = float(np.array(geom_mat["fs"]).ravel()[0])
    ns = int(np.array(geom_mat["rf_data"]).shape[0])
    z_max = (ns - 1) * c / (2.0 * fs)
    return np.linspace(z_min, z_max, nz, dtype=np.float64)


def resample_axial(rf: np.ndarray, z_src: np.ndarray, z_dst: np.ndarray,
                   fc: float, c: float, taper_m: float = 2e-3) -> np.ndarray:
    """Resample rf (Nz, Nx) from z_src to z_dst by demodulation + linear interp.

    ``taper_m`` applies a raised cosine over the last millimetres of the
    valid segment so that the zero padding (z > z_src[-1]) does not create a
    step: the subsequent Hilbert would turn it into ringing. The deepest
    scatterers are at 44 mm (points) / 45 mm (cysts), so a 48 -> 50 mm taper
    does not erase useful signal.
    """
    analytic = hilbert(rf.astype(np.float64), axis=0)            # (Nz, Nx) complex
    k = 2.0 * np.pi * fc * 2.0 / c                                # rad/m (round trip)
    base = analytic * np.exp(-1j * k * z_src[:, None])            # baseband, smooth

    inside = (z_dst >= z_src[0]) & (z_dst <= z_src[-1])
    out = np.zeros((z_dst.size, rf.shape[1]), dtype=np.complex128)
    zi = z_dst[inside]
    for ix in range(rf.shape[1]):
        out[inside, ix] = (np.interp(zi, z_src, base[:, ix].real)
                           + 1j * np.interp(zi, z_src, base[:, ix].imag))
    out *= np.exp(+1j * k * z_dst[:, None])

    if taper_m > 0:
        w = np.ones(z_dst.size)
        w[~inside] = 0.0
        edge = z_src[-1]
        ramp = inside & (z_dst > edge - taper_m)
        w[ramp] = 0.5 * (1.0 + np.cos(np.pi * (z_dst[ramp] - (edge - taper_m)) / taper_m))
        out *= w[:, None]
    return out.real.astype(np.float32)


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    n_done = n_skip = 0

    for pt in args.phantom_types:
        src_dir = args.src / pt
        if not src_dir.exists():
            print(f"  WARNING: {src_dir} does not exist, skipped")
            continue
        for ph_dir in sorted(d for d in src_dir.iterdir() if d.is_dir()):
            src_npy = ph_dir / "compounded.npy"
            if not src_npy.exists():
                continue
            dst_npy = args.dst / pt / ph_dir.name / "compounded.npy"
            if dst_npy.exists() and not args.overwrite:
                n_skip += 1
                continue

            mat_path = args.data_root / pt / f"{ph_dir.name}.mat"
            if not mat_path.exists():
                print(f"  WARNING: {mat_path} missing, skipping {ph_dir.name}")
                continue
            m = sio.loadmat(mat_path, variable_names=["c", "fs", "fc", "rf_data"])
            c = float(np.array(m["c"]).ravel()[0])
            fc = float(np.array(m["fc"]).ravel()[0])

            rf = np.load(src_npy)
            nz, _ = rf.shape
            z_src = np.linspace(args.src_z_min, args.src_z_max, nz, dtype=np.float64)
            z_dst = train_grid_z(m, nz, args.train_z_min)

            out = resample_axial(rf, z_src, z_dst, fc, c)
            dst_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(dst_npy, out)
            n_done += 1
            if n_done % 25 == 0:
                print(f"  {pt}: {n_done} written ({time.perf_counter()-t0:.0f} s)")

    print(f"\nDone: {n_done} resampled, {n_skip} already existed "
          f"({(time.perf_counter()-t0)/60:.1f} min) -> {args.dst}")


if __name__ == "__main__":
    main()
