"""Precompute Schiffner DAS-Fd-F targets for Field II multi-angle phantoms.

For each phantom × steering-angle, run ``schiffner_das.das_pw`` to produce a
single-angle beamformed image, then coherent-sum across all 75 angles to
obtain the compounded target used as supervised label for Idea 1.2
(Strategy A residual with ``H_Fdf``).

This replaces the LSQR baseline used in earlier phases. Schiffner DAS is the
gold-standard reference: it implements the same frequency-dependent F-number
(GratingAngleLB) that our ``OpHFdf`` operator uses, so the target lies in
the range of the forward operator that the EDN will learn to correct.

t_start handling
----------------
Field II returns rf_data starting at ``t = t_start`` (potentially negative
for steered angles). Schiffner accepts an ``index_t0`` parameter that
encodes this offset natively — no RF reshape needed (cleaner than the
``apply_t_start_shift`` used in the LSQR path).

Normalization
-------------
Raw Field II rf_data is at voltage scale ~1e-23. The LSQR pipeline divides
each angle's RF by ``max|rf_a|`` before solving, so targets land at unit-ish
scale (~1e-1) ready for training. We mirror that here: per-angle normalize
before DAS, save the scale to the CSV, and accumulate the normalized images
for the compounded target. This keeps Schiffner targets drop-in compatible
with the existing LSQR-target consumers (e.g. train_strategy_v3_multiangle).

Output layout::

    <output_dir>/{points,cysts}/<stem>/
        compounded.npy           (Nz, Nx) float32 — real(Σ_a image_a)
        per_angle/angle_XXX.npy  (Nz, Nx) float32 — optional, --save_per_angle
    <output_dir>/stats_<RANGE>.csv

Usage::

    python precompute_schiffner_targets_fieldii.py \\
        --data_root "datasets/Field II/field_ii_data_multiangle" \\
        --output_dir results_schiffner_targets_fieldii \\
        --range_start 0 --range_end 600

Resumable: per-phantom outputs are skipped if ``compounded.npy`` exists.
SLURM parallelism: use --range_start / --range_end in an array job.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

# Schiffner package lives in hyphenated dir; same path trick as
# zhang_dnn/measurement_operator_fdf.py:44.
_SCHIFFNER_ROOT = Path(__file__).resolve().parents[3] / "code"
if str(_SCHIFFNER_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCHIFFNER_ROOT))

from schiffner_das.beamforming import das_pw  # noqa: E402
from schiffner_das.f_numbers import GratingAngleLB  # noqa: E402
from schiffner_das.normalizations import NormalizationOn  # noqa: E402
from schiffner_das.windows import Tukey  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="root with points/ and cysts/ subdirs of multi-angle .mat")
    p.add_argument("--output_dir", default="results_schiffner_targets_fieldii")
    p.add_argument("--phantom_types", nargs="+", default=["points", "cysts"])
    p.add_argument("--nz", type=int, default=612)
    p.add_argument("--nx", type=int, default=388)
    p.add_argument("--z_min", type=float, default=5e-3)
    p.add_argument("--z_max", type=float, default=50e-3)
    p.add_argument("--x_min", type=float, default=-19.2e-3)
    p.add_argument("--x_max", type=float, default=19.2e-3)
    p.add_argument("--pitch", type=float, default=300e-6)
    p.add_argument("--fs", type=float, default=20.832e6)
    p.add_argument("--c", type=float, default=1540.0)
    p.add_argument("--n_elements", type=int, default=128)
    p.add_argument("--element_width", type=float, default=270e-6,
                   help="Field II default for the 300µm-pitch probe is 270µm (90% of pitch).")
    p.add_argument("--f_number_grating_deg", type=float, default=60.0)
    p.add_argument("--f_number_lb", type=float, default=1.5)
    p.add_argument("--tukey_roll", type=float, default=0.2)
    p.add_argument("--range_start", type=int, default=0,
                   help="phantom index (across all collected phantoms) to start from (inclusive)")
    p.add_argument("--range_end", type=int, default=None,
                   help="phantom index to stop at (exclusive). Default = all.")
    p.add_argument("--max_angles", type=int, default=None,
                   help="cap angles per phantom (debug)")
    p.add_argument("--save_per_angle", action="store_true",
                   help="also persist each angle image (43 GB total at full grid; off by default)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--z_chunk", type=int, default=32,
                   help="passed to schiffner_das.das_pw (peak-memory control)")
    p.add_argument("--x_batch", type=int, default=None,
                   help="passed to schiffner_das.das_pw (None=auto)")
    return p.parse_args()


def collect_files(data_root: Path, phantom_types: list[str]) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    for pt in phantom_types:
        sub = data_root / pt
        if not sub.exists():
            print(f"  WARNING: {sub} not found, skipping")
            continue
        for fp in sorted(sub.glob("*.mat")):
            pairs.append((pt, fp))
    return pairs


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    grid_z = torch.linspace(args.z_min, args.z_max, args.nz, dtype=torch.float64)
    grid_x = torch.linspace(args.x_min, args.x_max, args.nx, dtype=torch.float64)

    # Shared Schiffner objects (stateless: rebuilt cost is trivial but reuse is cleaner)
    f_number = GratingAngleLB(args.f_number_grating_deg, args.f_number_lb)
    window = Tukey(args.tukey_roll)
    normalization = NormalizationOn()

    files = collect_files(data_root, args.phantom_types)
    if not files:
        raise RuntimeError(f"No .mat files under {data_root}")

    range_end = args.range_end if args.range_end is not None else len(files)
    files = files[args.range_start:range_end]
    print(f"Processing phantoms [{args.range_start}, {range_end}) — {len(files)} files")
    print(f"Device: {device}")
    print(f"Grid: ({args.nz}, {args.nx}) over z=[{args.z_min*1e3:.1f}, {args.z_max*1e3:.1f}] mm, "
          f"x=[{args.x_min*1e3:+.1f}, {args.x_max*1e3:+.1f}] mm")

    csv_path = output_dir / f"stats_{args.range_start:04d}_{range_end:04d}.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["phantom_type", "stem", "angle_idx", "angle_deg",
                     "t_start_us", "index_t0", "scale", "img_absmax", "das_s"])

    t0 = time.perf_counter()
    n_done_phantoms = 0
    n_total_angles = 0

    # First pass: total angle-count for ETA
    for pt, fp in files:
        m = sio.loadmat(fp, variable_names=["angles_deg"])
        n_ang = np.array(m["angles_deg"]).reshape(-1).size
        if args.max_angles:
            n_ang = min(n_ang, args.max_angles)
        n_total_angles += n_ang
    print(f"Total DAS-Fd-F runs to do (worst case): {n_total_angles}\n")
    n_done_angles = 0

    for pt, fp in files:
        ph_dir = output_dir / pt / fp.stem
        comp_path = ph_dir / "compounded.npy"
        if comp_path.exists():
            # already done; tally angles for ETA accuracy
            m = sio.loadmat(fp, variable_names=["angles_deg"])
            n_ang = np.array(m["angles_deg"]).reshape(-1).size
            if args.max_angles:
                n_ang = min(n_ang, args.max_angles)
            n_done_angles += n_ang
            n_done_phantoms += 1
            print(f"=== {pt}/{fp.stem} — skipped (compounded.npy exists)")
            continue

        m = sio.loadmat(fp)
        rf_all = m["rf_data"]                                  # (D, K, n_angles), float32
        angles_deg = np.array(m["angles_deg"]).reshape(-1)
        ts_arr = np.array(m["t_start"]).reshape(-1)
        fs_local = float(np.array(m["fs"]).squeeze())
        if abs(fs_local - args.fs) > 1.0:
            print(f"  NOTE: fs mismatch — file={fs_local:.0f}, arg={args.fs:.0f}; "
                  f"using file fs.")
        fs = fs_local

        n_angles = rf_all.shape[2]
        if args.max_angles:
            n_angles = min(n_angles, args.max_angles)

        ph_dir.mkdir(parents=True, exist_ok=True)
        if args.save_per_angle:
            (ph_dir / "per_angle").mkdir(exist_ok=True)

        print(f"\n=== {pt}/{fp.stem}  (n_angles={n_angles}) ===")
        compounded = torch.zeros(args.nz, args.nx, dtype=torch.cdouble, device=device)

        for a in range(n_angles):
            angle_deg = float(angles_deg[a])
            angle_rad = math.radians(angle_deg)
            t_start = float(ts_arr[a])
            index_t0 = int(round(t_start * fs))

            rf_np = rf_all[:, :, a].astype(np.float64)         # (D, K)
            scale = float(np.max(np.abs(rf_np))) + 1e-30
            rf = torch.from_numpy(rf_np / scale)                # normalized

            t_d0 = time.perf_counter()
            result = das_pw(
                positions_x=grid_x,
                positions_z=grid_z,
                data_RF=rf,
                f_s=fs,
                steering_angle=angle_rad,
                element_width=args.element_width,
                element_pitch=args.pitch,
                c_0=args.c,
                index_t0=index_t0,
                window=window,
                F_number=f_number,
                normalization=normalization,
                z_chunk=args.z_chunk,
                x_batch=args.x_batch,
                device=device,
            )
            t_d = time.perf_counter() - t_d0

            img = result.image                                  # (Nz, Nx) complex
            compounded = compounded + img.to(compounded.device)

            if args.save_per_angle:
                np.save(
                    ph_dir / "per_angle" / f"angle_{a:03d}.npy",
                    img.real.cpu().numpy().astype(np.float32),
                )

            img_absmax = float(img.abs().max().item())
            writer.writerow([pt, fp.stem, a, f"{angle_deg:+.4f}",
                             f"{t_start*1e6:+.4f}", index_t0,
                             f"{scale:.6e}", f"{img_absmax:.6e}", f"{t_d:.2f}"])
            csv_file.flush()
            n_done_angles += 1

            if a == 0 or (a + 1) % 25 == 0 or a == n_angles - 1:
                elapsed = time.perf_counter() - t0
                avg = elapsed / max(n_done_angles, 1)
                remaining = (n_total_angles - n_done_angles) * avg
                print(f"  [{n_done_angles}/{n_total_angles}] ang={angle_deg:+.2f}° "
                      f"t0_idx={index_t0:+4d} das={t_d:.2f}s  ETA {remaining/60:.1f}min")

        # Save coherent compounded real part
        np.save(comp_path, compounded.real.cpu().numpy().astype(np.float32))
        n_done_phantoms += 1

    csv_file.close()
    print(f"\nDone. Stats: {csv_path}")
    print(f"Phantoms processed: {n_done_phantoms}/{len(files)}")
    print(f"Total time: {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
