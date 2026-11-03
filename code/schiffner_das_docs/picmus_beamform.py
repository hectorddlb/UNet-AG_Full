#!/usr/bin/env python3
"""
Baseline 2 — Fd-F beamforming on PICMUS datasets.

This script applies Schiffner's frequency-dependent F-number to all four
PICMUS phantoms and generates validation figures.

Experiments performed per phantom:
  A. F-number comparison (1 PW, 0°): Full aperture vs Fixed F=1.5 vs Fd-F
  B. CPWC quality ladder (Fd-F): 1 PW → 3 PW → 11 PW → 75 PW

Usage
-----
    # Full grid (387×609) — ~17 min per beamform on CPU
    python picmus_beamform.py --picmus PICMUS --out results_picmus

    # Quick validation (1/4 grid) — ~2 min per beamform
    python picmus_beamform.py --picmus PICMUS --out results_picmus --quick

    # Single phantom only
    python picmus_beamform.py --picmus PICMUS --out results_picmus --quick --only simu_resolution

References
----------
[1] Schiffner & Schmitz, IEEE TUFFC 70(9), 2023.
[2] Liebgott et al., "PICMUS," IEEE IUS 2016.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schiffner_das import (
    ConstantFNumber,
    GratingAngleLB,
    NormalizationOff,
    NormalizationOn,
    Tukey,
    das_pw,
    load_picmus_hdf5,
    load_picmus_scan,
    to_bmode_db,
)


# =====================================================================
# Phantom registry
# =====================================================================
@dataclass
class PhantomSpec:
    """Specification for one PICMUS phantom."""
    key: str                # short name for CLI and filenames
    label: str              # display name for figure titles
    rf_path: str            # relative path to RF HDF5
    scan_path: str          # relative path to scan HDF5
    phantom_path: str       # relative path to phantom HDF5


PHANTOMS = [
    PhantomSpec(
        key="simu_resolution",
        label="Simulation — Resolution",
        rf_path="database/simulation/resolution_distorsion/resolution_distorsion_simu_dataset_rf.hdf5",
        scan_path="database/simulation/resolution_distorsion/resolution_distorsion_simu_scan.hdf5",
        phantom_path="database/simulation/resolution_distorsion/resolution_distorsion_simu_phantom.hdf5",
    ),
    PhantomSpec(
        key="simu_contrast",
        label="Simulation — Contrast",
        rf_path="database/simulation/contrast_speckle/contrast_speckle_simu_dataset_rf.hdf5",
        scan_path="database/simulation/contrast_speckle/contrast_speckle_simu_scan.hdf5",
        phantom_path="database/simulation/contrast_speckle/contrast_speckle_simu_phantom.hdf5",
    ),
    PhantomSpec(
        key="expe_resolution",
        label="Experimental — Resolution",
        rf_path="database/experiments/resolution_distorsion/resolution_distorsion_expe_dataset_rf.hdf5",
        scan_path="database/experiments/resolution_distorsion/resolution_distorsion_expe_scan.hdf5",
        phantom_path="database/experiments/resolution_distorsion/resolution_distorsion_expe_phantom.hdf5",
    ),
    PhantomSpec(
        key="expe_contrast",
        label="Experimental — Contrast",
        rf_path="database/experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5",
        scan_path="database/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5",
        phantom_path="database/experiments/contrast_speckle/contrast_speckle_expe_phantom.hdf5",
    ),
]

# CPWC configurations: (n_angles, label, angle_indices_fn)
# angle_indices_fn receives n_total and returns a list of indices
CPWC_CONFIGS = [
    (1,  "1 PW",  lambda n: [n // 2]),                                  # center (0°)
    (3,  "3 PW",  lambda n: [n // 4, n // 2, 3 * n // 4]),             # ≈ -8°, 0°, +8°
    (11, "11 PW", lambda n: list(np.linspace(0, n - 1, 11).astype(int))),
    (75, "75 PW", lambda n: list(range(n))),
]

DYNAMIC_RANGE_DB = 60.0


# =====================================================================
# Helpers
# =====================================================================
def subsample_grid(x: torch.Tensor, z: torch.Tensor, factor: int):
    """Subsample a grid by a given factor (for quick mode)."""
    return x[::factor], z[::factor]


def select_angle_indices(n_total: int, n_pw: int, indices_fn):
    """Select angle indices for a given CPWC configuration."""
    indices = indices_fn(n_total)
    return [i for i in indices if 0 <= i < n_total]


def beamform_compound(
    ds, grid_x, grid_z, angle_indices, z_chunk=32, **bf_kwargs,
):
    """Beamform and coherently compound a list of steering angles."""
    images = []
    for i, aidx in enumerate(angle_indices):
        angle_rad = float(ds.steering_angles_rad[aidx])
        result = das_pw(
            grid_x, grid_z,
            ds.rf_for_angle(aidx),
            ds.f_s, angle_rad,
            ds.element_width, ds.element_pitch, ds.c_avg,
            f_bounds=(ds.f_lb, ds.f_ub),
            index_t0=ds.index_t0,
            z_chunk=z_chunk,
            **bf_kwargs,
        )
        images.append(result.image)

    compound = torch.stack(images, dim=0).sum(dim=0)
    return compound, images


def plot_grid(
    images, labels, grid_x, grid_z,
    suptitle="", save_path=None, ncols=None,
):
    """Plot a grid of B-mode images."""
    n = len(images)
    if ncols is None:
        ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 7 * nrows),
                             squeeze=False)
    xmm = grid_x.numpy() * 1e3
    zmm = grid_z.numpy() * 1e3

    for i in range(nrows * ncols):
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        if i < n:
            db = to_bmode_db(images[i], DYNAMIC_RANGE_DB)
            im = ax.imshow(
                db,
                extent=[xmm[0], xmm[-1], zmm[-1], zmm[0]],
                aspect="auto", cmap="gray",
                vmin=-DYNAMIC_RANGE_DB, vmax=0,
            )
            ax.set_title(labels[i], fontsize=12)
            ax.set_xlabel("Lateral (mm)")
            if col == 0:
                ax.set_ylabel("Axial (mm)")
            plt.colorbar(im, ax=ax, shrink=0.7, label="dB")
        else:
            ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"  → Saved: {save_path}")
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================
def process_phantom(
    spec: PhantomSpec,
    picmus_root: Path,
    out_dir: Path,
    quick: bool = False,
    skip_cpwc_75: bool = False,
    z_chunk: int = 32,
):
    """Process one PICMUS phantom: F-number comparison + CPWC ladder."""
    header = f"  {spec.label}  ({spec.key})"
    print("\n" + "═" * 70)
    print(header)
    print("═" * 70)

    # ── Load data ────────────────────────────────────────────────
    rf_path = picmus_root / spec.rf_path
    scan_path = picmus_root / spec.scan_path

    if not rf_path.exists():
        print(f"  WARNING: RF file not found: {rf_path} — skipping.")
        return
    if not scan_path.exists():
        print(f"  WARNING: Scan file not found: {scan_path} — skipping.")
        return

    ds = load_picmus_hdf5(rf_path)
    grid = load_picmus_scan(scan_path)
    print(f"  RF: {tuple(ds.data_RF.shape)}  |  Grid: {grid.N_x}×{grid.N_z}")
    print(f"  f_s={ds.f_s/1e6:.3f} MHz  pitch={ds.element_pitch*1e6:.1f} µm  "
          f"c={ds.c_avg:.1f} m/s  t0={ds.index_t0} samples")

    gx, gz = grid.x, grid.z
    if quick:
        gx, gz = subsample_grid(gx, gz, 4)
        print(f"  [QUICK] Grid subsampled to {len(gx)}×{len(gz)}")

    phantom_out = out_dir / spec.key
    phantom_out.mkdir(parents=True, exist_ok=True)

    n_total_angles = ds.n_angles
    idx_0deg = n_total_angles // 2  # center angle ≈ 0°
    angle_0_rad = float(ds.steering_angles_rad[idx_0deg])
    angle_0_deg = float(ds.steering_angles_deg[idx_0deg])

    # ── A. F-number comparison (1 PW at 0°) ──────────────────────
    print(f"\n  ── A. F-number comparison (angle={angle_0_deg:.1f}°) ──")

    fnumber_configs = [
        ("Full aperture (F=0)",  ConstantFNumber(0)),
        ("Fixed F=1.5",          ConstantFNumber(1.5)),
        ("Proposed Fd-F",        GratingAngleLB(60, 1.5)),
    ]

    images_fn = []
    labels_fn = []
    for label, fn in fnumber_configs:
        result = das_pw(
            gx, gz,
            ds.rf_for_angle(idx_0deg),
            ds.f_s, angle_0_rad,
            ds.element_width, ds.element_pitch, ds.c_avg,
            f_bounds=(ds.f_lb, ds.f_ub),
            index_t0=ds.index_t0,
            F_number=fn,
            z_chunk=z_chunk,
        )
        images_fn.append(result.image)
        labels_fn.append(label)

    plot_grid(
        images_fn, labels_fn, gx, gz,
        suptitle=f"{spec.label} — F-number comparison (1 PW, {angle_0_deg:.0f}°)",
        save_path=phantom_out / "A_fnumber_comparison.png",
    )

    # ── B. CPWC quality ladder (with Fd-F) ────────────────────────
    print(f"\n  ── B. CPWC ladder (Fd-F, χ_lb=60°) ──")

    fn_fdf = GratingAngleLB(60, 1.5)
    images_cpwc = []
    labels_cpwc = []

    for n_pw, pw_label, idx_fn in CPWC_CONFIGS:
        if skip_cpwc_75 and n_pw == 75:
            print(f"  [{pw_label}] skipped (--skip-cpwc-75)")
            continue

        indices = select_angle_indices(n_total_angles, n_pw, idx_fn)
        actual_n = len(indices)
        angles_used = [f"{ds.steering_angles_deg[i]:.1f}°" for i in indices]

        print(f"  [{pw_label}] {actual_n} angles: "
              f"[{angles_used[0]}, ..., {angles_used[-1]}]"
              if actual_n > 2 else f"  [{pw_label}] {actual_n} angle(s): {angles_used}")

        compound, _ = beamform_compound(
            ds, gx, gz, indices,
            z_chunk=z_chunk,
            F_number=fn_fdf,
        )
        images_cpwc.append(compound)
        labels_cpwc.append(f"Fd-F {pw_label}")

    if images_cpwc:
        plot_grid(
            images_cpwc, labels_cpwc, gx, gz,
            suptitle=f"{spec.label} — CPWC with Fd-F",
            save_path=phantom_out / "B_cpwc_ladder.png",
        )

    # ── C. Save individual 1-PW Fd-F image for DL pipeline ───────
    # This is the Baseline 2 image that will be compared against
    # Baseline 1 (DAS-1PW) and the proposed model (Zhang+Fd-F).
    fdf_1pw = images_fn[2]  # the "Proposed Fd-F" from experiment A
    save_path_pt = phantom_out / "baseline2_fdf_1pw.pt"
    torch.save({
        "image": fdf_1pw,
        "grid_x": gx,
        "grid_z": gz,
        "angle_deg": angle_0_deg,
        "f_number": "GratingAngleLB(60, 1.5)",
        "phantom": spec.key,
    }, save_path_pt)
    print(f"  → Saved tensor: {save_path_pt}")

    print(f"\n  {spec.key} complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Baseline 2 — Fd-F beamforming on PICMUS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--picmus", default="PICMUS",
        help="Root directory of PICMUS data (default: PICMUS)",
    )
    parser.add_argument(
        "--out", default="results_picmus",
        help="Output directory (default: results_picmus)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Subsample grid by 4× for fast validation (~2 min/beamform)",
    )
    parser.add_argument(
        "--only", default=None,
        help="Process only one phantom (e.g. simu_resolution, expe_contrast)",
    )
    parser.add_argument(
        "--skip-cpwc-75", action="store_true",
        help="Skip the 75-PW compound (very slow on full grid)",
    )
    parser.add_argument(
        "--z-chunk", type=int, default=32,
        help="Z-chunk size for memory control (default: 32)",
    )
    args = parser.parse_args()

    picmus_root = Path(args.picmus)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not picmus_root.exists():
        print(f"ERROR: PICMUS root not found: {picmus_root}")
        sys.exit(1)

    print("=" * 70)
    print("  Baseline 2 — Fd-F Beamforming on PICMUS")
    print("=" * 70)
    print(f"  PICMUS root : {picmus_root.resolve()}")
    print(f"  Output      : {out_dir.resolve()}")
    print(f"  Quick mode  : {args.quick}")
    print(f"  z_chunk     : {args.z_chunk}")

    t_total = time.perf_counter()

    # Select phantoms
    if args.only:
        phantoms = [p for p in PHANTOMS if p.key == args.only]
        if not phantoms:
            valid = ", ".join(p.key for p in PHANTOMS)
            print(f"ERROR: Unknown phantom '{args.only}'. Valid: {valid}")
            sys.exit(1)
    else:
        phantoms = PHANTOMS

    for spec in phantoms:
        process_phantom(
            spec, picmus_root, out_dir,
            quick=args.quick,
            skip_cpwc_75=args.skip_cpwc_75,
            z_chunk=args.z_chunk,
        )

    elapsed = time.perf_counter() - t_total
    hours = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    print("\n" + "=" * 70)
    print(f"  All done!  Total time: {hours}h {mins}m")
    print(f"  Results in: {out_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()