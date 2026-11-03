"""Generate Schiffner Fd-F 75 PW reconstructions on the 4 PICMUS phantoms.

Saves, per phantom:
  - <out>/<key>_schiffner75pw.npy   (Nz, Nx) complex64 compounded image
  - <out>/<key>_schiffner75pw.png   B-mode preview (-60 dB DR)

Plus:
  - <out>/schiffner_fdf_75pw_picmus_4panel.png   2x2 grid for slide use

GPU-enabled (auto). On TITAN RTX the full job runs in ~2 min total.
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

# Make the Schiffner library importable from anywhere.
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]          # repository root
sys.path.insert(0, str(_REPO / "code"))

from schiffner_das import (  # noqa: E402
    GratingAngleLB,
    das_pw,
    load_picmus_hdf5,
    load_picmus_scan,
    to_bmode_db,
)

DYNAMIC_RANGE_DB = 60.0


@dataclass
class PhantomSpec:
    key: str
    label: str
    rf_path: str
    scan_path: str


PHANTOMS = [
    PhantomSpec(
        key="sim_resolution",
        label="Simulation — Resolution",
        rf_path="database/simulation/resolution_distorsion/resolution_distorsion_simu_dataset_rf.hdf5",
        scan_path="database/simulation/resolution_distorsion/resolution_distorsion_simu_scan.hdf5",
    ),
    PhantomSpec(
        key="sim_contrast",
        label="Simulation — Contrast",
        rf_path="database/simulation/contrast_speckle/contrast_speckle_simu_dataset_rf.hdf5",
        scan_path="database/simulation/contrast_speckle/contrast_speckle_simu_scan.hdf5",
    ),
    PhantomSpec(
        key="exp_resolution",
        label="Experimental — Resolution",
        rf_path="database/experiments/resolution_distorsion/resolution_distorsion_expe_dataset_rf.hdf5",
        scan_path="database/experiments/resolution_distorsion/resolution_distorsion_expe_scan.hdf5",
    ),
    PhantomSpec(
        key="exp_contrast",
        label="Experimental — Contrast",
        rf_path="database/experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5",
        scan_path="database/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5",
    ),
]


def beamform_75pw_schiffner(ds, gx, gz, *, device: str, z_chunk: int = 32):
    """Coherently compound all available PWs with Schiffner Fd-F."""
    fn_fdf = GratingAngleLB(60, 1.5)
    images = []
    n = ds.n_angles
    for aidx in range(n):
        angle_rad = float(ds.steering_angles_rad[aidx])
        res = das_pw(
            gx, gz,
            ds.rf_for_angle(aidx),
            ds.f_s, angle_rad,
            ds.element_width, ds.element_pitch, ds.c_avg,
            f_bounds=(ds.f_lb, ds.f_ub),
            index_t0=ds.index_t0,
            F_number=fn_fdf,
            z_chunk=z_chunk,
            device=device,
        )
        images.append(res.image)
    return torch.stack(images, dim=0).sum(dim=0)


def save_bmode_single(image, gx, gz, title, save_path):
    db = to_bmode_db(image, DYNAMIC_RANGE_DB)
    xmm = gx.detach().cpu().numpy() * 1e3
    zmm = gz.detach().cpu().numpy() * 1e3
    fig, ax = plt.subplots(figsize=(5.5, 7))
    im = ax.imshow(
        db,
        extent=[xmm[0], xmm[-1], zmm[-1], zmm[0]],
        aspect="auto", cmap="gray",
        vmin=-DYNAMIC_RANGE_DB, vmax=0,
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Lateral (mm)")
    ax.set_ylabel("Axial (mm)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="dB")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_4panel(images, labels, grids_xz, save_path, *, suptitle):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for ax, img, lbl, (gx, gz) in zip(axes.ravel(), images, labels, grids_xz):
        db = to_bmode_db(img, DYNAMIC_RANGE_DB)
        xmm = gx.detach().cpu().numpy() * 1e3
        zmm = gz.detach().cpu().numpy() * 1e3
        im = ax.imshow(
            db,
            extent=[xmm[0], xmm[-1], zmm[-1], zmm[0]],
            aspect="auto", cmap="gray",
            vmin=-DYNAMIC_RANGE_DB, vmax=0,
        )
        ax.set_title(lbl, fontsize=12)
        ax.set_xlabel("Lateral (mm)")
        ax.set_ylabel("Axial (mm)")
        plt.colorbar(im, ax=ax, shrink=0.75, label="dB")
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--picmus", default="datasets/PICMUS", type=Path)
    p.add_argument("--out", default="results_v2/schiffner_fdf_75pw_picmus", type=Path)
    p.add_argument("--quick", action="store_true",
                   help="Subsample grid by 4x for fast preview")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--z-chunk", type=int, default=32)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"PICMUS root  : {args.picmus.resolve()}")
    print(f"Output dir   : {args.out.resolve()}")
    print(f"Device       : {args.device}")
    print(f"Quick mode   : {args.quick}")
    print()

    images_for_panel = []
    labels_for_panel = []
    grids_for_panel = []

    t_total = time.perf_counter()
    for spec in PHANTOMS:
        print(f"── {spec.label} ({spec.key}) ─────────────────────")
        t0 = time.perf_counter()

        ds = load_picmus_hdf5(args.picmus / spec.rf_path)
        grid = load_picmus_scan(args.picmus / spec.scan_path)
        gx, gz = grid.x, grid.z
        if args.quick:
            gx, gz = gx[::4], gz[::4]
        print(f"  Grid: {len(gz)} x {len(gx)}, angles: {ds.n_angles}")

        img = beamform_75pw_schiffner(ds, gx, gz,
                                      device=args.device, z_chunk=args.z_chunk)
        npy_path = args.out / f"{spec.key}_schiffner75pw.npy"
        png_path = args.out / f"{spec.key}_schiffner75pw.png"
        np.save(npy_path, img.detach().cpu().numpy())
        save_bmode_single(
            img, gx, gz,
            title=f"Schiffner Fd-F 75 PW — {spec.label}",
            save_path=png_path,
        )
        dt = time.perf_counter() - t0
        print(f"  {dt:.1f}s  →  {png_path.name}")

        images_for_panel.append(img)
        labels_for_panel.append(spec.label)
        grids_for_panel.append((gx, gz))

    panel_path = args.out / "schiffner_fdf_75pw_picmus_4panel.png"
    save_4panel(
        images_for_panel, labels_for_panel, grids_for_panel,
        save_path=panel_path,
        suptitle="Schiffner Fd-F  ·  75 compounded plane waves  ·  PICMUS",
    )
    print(f"\n4-panel    →  {panel_path}")
    print(f"Total      :  {time.perf_counter() - t_total:.1f}s")


if __name__ == "__main__":
    main()
