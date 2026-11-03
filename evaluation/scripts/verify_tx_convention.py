"""Verify the plane-wave transmit-time convention for PICMUS in das_pw.

Reconstructs sim_resolution with 3 steering angles (min, 0, max) under three
conventions and compares single-angle point positions against the 0-degree
reference (identical in all variants) and the phantom ground truth.

The correct convention keeps every point at the same (x, z) for all angles,
so the 3-angle compound sharpens instead of smearing.

Run from the repository root:
    python evaluation/scripts/verify_tx_convention.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("THESIS_DATASETS", ROOT / "data"))  # PICMUS/ and Field II/, see data/README.md
sys.path.insert(0, str(ROOT / "code"))

from schiffner_das import (  # noqa: E402
    GratingAngleLB,
    das_pw,
    load_picmus_hdf5,
    load_picmus_scan,
    to_bmode_db,
)

OUT = ROOT / "evaluation" / "verification"
OUT.mkdir(parents=True, exist_ok=True)

RF = DATASETS / "PICMUS/database/simulation/resolution_distorsion/resolution_distorsion_simu_dataset_rf.hdf5"
SCAN = DATASETS / "PICMUS/database/simulation/resolution_distorsion/resolution_distorsion_simu_scan.hdf5"
PHANTOM = DATASETS / "PICMUS/database/simulation/resolution_distorsion/resolution_distorsion_simu_phantom.hdf5"

VARIANTS = {
    "A_actual_sign-1_edge":  dict(tx_sign=-1.0, tx_reference="edge"),
    "B_picmus_sign+1_origin": dict(tx_sign=+1.0, tx_reference="origin"),
    "C_sign-1_origin":       dict(tx_sign=-1.0, tx_reference="origin"),
}


def main() -> None:
    ds = load_picmus_hdf5(RF)
    grid = load_picmus_scan(SCAN)
    gx, gz = grid.x[::4], grid.z[::4]
    print(ds.summary())
    print(f"initial_time_s = {ds.initial_time_s}  index_t0 = {ds.index_t0}")
    print(f"quick grid: {len(gz)} x {len(gx)}")

    # ground-truth scatterers if available
    try:
        import h5py
        with h5py.File(PHANTOM, "r") as f:
            raw = {}
            f.visititems(lambda n, o: raw.__setitem__(n, np.array(o))
                         if hasattr(o, "shape") else None)
        for k, v in raw.items():
            if "position" in k.lower() or "phantom" in k.lower():
                print("phantom key:", k, v.shape if hasattr(v, "shape") else v)
    except Exception as e:  # noqa: BLE001
        print("phantom read failed:", e)

    n = ds.n_angles
    angle_ids = [0, n // 2, n - 1]
    fn = GratingAngleLB(60, 1.5)

    results = {}
    for vname, kw in VARIANTS.items():
        per_angle = []
        for aidx in angle_ids:
            res = das_pw(
                gx, gz, ds.rf_for_angle(aidx), ds.f_s,
                float(ds.steering_angles_rad[aidx]),
                ds.element_width, ds.element_pitch, ds.c_avg,
                f_bounds=(ds.f_lb, ds.f_ub), index_t0=ds.index_t0,
                F_number=fn, z_chunk=64, device="cpu", **kw,
            )
            per_angle.append(res.image)
        results[vname] = per_angle
        np.save(OUT / f"verify_{vname}.npy",
                torch.stack(per_angle).numpy())

    # Figure: rows = variants, cols = angle(min), 0deg, angle(max), compound3
    fig, axes = plt.subplots(len(VARIANTS), 4,
                             figsize=(16, 4.5 * len(VARIANTS)))
    adeg = ds.steering_angles_deg
    for i, (vname, per_angle) in enumerate(results.items()):
        comp = torch.stack(per_angle).sum(dim=0)
        imgs = per_angle + [comp]
        labels = [f"{adeg[angle_ids[0]]:+.1f} deg", "0 deg",
                  f"{adeg[angle_ids[-1]]:+.1f} deg", "compound 3 ang"]
        for j, (img, lbl) in enumerate(zip(imgs, labels)):
            ax = axes[i, j]
            db = to_bmode_db(img, 60.0)
            ax.imshow(db, cmap="gray", vmin=-60, vmax=0, aspect="auto",
                      extent=[gx[0] * 1e3, gx[-1] * 1e3,
                              gz[-1] * 1e3, gz[0] * 1e3])
            ax.set_title(f"{vname}\n{lbl}" if j == 0 else lbl, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "verify_tx_convention.png", dpi=130)
    print(f"wrote {OUT / 'verify_tx_convention.png'}")


if __name__ == "__main__":
    main()
