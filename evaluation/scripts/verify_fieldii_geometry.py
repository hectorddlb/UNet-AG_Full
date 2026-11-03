"""Check the Field II figures quoted in the thesis on partition, grid and t_start.

1. Deterministic partition of the 300 point phantoms (field_ii_split, seed
   42): 240 / 30 / 30, and how many of each part have a target in
   data/targets_schiffner_75pw (231 / 28 / 28: the trainer silently drops the
   samples without a target, hence the 231 steps per epoch).
2. Trainer grid: built ONCE from the first phantom of the training subset
   (b1_v5_pulse_aug.py, ``sample0``), from 5 mm to (n_s - 1) c / (2 f_s). The
   targets were beamformed on 5-50 mm: same number of rows (612), different
   scale.
3. Distribution of t_start · c (in mm) over the training phantoms.
4. Spread of the initial time across the 75 angles of each phantom (the
   reason for using each angle's t_start when compounding the targets).

Usage (from the repository root; Field II in $THESIS_DATASETS):
    python evaluation/scripts/verify_fieldii_geometry.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("THESIS_DATASETS", ROOT / "data"))  # PICMUS/ and Field II/, see data/README.md
sys.path.insert(0, str(ROOT / "code"))

from unet_ag.data.manifest import field_ii_split  # noqa: E402

NZ = 612


def _scalar(m: dict, key: str) -> float:
    return float(np.asarray(m[key]).ravel()[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path, default=DATASETS / "field_ii_data_multiangle")
    p.add_argument("--targets", type=Path, default=ROOT / "data" / "targets_schiffner_75pw")
    p.add_argument("--out", type=Path,
                   default=ROOT / "evaluation" / "verification" / "fieldii_partition_grid_tstart.json")
    args = p.parse_args()

    split = field_ii_split(args.data_root, ["points"])["points"]
    has_t = lambda f: (args.targets / "points" / f.stem / "compounded.npy").exists()  # noqa: E731
    parts = {k: getattr(split, k) for k in ("train", "val", "test")}
    counts = {k: dict(phantoms=len(v), with_target=sum(has_t(f) for f in v))
              for k, v in parts.items()}

    # trainer grid: first phantom of the training subset
    first = parts["train"][0]
    m0 = sio.loadmat(first, variable_names=["rf_data", "fs", "c"])
    ns = int(np.asarray(m0["rf_data"]).shape[0])
    fs, c = _scalar(m0, "fs"), _scalar(m0, "c")
    z_max = (ns - 1) * c / (2.0 * fs)
    dz_e = (z_max - 5e-3) / (NZ - 1)
    dz_t = (50e-3 - 5e-3) / (NZ - 1)
    z45 = 45e-3
    row45 = (z45 - 5e-3) / dz_e
    off45 = row45 * (dz_e - dz_t)

    # t_start * c (angle 0) on the training phantoms
    shifts, spreads = [], []
    for f in parts["train"]:
        m = sio.loadmat(f, variable_names=["t_start", "angles_deg", "c"])
        ang = np.asarray(m["angles_deg"]).ravel()
        ts = np.asarray(m["t_start"]).ravel()
        t0 = ts[int(np.argmin(np.abs(ang)))]
        shifts.append(float(t0) * _scalar(m, "c") * 1e3)
        spreads.append(float(ts.max() - ts.min()))
    shifts, spreads = np.asarray(shifts), np.asarray(spreads)

    out = dict(
        partition=counts,
        trainer_grid=dict(first_phantom=first.stem, n_s=ns, fs=fs, c=c,
                          z_max_mm=z_max * 1e3),
        target_grid_mm=[5.0, 50.0],
        relative_scale_error=1.0 - dz_t / dz_e,
        shift_at_45mm=dict(mm=off45 * 1e3, trainer_rows=off45 / dz_e,
                           target_rows=off45 / dz_t),
        t_start_times_c_mm_training=dict(n=int(shifts.size), min=float(shifts.min()),
                                         max=float(shifts.max()),
                                         median=float(np.median(shifts))),
        t_start_spread_across_angles=dict(
            median_us=float(np.median(spreads) * 1e6),
            min_us=float(spreads.min() * 1e6), max_us=float(spreads.max() * 1e6),
            median_axial_mm=float(np.median(spreads) * c / 2 * 1e3)),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
