"""Common input for the MATLAB validation of the Fd-F beamformer.

Packs into a ``.mat`` the RF of the 75 angles of the PICMUS experimental
contrast scenario, its geometry and the comparison grid (612 x 388 points,
z from 5 to 50 mm and x within ±19.2 mm), so that ``das_pw.m`` and the Python
port read exactly the same data.

Usage (from the repository root):
    python evaluation/matlab_validation/prepare_input.py
    -> evaluation/matlab_validation/work/exp_contrast_rf.mat   (~255 MB, not versioned)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("THESIS_DATASETS", ROOT / "data"))  # PICMUS/ and Field II/, see data/README.md
sys.path.insert(0, str(ROOT / "code"))

from schiffner_das import load_picmus_hdf5  # noqa: E402

RF_REL = "experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5"
GZ = np.linspace(5e-3, 50e-3, 612)
GX = np.linspace(-128 * 300e-6 / 2, 128 * 300e-6 / 2, 388)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--picmus", type=Path, default=DATASETS / "PICMUS")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent / "work" / "exp_contrast_rf.mat")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ds = load_picmus_hdf5(args.picmus / "database" / RF_REL)
    rf = np.stack([ds.rf_for_angle(a).numpy() for a in range(len(ds.steering_angles_rad))],
                  -1).astype(np.float64)
    sio.savemat(args.out, dict(
        data_RF=rf, angles=np.asarray(ds.steering_angles_rad, float), fs=float(ds.f_s),
        pitch=float(ds.element_pitch), width=float(ds.element_width), c0=float(ds.c_avg),
        positions_z=GZ, positions_x=GX, t0=float(ds.initial_time_s)), do_compression=False)
    print(f"wrote {args.out}: RF {rf.shape}, central angle (1-based) = "
          f"{int(np.argmin(np.abs(np.asarray(ds.steering_angles_rad)))) + 1}")


if __name__ == "__main__":
    main()
