"""Regenerate the classical PICMUS reconstructions with the CORRECT
plane-wave transmit-time convention (tx_sign=+1, tx_reference="origin").

Background: the das_pw port used Schiffner's data_RF.mat convention
(esx = -sin(angle), t=0 at the lagging edge element). PICMUS and the
Field II simulations of this thesis emit the wavefront through the array
center at t=0 with delays x*sin(angle)/c, so every off-axis angle was
misaligned laterally AND axially, destroying the 75-PW coherent compound
(the "vertical stripe" artefacts of the old figures).

Methods generated per phantom (complex64 npy on the native scan grid):
  das1pw       : central angle, ConstantFNumber(0) = full aperture
  schiffner1pw : central angle, GratingAngleLB(60, 1.5) Fd-F
  das75pw      : all angles, ConstantFNumber(0), coherent sum
  schiffner75  : all angles, GratingAngleLB(60, 1.5) Fd-F, coherent sum

schiffner1pw isolates what the Fd-F apodization contributes FOR THE SAME
INPUT: same angle and same cost as das1pw (596 GFLOP on the experimental
scenarios), only the F-number changes. It is also the correct baseline of
UNet-AG, because the defaults of OpHFdf --the operator that produces its
input-- are exactly GratingAngleLB(60, 1.5) + Tukey(0.2) + NormalizationOn:
with the out_conv residual initialized to zero, this image IS the model
output at epoch 0.

Output:
  evaluation/npys/<method>/<key>_<method>.npy      (complex128)
  evaluation/npys/<method>/<key>_<method>.png      (B-mode preview)

Run from the repository root (GPU strongly recommended; PICMUS under
$THESIS_DATASETS/PICMUS):
  python evaluation/scripts/gen_classical_picmus_fixed.py --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
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
    ConstantFNumber,
    GratingAngleLB,
    das_pw,
    load_picmus_hdf5,
    load_picmus_scan,
    to_bmode_db,
)

DR_DB = 60.0

# Correct convention for PICMUS (and Field II) data:
TX_KW = dict(tx_sign=+1.0, tx_reference="origin")


@dataclass
class PhantomSpec:
    key: str
    label: str
    rf_path: str
    scan_path: str


PHANTOMS = [
    PhantomSpec("sim_resolution", "Simulation — Resolution",
                "database/simulation/resolution_distorsion/resolution_distorsion_simu_dataset_rf.hdf5",
                "database/simulation/resolution_distorsion/resolution_distorsion_simu_scan.hdf5"),
    PhantomSpec("sim_contrast", "Simulation — Contrast",
                "database/simulation/contrast_speckle/contrast_speckle_simu_dataset_rf.hdf5",
                "database/simulation/contrast_speckle/contrast_speckle_simu_scan.hdf5"),
    PhantomSpec("exp_resolution", "Experimental — Resolution",
                "database/experiments/resolution_distorsion/resolution_distorsion_expe_dataset_rf.hdf5",
                "database/experiments/resolution_distorsion/resolution_distorsion_expe_scan.hdf5"),
    PhantomSpec("exp_contrast", "Experimental — Contrast",
                "database/experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5",
                "database/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5"),
]


def _index_t0(ds) -> int:
    """das_pw semantics: sample index where t=0 falls => -t_start*fs."""
    return -int(round(ds.initial_time_s * ds.f_s))


def _save_png(img_c, gx, gz, title, path):
    db = to_bmode_db(img_c, DR_DB)
    fig, ax = plt.subplots(figsize=(5.5, 7))
    im = ax.imshow(db, extent=[gx[0] * 1e3, gx[-1] * 1e3,
                               gz[-1] * 1e3, gz[0] * 1e3],
                   aspect="equal", cmap="gray", vmin=-DR_DB, vmax=0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Lateral (mm)")
    ax.set_ylabel("Axial (mm)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="dB")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def beamform_compound(ds, gx, gz, fnum, angle_ids, *, device, z_chunk):
    total = None
    for aidx in angle_ids:
        res = das_pw(
            gx, gz, ds.rf_for_angle(aidx), ds.f_s,
            float(ds.steering_angles_rad[aidx]),
            ds.element_width, ds.element_pitch, ds.c_avg,
            f_bounds=(ds.f_lb, ds.f_ub), index_t0=_index_t0(ds),
            F_number=fnum, z_chunk=z_chunk, device=device, **TX_KW,
        )
        total = res.image if total is None else total + res.image
    return total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--picmus", type=Path, default=DATASETS / "PICMUS")
    p.add_argument("--out", type=Path,
                   default=ROOT / "evaluation" / "npys")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--z_chunk", type=int, default=32)
    p.add_argument("--only", default=None,
                   help="restrict to one phantom key (debug)")
    args = p.parse_args()

    methods = {
        "das1pw": ("DAS 1 PW (F=0)", ConstantFNumber(0.0)),
        "schiffner1pw": ("Schiffner Fd-F 1 PW", GratingAngleLB(60.0, 1.5)),
        "das75pw": ("DAS 75 PW (F=0)", ConstantFNumber(0.0)),
        "schiffner75pw": ("Schiffner Fd-F 75 PW", GratingAngleLB(60.0, 1.5)),
    }
    for m in methods:
        (args.out / m).mkdir(parents=True, exist_ok=True)

    t_all = time.perf_counter()
    for spec in PHANTOMS:
        if args.only and spec.key != args.only:
            continue
        print(f"\n================ {spec.label} ({spec.key}) ================")
        ds = load_picmus_hdf5(args.picmus / spec.rf_path)
        grid = load_picmus_scan(args.picmus / spec.scan_path)
        gx, gz = grid.x, grid.z
        print(f"grid {len(gz)}x{len(gx)}  angles={ds.n_angles}  "
              f"initial_time={ds.initial_time_s}  index_t0(correct)={_index_t0(ds)}")

        angles = np.asarray(ds.steering_angles_rad)
        central = int(np.argmin(np.abs(angles)))

        for mkey, (mlabel, fnum) in methods.items():
            npy = args.out / mkey / f"{spec.key}_{mkey}.npy"
            if npy.exists():
                print(f"  {mkey}: exists, skip")
                continue
            t0 = time.perf_counter()
            # The 1 PW methods use only the central angle; the 75 PW ones, all of them.
            ids = [central] if mkey.endswith("1pw") else list(range(ds.n_angles))
            img = beamform_compound(ds, gx, gz, fnum, ids,
                                    device=args.device, z_chunk=args.z_chunk)
            np.save(npy, img.detach().cpu().numpy())
            _save_png(img, gx.numpy(), gz.numpy(),
                      f"{mlabel} — {spec.label}",
                      args.out / mkey / f"{spec.key}_{mkey}.png")
            print(f"  {mkey}: {time.perf_counter() - t0:.1f}s -> {npy.name}")

    print(f"\nDONE in {(time.perf_counter() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
