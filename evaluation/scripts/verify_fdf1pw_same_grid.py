"""Compare das_pw(Fd-F, 1 PW) with OpHFdf.adjoint ON THE SAME GRID.

Why this script is needed when verify_schiffner1pw_vs_adjoint.py already
exists: that one compared the canonical schiffner1pw npy (computed on the
PICMUS grid and then interpolated to the evaluation one) against the adjoint
(computed directly on the evaluation grid). The resampling enters only one
side of that comparison, so it inflates its RMSE and nothing can be concluded
from it coming out larger than that of Fd-F vs full aperture.

Here das_pw is computed DIRECTLY on the 612x388 evaluation grid, which is the
same one the adjoint uses. With no resampling anywhere, whatever difference
remains is real: either they are the same beamformer or they are not.

Three things are compared on that common grid:
  das_pw(GratingAngleLB)  vs  OpHFdf.adjoint      <- the question
  das_pw(GratingAngleLB)  vs  das_pw(F=0)         <- reference: how much the
                                                     apodization changes
  das_pw(F=0)             vs  OpHFdf.adjoint      <- control

The reported means (correlation 0.931 / 0.823 / 0.761) are saved in
evaluation/verification/correlation_fdf1pw_adjoint.json.

Usage
-----
    python evaluation/scripts/verify_fdf1pw_same_grid.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("THESIS_DATASETS", ROOT / "data"))  # PICMUS/ and Field II/, see data/README.md
sys.path.insert(0, str(ROOT / "code"))

from schiffner_das import (  # noqa: E402
    ConstantFNumber,
    GratingAngleLB,
    NormalizationOn,
    Tukey,
    das_pw,
    load_picmus_hdf5,
)
from unet_ag.data.rf_loader import PICMUSLoader  # noqa: E402
from unet_ag.eval.evaluate_fieldii import _build_grid  # noqa: E402
from unet_ag.physics.h_fdf_wrapper import make_op_for_angle  # noqa: E402

DR_DB = 60.0
TX_KW = dict(tx_sign=+1.0, tx_reference="origin")

PHANTOMS = [
    ("sim_resolution", "sim", "resolution_distortion",
     "simulation/resolution_distorsion/resolution_distorsion_simu_dataset_rf.hdf5"),
    ("sim_contrast", "sim", "contrast_speckle",
     "simulation/contrast_speckle/contrast_speckle_simu_dataset_rf.hdf5"),
    ("exp_resolution", "exp", "resolution_distortion",
     "experiments/resolution_distorsion/resolution_distorsion_expe_dataset_rf.hdf5"),
    ("exp_contrast", "exp", "contrast_speckle",
     "experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5"),
]


def env_log01(img) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    env = np.abs(img).astype(np.float64)
    env = env / (env.max() + 1e-30)
    db = np.clip(20.0 * np.log10(env + 1e-30), -DR_DB, 0.0)
    return ((db + DR_DB) / DR_DB).astype(np.float32)


def stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    return (float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
            float(np.sqrt(np.mean((a - b) ** 2))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--picmus", type=Path, default=DATASETS / "PICMUS")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "evaluation" / "verification" / "correlation_fdf1pw_adjoint.json")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"device = {dev}\ncommon grid = 612 x 388 (the evaluation one)\n")

    acc = {"fdf_adj": [], "fdf_das0": [], "das0_adj": []}
    for key, mode, phantom, rf_rel in PHANTOMS:
        # grid and geometry through the model's path
        loader = PICMUSLoader(args.picmus, mode=mode, phantom=phantom, normalize=True)
        sample = loader[0]
        gz_t, gx_t = _build_grid(sample.geometry, 612, 388)

        # adjoint (the input of UNet-AG)
        op = make_op_for_angle(0.0, sample.geometry, gz_t, gx_t, device=dev)
        y = sample.rf_1pw.to(dev).to(op.float_dtype)
        x0 = op.adjoint(y)
        adj = env_log01(x0.real if x0.is_complex() else x0)

        # das_pw on THE SAME grid, central angle
        ds = load_picmus_hdf5(args.picmus / "database" / rf_rel)
        angles = np.asarray(ds.steering_angles_rad)
        central = int(np.argmin(np.abs(angles)))
        index_t0 = -int(round(ds.initial_time_s * ds.f_s))
        gz64 = gz_t.to(torch.float64)
        gx64 = gx_t.to(torch.float64)

        def beamform(fnum):
            res = das_pw(
                gx64, gz64, ds.rf_for_angle(central), ds.f_s,
                float(angles[central]), ds.element_width, ds.element_pitch,
                ds.c_avg, f_bounds=(ds.f_lb, ds.f_ub), index_t0=index_t0,
                window=Tukey(0.2), F_number=fnum, normalization=NormalizationOn(),
                device=dev, **TX_KW,
            )
            return env_log01(res.image)

        fdf = beamform(GratingAngleLB(60.0, 1.5))
        das0 = beamform(ConstantFNumber(0.0))

        c1, r1 = stats(fdf, adj)
        c2, r2 = stats(fdf, das0)
        c3, r3 = stats(das0, adj)
        acc["fdf_adj"].append((c1, r1))
        acc["fdf_das0"].append((c2, r2))
        acc["das0_adj"].append((c3, r3))

        print(f"=== {key} ===")
        print(f"  Fd-F      vs adjoint : corr={c1:.4f}  rmse={r1:.4f}")
        print(f"  Fd-F      vs DAS F=0 : corr={c2:.4f}  rmse={r2:.4f}")
        print(f"  DAS F=0   vs adjoint : corr={c3:.4f}  rmse={r3:.4f}")
        print()

    print("=" * 66)
    summary = {}
    for k, lab in [("fdf_adj", "Fd-F    vs adjoint"),
                   ("fdf_das0", "Fd-F    vs DAS F=0"),
                   ("das0_adj", "DAS F=0 vs adjoint")]:
        c = float(np.mean([v[0] for v in acc[k]]))
        r = float(np.mean([v[1] for v in acc[k]]))
        summary[k] = dict(corr_mean=c, rmse_mean=r,
                          per_phantom={key: dict(corr=v[0], rmse=v[1])
                                       for (key, *_), v in zip(PHANTOMS, acc[k])})
        print(f"  {lab} : mean corr={c:.4f}   mean rmse={r:.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"  wrote {args.out}")
    print()
    print("  If 'Fd-F vs adjoint' has corr ~1 and an rmse far below the other")
    print("  two, the adjoint IS Schiffner Fd-F 1 PW and that row of the table is")
    print("  the exact baseline of UNet-AG. Otherwise they are different")
    print("  beamformers and must be reported as such.")


if __name__ == "__main__":
    main()
