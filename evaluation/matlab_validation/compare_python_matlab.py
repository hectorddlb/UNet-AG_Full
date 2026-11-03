"""Validation of the Python port of the Fd-F beamformer against das_pw.m.

Compares, on the PICMUS experimental contrast scenario and the 612 x 388 point
grid of prepare_input.py, the complex image of the MATLAB reference
implementation with that of the port (``schiffner_das.das_pw``) run with the
same parameters: Tukey 0.2, normalization by the sum of weights, 2.25 to
6.75 MHz band, zero start index, F-number GratingAngleLB(60, 1.5) or full
aperture, no antialiasing, in float64.

The reference only supports its native transmit convention (edge, -sin θ), so
the port is run with that same convention: at 0° it matches the PICMUS one;
with 75 angles the comparison validates the numerical equivalence, not the
75 plane wave comparators of the evaluation (which use the corrected
convention).

Metrics: relative L2 error of the complex image, maximum difference relative
to the peak, and gCNR and CNR (dB) differences between both implementations,
computed with the SAME regions of interest for both: an inner disc of radius
0.75 r in each inclusion and a background ring from 1.25 r to 1.75 r that
excludes the other inclusions, on the dB envelope clipped at 60 dB and mapped
to [0, 1].

MATLAB inputs: the ``.mat`` files of run_das_pw_matlab.m in work/ if they
exist; otherwise, the images already computed in results/matlab_<case>.npy.

Usage (from the repository root; a GPU is advisable for 75 angles):
    python evaluation/matlab_validation/prepare_input.py
    python evaluation/matlab_validation/compare_python_matlab.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import torch

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("THESIS_DATASETS", ROOT / "data"))  # PICMUS/ and Field II/, see data/README.md
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from schiffner_das import ConstantFNumber, GratingAngleLB, das_pw  # noqa: E402
from schiffner_das.normalizations import NormalizationOn  # noqa: E402
from schiffner_das.windows import Tukey  # noqa: E402

PHANTOM_REL = "experiments/contrast_speckle/contrast_speckle_expe_phantom.hdf5"
DR = 60.0


def to_I(E: np.ndarray) -> np.ndarray:
    e = E / (E.max() + 1e-300)
    return (np.clip(20 * np.log10(e + 1e-300), -DR, 0) + DR) / DR


def gcnr(a: np.ndarray, b: np.ndarray) -> float:
    ha, _ = np.histogram(a, bins=256, range=(0.0, 1.0))
    hb, _ = np.histogram(b, bins=256, range=(0.0, 1.0))
    return float(1.0 - np.minimum(ha / ha.sum(), hb / hb.sum()).sum())


def cnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(20 * np.log10(abs(a.mean() - b.mean()) / np.sqrt(a.var() + b.var())))


def roi_masks(phantom_hdf5: Path, gz_mm: np.ndarray, gx_mm: np.ndarray):
    """Interior (0.75 r disc) and background (1.25-1.75 r ring without other inclusions)."""
    g = h5py.File(phantom_hdf5, "r")["US/US_DATASET0000"]
    cx = np.ravel(g["phantom_occlusionCenterX"][()]) * 1e3
    cz = np.ravel(g["phantom_occlusionCenterZ"][()]) * 1e3
    rr = np.ravel(g["phantom_occlusionDiameter"][()]) * 1e3 / 2
    zz, xx = np.meshgrid(gz_mm, gx_mm, indexing="ij")
    d = [np.hypot(zz - z, xx - x) for z, x in zip(cz, cx)]
    inner, ring = np.zeros_like(zz, bool), np.zeros_like(zz, bool)
    for i, r in enumerate(rr):
        other = np.zeros_like(zz, bool)
        for j, r2 in enumerate(rr):
            if j != i:
                other |= d[j] <= 1.25 * r2
        inner |= d[i] <= 0.75 * r
        ring |= (d[i] >= 1.25 * r) & (d[i] <= 1.75 * r) & ~other
    return inner, ring


def port(S: dict, kind: str, ids, device: str) -> np.ndarray:
    F = GratingAngleLB(60.0, 1.5) if kind == "fdf" else ConstantFNumber(0.0)
    gz, gx = np.ravel(S["positions_z"]), np.ravel(S["positions_x"])
    rf, ang = S["data_RF"], np.ravel(S["angles"])
    tot = None
    for a in ids:
        r = das_pw(torch.from_numpy(gx), torch.from_numpy(gz),
                   torch.from_numpy(np.ascontiguousarray(rf[:, :, a])),
                   S["fs"].item(), float(ang[a]), S["width"].item(), S["pitch"].item(),
                   S["c0"].item(), f_bounds=(2.25e6, 6.75e6), index_t0=0,
                   window=Tukey(0.2), F_number=F, normalization=NormalizationOn(),
                   z_chunk=32, device=device, tx_sign=-1.0, tx_reference="edge",
                   precision="float64")
        im = r.image.detach().cpu().numpy()
        tot = im if tot is None else tot + im
    return tot


def matlab_image(case: str, mat_files: list[Path]) -> np.ndarray:
    """Sum of the images of the MATLAB ``.mat`` files, or the one already saved in results/."""
    if mat_files and all(f.exists() for f in mat_files):
        tot = None
        for f in mat_files:
            with h5py.File(f, "r") as h:
                d = h["imgs"][()]
            im = (d["real"] + 1j * d["imag"]) if d.dtype.names else d
            im = np.transpose(im, (0, 2, 1)) if im.ndim == 3 else im.T   # v7.3: column order
            s = im.sum(0) if im.ndim == 3 else im
            tot = s if tot is None else tot + s
        return tot
    return np.load(HERE / "results" / f"matlab_{case}.npy")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=HERE / "work" / "exp_contrast_rf.mat")
    p.add_argument("--picmus", type=Path, default=DATASETS / "PICMUS")
    p.add_argument("--out", type=Path, default=HERE / "results" / "matlab_validation.csv")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device

    S = sio.loadmat(args.input)
    gz_mm, gx_mm = np.ravel(S["positions_z"]) * 1e3, np.ravel(S["positions_x"]) * 1e3
    inner, ring = roi_masks(args.picmus / "database" / PHANTOM_REL, gz_mm, gx_mm)
    n_ang = np.ravel(S["angles"]).size
    c0 = int(np.argmin(np.abs(np.ravel(S["angles"]))))
    work = HERE / "work"
    cases = [("fdf_1pw", "fdf", [c0], [work / "matlab_fdf_1pw.mat"]),
             ("das_1pw", "das", [c0], [work / "matlab_das_1pw.mat"]),
             ("fdf_75pw_native", "fdf", range(n_ang), sorted(work.glob("matlab_fdf_75_*.mat"))),
             ("das_75pw_native", "das", range(n_ang), sorted(work.glob("matlab_das_75_*.mat")))]
    rows = []
    for name, kind, ids, files in cases:
        Mi, Pi = matlab_image(name, files), port(S, kind, ids, device)
        assert Mi.shape == Pi.shape == (gz_mm.size, gx_mm.size), (Mi.shape, Pi.shape)
        IM, IP = to_I(np.abs(Mi)), to_I(np.abs(Pi))
        row = dict(case=name, n_angles=len(list(ids)),
                   rel_err_L2=float(np.linalg.norm(Mi - Pi) / np.linalg.norm(Mi)),
                   max_abs_diff=float(np.abs(Mi - Pi).max()),
                   max_diff_rel_to_peak=float(np.abs(Mi - Pi).max() / np.abs(Mi).max()),
                   gcnr_matlab=gcnr(IM[inner], IM[ring]), gcnr_port=gcnr(IP[inner], IP[ring]),
                   cnr_matlab_db=cnr(IM[inner], IM[ring]), cnr_port_db=cnr(IP[inner], IP[ring]))
        row["gcnr_diff"] = row["gcnr_port"] - row["gcnr_matlab"]
        row["cnr_diff_db"] = row["cnr_port_db"] - row["cnr_matlab_db"]
        rows.append(row)
        print(f"{name:16s} rel_err_L2={row['rel_err_L2']:.2e}  "
              f"max_diff/peak={row['max_diff_rel_to_peak']:.2e}  "
              f"|ΔgCNR|={abs(row['gcnr_diff']):.1e}  |ΔCNR|={abs(row['cnr_diff_db']):.1e} dB",
              flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
