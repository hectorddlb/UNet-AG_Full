"""Compute PICMUS metrics for a saved complex .npy reconstruction.

Given a method whose .npy files live in ``<results_dir>/<key>_<suffix>.npy``
(complex64, on the native PICMUS scan grid), report:

  resolution_distortion -> FWHM_lat_med_mm, n_resolved_lat,
                           FWHM_ax_med_mm,  n_resolved_ax, n_total
  contrast_speckle      -> CNR_dB, gCNR, n_cysts

The .npy is expected to be the coherent compound image (complex), aligned to
the scan grid stored in each phantom's ``*_scan.hdf5``. For DAS 75 PW this
matches ``gen_das_75pw_picmus.py``.

Writes:
  - one CSV with one row per phantom
  - prints a small markdown-style table to stdout
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]          # repository root
sys.path.insert(0, str(_REPO / "code"))

from schiffner_das import load_picmus_scan  # noqa: E402

from unet_ag.eval.masks_picmus import cyst_masks, points_wire_targets  # noqa: E402
from unet_ag.metrics.cnr import compute_cnr  # noqa: E402
from unet_ag.metrics.fwhm import aggregate_fwhms_per_wire  # noqa: E402
from unet_ag.metrics.gcnr import compute_gcnr  # noqa: E402


@dataclass
class PhantomSpec:
    key: str
    label: str
    scan_path: str
    phantom_path: str
    phantom_type: str   # "resolution_distortion" | "contrast_speckle"


PHANTOMS = [
    PhantomSpec(
        key="sim_resolution", label="Simulation - Resolution",
        scan_path="database/simulation/resolution_distorsion/resolution_distorsion_simu_scan.hdf5",
        phantom_path="database/simulation/resolution_distorsion/resolution_distorsion_simu_phantom.hdf5",
        phantom_type="resolution_distortion",
    ),
    PhantomSpec(
        key="sim_contrast", label="Simulation - Contrast",
        scan_path="database/simulation/contrast_speckle/contrast_speckle_simu_scan.hdf5",
        phantom_path="database/simulation/contrast_speckle/contrast_speckle_simu_phantom.hdf5",
        phantom_type="contrast_speckle",
    ),
    PhantomSpec(
        key="exp_resolution", label="Experimental - Resolution",
        scan_path="database/experiments/resolution_distorsion/resolution_distorsion_expe_scan.hdf5",
        phantom_path="database/experiments/resolution_distorsion/resolution_distorsion_expe_phantom.hdf5",
        phantom_type="resolution_distortion",
    ),
    PhantomSpec(
        key="exp_contrast", label="Experimental - Contrast",
        scan_path="database/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5",
        phantom_path="database/experiments/contrast_speckle/contrast_speckle_expe_phantom.hdf5",
        phantom_type="contrast_speckle",
    ),
]


def _envelope_from_npy(arr: np.ndarray) -> np.ndarray:
    """Return non-negative envelope from a stored .npy.

    - complex -> |complex|
    - real    -> assumed to already be envelope (no Hilbert applied)
    """
    if np.iscomplexobj(arr):
        return np.abs(arr).astype(np.float32)
    return arr.astype(np.float32)


def eval_one_phantom(npy_path: Path, scan_path: Path, phantom_path: Path,
                     phantom_type: str) -> dict:
    arr = np.load(npy_path)
    env = _envelope_from_npy(arr)

    grid = load_picmus_scan(scan_path)
    gx_m = grid.x.detach().cpu().numpy().astype(np.float64)
    gz_m = grid.z.detach().cpu().numpy().astype(np.float64)
    if env.shape != (gz_m.size, gx_m.size):
        raise ValueError(
            f"shape mismatch: npy {env.shape} vs scan grid ({gz_m.size}, {gx_m.size})"
        )

    out: dict = {}
    if phantom_type == "resolution_distortion":
        wires = points_wire_targets(phantom_path)
        dz_mm = float((gz_m[1] - gz_m[0]) * 1e3)
        dx_mm = float((gx_m[1] - gx_m[0]) * 1e3)
        res = aggregate_fwhms_per_wire(
            env, wires, dx_mm=dx_mm, dz_mm=dz_mm,
            grid_z_mm=gz_m * 1e3, grid_x_mm=gx_m * 1e3,
            search_radius_mm=1.5,
        )
        out["fwhm_lat_med_mm"] = res["lat_median"]
        out["fwhm_ax_med_mm"] = res["ax_median"]
        out["fwhm_n_resolved_lat"] = int(res["n_resolved_lat"])
        out["fwhm_n_resolved_ax"] = int(res["n_resolved_ax"])
        out["fwhm_n_total"] = int(res["n_total"])
    else:  # contrast_speckle
        mi, mo, cysts = cyst_masks(phantom_path, gz_m, gx_m)
        if not cysts or mo.sum() == 0:
            out["cnr_db"] = float("nan")
            out["gcnr"] = float("nan")
            out["n_cysts"] = 0
        else:
            out["cnr_db"] = compute_cnr(env, mi, mo, on_envelope=False)
            out["gcnr"] = compute_gcnr(env, mi, mo, on_envelope=False)
            out["n_cysts"] = len(cysts)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--picmus", type=Path, default=Path("datasets/PICMUS"))
    p.add_argument("--npy_dir", type=Path, required=True,
                   help="Dir containing <key>_<suffix>.npy files")
    p.add_argument("--suffix", type=str, required=True,
                   help="File-name suffix between key_ and .npy (e.g. das75pw)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output CSV path")
    p.add_argument("--method_label", type=str, default="",
                   help="Optional label printed at top of stdout table")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    print(f"method        : {args.method_label or args.suffix}")
    print(f"npy dir       : {args.npy_dir.resolve()}")
    print(f"PICMUS root   : {args.picmus.resolve()}")
    print()
    for spec in PHANTOMS:
        npy_path = args.npy_dir / f"{spec.key}_{args.suffix}.npy"
        scan_path = args.picmus / spec.scan_path
        phantom_path = args.picmus / spec.phantom_path
        if not npy_path.exists():
            print(f"  SKIP {spec.key}: missing {npy_path}")
            continue
        if not scan_path.exists() or not phantom_path.exists():
            print(f"  SKIP {spec.key}: missing scan or phantom HDF5")
            continue

        metrics = eval_one_phantom(npy_path, scan_path, phantom_path,
                                   spec.phantom_type)
        row = {
            "key": spec.key, "label": spec.label,
            "phantom_type": spec.phantom_type,
            "npy": str(npy_path),
        }
        row.update(metrics)
        rows.append(row)

        print(f"=== {spec.label} ({spec.key}) ===")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:24s} = {v:.4f}")
            else:
                print(f"  {k:24s} = {v}")

    if not rows:
        print("No rows computed.")
        return

    all_keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=all_keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {args.out}")

    # Compact summary table
    print("\n=== SUMMARY ===")
    print(f"{'phantom':32s}  metric             value")
    for r in rows:
        for k in ("fwhm_lat_med_mm", "fwhm_ax_med_mm",
                  "fwhm_n_resolved_lat", "fwhm_n_resolved_ax", "fwhm_n_total",
                  "cnr_db", "gcnr", "n_cysts"):
            if k in r:
                v = r[k]
                fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
                print(f"{r['label']:32s}  {k:18s} {fmt}")


if __name__ == "__main__":
    main()
