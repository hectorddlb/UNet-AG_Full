"""Phase E — evaluates the DDRM restorations with the EXACT unet_ag metrics.

Chain per image (same role as ``abs(mean(x))`` + log in Zhang's MATLAB, but
ending in the anchor05 metrics, without touching a single line of them):

    x_mean (256², signed reflectivity) → |x| → /max → 20·log10 → clip 60 dB
    → [0,1] → bilinear interpolation to the 612×388 eval grid (fill 0)
    → unet_ag.eval.evaluate_picmus._eval_image (exact masks/FWHM/gCNR)

Gate G-E1: with ``--eval_by`` it also evaluates the Phase A env_log(By) through
the same interpolation chain — its metrics must be of the order of the known
DAS_1PW (sim/res FWHM_lat ≈ 1.4 mm, exp/res ≈ 2.5 mm, gCNR ≈ 0.58).

Usage:
    python zhang_drus/scripts/evaluate_ddrm.py \
        --pred_dir zhang_drus/results_drus/phaseD_ft287_ext \
        --phase_a_dir zhang_drus/results_drus/phaseA_ext \
        --picmus_root datasets/PICMUS \
        --out zhang_drus/results_drus/phaseD_ft287_ext/eval_ddrm.csv [--eval_by]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import hilbert

REPO_ROOT = Path(__file__).resolve().parents[3] / "code"
sys.path.insert(0, str(REPO_ROOT))

from unet_ag.data.rf_loader import PICMUSLoader
from unet_ag.eval.evaluate_fieldii import _build_grid
from unet_ag.eval.evaluate_picmus import _eval_image

DYN_RANGE_DB = 60.0
EVAL_NZ, EVAL_NX = 612, 388

# short paper phantom name → (mode, phantom) of the PICMUSLoader
PHANTOM_MAP = {
    "simu_reso": ("sim", "resolution_distortion"),
    "simu_cont": ("sim", "contrast_speckle"),
    "expe_reso": ("exp", "resolution_distortion"),
    "expe_cont": ("exp", "contrast_speckle"),
}


def envelope_db01(img_signed: np.ndarray, axial_hilbert: bool = False) -> np.ndarray:
    """signed signal → env_log [0,1]. For DDRM x, |x| is enough (no carrier);
    for By (RF domain with carrier) use an axial hilbert."""
    env = np.abs(hilbert(img_signed, axis=0)) if axial_hilbert else np.abs(img_signed)
    env = env / (env.max() + 1e-30)
    db = np.clip(20.0 * np.log10(env + 1e-30), -DYN_RANGE_DB, 0.0)
    return ((db + DYN_RANGE_DB) / DYN_RANGE_DB).astype(np.float32)


def to_eval_grid(img01: np.ndarray, drus_z: np.ndarray, drus_x: np.ndarray,
                 eval_z: np.ndarray, eval_x: np.ndarray) -> np.ndarray:
    interp = RegularGridInterpolator((drus_z, drus_x), img01,
                                     bounds_error=False, fill_value=0.0)
    zz, xx = np.meshgrid(eval_z, eval_x, indexing="ij")
    return interp(np.stack([zz.ravel(), xx.ravel()], axis=1)).reshape(zz.shape).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=Path, required=True)
    ap.add_argument("--phase_a_dir", type=Path, required=True,
                    help="Phase A dir (grid.npz + By_*.npy for --eval_by)")
    ap.add_argument("--picmus_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--eval_by", action="store_true", help="gate G-E1: evaluates env_log(By)")
    args = ap.parse_args()

    grid = np.load(args.phase_a_dir / "grid.npz")
    drus_x, drus_z = grid["x_axis"], grid["z_axis"]

    manifest_path = args.pred_dir / "manifest.json"
    runs = json.loads(manifest_path.read_text())["runs"] if manifest_path.exists() else []
    if not runs:
        print("[eval] no manifest.json — gate G-E1 only (--eval_by)", flush=True)

    # geometries and eval grids per phantom (once)
    ctx = {}
    for phantom, (mode, ph) in PHANTOM_MAP.items():
        loader = PICMUSLoader(args.picmus_root, mode=mode, phantom=ph)
        sample = loader[0]
        gz, gx = _build_grid(sample.geometry, EVAL_NZ, EVAL_NX)
        ctx[phantom] = (loader, ph, gz.numpy().astype(np.float64), gx.numpy().astype(np.float64))

    rows = []
    if args.eval_by:
        for phantom, (loader, ph, gz, gx) in ctx.items():
            by = np.load(args.phase_a_dir / f"By_{phantom}.npy").reshape(len(drus_z), len(drus_x))
            im = to_eval_grid(envelope_db01(by, axial_hilbert=True), drus_z, drus_x, gz, gx)
            m = _eval_image(im, loader.phantom_path, ph, gz, gx)
            rows.append({"phantom": phantom, "method": "By_env (gate G-E1)",
                         "gamma": "", "seed": "", **m})
            print(f"[G-E1] {phantom}: {m}", flush=True)

    for run in runs:
        phantom = run["phantom"]
        loader, ph, gz, gx = ctx[phantom]
        x = np.load(args.pred_dir / run["file"])          # (256,256) firmado
        im = to_eval_grid(envelope_db01(x), drus_z, drus_x, gz, gx)
        m = _eval_image(im, loader.phantom_path, ph, gz, gx)
        rows.append({"phantom": phantom, "method": "DRUS",
                     "gamma": run["gamma"], "seed": run["seed"], **m})
        print(f"[eval] {phantom} γ={run['gamma']} s={run['seed']}: {m}", flush=True)

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in
                  ("phantom", "method", "gamma", "seed"), k))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[done] {len(rows)} rows → {args.out}", flush=True)

    # summary: best γ per phantom according to the metric of its task
    print("\n== best γ per phantom (DRUS only) ==", flush=True)
    for phantom in PHANTOM_MAP:
        cand = [r for r in rows if r["phantom"] == phantom and r["method"] == "DRUS"]
        if not cand:
            continue
        if "reso" in phantom:
            best = max(cand, key=lambda r: (r.get("fwhm_n_resolved_lat", 0),
                                            -(r.get("fwhm_lat_med_mm") or 9e9)))
            print(f"  {phantom}: γ={best['gamma']} s={best['seed']} → "
                  f"n_lat={best.get('fwhm_n_resolved_lat')} "
                  f"FWHM_lat={best.get('fwhm_lat_med_mm')}", flush=True)
        else:
            best = max(cand, key=lambda r: r.get("gcnr") or -1)
            print(f"  {phantom}: γ={best['gamma']} s={best['seed']} → "
                  f"gCNR={best.get('gcnr')}", flush=True)


if __name__ == "__main__":
    main()
