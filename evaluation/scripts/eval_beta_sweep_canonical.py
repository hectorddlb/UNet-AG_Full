"""Beta sweep {0.3..0.7} measured with the CANONICAL PROTOCOL of the thesis.

CONTEXT
-------
The choice of beta rests on a curve over {0.3, 0.4, 0.5, 0.6, 0.7}. The
existing CSVs of that sweep (``results_v2/b1_v5_*/eval_picmus.csv``) come from
the PER-RUN evaluator, whose "DAS 1 PW" reference is the Fd-F adjoint and not
the full-aperture DAS of the canonical table. Putting that curve next to the
main table would mix the two sources exactly in the figure that supports the
choice of beta.

Since the five ``best.pt`` are still intact, this script re-runs inference on
the 4 PICMUS phantoms with each one and measures them with ``_eval_image`` on
the 612x388 grid -- exactly the same chain as ``eval_and_figs_final.py``.

The five runs share their configuration (20 epochs, seed 42, points,
mse=0.7 l1=0.2 ssim=0.1, pulse aug prob=0.7); the only variable is anchor_w.
Checked in the training logs 192071 / 198399 / 193078 / 198402 / 198405.

CONSISTENCY CHECK
-----------------
beta=0.50 is the main model of the thesis, so its row MUST reproduce the
``b1v5,s42`` row of ``metrics_picmus_final.csv``. The script checks it and
aborts with code != 0 if it does not match within tolerance: if that row does
not come out the same, the chain is not the canonical one and the curve is
not publishable.

Usage
-----
    python evaluation/scripts/eval_beta_sweep_canonical.py --device cuda
"""
from __future__ import annotations

import argparse
import os
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("THESIS_DATASETS", ROOT / "data"))  # PICMUS/ and Field II/, see data/README.md
sys.path.insert(0, str(ROOT / "evaluation" / "scripts"))
sys.path.insert(0, str(ROOT / "code"))

from eval_and_figs_final import PHANTOMS  # noqa: E402
from unet_ag.data.rf_loader import PICMUSLoader  # noqa: E402
from unet_ag.eval.evaluate_fieldii import (  # noqa: E402
    _build_grid,
    _load_b1_v3_ckpt,
    _reconstruct_b1,
)
from unet_ag.eval.evaluate_picmus import _eval_image  # noqa: E402

# beta -> results_v2 subdirectory (mapping taken from the training logs)
SWEEP = {
    0.30: "b1_v5_pulse",
    0.40: "b1_v5_anchor04",
    0.50: "b1_v5_anchor05",
    0.60: "b1_v5_anchor06",
    0.70: "b1_v5_anchor07",
}

NPYS = ROOT / "evaluation" / "npys" / "beta_sweep"
METR = ROOT / "evaluation" / "metrics"

# Canonical row of b1v5 seed 42 (metrics_picmus_final.csv) for the check.
CANON_S42 = {
    ("sim_resolution", "fwhm_lat_med_mm"): 1.375744104385376,
    ("exp_resolution", "fwhm_lat_med_mm"): 2.7551991939544678,
    ("sim_contrast",   "gcnr"):            0.5576750210072512,
    ("exp_contrast",   "gcnr"):            0.7902109395544537,
}
TOL = 1e-3


def main() -> int:
    p = argparse.ArgumentParser()
    # The 5 checkpoints of the sweep are NOT included in the repository
    # (203 MB of intermediate models). The resulting CSV is, in
    # evaluation/metrics/metrics_beta_sweep.csv, and it is enough to redraw
    # the figure. To regenerate it, retrain the sweep or point --sweep_root
    # to a directory with the best.pt files.
    p.add_argument("--sweep_root", type=Path,
                   default=ROOT / "results" / "beta_sweep")
    p.add_argument("--picmus", type=Path, default=DATASETS / "PICMUS")
    p.add_argument("--nz", type=int, default=612)
    p.add_argument("--nx", type=int, default=388)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    print(f"device = {device}   sweep_root = {args.sweep_root}")

    missing = [n for n in SWEEP.values()
               if not (args.sweep_root / n / "checkpoints" / "best.pt").exists()]
    if missing:
        print(f"ERROR: faltan checkpoints: {missing}")
        return 2

    # The loaders and grids do not depend on beta: they are built once.
    ctx = {}
    for key, mode, phantom, _label in PHANTOMS:
        loader = PICMUSLoader(args.picmus, mode=mode, phantom=phantom, normalize=True)
        sample = loader[0]
        gz_t, gx_t = _build_grid(sample.geometry, args.nz, args.nx)
        ctx[key] = (loader, sample, gz_t, gx_t, phantom)

    rows = []
    for beta, run in SWEEP.items():
        ckpt = args.sweep_root / run / "checkpoints" / "best.pt"
        model = _load_b1_v3_ckpt(ckpt, device, base_ch=32)
        out_dir = NPYS / f"beta{int(round(beta * 100)):03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n== beta={beta:.2f}  run={run}")
        for key, _mode, _ph, _lab in PHANTOMS:
            loader, sample, gz_t, gx_t, phantom = ctx[key]
            t0 = time.perf_counter()
            pred_env, _das_env = _reconstruct_b1(model, sample, gz_t, gx_t, device)
            np.save(out_dir / f"{key}_b1v5.npy", pred_env.astype(np.float32))
            met = _eval_image(pred_env, loader.phantom_path, phantom,
                              gz_t.numpy().astype(np.float64),
                              gx_t.numpy().astype(np.float64))
            rows.append({"beta": f"{beta:.2f}", "run": run, "phantom": key, **met})
            print(f"  {key:16s} {time.perf_counter() - t0:6.1f}s  "
                  + "  ".join(f"{k}={v:.4f}" for k, v in met.items()
                              if isinstance(v, float) and v == v))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    METR.mkdir(parents=True, exist_ok=True)
    out_csv = METR / "metrics_beta_sweep.csv"
    cols = sorted({k for r in rows for k in r} - {"beta", "run", "phantom"})
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["beta", "run", "phantom"] + cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}")

    # ── summary table ───────────────────────────────────────────────────
    by = {(r["beta"], r["phantom"]): r for r in rows}

    def g(beta, ph, col):
        v = by.get((f"{beta:.2f}", ph), {}).get(col, "")
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    print("\n" + "=" * 78)
    print("Beta sweep — canonical protocol (env_log 612x388, _eval_image)")
    print(f"{'beta':>6s} {'sim FWHM_lat':>13s} {'exp FWHM_lat':>13s} "
          f"{'sim gCNR':>10s} {'exp gCNR':>10s} {'sim CNR dB':>11s} {'exp CNR dB':>11s}")
    print("-" * 78)
    for beta in SWEEP:
        print(f"{beta:6.2f} {g(beta,'sim_resolution','fwhm_lat_med_mm'):13.3f} "
              f"{g(beta,'exp_resolution','fwhm_lat_med_mm'):13.3f} "
              f"{g(beta,'sim_contrast','gcnr'):10.3f} "
              f"{g(beta,'exp_contrast','gcnr'):10.3f} "
              f"{g(beta,'sim_contrast','cnr_db'):11.3f} "
              f"{g(beta,'exp_contrast','cnr_db'):11.3f}")
    print("-" * 78)
    print("  Canonical DAS 1 PW reference: sim 1.236 / exp 3.728 / "
          "gCNR 0.558 / 0.528 / CNR 0.744 / -3.974")

    # ── consistency check against the canonical seed 42 row ─────────────
    print("\nCheck: beta=0.50 must reproduce the canonical b1v5,s42 row")
    bad = []
    for (ph, col), ref in CANON_S42.items():
        got = g(0.50, ph, col)
        ok = abs(got - ref) <= TOL
        print(f"  {ph:16s} {col:18s} got={got:.6f}  canonical={ref:.6f}  "
              f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            bad.append((ph, col, got, ref))
    if bad:
        print("\nFAIL: the chain does not reproduce the canonical row. The curve is NOT")
        print("publishable next to the main table until this is resolved.")
        return 1
    print("\nOK — same chain as the canonical table. Curve publishable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
