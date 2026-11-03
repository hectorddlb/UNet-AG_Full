"""Measure, with the canonical protocol, the TWO candidate anchor images.

CONTEXT
-------
The anchor term of the loss is ``(pred - x_das).abs().mean()`` with
``x_das = inputs["x0"]``, and ``x0`` is ``_envelope_log_compress(OpHFdf.adjoint)``.
The OpHFdf defaults are GratingAngleLB(60,1.5) + Tukey(0.2) + NormalizationOn,
so **the anchor ALREADY is a 1 plane wave Fd-F image**, even though the
variable is called ``x_das`` and the comment talks about the "DAS_1PW floor".
Changing the anchor "from DAS to Fd-F" is not an experiment: there is nothing
to change.

What ARE two different images (correlation 0.931 on the common grid, see
verify_fdf1pw_same_grid.py) are the two Fd-F routes:

  A) OpHFdf.adjoint      -> the adjoint of the measurement operator. It is the
                            current anchor and the model input.
  B) das_pw(Fd-F, 1 PW)  -> the classical beamformer. It is the
                            "Schiffner Fd-F 1 PW" row of the canonical table.

Anchoring to B instead of A WOULD be a real experiment. But it is only worth
spending 9 h of GPU if B is BETTER than A: the anchor pulls the model towards
the image it is given, so anchoring to a worse image is counterproductive.

This script answers that by measuring A with the same protocol B was already
measured with, reusing the npys that gen_b1v5_picmus_seeds.py already saved in
``npys/das1pw_adjoint/`` (env_log [0,1] on the 612x388 grid: neither GPU nor
resampling is needed).

Usage
-----
    python evaluation/scripts/eval_anchor_candidates.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("THESIS_DATASETS", ROOT / "data"))  # PICMUS/ and Field II/, see data/README.md
sys.path.insert(0, str(ROOT / "evaluation" / "scripts"))
sys.path.insert(0, str(ROOT / "code"))

from eval_and_figs_final import PHANTOMS  # noqa: E402
from unet_ag.data.rf_loader import PICMUSLoader  # noqa: E402
from unet_ag.eval.evaluate_fieldii import _build_grid  # noqa: E402
from unet_ag.eval.evaluate_picmus import _eval_image  # noqa: E402

NPYS = ROOT / "evaluation" / "npys"
METR = ROOT / "evaluation" / "metrics"


def main() -> None:
    rows = []
    for key, mode, phantom, label in PHANTOMS:
        p = NPYS / "das1pw_adjoint" / f"{key}_das1pwadj.npy"
        if not p.exists():
            print(f"MISSING {p}")
            return
        loader = PICMUSLoader(DATASETS / "PICMUS", mode=mode,
                              phantom=phantom, normalize=True)
        gz_t, gx_t = _build_grid(loader[0].geometry, 612, 388)
        gz, gx = gz_t.numpy().astype(np.float64), gx_t.numpy().astype(np.float64)
        img = np.load(p).astype(np.float32)
        met = _eval_image(img, loader.phantom_path, phantom, gz, gx)
        rows.append({"method": "das1pw_adjoint", "seed": "", "phantom": key, **met})
        print(f"{key:16s} {met}")

    out = METR / "metrics_anchor_candidates.csv"
    cols = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "seed", "phantom"] +
                           [c for c in cols if c not in ("method", "seed", "phantom")])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")

    # ── comparison with the canonical das_pw(Fd-F, 1 PW) row ────────────
    canon = {}
    with open(METR / "metrics_picmus_final.csv") as fh:
        for r in csv.DictReader(fh):
            if r["method"] == "schiffner1pw":
                canon[r["phantom"]] = r

    def g(d, k):
        v = d.get(k, "")
        return float(v) if v not in ("", None) else float("nan")

    print("\n" + "=" * 74)
    print("A = OpHFdf.adjoint (CURRENT anchor)   B = das_pw Fd-F 1 PW (canonical row)")
    print(f"{'metric':22s} {'A (adjoint)':>12s} {'B (das_pw)':>12s}  {'better':>8s}")
    print("-" * 74)
    spec = [("sim_resolution", "fwhm_lat_med_mm", "sim FWHM_lat", True),
            ("exp_resolution", "fwhm_lat_med_mm", "exp FWHM_lat", True),
            ("sim_contrast",   "gcnr",            "sim gCNR",     False),
            ("exp_contrast",   "gcnr",            "exp gCNR",     False),
            ("sim_contrast",   "cnr_db",          "sim CNR dB",   False),
            ("exp_contrast",   "cnr_db",          "exp CNR dB",   False)]
    wins = {"A": 0, "B": 0}
    by_row = {r["phantom"]: r for r in rows}
    for ph, col, lab, lower in spec:
        a = g(by_row.get(ph, {}), col)
        b = g(canon.get(ph, {}), col)
        if a != a or b != b:
            continue
        better = "A" if ((a < b) if lower else (a > b)) else "B"
        wins[better] += 1
        print(f"{lab:22s} {a:12.3f} {b:12.3f}  {better:>8s}")
    print("-" * 74)
    print(f"  A wins {wins['A']}/6   B wins {wins['B']}/6")
    print()
    print("  If B wins comfortably, anchoring to das_pw(Fd-F) instead of the adjoint")
    print("  is a motivated experiment. If A wins, the current anchor already is the")
    print("  better of the two and there is nothing to test.")


if __name__ == "__main__":
    main()
