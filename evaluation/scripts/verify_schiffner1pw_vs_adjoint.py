"""Check whether "Schiffner Fd-F 1 PW" is the input of UNet-AG.

RATIONALE
---------
There are two code paths that apply the SAME Fd-F apodization to ONE single
plane wave, and the project treats them as different things:

  A) ``schiffner_das.das_pw(1 angle, GratingAngleLB(60,1.5), Tukey(0.2),
     NormalizationOn)``  -> the classical method, canonical protocol.
  B) ``OpHFdf.adjoint``  -> the adjoint of the measurement operator, whose
     DEFAULTS are exactly GratingAngleLB(60,1.5) + Tukey(0.2) + NormalizationOn.
     It is what produces ``x0``, the INPUT of UNet-AG, and therefore --with
     the ``out_conv`` residual initialized to zero-- its output at epoch 0.

If A and B match, the "Schiffner Fd-F 1 PW" row of the table is not just a
fourth 1 PW method: it is the exact baseline of the proposed model, and the
comparison UNet-AG vs that row measures what the NETWORK contributes,
separately from what the apodization contributes.

It also clears up a real ambiguity of the project: the ``das_1pw`` column of
the per-run ``eval_picmus.csv`` is computed with path B (adjoint, Fd-F),
whereas ``das1pw`` of the canonical table uses ``ConstantFNumber(0)``
--full aperture--. They are two different beamformers with the same label,
and this script quantifies how far apart they are.

WHAT IT COMPARES
----------------
The three images, all brought to the canonical protocol (env_log [0,1] on the
612x388 evaluation grid):

  schiffner1pw   npys/schiffner1pw/<key>_schiffner1pw.npy   (complex, scan grid)
  adjoint        npys/das1pw_adjoint/<key>_das1pwadj.npy    (already env_log 612x388)
  das1pw         npys/das1pw/<key>_das1pw.npy               (complex, scan grid)

Usage
-----
    python evaluation/scripts/verify_schiffner1pw_vs_adjoint.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("THESIS_DATASETS", ROOT / "data"))  # PICMUS/ and Field II/, see data/README.md
sys.path.insert(0, str(ROOT / "evaluation" / "scripts"))
sys.path.insert(0, str(ROOT / "code"))

from eval_and_figs_final import (  # noqa: E402
    PHANTOMS,
    env_log01,
    interp_to,
    load_scan_axes,
)
from unet_ag.data.rf_loader import PICMUSLoader  # noqa: E402
from unet_ag.eval.evaluate_fieldii import _build_grid  # noqa: E402

NPYS = ROOT / "evaluation" / "npys"


def to_eval_grid(path: Path, key: str, gz: np.ndarray, gx: np.ndarray) -> np.ndarray:
    """Complex on the scan grid -> env_log [0,1] on the evaluation grid."""
    src_z, src_x = load_scan_axes(key)
    return interp_to(env_log01(np.load(path)), src_z, src_x, gz, gx)


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    return {
        "corr": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "max_abs": float(np.abs(d).max()),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
    }


def main() -> None:
    print("Comparison on the canonical evaluation grid (612x388, env_log)\n")
    rows_ = []
    for key, mode, phantom, label in PHANTOMS:
        loader = PICMUSLoader(DATASETS / "PICMUS", mode=mode,
                              phantom=phantom, normalize=True)
        gz_t, gx_t = _build_grid(loader[0].geometry, 612, 388)
        gz, gx = gz_t.numpy().astype(np.float64), gx_t.numpy().astype(np.float64)

        p_s1 = NPYS / "schiffner1pw" / f"{key}_schiffner1pw.npy"
        p_adj = NPYS / "das1pw_adjoint" / f"{key}_das1pwadj.npy"
        p_das = NPYS / "das1pw" / f"{key}_das1pw.npy"
        for p in (p_s1, p_adj, p_das):
            if not p.exists():
                print(f"MISSING {p} — run gen_classical_picmus_fixed.py first")
                return

        s1 = to_eval_grid(p_s1, key, gz, gx)
        das = to_eval_grid(p_das, key, gz, gx)
        adj = np.load(p_adj).astype(np.float32)

        a = stats(s1, adj)   # the two 1 PW Fd-F paths
        b = stats(s1, das)   # Fd-F vs full aperture
        rows_.append((label, a, b))

        print(f"=== {label} ({key}) ===")
        print(f"  das_pw(Fd-F,1PW)  vs  OpHFdf.adjoint : "
              f"corr={a['corr']:.4f}  rmse={a['rmse']:.4f}  max|d|={a['max_abs']:.4f}")
        print(f"  das_pw(Fd-F,1PW)  vs  DAS 1 PW (F=0) : "
              f"corr={b['corr']:.4f}  rmse={b['rmse']:.4f}  max|d|={b['max_abs']:.4f}")
        print()

    print("=" * 72)
    print("READING")
    print("  If corr(Fd-F, adjoint) ~ 1 and its rmse << rmse(Fd-F, F=0), the two")
    print("  paths are the same beamformer and the input of UNet-AG IS")
    print("  Schiffner Fd-F 1 PW. If not, they are different methods and must be")
    print("  reported separately.")
    m_adj = float(np.mean([a["rmse"] for _l, a, _b in rows_]))
    m_das = float(np.mean([b["rmse"] for _l, _a, b in rows_]))
    c_adj = float(np.mean([a["corr"] for _l, a, _b in rows_]))
    print(f"\n  mean rmse(Fd-F, adjoint)    = {m_adj:.4f}   mean corr = {c_adj:.4f}")
    print(f"  mean rmse(Fd-F, DAS F=0)    = {m_das:.4f}")
    if m_das > 0:
        print(f"  ratio                        = {m_das / max(m_adj, 1e-12):.1f}x")


if __name__ == "__main__":
    main()
