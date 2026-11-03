"""DRUS (prior without fine-tuning) on the 4 PICMUS phantoms, with the thesis γ.

Uses the ``drus`` package (``drus.inference.DRUS``), with the evaluation
conditions: prior published by OpenAI without fine-tuning, γ per phantom
(``THESIS_GAMMA``), sampling seed 0, 50 DDRM steps.

Outputs:
  evaluation/npys/drus/x_<ph>_g<γ>_s<seed>.npy   signed output, 256x256
  evaluation/npys/drus/grid.npz                  native grid axes

Run from the repository root, with the DRUS artifacts built
(``drus prepare ...``, see methods/04_drus/README.md):
  python evaluation/scripts/gen_drus_picmus.py
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("THESIS_DATASETS", ROOT / "data"))  # PICMUS/ and Field II/, see data/README.md
sys.path.insert(0, str(ROOT / "code"))

from drus.inference import DRUS, THESIS_GAMMA  # noqa: E402
from drus.observations import PHANTOMS, rf_path  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--picmus", type=Path, default=DATASETS / "PICMUS")
    p.add_argument("--artifacts", type=Path, default=None,
                   help="DRUS artifacts root (default $DRUS_ARTIFACTS)")
    p.add_argument("--out", type=Path, default=ROOT / "evaluation" / "npys" / "drus")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model = DRUS.from_artifacts(args.artifacts, device=args.device)
    shutil.copyfile(model.artifacts.operator / "grid.npz", args.out / "grid.npz")
    t_all = time.perf_counter()
    for ph in PHANTOMS:
        gamma = THESIS_GAMMA[ph]
        for seed in args.seeds:
            res = model.reconstruct_picmus(rf_path(args.picmus, ph), phantom=ph,
                                            gamma=gamma, seed=seed)
            out = args.out / f"x_{ph}_g{gamma:g}_s{seed}.npy"
            np.save(out, res.x)
            print(f"  {ph}: γ={gamma:g} seed={seed} {res.seconds:.0f}s -> {out.name}",
                  flush=True)
    print(f"DONE in {(time.perf_counter() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
