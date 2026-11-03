"""B1-v5 anchor05 inference on the 4 PICMUS phantoms, for the 3 seeds.

Outputs env_log [0,1] float32 (612x388 eval grid) npys:
  evaluation/npys/b1v5_<seed>/<key>_b1v5.npy
  evaluation/npys/das1pw_adjoint/<key>_das1pwadj.npy   (Fd-F adjoint: input and anchor)

Run from the repository root (PICMUS under $THESIS_DATASETS/PICMUS):
  python evaluation/scripts/gen_b1v5_picmus_seeds.py --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("THESIS_DATASETS", ROOT / "data"))  # PICMUS/ and Field II/, see data/README.md
sys.path.insert(0, str(ROOT / "code"))

from unet_ag.data.rf_loader import PICMUSLoader  # noqa: E402
from unet_ag.eval.evaluate_fieldii import (  # noqa: E402
    _build_grid,
    _load_b1_v3_ckpt,
    _reconstruct_b1,
)

SEEDS = {
    "s42": "methods/05_unet_ag_proposed/checkpoints/anchor05_s42.pt",
    "s1":  "methods/05_unet_ag_proposed/checkpoints/anchor05_s1.pt",
    "s2":  "methods/05_unet_ag_proposed/checkpoints/anchor05_s2.pt",
}

PHANTOMS = [
    ("sim_resolution", "sim", "resolution_distortion"),
    ("sim_contrast",   "sim", "contrast_speckle"),
    ("exp_resolution", "exp", "resolution_distortion"),
    ("exp_contrast",   "exp", "contrast_speckle"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--picmus", type=Path, default=DATASETS / "PICMUS")
    p.add_argument("--out", type=Path, default=ROOT / "evaluation" / "npys")
    p.add_argument("--nz", type=int, default=612)
    p.add_argument("--nx", type=int, default=388)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    print(f"device = {device}")

    for seed, ckpt_rel in SEEDS.items():
        ckpt = ROOT / ckpt_rel
        if not ckpt.exists():
            raise FileNotFoundError(f"missing checkpoint {ckpt}")
        out_dir = args.out / f"b1v5_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        model = _load_b1_v3_ckpt(ckpt, device, base_ch=32)
        print(f"== seed {seed}  ckpt={ckpt_rel}")
        for key, mode, phantom in PHANTOMS:
            t0 = time.perf_counter()
            loader = PICMUSLoader(args.picmus, mode=mode, phantom=phantom,
                                  normalize=True)
            sample = loader[0]
            grid_z, grid_x = _build_grid(sample.geometry, args.nz, args.nx)
            pred_env, das_env = _reconstruct_b1(model, sample, grid_z, grid_x,
                                                device)
            np.save(out_dir / f"{key}_b1v5.npy", pred_env.astype(np.float32))
            # DAS 1PW baseline of the SAME pipeline (adjoint), once per phantom
            das_path = args.out / "das1pw_adjoint" / f"{key}_das1pwadj.npy"
            das_path.parent.mkdir(parents=True, exist_ok=True)
            if not das_path.exists():
                np.save(das_path, das_env.astype(np.float32))
            print(f"  {key}: {time.perf_counter() - t0:.1f}s "
                  f"(pred min={pred_env.min():.3f} max={pred_env.max():.3f})")

    print("DONE")


if __name__ == "__main__":
    main()
