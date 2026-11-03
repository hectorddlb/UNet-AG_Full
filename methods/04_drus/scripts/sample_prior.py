"""Gate G-C1 — unconditional samples of the prior (pre/post fine-tuning).

The samples of the base ckpt should look ImageNet-ish; those of the FT ckpt
should look like US images (Schiffner speckle). Visual comparison.

Usage:
    python zhang_drus/scripts/sample_prior.py \
        --ckpt zhang_drus/results_drus/ft_prior_schiffner287/ema_0.9999_006000.pt \
        --out zhang_drus/results_drus/ft_prior_schiffner287/samples_ema6000.png \
        --n 4 --steps 250
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3] / "code"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from guided_diffusion.script_util import create_model_and_diffusion
from drus.prior import DIFFUSION_FLAGS, MODEL_FLAGS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=str, default="250")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    flags = {**MODEL_FLAGS, **DIFFUSION_FLAGS, "timestep_respacing": args.steps}
    model, diffusion = create_model_and_diffusion(**flags)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    if flags["use_fp16"]:
        model.convert_to_fp16()
    model.eval()

    torch.manual_seed(args.seed)
    t0 = time.time()
    with torch.no_grad():
        x = diffusion.p_sample_loop(
            model, (args.n, 3, 256, 256), clip_denoised=True, progress=True)
    x = x.clamp(-1, 1).cpu().numpy()
    print(f"[sample] {args.n} samples in {time.time()-t0:.0f}s", flush=True)

    fig, axes = plt.subplots(1, args.n, figsize=(4 * args.n, 4.4))
    axes = np.atleast_1d(axes)
    for ax, img in zip(axes, x):
        ax.imshow(img.mean(axis=0), cmap="gray", vmin=-1, vmax=1)
        ax.axis("off")
    fig.suptitle(f"Unconditional samples — {args.ckpt.name} ({args.steps} steps)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"[sample] guardado {args.out}", flush=True)


if __name__ == "__main__":
    main()
