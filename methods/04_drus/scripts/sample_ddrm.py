"""Phase D — DDRM restoration of the 4 PICMUS phantoms with a γ grid and multiple seeds.

Usage:
    python zhang_drus/scripts/sample_ddrm.py \
        --svd_dir zhang_drus/results_drus/phaseB_ext \
        --ckpt zhang_drus/results_drus/ft_prior_schiffner287/ema_0.9999_006000.pt \
        --out zhang_drus/results_drus/phaseD_ft287_ext \
        [--phantoms simu_reso simu_cont expe_reso expe_cont] \
        [--gammas 15 20 25 50 75 100] [--seeds 0 1 2] [--timesteps 50]

For each (phantom, γ, seed) it saves ``x_{phantom}_g{γ}_s{seed}.npy`` (x_final
channel mean, (256,256) float32, prior domain [-1,1]) and an inspection PNG.
The γ grid plays the role of the anchor05 β sweep; the paper's γ for DRUS are
[25, 50, 75, 20] per phantom.
"""

from __future__ import annotations

import argparse
import json
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
from drus.ddrm_sampler import SpectralOps, ddrm_sample, expand_uty
from drus.prior import DIFFUSION_FLAGS, MODEL_FLAGS

PHANTOMS_DEFAULT = ["simu_reso", "simu_cont", "expe_reso", "expe_cont"]


def load_model(ckpt: Path, device: str):
    model, _ = create_model_and_diffusion(**MODEL_FLAGS, **DIFFUSION_FLAGS)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device)
    if MODEL_FLAGS["use_fp16"]:
        model.convert_to_fp16()
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--svd_dir", type=Path, required=True, help="Phase B dir (S/Vt/Uty_*)")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phantoms", nargs="+", default=PHANTOMS_DEFAULT,
                    help="explicit list, or 'all' to glob the Uty_*.npy of svd_dir")
    ap.add_argument("--gammas", nargs="+", type=float, default=[15, 20, 25, 50, 75, 100])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--timesteps", type=int, default=50)
    ap.add_argument("--eta", type=float, default=0.85)
    ap.add_argument("--etaB", type=float, default=1.0)
    ap.add_argument("--no_png", action="store_true", help="skip PNGs (useful in massive runs)")
    ap.add_argument("--skip_existing", action="store_true",
                    help="skip (phantom,γ,seed) whose .npy already exists — resumable")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.phantoms == ["all"]:
        args.phantoms = sorted(p.stem[len("Uty_"):] for p in args.svd_dir.glob("Uty_*.npy"))
        print(f"[ddrm] 'all' mode: {len(args.phantoms)} phantoms from {args.svd_dir}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ddrm] cargando SVD de {args.svd_dir} …", flush=True)
    t0 = time.time()
    s = np.load(args.svd_dir / "S.npy")
    vt = np.load(args.svd_dir / "Vt.npy")            # 17 GB → RAM
    ops = SpectralOps(s, vt, spectral_device="cpu")
    print(f"[ddrm] SVD in RAM ({time.time()-t0:.0f}s) | model: {args.ckpt.name}", flush=True)
    model = load_model(args.ckpt, dev)

    betas = np.linspace(1e-4, 0.02, 1000)
    manifest = []
    for phantom in args.phantoms:
        uty = expand_uty(np.load(args.svd_dir / f"Uty_{phantom}.npy"))
        for gamma in args.gammas:
            for seed in args.seeds:
                tag = f"{phantom}_g{gamma:g}_s{seed}"
                if args.skip_existing and (args.out / f"x_{tag}.npy").exists():
                    continue
                res = ddrm_sample(model, betas, ops, uty, gamma=max(gamma, 0.1),
                                  timesteps=args.timesteps, eta=args.eta,
                                  eta_b=args.etaB, seed=seed, model_device=dev,
                                  verbose=False)
                np.save(args.out / f"x_{tag}.npy", res.x_mean)
                if not args.no_png:
                    plt.imsave(args.out / f"x_{tag}.png",
                               np.clip(res.x_mean, -1, 1), cmap="gray", vmin=-1, vmax=1)
                manifest.append({"phantom": phantom, "gamma": gamma, "seed": seed,
                                 "file": f"x_{tag}.npy", "seconds": round(res.seconds, 1)})
                print(f"[ddrm] {tag}: {res.seconds:.0f}s", flush=True)

    (args.out / "manifest.json").write_text(json.dumps({
        "ckpt": str(args.ckpt), "svd_dir": str(args.svd_dir),
        "timesteps": args.timesteps, "eta": args.eta, "etaB": args.etaB,
        "runs": manifest}, indent=2))
    print(f"[done] {len(manifest)} restorations in {args.out}", flush=True)


if __name__ == "__main__":
    main()
