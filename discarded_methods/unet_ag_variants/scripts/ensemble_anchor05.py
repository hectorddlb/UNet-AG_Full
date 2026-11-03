"""Ensemble inference: mean of the env+log predictions of N checkpoints.

By default: anchor05 (seed=42, seed=1, seed=2) → simple mean in the
envelope+log [0,1] image. Evaluates on PICMUS (4 phantoms) + (optionally)
Field II points/cysts. CPU-friendly: runs on the login node, does not touch
the GPU queue.

Usage:
  python -m unet_ag.scripts.ensemble_anchor05 \\
      --ckpts results_v2/b1_v5_anchor05/checkpoints/best.pt \\
              results_v2/b1_v5_anchor05_s1/checkpoints/best.pt \\
              results_v2/b1_v5_anchor05_s2/checkpoints/best.pt \\
      --picmus_root datasets/PICMUS \\
      --out_dir results_v2/ensemble_anchor05_3seed

Outputs:
  ensemble_picmus.csv       metrics (sim+exp × res+contrast)
  picmus_<mode>_<ph>_ensemble_bmode.png   3 ckpts + ensemble + DAS
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data.rf_loader import PICMUSLoader
from ..eval.evaluate_fieldii import _build_grid, _load_b1_v3_ckpt, _reconstruct_b1
from ..eval.evaluate_picmus import _eval_cysts, _eval_points
from ..eval.masks_picmus import points_wire_targets


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="Paths to checkpoint files to ensemble.")
    p.add_argument("--picmus_root", type=Path, required=True)
    p.add_argument("--out_dir",     type=Path, required=True)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--nz", type=int, default=612)
    p.add_argument("--nx", type=int, default=388)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"device = {device}")
    ckpt_paths = [Path(c) for c in args.ckpts]
    print(f"ensemble of {len(ckpt_paths)} ckpts:")
    for c in ckpt_paths:
        print(f"  - {c}")

    models = [_load_b1_v3_ckpt(c, device, base_ch=args.base_ch) for c in ckpt_paths]

    combos = [
        ("sim", "resolution_distortion"),
        ("sim", "contrast_speckle"),
        ("exp", "resolution_distortion"),
        ("exp", "contrast_speckle"),
    ]

    rows = []
    for mode, ph in combos:
        try:
            loader = PICMUSLoader(args.picmus_root, mode=mode, phantom=ph,
                                  normalize=True)
        except FileNotFoundError as e:
            print(f"  SKIP {mode}/{ph} — {e}")
            continue
        print(f"\n=== {mode}/{ph} ===")
        sample = loader[0]
        grid_z, grid_x = _build_grid(sample.geometry, args.nz, args.nx)
        grid_z_m, grid_x_m = grid_z.numpy(), grid_x.numpy()

        env_stack = []
        das_env = None
        for i, mdl in enumerate(models):
            pred_env, das_env_i = _reconstruct_b1(mdl, sample, grid_z, grid_x, device)
            env_stack.append(pred_env)
            if das_env is None:
                das_env = das_env_i
            print(f"  ckpt[{i}] env median={pred_env.mean():.3f}")
        env_arr = np.stack(env_stack, axis=0)
        ens_env = env_arr.mean(axis=0)

        if ph == "resolution_distortion":
            m_ens = _eval_points(ens_env, loader.phantom_path, grid_z_m, grid_x_m)
            m_das = _eval_points(das_env,  loader.phantom_path, grid_z_m, grid_x_m)
        else:
            m_ens = _eval_cysts(ens_env, loader.phantom_path, grid_z_m, grid_x_m)
            m_das = _eval_cysts(das_env,  loader.phantom_path, grid_z_m, grid_x_m)

        row = {"mode": mode, "phantom": ph}
        for k, v in m_ens.items():
            row[f"ens_{k}"] = v
        for k, v in m_das.items():
            row[f"das_{k}"] = v
        rows.append(row)
        print(f"  ensemble: {m_ens}")
        print(f"  das     : {m_das}")

        # B-mode figure: ckpt0..ckptN + ensemble + das
        gz_mm = grid_z.numpy() * 1e3
        gx_mm = grid_x.numpy() * 1e3
        wires = points_wire_targets(loader.phantom_path) if ph == "resolution_distortion" else None
        n_panels = len(models) + 2
        fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 6.4),
                                 constrained_layout=True)
        extent = [gx_mm[0], gx_mm[-1], gz_mm[-1], gz_mm[0]]
        for i, img in enumerate(env_stack):
            axes[i].imshow(img, cmap="gray", vmin=0, vmax=1, extent=extent, aspect="equal")
            axes[i].set_title(f"ckpt[{i}]")
        axes[len(models)].imshow(ens_env, cmap="gray", vmin=0, vmax=1, extent=extent, aspect="equal")
        axes[len(models)].set_title("ENSEMBLE", fontweight="bold")
        axes[len(models) + 1].imshow(das_env, cmap="gray", vmin=0, vmax=1, extent=extent, aspect="equal")
        axes[len(models) + 1].set_title("DAS_1PW")
        for ax in axes:
            if wires:
                ax.scatter([w[1] for w in wires], [w[0] for w in wires],
                           marker="+", c="red", s=50, linewidths=0.9)
            ax.set_xlabel("x (mm)")
        axes[0].set_ylabel("z (mm)")
        fig.suptitle(f"{mode}/{ph}  —  ensemble {len(models)} ckpts", fontsize=11)
        fig.savefig(args.out_dir / f"picmus_{mode}_{ph}_ensemble_bmode.png", dpi=120)
        plt.close(fig)
        print(f"  fig → picmus_{mode}_{ph}_ensemble_bmode.png")

    if rows:
        keys = list(rows[0].keys())
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(args.out_dir / "ensemble_picmus.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows → {args.out_dir/'ensemble_picmus.csv'}")
    print("DONE.")


if __name__ == "__main__":
    main()
