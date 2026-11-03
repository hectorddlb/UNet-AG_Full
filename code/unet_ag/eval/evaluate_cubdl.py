"""Evaluate a trained B1 model on CUBDL OOD subset (in-vivo + diverse phantoms).

CUBDL has no canonical geometric masks (cysts/wires) shared across files, so the
eval is **distributional** rather than point-by-point:

  - Reconstructs B1 prediction + 1-PW DAS baseline per phantom.
  - Computes envelope statistics on both: median, p95, dynamic-range proxy.
  - Quantifies how much B1 deviates from DAS in OOD (SSIM, MSE on env+log).
  - Writes one CSV row per phantom + (optional) B-mode PNG pairs.

CLI:
  python -m unet_ag.eval.evaluate_cubdl \\
      --ckpt        results_v2/b1_v5_anchor05/checkpoints/best.pt \\
      --cubdl_root  datasets/CUBDL \\
      --out         results_v2/b1_v5_anchor05/eval_cubdl.csv \\
      --save_bmode_dir results_v2/b1_v5_anchor05/cubdl_bmodes \\
      --max_phantoms 20
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from ..data.rf_loader import CUBDLLoader
from .evaluate_fieldii import (
    _build_grid,
    _envelope_log_compress,
    _load_b1_v3_ckpt,
    _reconstruct_b1,
)


def _ssim_2d(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM on env+log compressed images already in [0,1].

    Uses the structural-similarity defaults from scikit-image (data_range=1).
    Falls back to a manual implementation if scikit-image is unavailable.
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        return float(ssim(a, b, data_range=1.0))
    except Exception:
        mu_a, mu_b = a.mean(), b.mean()
        va, vb = a.var(), b.var()
        cov = ((a - mu_a) * (b - mu_b)).mean()
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
        den = (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
        return float(num / den) if den > 0 else float("nan")


def _envelope_stats(env_img: np.ndarray) -> dict:
    """Statistics on env+log [0,1] image."""
    flat = env_img.reshape(-1)
    return {
        "env_median": float(np.median(flat)),
        "env_p95":    float(np.percentile(flat, 95)),
        "env_p05":    float(np.percentile(flat, 5)),
        "env_dyn":    float(np.percentile(flat, 95) - np.percentile(flat, 5)),
    }


def _save_bmode_pair(out_dir: Path, phantom_id: str,
                     pred_env: np.ndarray, das_env: np.ndarray,
                     gz_mm: np.ndarray, gx_mm: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    extent = [gx_mm[0], gx_mm[-1], gz_mm[-1], gz_mm[0]]
    fig, axes = plt.subplots(1, 2, figsize=(8, 6), constrained_layout=True)
    axes[0].imshow(das_env, cmap="gray", vmin=0.0, vmax=1.0,
                   extent=extent, aspect="equal")
    axes[0].set_title("DAS 1 PW")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("z (mm)")
    axes[1].imshow(pred_env, cmap="gray", vmin=0.0, vmax=1.0,
                   extent=extent, aspect="equal")
    axes[1].set_title("B1 prediction")
    axes[1].set_xlabel("x (mm)")
    fig.suptitle(phantom_id, fontsize=10)
    fig.savefig(out_dir / f"{phantom_id}.png", dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       type=Path, required=True)
    p.add_argument("--cubdl_root", type=Path, required=True)
    p.add_argument("--out",        type=Path, required=True)
    p.add_argument("--save_bmode_dir", type=Path, default=None)
    p.add_argument("--institutions", nargs="*", default=None,
                   help="Filter by file-prefix institution (e.g. INS UFL MYO OSL JHU).")
    p.add_argument("--max_phantoms", type=int, default=None)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--nz", type=int, default=612)
    p.add_argument("--nx", type=int, default=388)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"device = {device}")
    print(f"ckpt   = {args.ckpt}")

    model = _load_b1_v3_ckpt(args.ckpt, device, base_ch=args.base_ch)

    loader = CUBDLLoader(args.cubdl_root, normalize=True, keep_all_angles=False,
                         institutions=args.institutions)
    n = len(loader) if args.max_phantoms is None else min(len(loader), args.max_phantoms)
    print(f"CUBDL: {len(loader)} files found, evaluating first {n}.")

    rows: list[dict] = []
    for i in range(n):
        try:
            sample = loader[i]
        except Exception as e:
            print(f"  [{i}] LOAD ERROR — {e}")
            continue
        phantom_id = sample.phantom_id
        institution = phantom_id.split("__")[-1][:3].upper()
        print(f"\n=== [{i+1}/{n}] {phantom_id}  (inst={institution}) ===")

        grid_z, grid_x = _build_grid(sample.geometry, args.nz, args.nx)
        try:
            pred_env, das_env = _reconstruct_b1(model, sample, grid_z, grid_x, device)
        except Exception as e:
            print(f"  RECONSTRUCT ERROR — {e}")
            continue

        s_pred = _envelope_stats(pred_env)
        s_das  = _envelope_stats(das_env)
        ssim_v = _ssim_2d(pred_env, das_env)
        mse_v  = float(np.mean((pred_env - das_env) ** 2))

        row = {
            "phantom_id":  phantom_id,
            "institution": institution,
            "n_samples":   sample.geometry["n_samples"],
            "fc_mhz":      sample.geometry["fc"] / 1e6,
            "fs_mhz":      sample.geometry["sampling_freq_hz"] / 1e6,
            "pitch_um":    sample.geometry["pitch_m"] * 1e6,
            "ssim_pred_vs_das": ssim_v,
            "mse_pred_vs_das":  mse_v,
        }
        for k, v in s_pred.items():
            row[f"b1_{k}"] = v
        for k, v in s_das.items():
            row[f"das_{k}"] = v
        rows.append(row)
        print(f"  ssim={ssim_v:.3f}  mse={mse_v:.4f}  "
              f"b1_med={s_pred['env_median']:.3f}  das_med={s_das['env_median']:.3f}")

        if args.save_bmode_dir is not None:
            try:
                _save_bmode_pair(args.save_bmode_dir, phantom_id, pred_env, das_env,
                                 grid_z.numpy() * 1e3, grid_x.numpy() * 1e3)
            except Exception as e:
                print(f"  BMODE SAVE ERROR — {e}")

    if not rows:
        print("\nNo phantoms evaluated — exiting without writing CSV.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    insts: dict[str, list[float]] = {}
    for r in rows:
        insts.setdefault(r["institution"], []).append(r["ssim_pred_vs_das"])
    print("\n--- Summary by institution ---")
    for k in sorted(insts):
        vs = np.asarray(insts[k])
        print(f"  {k}: n={len(vs):3d}  ssim mean={vs.mean():.3f}  std={vs.std(ddof=1) if len(vs)>1 else 0:.3f}")
    print(f"\nWrote {len(rows)} rows → {args.out}")


if __name__ == "__main__":
    main()
