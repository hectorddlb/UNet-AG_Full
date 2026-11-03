"""Phase C.1 — generates the fine-tuning dataset (287 Schiffner75 → 256×256 [0,1])
and a preview grid for visual inspection.

Usage:
    python zhang_drus/scripts/make_ft_dataset.py \
        --target_root results_schiffner_targets_fieldii \
        --out zhang_drus/results_drus/ft_data_schiffner287
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from drus.ft_dataset import build_ft_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--domain", choices=["signed", "envlog"], default="signed")
    args = ap.parse_args()

    n = build_ft_dataset(args.target_root, args.out, args.size, domain=args.domain)
    print(f"[ft_data] {n} images ({args.domain}) written to {args.out}", flush=True)

    files = sorted(args.out.glob("*.npy"))[:36]
    fig, axes = plt.subplots(6, 6, figsize=(12, 12))
    for ax, f in zip(axes.ravel(), files):
        img = np.load(f)
        if args.domain == "signed":
            # dB preview of the envelope to judge whether it "looks like a B-mode"
            env = np.abs(img)
            img = np.clip(20 * np.log10(env / (env.max() + 1e-30) + 1e-30), -60, 0)
            ax.imshow(img, cmap="gray", vmin=-60, vmax=0)
        else:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f.stem, fontsize=6)
        ax.axis("off")
    fig.suptitle(f"FT dataset preview — {n} Schiffner75 images [{args.domain}]")
    fig.savefig(args.out / "preview_grid.png", dpi=110, bbox_inches="tight")
    print(f"[ft_data] preview: {args.out}/preview_grid.png", flush=True)


if __name__ == "__main__":
    main()
