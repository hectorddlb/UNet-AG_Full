"""Phase B — dense BH + full SVD. Two stages to isolate memory:

  stage bh  : BH = B @ H by column blocks (GPU if available, otherwise CPU
              torch), written directly to BH.npy via memmap (peak RAM ~4 GB).
  stage svd : loads BH.npy (16 GiB) and runs MKL ILP64 sgesdd (~64 GiB of
              workspace). Automatic fallback to the Gram path (ssyevd of BHᵀBH,
              ~52 GiB peak) if the available RAM is not enough. Gate G-B1 +
              spectrum + U^T·By of the 4 phantoms.

The implementation lives in ``drus.preparation`` (the same one used by
``drus prepare svd``); this script keeps the per-stage interface with which it
was run on the cluster (``slurm/svd_bh.sbatch``).

Usage (inside the sbatch):
    python methods/04_drus/scripts/run_svd_bh.py --stage bh  --in_dir <operator> --out_dir <svd>
    python methods/04_drus/scripts/run_svd_bh.py --stage svd --in_dir <operator> --out_dir <svd>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from drus.preparation import stage_bh, stage_svd  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["bh", "svd"], required=True)
    ap.add_argument("--in_dir", type=Path, required=True, help="Phase A dir (B.npz/H.npz/By_*.npy)")
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "bh":
        stage_bh(args.in_dir, args.out_dir)
    else:
        stage_svd(args.in_dir, args.out_dir)


if __name__ == "__main__":
    main()
