"""Phase F.3 (prep) — packs the DRUS-noFT restorations as the trainer's target_root.

The DDRM teacher (``sample_ddrm.py --phantoms all``) leaves flat files
``x_<phantom>_g<γ>_s<seed>.npy`` (256×256, prior domain [-1,1]). The
``b1_v5_pulse_aug`` trainer expects targets in
``<root>/<ptype>/<phantom_id>/compounded.npy`` (like the Schiffner ones). This
script bridges the two:

    x_points_0001_g25_s0.npy  →  <out>/points/points_0001/compounded.npy
    x_cysts_0042_g25_s0.npy   →  <out>/cysts/cysts_0042/compounded.npy

IMPORTANT: the scale is NOT touched. The DRUS output already is an enveloped,
log-compressed B-mode in [-1,1]; the trainer consumes it with
``--target_is_bmode``, which only remaps [-1,1]→[0,1]. Re-applying
hilbert/log-compress would destroy it. The .npy is saved as is (float32,
256×256); the trainer does the bilinear resize to the 612×388 grid.

Usage:
    python zhang_drus/scripts/make_drus_targets.py \
        --teacher_dir zhang_drus/results_drus/fieldii_teacher_noft \
        --out         unet_ag/targets_drus_noft \
        --gamma 25 --seed 0
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

_RE = re.compile(r"^x_(?P<pid>(?:points|cysts)_\d+)_g(?P<g>[\d.]+)_s(?P<s>\d+)$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_dir", type=Path, required=True,
                    help="dir with x_<id>_g<γ>_s<seed>.npy (output of sample_ddrm)")
    ap.add_argument("--out", type=Path, required=True,
                    help="destination target_root (<out>/<ptype>/<id>/compounded.npy)")
    ap.add_argument("--gamma", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gtag = f"g{args.gamma:g}"
    stag = f"s{args.seed}"
    n_pts = n_cyst = 0
    skipped: list[str] = []
    for f in sorted(args.teacher_dir.glob("x_*.npy")):
        m = _RE.match(f.stem)
        if not m:
            skipped.append(f.name)
            continue
        if f"g{float(m['g']):g}" != gtag or f"s{m['s']}" != stag:
            continue
        pid = m["pid"]
        ptype = "points" if pid.startswith("points") else "cysts"
        arr = np.load(f).astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if lo < -1.5 or hi > 1.5:
            raise SystemExit(
                f"[abort] {f.name} outside the prior range [-1,1]: min={lo:.3f} max={hi:.3f} "
                "— is it really DRUS output and not RF? NOT packing.")
        dst = args.out / ptype / pid / "compounded.npy"
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, arr)
        if ptype == "points":
            n_pts += 1
        else:
            n_cyst += 1

    manifest = {"teacher_dir": str(args.teacher_dir), "gamma": args.gamma,
                "seed": args.seed, "n_points": n_pts, "n_cysts": n_cyst,
                "scale": "prior [-1,1] preserved (consume with --target_is_bmode)"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] {n_pts} points + {n_cyst} cysts → {args.out}"
          + (f"  | {len(skipped)} files ignored" if skipped else ""))


if __name__ == "__main__":
    main()
