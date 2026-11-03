# -*- coding: utf-8 -*-
"""Command-line interface of the DRUS replica.

    drus info
    drus prepare operator --picmus <PICMUS root>
    drus prepare svd
    drus prepare prior
    drus reconstruct --picmus <file.hdf5> --out bmode.png

After `pip install ".[drus]"` the `drus` command is available on the PATH. The
artifacts (operator, SVD and prior) are located with --artifacts or with the
DRUS_ARTIFACTS variable; see `drus.preparation`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .inference import DRUS, DYN_RANGE_DB, THESIS_GAMMA
from .observations import PHANTOMS, phantom_from_filename
from .preparation import (GRID_NX, GRID_NZ, GRID_X_MM, GRID_Z_MM,
                          build_operator, download_prior, picmus_root_from,
                          resolve_artifacts, stage_bh, stage_svd)


def _size_gib(p: Path) -> float:
    if p.is_file():
        return p.stat().st_size / 2**30
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2**30
    return 0.0


def _save_png(res, out: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    db = res.bmode * DYN_RANGE_DB - DYN_RANGE_DB
    fig, ax = plt.subplots(figsize=(4.8, 5.0))
    im = ax.imshow(db, cmap="gray", vmin=-DYN_RANGE_DB, vmax=0.0, aspect="equal",
                   extent=[res.x_axis[0] * 1e3, res.x_axis[-1] * 1e3,
                           res.z_axis[-1] * 1e3, res.z_axis[0] * 1e3])
    ax.set_xlabel("Lateral (mm)")
    ax.set_ylabel("Depth (mm)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03).set_label("dB")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _cmd_info(args) -> int:
    try:
        art = resolve_artifacts(args.artifacts, require=True)
    except FileNotFoundError:
        art = resolve_artifacts(args.artifacts, require=False)
    st = art.status()
    print("DRUS — replica of the diffusion method (Zhang et al., 2023)")
    print(f"  artifacts  : {art.root}")
    for name, path in (("operator", art.operator), ("svd", art.svd),
                       ("prior", art.prior)):
        mark = "ok     " if st[name] else "missing"
        print(f"    [{mark}] {name:9s} {path}  ({_size_gib(path):.1f} GiB)")
    print(f"  grid       : {GRID_NZ} x {GRID_NX}, x ∈ {list(GRID_X_MM)} mm, "
          f"z ∈ {list(GRID_Z_MM)} mm")
    print("  thesis γ per phantom (seed 0, prior without fine-tuning):")
    for k, g in THESIS_GAMMA.items():
        print(f"    {k:10s} γ = {g:g}")
    if art.missing():
        print("\nArtifacts missing. See `drus prepare --help`.")
        return 1
    return 0


def _cmd_prepare(args) -> int:
    art = resolve_artifacts(args.artifacts, require=False)
    if args.what == "operator":
        build_operator(picmus_root_from(args.picmus), art.operator)
    elif args.what == "svd":
        if not (art.operator / "B.npz").exists():
            print(f"The operator is missing in {art.operator}: "
                  f"run `drus prepare operator` first.", file=sys.stderr)
            return 1
        if args.stage in ("bh", "all"):
            stage_bh(art.operator, art.svd)
        if args.stage in ("svd", "all"):
            stage_svd(art.operator, art.svd)
            if not args.keep_bh:
                (art.svd / "BH.npy").unlink(missing_ok=True)
    elif args.what == "prior":
        download_prior(art.prior)
    print(f"  artifacts in {art.root}")
    return 0


def _cmd_reconstruct(args) -> int:
    try:
        model = DRUS.from_artifacts(args.artifacts, device=args.device)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    phantom = args.phantom or phantom_from_filename(args.picmus)
    print(f"Reconstructing {Path(args.picmus).name} ({phantom}, "
          f"γ = {args.gamma if args.gamma is not None else THESIS_GAMMA[phantom]:g}, "
          f"seed {args.seed}, {model.device}) ...", flush=True)
    res = model.reconstruct_picmus(args.picmus, phantom=phantom, gamma=args.gamma,
                                   seed=args.seed)
    out = Path(args.out)
    if out.suffix.lower() == ".npy":
        np.save(out, res.bmode)
    else:
        _save_png(res, out, f"DRUS — {phantom} (γ = {res.gamma:g})")
    print(f"  wrote {out}  ({res.seconds:.0f} s of sampling)")
    if args.save_x:
        np.save(args.save_x, res.x)
        print(f"  wrote {args.save_x} (signed output, prior domain)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="drus",
        description="DRUS: diffusion (DDRM) B-mode reconstruction from one plane wave.")
    p.add_argument("--artifacts", type=Path, default=None,
                   help="artifacts root (default $DRUS_ARTIFACTS)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("info", help="status of the artifacts and the thesis parameters")
    pi.set_defaults(func=_cmd_info)

    pp = sub.add_parser("prepare", help="build the artifacts (once)")
    pp.add_argument("what", choices=["operator", "svd", "prior"])
    pp.add_argument("--picmus", type=Path, default=None,
                    help="PICMUS root (default $THESIS_DATASETS/PICMUS)")
    pp.add_argument("--stage", choices=["bh", "svd", "all"], default="all",
                    help="svd: stages to run (on a cluster it is better to split them)")
    pp.add_argument("--keep-bh", action="store_true",
                    help="svd: keep BH.npy (17 GB) when done")
    pp.set_defaults(func=_cmd_prepare)

    pr = sub.add_parser("reconstruct", help="reconstruct a PICMUS phantom")
    pr.add_argument("--picmus", type=Path, required=True,
                    help="PICMUS RF file (*_dataset_rf.hdf5)")
    pr.add_argument("--phantom", choices=list(PHANTOMS), default=None,
                    help="phantom (by default inferred from the file name)")
    pr.add_argument("--gamma", type=float, default=None,
                    help="sampler γ (default: the thesis one)")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--out", type=Path, default=Path("bmode_drus.png"),
                    help="output .png or .npy (log envelope in [0, 1])")
    pr.add_argument("--save-x", type=Path, default=None,
                    help="also save the signed output (.npy)")
    pr.add_argument("--device", default="auto")
    pr.set_defaults(func=_cmd_reconstruct)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
