# -*- coding: utf-8 -*-
"""Command-line interface of UNet-AG.

    unetag info
    unetag reconstruct --picmus <file.hdf5> --out bmode.png
    unetag reconstruct --rf data.npy --geometry geo.json --out bmode.png

After `pip install .` the `unetag` command is available on the PATH.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .inference import DYN_RANGE_DB, Geometry, UNetAG, load_picmus_rf, to_decibels


def _save_png(bmode: np.ndarray, out: Path, gz_mm=None, gx_mm=None,
              title: str | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    db = to_decibels(bmode)
    fig, ax = plt.subplots(figsize=(4.6, 5.4))
    if gz_mm is not None and gx_mm is not None:
        n = int(np.searchsorted(gz_mm, 55.0, side="right"))
        im = ax.imshow(db[:n], cmap="gray", vmin=-DYN_RANGE_DB, vmax=0.0,
                       aspect="equal",
                       extent=[gx_mm[0], gx_mm[-1], gz_mm[n - 1], gz_mm[0]])
        ax.set_xlabel("Lateral (mm)")
        ax.set_ylabel("Depth (mm)")
    else:
        im = ax.imshow(db, cmap="gray", vmin=-DYN_RANGE_DB, vmax=0.0)
        ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03).set_label("dB")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _cmd_info(args) -> int:
    try:
        m = UNetAG.from_pretrained(seed=args.seed, device=args.device)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"UNet-AG — B-mode reconstruction from 1 plane wave")
    print(f"  checkpoint : {m.checkpoint}")
    print(f"  seed       : {args.seed}")
    print(f"  parameters : {m.n_parameters:,}")
    print(f"  device     : {m.device}")
    print(f"  grid       : {m.nz} x {m.nx}")
    print(f"  gain γ     : {float(m.model.gain):.4f}")
    return 0


def _cmd_reconstruct(args) -> int:
    if not args.picmus and not args.rf:
        print("Give --picmus <hdf5> or --rf <npy> --geometry <json>",
              file=sys.stderr)
        return 2
    try:
        model = UNetAG(args.ckpt, device=args.device) if args.ckpt \
            else UNetAG.from_pretrained(seed=args.seed, device=args.device)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    if args.picmus:
        rf, geom = load_picmus_rf(args.picmus, angle_index=args.angle_index)
        label = Path(args.picmus).stem
    else:
        rf = np.load(args.rf)
        if args.geometry:
            geom = Geometry(**json.loads(Path(args.geometry).read_text()))
        else:
            print("Warning: without --geometry the PICMUS geometry is assumed "
                  "(pitch 300 µm, fs 20.832 MHz, c 1540 m/s).", file=sys.stderr)
            geom = Geometry(n_elements=rf.shape[1], n_samples=rf.shape[0],
                            pitch_m=300e-6, sampling_freq_hz=20.832e6)
        label = Path(args.rf).stem

    print(f"Reconstructing {label}  "
          f"(RF {rf.shape[0]}x{rf.shape[1]}, {model.device}) ...")
    bmode, adjoint = model.reconstruct(rf, geom, return_adjoint=True)

    out = Path(args.out)
    if out.suffix.lower() == ".npy":
        np.save(out, bmode)
    else:
        gz, gx = model.grid(geom)
        _save_png(bmode, out, gz * 1e3, gx * 1e3, title=f"UNet-AG — {label}")
    print(f"  wrote {out}")

    if args.save_adjoint:
        a = Path(args.save_adjoint)
        if a.suffix.lower() == ".npy":
            np.save(a, adjoint)
        else:
            gz, gx = model.grid(geom)
            _save_png(adjoint, a, gz * 1e3, gx * 1e3,
                      title=f"Fd-F adjoint — {label}")
        print(f"  wrote {a} (initial solution)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="unetag",
        description="UNet-AG: B-mode reconstruction from a single plane wave.")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = dict(seed="s42", device="auto")
    pi = sub.add_parser("info", help="show the loaded model")
    pi.add_argument("--seed", default=common["seed"], choices=["s42", "s1", "s2"])
    pi.add_argument("--device", default=common["device"])
    pi.set_defaults(func=_cmd_info)

    pr = sub.add_parser("reconstruct", help="reconstruct a B-mode image")
    src = pr.add_argument_group("input")
    src.add_argument("--picmus", type=Path, help="PICMUS RF file (.hdf5)")
    src.add_argument("--rf", type=Path, help="raw RF (n_samples, n_elements) .npy")
    src.add_argument("--geometry", type=Path, help="geometry as JSON (with --rf)")
    src.add_argument("--angle-index", type=int, default=None,
                     help="PICMUS angle index (default: the 0° one)")
    pr.add_argument("--out", type=Path, default=Path("bmode.png"),
                    help="output .png or .npy (default bmode.png)")
    pr.add_argument("--save-adjoint", type=Path, default=None,
                    help="also save the initial solution (Fd-F adjoint)")
    pr.add_argument("--ckpt", type=Path, default=None,
                    help="alternative checkpoint")
    pr.add_argument("--seed", default=common["seed"], choices=["s42", "s1", "s2"])
    pr.add_argument("--device", default=common["device"])
    pr.set_defaults(func=_cmd_reconstruct)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
