"""Phase A — builds H/B (256×256 DRUS grid) and the By observations of the 4
PICMUS phantoms; runs gates G-A1 (comparison vs the official DAS npw=1) and
G-A2 (coverage of the ROIs inside the grid).

Usage:
    python -m zhang_drus.scripts.build_by_picmus \
        --picmus_root datasets/PICMUS --out zhang_drus/results_drus/phaseA [--nx 256 --nz 256]

Salidas en --out:
    H.npz, B.npz, grid.npz, meta.json
    By_{phantom}.npy               (vector Npix, float64)
    bmode_{phantom}.png            (env_log(By) only)
    gateA1_{phantom}.png           (side-by-side vs the official DAS 1PW + Pearson r)
    gateA2_report.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from drus.observations import PHANTOMS, compute_by, load_picmus_rf
from drus.physics_picmus import build_h_and_b, build_scan_grid

DYN_RANGE_DB = 60.0


def env_log_db(img: np.ndarray) -> np.ndarray:
    """Envelope (axial hilbert, axis=0) → dB normalized to [-DYN, 0]."""
    env = np.abs(hilbert(img, axis=0))
    env /= env.max() + 1e-30
    return np.clip(20.0 * np.log10(env + 1e-30), -DYN_RANGE_DB, 0.0)


def load_official_das(picmus_root: Path, phantom: str, npw: int = 1):
    acq, ph, tag = PHANTOMS[phantom]
    p = picmus_root / "reconstructed_image" / acq / ph / f"{ph}_{tag}_img_from_rf.hdf5"
    with h5py.File(p, "r") as f:
        g = f["US/US_DATASET0000"]
        slots = np.asarray(g["number_plane_waves"]).ravel()
        idx = int(np.where(slots == npw)[0][0])
        img = np.asarray(g["data/real"][idx])            # (X, Z) as stored
        x_ax = np.asarray(g["scan/x_axis"]).ravel()
        z_ax = np.asarray(g["scan/z_axis"]).ravel()
    if img.shape == (len(x_ax), len(z_ax)):
        img = img.T                                       # → (Z, X)
    mag = np.abs(img)
    mag /= mag.max() + 1e-30
    db = np.clip(20.0 * np.log10(mag + 1e-30), -DYN_RANGE_DB, 0.0)
    return db, z_ax, x_ax


def gate_a1(by_db: np.ndarray, picmus_root: Path, phantom: str,
            x_axis: np.ndarray, z_axis: np.ndarray, out_png: Path) -> float:
    das_db, das_z, das_x = load_official_das(picmus_root, phantom, npw=1)
    interp = RegularGridInterpolator((das_z, das_x), das_db,
                                     bounds_error=False, fill_value=-DYN_RANGE_DB)
    zz, xx = np.meshgrid(z_axis, x_axis, indexing="ij")
    das_on_grid = interp(np.stack([zz.ravel(), xx.ravel()], axis=1)).reshape(zz.shape)

    # Pearson on the interior region (avoids edges, TGC clamp and fill)
    mask = (zz > 0.012) & (zz < 0.044) & (np.abs(xx) < 0.017)
    r = float(np.corrcoef(by_db[mask], das_on_grid[mask])[0, 1])

    extent = [x_axis[0] * 1e3, x_axis[-1] * 1e3, z_axis[-1] * 1e3, z_axis[0] * 1e3]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
    for ax, im, title in [(axes[0], by_db, f"env_log(By) — {phantom}"),
                          (axes[1], das_on_grid, "official DAS 1PW (interp)")]:
        h = ax.imshow(im, cmap="gray", vmin=-DYN_RANGE_DB, vmax=0, extent=extent, aspect="equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x [mm]")
    axes[0].set_ylabel("z [mm]")
    fig.suptitle(f"Gate G-A1 {phantom}: Pearson r = {r:.3f} (interior region)")
    fig.colorbar(h, ax=axes, shrink=0.8, label="dB")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return r


def gate_a2(picmus_root: Path, x_axis: np.ndarray, z_axis: np.ndarray) -> str:
    lines = [f"Gate G-A2 — grid x[{x_axis[0]*1e3:.1f},{x_axis[-1]*1e3:.1f}] mm, "
             f"z[{z_axis[0]*1e3:.1f},{z_axis[-1]*1e3:.1f}] mm"]
    for phantom, (acq, ph, tag) in PHANTOMS.items():
        p = picmus_root / "database" / acq / ph / f"{ph}_{tag}_phantom.hdf5"
        if not p.exists():
            lines.append(f"  {phantom}: phantom.hdf5 does NOT exist — coverage cannot be checked here "
                         f"(masks_picmus.py defines the exp ROIs)")
            continue
        with h5py.File(p, "r") as f:
            g = f["US/US_DATASET0000"]
            if "scatterers_positions" in g:
                pos = np.asarray(g["scatterers_positions"])          # (3, N)
                inside = ((pos[0] >= x_axis[0]) & (pos[0] <= x_axis[-1]) &
                          (pos[2] >= z_axis[0]) & (pos[2] <= z_axis[-1]))
                lines.append(f"  {phantom}: {inside.sum()}/{pos.shape[1]} scatterers inside; "
                             f"z=[{pos[2].min()*1e3:.1f},{pos[2].max()*1e3:.1f}] mm, "
                             f"x=[{pos[0].min()*1e3:.1f},{pos[0].max()*1e3:.1f}] mm")
            for cx_k, cz_k, d_k in [("phantom_occlusionCenterX", "phantom_occlusionCenterZ",
                                     "phantom_occlusionDiameter")]:
                if cx_k in g:
                    cx = np.asarray(g[cx_k]).ravel()
                    cz = np.asarray(g[cz_k]).ravel()
                    dia = np.asarray(g[d_k]).ravel()
                    rad = (dia / 2 if len(dia) == len(cx) else np.full_like(cx, dia[0] / 2))
                    ok = ((cx - rad >= x_axis[0]) & (cx + rad <= x_axis[-1]) &
                          (cz - rad >= z_axis[0]) & (cz + rad <= z_axis[-1]))
                    lines.append(f"  {phantom}: {ok.sum()}/{len(cx)} cysts (with radius) inside; "
                                 f"z c ∈ [{cz.min()*1e3:.1f},{cz.max()*1e3:.1f}] mm, ⌀={dia[0]*1e3:.1f} mm")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--picmus_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--nx", type=int, default=256)
    ap.add_argument("--nz", type=int, default=256)
    ap.add_argument("--zlim_mm", type=float, nargs=2, default=[10.0, 46.0],
                    help="axial grid in mm; paper=[10,46], extended=[9,48] (covers 7/7 exp wires)")
    ap.add_argument("--xlim_mm", type=float, nargs=2, default=[-18.0, 18.0])
    ap.add_argument("--skip_matrices", action="store_true",
                    help="reuse the existing H.npz/B.npz in --out")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    x_axis, z_axis = build_scan_grid(
        args.nx, args.nz,
        x_lim=(args.xlim_mm[0] * 1e-3, args.xlim_mm[1] * 1e-3),
        z_lim=(args.zlim_mm[0] * 1e-3, args.zlim_mm[1] * 1e-3))

    # Reference geometry: sim/reso (like datasetReader(1,1,2) in MATLAB)
    ref = load_picmus_rf(args.picmus_root, "simu_reso")
    print(f"[geom] Ne={len(ref.probe_x)} fs={ref.fs/1e6:.4f} MHz c={ref.c} m/s", flush=True)

    if args.skip_matrices and (args.out / "B.npz").exists():
        B = sp.load_npz(args.out / "B.npz")
        meta = json.loads((args.out / "meta.json").read_text())
        big_k, k_min = meta["K"], meta["k_min"]
        print(f"[matrices] reutilizadas: K={big_k} k_min={k_min}", flush=True)
    else:
        t0 = time.time()
        mm = build_h_and_b(ref.probe_x, ref.fs, ref.c, x_axis, z_axis)
        print(f"[matrices] H {mm.H.shape} nnz={mm.H.nnz/1e6:.1f}M | "
              f"B {mm.B.shape} | K={mm.K} k_min={mm.k_min} | {time.time()-t0:.1f}s", flush=True)
        sp.save_npz(args.out / "H.npz", mm.H)
        sp.save_npz(args.out / "B.npz", mm.B)
        np.savez(args.out / "grid.npz", x_axis=x_axis, z_axis=z_axis)
        (args.out / "meta.json").write_text(json.dumps({
            "K": mm.K, "k_min": mm.k_min, "Ne": len(ref.probe_x),
            "fs": ref.fs, "c": ref.c, "nx": args.nx, "nz": args.nz,
            "zlim_mm": args.zlim_mm, "xlim_mm": args.xlim_mm,
            "rx_f_number": 1.4, "apod": "tukey25", "ros": 10,
        }, indent=2))
        B, big_k, k_min = mm.B, mm.K, mm.k_min

    report = [gate_a2(args.picmus_root, x_axis, z_axis)]
    print(report[0], flush=True)

    for phantom in PHANTOMS:
        rf = load_picmus_rf(args.picmus_root, phantom)
        assert abs(rf.fs - ref.fs) < 1 and abs(rf.c - ref.c) < 1e-6, f"geometry differs in {phantom}"
        by, _ = compute_by(B, rf, big_k, k_min, phantom)
        np.save(args.out / f"By_{phantom}.npy", by)

        img = by.reshape(args.nz, args.nx)
        db = env_log_db(img)
        plt.imsave(args.out / f"bmode_{phantom}.png", db, cmap="gray",
                   vmin=-DYN_RANGE_DB, vmax=0)
        r = gate_a1(db, args.picmus_root, phantom, x_axis, z_axis,
                    args.out / f"gateA1_{phantom}.png")
        line = f"[G-A1] {phantom}: Pearson r = {r:.3f}"
        print(line, flush=True)
        report.append(line)

    (args.out / "gateA2_report.txt").write_text("\n".join(report) + "\n")
    print("[done] Phase A complete", flush=True)


if __name__ == "__main__":
    main()
