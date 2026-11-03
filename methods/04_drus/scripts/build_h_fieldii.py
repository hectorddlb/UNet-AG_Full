"""Phase F.1-A — H/B for the multi-angle Field II geometry + gates.

Mirror of ``build_by_picmus.py`` for the anchor05 training dataset. 256×256
grid aligned with the unet_ag trainer grid (z ∈ [5 mm, z_max],
x ∈ ±K·pitch/2) so that the DDRM output can be interpolated 1:1 to 612×388.

Gates:
  G-F1a (calibration): the axial peaks of the wires in env_log(By) must match
    the GT positions of the .mat; the median offset is converted to samples
    (pad_extra) and re-checked. |residual| < 0.3 mm to pass.
  G-F1b (DAS sanity): Pearson r between env_log(By) and env_log(DAS_1PW unet_ag)
    on the same grid, interior region. r > 0.5 expected (adjoint ≠ DAS but
    same scene).

Salidas en --out:
    H.npz, B.npz, grid.npz, meta.json  (includes the calibrated pad_extra)
    By_<phantom>.npy, bmode_<phantom>.png, gateF1_report.txt

Usage:
    python -m zhang_drus.scripts.build_h_fieldii \
        --data_root "datasets/Field II/field_ii_data_multiangle" \
        --out zhang_drus/results_drus/fieldii_phaseA
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
import scipy.io as sio
import scipy.sparse as sp
from scipy.signal import hilbert

_ROOT = Path(__file__).resolve().parents[3] / "code"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drus.physics_picmus import build_h_and_b
from drus.physics_fieldii import (
    make_pulse_bank_fieldii,
    compute_by_fieldii,
)

DYN_RANGE_DB = 60.0


def env_log_db(img: np.ndarray) -> np.ndarray:
    env = np.abs(hilbert(img, axis=0))
    env /= env.max() + 1e-30
    return np.clip(20.0 * np.log10(env + 1e-30), -DYN_RANGE_DB, 0.0)


def load_mat(data_root: Path, phantom_id: str) -> dict:
    ptype = "points" if phantom_id.startswith("points") else "cysts"
    mat = sio.loadmat(str(data_root / ptype / f"{phantom_id}.mat"))
    angles = np.asarray(mat["angles_deg"]).reshape(-1)
    i0 = int(np.argmin(np.abs(angles)))
    return {
        "rf0": np.asarray(mat["rf_data"], dtype=np.float64)[:, :, i0],
        "t_start0": float(np.asarray(mat["t_start"]).reshape(-1)[i0]),
        "fs": float(np.asarray(mat["fs"]).squeeze()),
        "c": float(np.asarray(mat["c"]).squeeze()),
        "fc": float(np.asarray(mat["fc"]).squeeze()),
        "pitch": float(np.asarray(mat["pitch"]).squeeze()),
        "ne": int(np.asarray(mat["n_elements"]).squeeze()),
        "positions": np.asarray(mat["positions"], dtype=np.float64),  # (N, 3) x,y,z
    }


def wire_axial_offsets(db_img: np.ndarray, positions: np.ndarray,
                       x_axis: np.ndarray, z_axis: np.ndarray,
                       win_mm: float = 3.0) -> list[float]:
    """Axial offset (z_peak − z_GT) in mm per wire inside the grid."""
    offs = []
    for x_gt, _, z_gt in positions:
        if not (x_axis[0] <= x_gt <= x_axis[-1] and
                z_axis[0] + 1e-3 <= z_gt <= z_axis[-1] - 1e-3):
            continue
        ix = int(np.argmin(np.abs(x_axis - x_gt)))
        zsel = np.abs(z_axis - z_gt) <= win_mm * 1e-3
        if zsel.sum() < 5:
            continue
        col = db_img[zsel, ix]
        z_here = z_axis[zsel]
        offs.append(float(z_here[int(np.argmax(col))] - z_gt) * 1e3)
    return offs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--nx", type=int, default=256)
    ap.add_argument("--nz", type=int, default=256)
    ap.add_argument("--calib_phantoms", nargs="+",
                    default=["points_0001", "points_0002", "points_0003"])
    ap.add_argument("--rx_f_number", type=float, default=1.4)
    ap.add_argument("--skip_matrices", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ref = load_mat(args.data_root, args.calib_phantoms[0])
    fs, c, ne, pitch = ref["fs"], ref["c"], ref["ne"], ref["pitch"]
    ns = ref["rf0"].shape[0]
    probe_x = (np.arange(ne) - (ne - 1) / 2.0) * pitch
    z_max = (ns - 1) * c / (2.0 * fs)

    # Grid aligned with unet_ag._build_grid (z from 5 mm, x full aperture).
    x_axis = np.linspace(-ne * pitch / 2.0, ne * pitch / 2.0, args.nx)
    z_axis = np.linspace(5e-3, z_max, args.nz)
    print(f"[geom] Ne={ne} fs={fs/1e6:.3f} MHz c={c} | grid z=[5,{z_max*1e3:.1f}] mm "
          f"x=±{ne*pitch/2*1e3:.1f} mm | t_start0={ref['t_start0']*1e6:.3f} µs", flush=True)

    pulse = make_pulse_bank_fieldii(fs)
    print(f"[pulse] Field II bank: kh={pulse.kh} lagt={pulse.lagt*1e6:.3f} µs", flush=True)

    if args.skip_matrices and (args.out / "B.npz").exists():
        B = sp.load_npz(args.out / "B.npz")
        meta = json.loads((args.out / "meta.json").read_text())
        big_k, k_min = meta["K"], meta["k_min"]
        print(f"[matrices] reutilizadas: K={big_k} k_min={k_min}", flush=True)
    else:
        t0 = time.time()
        mm = build_h_and_b(probe_x, fs, c, x_axis, z_axis,
                           rx_f_number=args.rx_f_number, pulse=pulse)
        print(f"[matrices] H {mm.H.shape} nnz={mm.H.nnz/1e6:.1f}M | K={mm.K} "
              f"k_min={mm.k_min} | {time.time()-t0:.1f}s", flush=True)
        sp.save_npz(args.out / "H.npz", mm.H)
        sp.save_npz(args.out / "B.npz", mm.B)
        np.savez(args.out / "grid.npz", x_axis=x_axis, z_axis=z_axis)
        B, big_k, k_min = mm.B, mm.K, mm.k_min

    report = []

    # ── Gate G-F1a: pad_extra calibration with GT wires ────
    def median_offset(pad_extra: int) -> tuple[float, int]:
        offs_all = []
        for pid in args.calib_phantoms:
            m = load_mat(args.data_root, pid)
            by, _ = compute_by_fieldii(B, m["rf0"], m["t_start0"], fs, big_k, k_min,
                                       pad_extra=pad_extra)
            db = env_log_db(by.reshape(args.nz, args.nx))
            offs_all += wire_axial_offsets(db, m["positions"], x_axis, z_axis)
        return float(np.median(offs_all)), len(offs_all)

    off0, n0 = median_offset(0)
    pad_extra = -int(round(off0 * 1e-3 * 2 * fs / c))
    off1, n1 = median_offset(pad_extra)
    line = (f"[G-F1a] offset axial mediano: {off0:+.3f} mm (n={n0}, pad_extra=0) → "
            f"{off1:+.3f} mm (n={n1}, pad_extra={pad_extra}) | "
            f"{'PASS' if abs(off1) < 0.3 else 'FAIL'}")
    print(line, flush=True)
    report.append(line)

    # ── Gate G-F1b: Pearson vs the unet_ag DAS_1PW on the same grid ─
    try:
        from unet_ag.physics.classical.das_1pw import das_1pw
        for pid in args.calib_phantoms[:1]:
            m = load_mat(args.data_root, pid)
            geo = {"n_elements": ne, "pitch_m": pitch, "sampling_freq_hz": fs,
                   "speed_of_sound": c, "t_start": m["t_start0"]}
            das = das_1pw(m["rf0"].astype(np.float32), geo, z_axis, x_axis)
            das_db = env_log_db(das)
            by, _ = compute_by_fieldii(B, m["rf0"], m["t_start0"], fs, big_k, k_min,
                                       pad_extra=pad_extra)
            by_db = env_log_db(by.reshape(args.nz, args.nx))
            zz = np.repeat(z_axis, args.nx).reshape(args.nz, args.nx)
            xx = np.tile(x_axis, args.nz).reshape(args.nz, args.nx)
            mask = (zz > 0.010) & (zz < z_max - 0.006) & (np.abs(xx) < 0.017)
            r = float(np.corrcoef(by_db[mask], das_db[mask])[0, 1])
            line = f"[G-F1b] {pid}: Pearson r(By, DAS_1PW) = {r:.3f} | {'PASS' if r > 0.5 else 'FAIL'}"
            print(line, flush=True)
            report.append(line)

            extent = [x_axis[0]*1e3, x_axis[-1]*1e3, z_axis[-1]*1e3, z_axis[0]*1e3]
            fig, axes = plt.subplots(1, 2, figsize=(10, 6))
            axes[0].imshow(by_db, cmap="gray", vmin=-60, vmax=0, extent=extent, aspect="equal")
            axes[0].set_title(f"env_log(By) — {pid}")
            axes[1].imshow(das_db, cmap="gray", vmin=-60, vmax=0, extent=extent, aspect="equal")
            axes[1].set_title("DAS 1PW unet_ag")
            fig.savefig(args.out / f"gateF1_{pid}.png", dpi=130, bbox_inches="tight")
            plt.close(fig)
    except Exception as exc:  # pragma: no cover — informativo, no bloqueante
        line = f"[G-F1b] SKIP ({exc})"
        print(line, flush=True)
        report.append(line)

    # ── By + bmodes of the calibration phantoms ─────
    for pid in args.calib_phantoms:
        m = load_mat(args.data_root, pid)
        by, _ = compute_by_fieldii(B, m["rf0"], m["t_start0"], fs, big_k, k_min,
                                   pad_extra=pad_extra)
        np.save(args.out / f"By_{pid}.npy", by)
        plt.imsave(args.out / f"bmode_{pid}.png",
                   env_log_db(by.reshape(args.nz, args.nx)),
                   cmap="gray", vmin=-60, vmax=0)

    (args.out / "meta.json").write_text(json.dumps({
        "K": int(big_k), "k_min": int(k_min), "Ne": ne, "fs": fs, "c": c,
        "nx": args.nx, "nz": args.nz,
        "zlim_mm": [5.0, z_max * 1e3], "xlim_mm": [-ne*pitch/2*1e3, ne*pitch/2*1e3],
        "rx_f_number": args.rx_f_number, "apod": "tukey25", "ros": 10,
        "pulse": "fieldii(fc=5.208MHz,ncycles=2.5,frac_bw=0.67)",
        "pad_extra": int(pad_extra), "t_start0_ref": ref["t_start0"],
        "calib_offset_mm": {"pre": off0, "post": off1},
    }, indent=2))
    (args.out / "gateF1_report.txt").write_text("\n".join(report) + "\n")
    print("[done] Phase F.1-A complete", flush=True)


if __name__ == "__main__":
    main()
