"""Generate DAS_1PW (Schiffner library, F_number=0 = full aperture) and
B1-v5_anchor05 inference .npy files for the 4 PICMUS phantoms.

Outputs (env_log [0,1] B-mode, float32, shape (Nz, Nx)):
  results_v2/das_1pw_picmus/{key}_das.npy
  results_v2/b1_v5_anchor05_inference/{key}_b1v5.npy

Where ``{key}`` is one of:
  sim_resolution, sim_contrast, exp_resolution, exp_contrast

For the model side, ``(Nz, Nx)`` matches the eval grid (default 612 x 388),
identical to the published ``evaluate_picmus`` configuration. For DAS_1PW we
use the native PICMUS scan grid (matches Schiffner 4-panel) — comparison is
qualitative B-mode, not pixel-aligned.

Run from repo root. GPU recommended (CUDA auto). No plotting.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.signal import hilbert

# Schiffner library import
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]          # repository root
sys.path.insert(0, str(_REPO / "code"))

from schiffner_das import (  # noqa: E402
    ConstantFNumber,
    das_pw,
    load_picmus_hdf5,
    load_picmus_scan,
)

from unet_ag.data.rf_loader import PICMUSLoader  # noqa: E402
from unet_ag.eval.evaluate_fieldii import (  # noqa: E402
    _build_grid,
    _load_b1_v3_ckpt,
    _reconstruct_b1,
)


DYN_RANGE_DB = 60.0


@dataclass
class PhantomSpec:
    key: str
    rf_path: str
    scan_path: str
    loader_mode: str
    loader_phantom: str


PHANTOMS = [
    PhantomSpec(
        key="sim_resolution",
        rf_path="database/simulation/resolution_distorsion/resolution_distorsion_simu_dataset_rf.hdf5",
        scan_path="database/simulation/resolution_distorsion/resolution_distorsion_simu_scan.hdf5",
        loader_mode="sim",
        loader_phantom="resolution_distortion",
    ),
    PhantomSpec(
        key="sim_contrast",
        rf_path="database/simulation/contrast_speckle/contrast_speckle_simu_dataset_rf.hdf5",
        scan_path="database/simulation/contrast_speckle/contrast_speckle_simu_scan.hdf5",
        loader_mode="sim",
        loader_phantom="contrast_speckle",
    ),
    PhantomSpec(
        key="exp_resolution",
        rf_path="database/experiments/resolution_distorsion/resolution_distorsion_expe_dataset_rf.hdf5",
        scan_path="database/experiments/resolution_distorsion/resolution_distorsion_expe_scan.hdf5",
        loader_mode="exp",
        loader_phantom="resolution_distortion",
    ),
    PhantomSpec(
        key="exp_contrast",
        rf_path="database/experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5",
        scan_path="database/experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5",
        loader_mode="exp",
        loader_phantom="contrast_speckle",
    ),
]


def _to_envlog_complex(image_complex: np.ndarray,
                       dyn_range_db: float = DYN_RANGE_DB) -> np.ndarray:
    """Envelope-log compression from a complex compound image."""
    env = np.abs(image_complex)
    env_max = float(env.max()) + 1e-12
    db = 20.0 * np.log10(env / env_max + 1e-30)
    db = np.clip(db, -dyn_range_db, 0.0)
    return ((db + dyn_range_db) / dyn_range_db).astype(np.float32)


def beamform_das_1pw(ds, gx, gz, *, device: str, z_chunk: int = 32):
    """DAS over the central plane wave (theta nearest 0) with F-number = 0
    (full aperture, no apodization gating)."""
    angles_raw = ds.steering_angles_rad
    if hasattr(angles_raw, "detach"):
        angles = angles_raw.detach().cpu().numpy()
    else:
        angles = np.asarray(angles_raw)
    aidx = int(np.argmin(np.abs(angles)))
    angle_rad = float(angles[aidx])
    fn0 = ConstantFNumber(0.0)
    res = das_pw(
        gx, gz,
        ds.rf_for_angle(aidx),
        ds.f_s, angle_rad,
        ds.element_width, ds.element_pitch, ds.c_avg,
        f_bounds=(ds.f_lb, ds.f_ub),
        index_t0=ds.index_t0,
        F_number=fn0,
        z_chunk=z_chunk,
        device=device,
    )
    return res.image, aidx


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--picmus", type=Path, default=Path("datasets/PICMUS"))
    p.add_argument("--out_das", type=Path,
                   default=Path("results_v2/das_1pw_picmus"))
    p.add_argument("--out_b1", type=Path,
                   default=Path("results_v2/b1_v5_anchor05_inference"))
    p.add_argument("--ckpt", type=Path,
                   default=Path("results_v2/b1_v5_anchor05/checkpoints/best.pt"))
    p.add_argument("--nz", type=int, default=612)
    p.add_argument("--nx", type=int, default=388)
    p.add_argument("--base_ch", type=int, default=32)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--z_chunk", type=int, default=32)
    args = p.parse_args()

    args.out_das.mkdir(parents=True, exist_ok=True)
    args.out_b1.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"device       = {device}")
    print(f"PICMUS root  = {args.picmus.resolve()}")
    print(f"ckpt         = {args.ckpt.resolve()}")
    print(f"out (DAS)    = {args.out_das.resolve()}")
    print(f"out (B1-v5)  = {args.out_b1.resolve()}")
    print()

    if not args.ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.ckpt}")

    # Load B1-v5 (B1V3Residual architecture)
    print("Loading B1-v5_anchor05 ...")
    model = _load_b1_v3_ckpt(args.ckpt, device, base_ch=args.base_ch)
    with torch.no_grad():
        g = model.gamma
    print(f"  gamma (loaded): gain={g['gain']:.4f}")
    print()

    t_total = time.perf_counter()
    for spec in PHANTOMS:
        print(f"---- {spec.key} ----------------------------------------")
        t0 = time.perf_counter()

        # ===== DAS 1 PW (Schiffner library, F_number=0) ==================
        ds = load_picmus_hdf5(args.picmus / spec.rf_path)
        grid = load_picmus_scan(args.picmus / spec.scan_path)
        gx, gz = grid.x, grid.z
        print(f"  scan grid: {len(gz)} x {len(gx)}; angles avail: {ds.n_angles}")

        img_c, aidx = beamform_das_1pw(ds, gx, gz, device=device,
                                       z_chunk=args.z_chunk)
        img_c_np = img_c.detach().cpu().numpy()
        env_das = _to_envlog_complex(img_c_np)
        das_npy = args.out_das / f"{spec.key}_das.npy"
        np.save(das_npy, env_das)
        print(f"  DAS_1PW  (theta_idx={aidx}, F=0): {env_das.shape} "
              f"-> {das_npy.name}")

        # ===== B1-v5 inference (eval grid 612 x 388) =====================
        loader = PICMUSLoader(args.picmus, mode=spec.loader_mode,
                              phantom=spec.loader_phantom, normalize=True)
        sample = loader[0]
        grid_z, grid_x = _build_grid(sample.geometry, args.nz, args.nx)
        pred_env, _ = _reconstruct_b1(model, sample, grid_z, grid_x, device)
        b1_npy = args.out_b1 / f"{spec.key}_b1v5.npy"
        np.save(b1_npy, pred_env.astype(np.float32))
        print(f"  B1-v5_anchor05 ({args.nz}x{args.nx}): "
              f"min={pred_env.min():.3f} max={pred_env.max():.3f} "
              f"-> {b1_npy.name}")

        print(f"  total: {time.perf_counter() - t0:.1f}s")
        print()

    print(f"All done in {time.perf_counter() - t_total:.1f}s")


if __name__ == "__main__":
    main()
