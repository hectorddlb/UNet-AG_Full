"""DDRM observations: load PICMUS 1PW (0°) RF + TGC + normalization + By.

Port of ``MATLAB/Results/02_picmus/BH/compute_By.m`` (DRUS-v1). The per-phantom
TGC/scale parameters come from the comments of the MATLAB script itself:

    tgc alpha : simu_reso=4.5, simu_cont=expe_reso=expe_cont=4.0
    scale     : simu_reso=10, simu_cont=30, expe_reso=10, expe_cont=30
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

# (acquisition_dir, phantom_dir, tag) by the paper's short name
PHANTOMS = {
    "simu_reso": ("simulation", "resolution_distorsion", "simu"),
    "simu_cont": ("simulation", "contrast_speckle", "simu"),
    "expe_reso": ("experiments", "resolution_distorsion", "expe"),
    "expe_cont": ("experiments", "contrast_speckle", "expe"),
}

TGC_ALPHA = {"simu_reso": 4.5, "simu_cont": 4.0, "expe_reso": 4.0, "expe_cont": 4.0}
NORM_SCALE = {"simu_reso": 10.0, "simu_cont": 30.0, "expe_reso": 10.0, "expe_cont": 30.0}

ZERO_DEG_IDX = 37  # transmission 38 in MATLAB (1-based) = angle 0° of 75


@dataclass
class PicmusRF:
    y: np.ndarray            # (D, Ne) — samples × elements, raw RF of the 0°
    fs: float
    c: float
    initial_time: float
    probe_x: np.ndarray      # (Ne,)
    angle_deg: float


def rf_path(picmus_root: Path, phantom: str) -> Path:
    acq, ph, tag = PHANTOMS[phantom]
    return picmus_root / "database" / acq / ph / f"{ph}_{tag}_dataset_rf.hdf5"


def phantom_from_filename(path) -> str:
    """Short phantom name from the PICMUS RF file.

    ``resolution_distorsion_expe_dataset_rf.hdf5`` -> ``"expe_reso"``. Raises
    ``ValueError`` if the name matches none of the four.
    """
    name = Path(path).name
    for short, (_acq, ph, tag) in PHANTOMS.items():
        if name.startswith(f"{ph}_{tag}_"):
            return short
    raise ValueError(f"no PICMUS phantom recognized in {name!r}; "
                     f"give it explicitly ({', '.join(PHANTOMS)})")


def load_picmus_rf(picmus_root: Path, phantom: str) -> PicmusRF:
    return load_picmus_rf_file(rf_path(picmus_root, phantom))


def load_picmus_rf_file(path: Path) -> PicmusRF:
    """Read the 0° angle RF of a PICMUS ``*_dataset_rf.hdf5`` file."""
    with h5py.File(path, "r") as f:
        g = f["US/US_DATASET0000"]
        data = g["data/real"][ZERO_DEG_IDX]          # (Ne, D)
        ang = float(np.degrees(g["angles"][ZERO_DEG_IDX]))
        fs = float(np.asarray(g["sampling_frequency"]).ravel()[0])
        c = float(np.asarray(g["sound_speed"]).ravel()[0])
        t0 = float(np.asarray(g["initial_time"]).ravel()[0])
        probe = g["probe_geometry"][:]
    assert abs(ang) < 0.1, f"transmission {ZERO_DEG_IDX} is not 0°: {ang}°"
    return PicmusRF(
        y=np.asarray(data, dtype=np.float64).T,      # (D, Ne)
        fs=fs, c=c, initial_time=t0,
        probe_x=np.asarray(probe[0], dtype=np.float64),
        angle_deg=ang,
    )


def compute_by(
    B: sp.csr_matrix,
    rf: PicmusRF,
    big_k: int,
    k_min: int,
    phantom: str,
    tgc_alpha: float | None = None,
    norm_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (By, y_processed). Replica of the Input section of compute_By.m.

    ``tgc_alpha`` and ``norm_scale`` default to the phantom values; giving
    them explicitly lets the chain be applied to your own data.
    """
    alpha = TGC_ALPHA[phantom] if tgc_alpha is None else float(tgc_alpha)
    scale = NORM_SCALE[phantom] if norm_scale is None else float(norm_scale)
    y = rf.y[:big_k, :].copy()                        # (K, Ne)
    if y.shape[0] < big_k:
        raise ValueError(f"RF of {phantom} has {rf.y.shape[0]} samples < K={big_k}")

    # Exponential TGC + near-field clamp (first k_min+150 samples)
    t = rf.initial_time + np.arange(big_k) / rf.fs
    tgc = np.exp(alpha * t / t.max())
    tgc[: k_min + 150] = tgc[k_min + 150]
    y *= tgc[:, None]

    # Global normalization + per-phantom scale
    y /= np.max(np.abs(y))
    y *= scale

    by = B @ y.flatten(order="F")                     # MATLAB's (:) = column-major
    return by.astype(np.float64), y
