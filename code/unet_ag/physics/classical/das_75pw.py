"""DAS 75 PW compounded — classical baseline ceiling (standard, not Fd-F)."""

from __future__ import annotations

import numpy as np

from .das_1pw import das_1pw


def das_75pw(
    rf_all: np.ndarray,
    angles_rad: np.ndarray,
    geometry: dict,
    grid_z: np.ndarray,
    grid_x: np.ndarray,
    f_number: float = 1.5,
    apod: str = "hanning",
    t_start_all: np.ndarray | None = None,
) -> np.ndarray:
    """Compound DAS over all angles.

    Parameters
    ----------
    rf_all : (D, K, n_angles)
    angles_rad : (n_angles,)
    t_start_all : (n_angles,) optional
        Per-angle Field II start times. Field II emits each steered PW with its
        own time origin (spread ~5 µs ≈ ±2 mm axial across ±16°), so compounding
        with the single ``geometry["t_start"]`` misaligns every off-axis angle.
    """
    if rf_all.ndim != 3:
        raise ValueError(f"rf_all must be (D, K, n_angles); got {rf_all.shape}")
    if rf_all.shape[-1] != angles_rad.size:
        raise ValueError(f"angle count mismatch: rf_all has {rf_all.shape[-1]}, angles_rad has {angles_rad.size}")
    if t_start_all is not None and np.asarray(t_start_all).size != angles_rad.size:
        raise ValueError(f"t_start_all size mismatch: {np.asarray(t_start_all).size} vs {angles_rad.size} angles")

    Nz, Nx = grid_z.size, grid_x.size
    acc = np.zeros((Nz, Nx), dtype=np.float64)
    for a_idx in range(angles_rad.size):
        geo_a = geometry
        if t_start_all is not None:
            geo_a = {**geometry, "t_start": float(np.asarray(t_start_all).reshape(-1)[a_idx])}
        img = das_1pw(
            rf_all[:, :, a_idx],
            geo_a,
            grid_z,
            grid_x,
            f_number=f_number,
            apod=apod,
            angle_rad=float(angles_rad[a_idx]),
        )
        acc += img
    acc /= float(angles_rad.size)
    return acc.astype(np.float32)
