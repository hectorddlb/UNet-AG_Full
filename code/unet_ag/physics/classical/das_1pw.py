"""DAS 1 PW (single plane-wave delay-and-sum) — classical baseline floor.

Standard PWUS DAS with steering angle, F-number aperture restriction and
Hanning (default) or rectangular apodization. Reference for the ``f_number=1.5``
default matches PICMUS evaluation kit.
"""

from __future__ import annotations

import numpy as np


def _element_positions(geometry: dict) -> np.ndarray:
    K = int(geometry["n_elements"])
    p = float(geometry["pitch_m"])
    return (np.arange(K) - (K - 1) / 2.0) * p


def _delayed_channels(
    rf: np.ndarray,
    geometry: dict,
    grid_z: np.ndarray,
    grid_x: np.ndarray,
    f_number: float,
    apod: str,
    angle_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute delayed channel signals s_k(z, x) and apodization weights.

    Returns
    -------
    s : (Nz, Nx, K) float32 — delayed RF samples (0 where invalid)
    valid : (Nz, Nx, K) bool — element participates in this pixel (apod ∧ in-range)
    """
    D, K = rf.shape
    fs = float(geometry["sampling_freq_hz"])
    c = float(geometry["speed_of_sound"])
    t_start = float(geometry.get("t_start", 0.0))
    xk = _element_positions(geometry)

    Nz, Nx = grid_z.size, grid_x.size
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))

    s = np.zeros((Nz, Nx, K), dtype=np.float32)
    valid = np.zeros((Nz, Nx, K), dtype=bool)

    for iz in range(Nz):
        z = float(grid_z[iz])
        half_w = z / (2.0 * float(f_number)) if f_number > 0 else np.inf

        t_tx = (grid_x * sin_a + z * cos_a) / c                  # (Nx,)
        dx = grid_x[:, None] - xk[None, :]                        # (Nx, K)
        t_rx = np.sqrt(dx * dx + z * z) / c                       # (Nx, K)
        t_tot = (t_tx[:, None] + t_rx) - t_start                  # (Nx, K)
        idx_i = np.round(t_tot * fs).astype(np.int64)
        in_range = (idx_i >= 0) & (idx_i < D)
        in_aper = np.abs(dx) < half_w
        v = in_range & in_aper

        if apod == "hanning":
            w = np.zeros_like(dx)
            mask = in_aper & (half_w > 0)
            w[mask] = 0.5 * (1.0 + np.cos(np.pi * dx[mask] / half_w))
        elif apod == "rect":
            w = in_aper.astype(np.float32)
        else:
            raise ValueError(f"unknown apod {apod!r}; use 'hanning' or 'rect'")

        idx_clip = np.clip(idx_i, 0, D - 1)
        k_idx = np.broadcast_to(np.arange(K), idx_clip.shape)
        samples = rf[idx_clip, k_idx]
        s[iz] = (samples * v * w).astype(np.float32)
        valid[iz] = v

    return s, valid


def das_1pw(
    rf: np.ndarray,
    geometry: dict,
    grid_z: np.ndarray,
    grid_x: np.ndarray,
    f_number: float = 1.5,
    apod: str = "hanning",
    angle_rad: float = 0.0,
) -> np.ndarray:
    s, _ = _delayed_channels(rf, geometry, grid_z, grid_x, f_number, apod, angle_rad)
    img = s.sum(axis=2)
    return np.ascontiguousarray(img, dtype=np.float32)
