"""Minimum Variance (MV) beamformer — Synnevåg et al. 2007.

Per-pixel adaptive weights from the subarray-averaged spatial covariance of
delayed channel data, with diagonal loading for numerical stability.
"""

from __future__ import annotations

import numpy as np

from .das_1pw import _delayed_channels


def mv_1pw(
    rf: np.ndarray,
    geometry: dict,
    grid_z: np.ndarray,
    grid_x: np.ndarray,
    subarray_len: int = 64,
    diag_load: float = 1e-3,
    f_number: float = 1.5,
    angle_rad: float = 0.0,
) -> np.ndarray:
    K = int(geometry["n_elements"])
    L = int(min(subarray_len, K))
    s, valid = _delayed_channels(rf, geometry, grid_z, grid_x, f_number, "rect", angle_rad)

    Nz, Nx = grid_z.size, grid_x.size
    img = np.zeros((Nz, Nx), dtype=np.float32)
    a = np.ones(L, dtype=np.float64)
    I_L = np.eye(L)
    n_sub = K - L + 1

    for iz in range(Nz):
        for ix in range(Nx):
            sk = s[iz, ix].astype(np.float64)
            if not valid[iz, ix].any():
                continue

            # Subarray covariance + diagonal loading
            R = np.zeros((L, L), dtype=np.float64)
            avg = np.zeros(L, dtype=np.float64)
            for l in range(n_sub):
                v = sk[l : l + L]
                R += np.outer(v, v)
                avg += v
            R /= n_sub
            avg /= n_sub

            trR = np.trace(R)
            if trR > 0:
                R += (diag_load * trR / L) * I_L
            else:
                # No signal at this pixel — fall back to DAS-equivalent.
                img[iz, ix] = float(sk.sum())
                continue

            try:
                R_inv_a = np.linalg.solve(R, a)
                denom = float(a @ R_inv_a)
                if abs(denom) < 1e-30:
                    img[iz, ix] = float(sk.sum())
                    continue
                w = R_inv_a / denom
                img[iz, ix] = float(w @ avg)
            except np.linalg.LinAlgError:
                img[iz, ix] = float(sk.sum())

    return img
