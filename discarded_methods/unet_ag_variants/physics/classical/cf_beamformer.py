"""Coherence Factor (CF) beamformer.

    CF(z, x) = |Σ_k s_k(z, x)|² / (N · Σ_k |s_k(z, x)|²)

where ``N`` is the number of elements that pass the F-number aperture mask
(equivalent to using the full ``K`` when no apodization is applied, and
satisfies CF ∈ [0, 1] by Cauchy–Schwarz).
"""

from __future__ import annotations

import numpy as np

from .das_1pw import _delayed_channels


def cf_1pw(
    rf: np.ndarray,
    geometry: dict,
    grid_z: np.ndarray,
    grid_x: np.ndarray,
    f_number: float = 1.5,
    angle_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (das, cf_map, das_x_cf), each (Nz, Nx) float32."""
    s, valid = _delayed_channels(rf, geometry, grid_z, grid_x, f_number, "rect", angle_rad)
    coh = s.sum(axis=2)
    incoh = (s * s).sum(axis=2)
    n_valid = valid.sum(axis=2).clip(min=1)

    cf_map = (coh * coh) / (n_valid * incoh + 1e-30)
    cf_map = np.clip(cf_map, 0.0, 1.0)
    das = coh
    das_cf = das * cf_map
    return (
        np.ascontiguousarray(das, dtype=np.float32),
        np.ascontiguousarray(cf_map, dtype=np.float32),
        np.ascontiguousarray(das_cf, dtype=np.float32),
    )
