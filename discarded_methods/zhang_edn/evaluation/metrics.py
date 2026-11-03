"""
Quantitative metrics: FWHM and CNR (Sec. 10, Zhang et al. 2021).
"""

from __future__ import annotations

import numpy as np


def compute_fwhm(
    profile_1d: np.ndarray,
    pixel_spacing_mm: float,
) -> float:
    """Full Width at Half Maximum of a 1-D profile.

    Parameters
    ----------
    profile_1d : envelope amplitude along one axis through the PSF peak.
    pixel_spacing_mm : distance between samples in mm.

    Returns
    -------
    FWHM in mm.
    """
    profile = np.abs(profile_1d)
    peak = profile.max()
    if peak == 0:
        return 0.0

    half_max = peak / 2.0
    above = profile >= half_max
    indices = np.where(above)[0]

    if len(indices) < 2:
        return pixel_spacing_mm  # single sample width

    return float(indices[-1] - indices[0]) * pixel_spacing_mm


def compute_cnr(
    image_envelope: np.ndarray,
    roi_inside: tuple[slice, slice],
    roi_outside: tuple[slice, slice],
) -> float:
    """Contrast-to-Noise Ratio in dB (Eq. 14).

    CNR = 20 * log10( |mu_in - mu_out| / sqrt((var_in + var_out) / 2) )

    Parameters
    ----------
    image_envelope : 2-D envelope (linear scale, NOT dB).
    roi_inside : (row_slice, col_slice) for the target region.
    roi_outside : (row_slice, col_slice) for the background region.

    Returns
    -------
    CNR in dB.
    """
    region_in = image_envelope[roi_inside]
    region_out = image_envelope[roi_outside]

    # Use intensity (envelope squared) for mean and variance
    int_in = region_in.ravel() ** 2
    int_out = region_out.ravel() ** 2

    mu_in = int_in.mean()
    mu_out = int_out.mean()
    var_in = int_in.var()
    var_out = int_out.var()

    denom = np.sqrt((var_in + var_out) / 2.0)
    if denom < 1e-30:
        return 0.0

    return float(20.0 * np.log10(np.abs(mu_in - mu_out) / denom + 1e-30))
