"""
Measurement matrix H construction (Eq. 1-6 of Zhang et al. 2021).

H encodes the linear forward model  y = H @ x  relating a beamformed image
x (Nz*Nx,) to the raw RF channel data y (D*K,).

The matrix is extremely sparse: at most K non-zeros per column (one per
element), so it is stored as ``scipy.sparse.csr_matrix``.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp


def _element_positions(n_elements: int, pitch_m: float) -> np.ndarray:
    """Centred element positions (K,) in metres."""
    k = np.arange(n_elements, dtype=np.float64)
    return (k - (n_elements - 1) / 2.0) * pitch_m


def build_measurement_matrix(
    grid_z: np.ndarray,
    grid_x: np.ndarray,
    n_elements: int,
    pitch_m: float,
    sampling_freq_hz: float,
    speed_of_sound: float,
    n_samples: int,
    angle_rad: float = 0.0,
) -> sp.csr_matrix:
    """Build the measurement matrix H per Eq. 1-6.

    Parameters
    ----------
    grid_z : (Nz,) axial positions in metres (positive, increasing).
    grid_x : (Nx,) lateral positions in metres.
    n_elements : K, number of transducer elements.
    pitch_m : element pitch in metres.
    sampling_freq_hz : fs, sampling frequency in Hz.
    speed_of_sound : c, speed of sound in m/s.
    n_samples : D, number of RF time samples.
    angle_rad : steering angle in radians (0 = non-steered).

    Returns
    -------
    H : scipy.sparse.csr_matrix of shape (D*K, Nz*Nx).
    """
    grid_z = np.asarray(grid_z, dtype=np.float64)
    grid_x = np.asarray(grid_x, dtype=np.float64)

    Nz = len(grid_z)
    Nx = len(grid_x)
    K = n_elements
    D = n_samples
    fs = sampling_freq_hz
    c = speed_of_sound

    x_elem = _element_positions(K, pitch_m)  # (K,)

    # Pre-compute transmit delays for all grid points.
    # Steered PW: τ_t = (z*cos(θ) + x*sin(θ)) / c
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Meshgrid: zz (Nz, Nx), xx (Nz, Nx)
    zz, xx = np.meshgrid(grid_z, grid_x, indexing="ij")
    tau_tx = (zz * cos_a + xx * sin_a) / c  # (Nz, Nx)

    # For Hanning weighting, find the closest element to each lateral position.
    # k_center[ix] = index of element closest to grid_x[ix]
    k_center = np.argmin(
        np.abs(x_elem[np.newaxis, :] - grid_x[:, np.newaxis]), axis=1
    )  # (Nx,)

    # Hanning half-width in element-index space.
    half_width = (K - 1) / 2.0

    # Build COO entries: row indices, column indices, values.
    rows = []
    cols = []
    vals = []

    # Iterate over elements (K=128 iterations — acceptable).
    for k in range(K):
        xk = x_elem[k]

        # Receive delay for all grid points: τ_r = sqrt(z² + (x - xk)²) / c
        tau_rx = np.sqrt(zz ** 2 + (xx - xk) ** 2) / c  # (Nz, Nx)

        # Total time-of-flight
        tau_tof = tau_tx + tau_rx  # (Nz, Nx)

        # Sample index (Eq. 5 inverted): d = round(τ_tof * fs)
        d_float = tau_tof * fs
        d_idx = np.round(d_float).astype(np.int64)  # (Nz, Nx)

        # Criterion Eq. 6: |d/fs - τ_tof| ≤ 1/(2*fs)
        # This is automatically satisfied by rounding (error ≤ 0.5 sample).
        # Only filter out-of-range samples.
        valid = (d_idx >= 0) & (d_idx < D)

        # Hanning weight: w = 0.5 * (1 + cos(π * (k - k_center) / half_width))
        # for |k - k_center| < half_width, else 0.
        # k_center is per lateral position, broadcast over Nz.
        k_dist = np.abs(k - k_center)  # (Nx,)
        # Broadcast to (Nz, Nx)
        k_dist_2d = np.broadcast_to(k_dist[np.newaxis, :], (Nz, Nx))

        in_window = k_dist_2d < half_width
        weight = np.where(
            in_window,
            0.5 * (1.0 + np.cos(math.pi * k_dist_2d / half_width)),
            0.0,
        )  # (Nz, Nx)

        mask = valid & (weight > 0)

        if not np.any(mask):
            continue

        # Extract entries
        iz, ix = np.nonzero(mask)
        d_vals = d_idx[iz, ix]
        w_vals = weight[iz, ix]

        # Row in H: d * K + k  (linearized RF index)
        row_idx = d_vals * K + k

        # Column in H: iz * Nx + ix  (linearized image index, row-major)
        col_idx = iz * Nx + ix

        rows.append(row_idx)
        cols.append(col_idx)
        vals.append(w_vals)

    # Assemble sparse matrix
    if len(rows) == 0:
        return sp.csr_matrix((D * K, Nz * Nx), dtype=np.float64)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)

    H = sp.coo_matrix((vals, (rows, cols)), shape=(D * K, Nz * Nx))
    return H.tocsr()
