"""DRUS physical model for PICMUS: H (forward) and B (apodized adjoint beamformer).

Exact port of ``MATLAB/Results/02_picmus/BH/compute_svdBH.m`` from the DRUS-v1
repo (Zhang et al. 2023). Deliberate, documented differences from the MATLAB:

1. **Pixel ordering**: MATLAB uses column-major with z varying fastest and
   then applies a permutation to V for the Python sampler. Here the image
   vector directly uses the numpy/torch row-major convention
   ``p = iz * Nx + ix`` (one grid row per z), so that ``x.reshape(Nz, Nx)``
   yields the image without any permutation.
2. **No L/R mirror trick**: MATLAB builds half the matrix and mirrors it for
   efficiency. Here the full grid is built, vectorized per element; the result
   is the same operator.

Everything else replicates the MATLAB operation by operation, including the
pulse bank with stride ``ros+1`` over a pulse sampled at ``dth = 1/(fs*ros)``
(a quirk of the original: the bank effectively resamples the pulse at fs/1.1;
it is replicated as is because that is how the paper results were generated).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# Grid del paper: 256×256, x ∈ [−18, 18] mm, z ∈ [10, 46] mm
# ---------------------------------------------------------------------------

def build_scan_grid(nx: int = 256, nz: int = 256,
                    x_lim: tuple[float, float] = (-0.018, 0.018),
                    z_lim: tuple[float, float] = (0.01, 0.046)):
    """The paper grid by default. The extended variant z_lim=(0.009, 0.048)
    covers the 7 wires of expe_reso (z∈[9.6, 47.0] mm) for the unet_ag n_lat/7
    eval."""
    x_axis = np.linspace(x_lim[0], x_lim[1], nx)
    z_axis = np.linspace(z_lim[0], z_lim[1], nz)
    return x_axis, z_axis


def _matlab_colon(lo: float, step: float, hi: float) -> np.ndarray:
    """Replica of MATLAB's ``lo:step:hi`` (includes hi if it falls on the mesh)."""
    n = int(math.floor((hi - lo) / step + 1e-10)) + 1
    return lo + np.arange(n) * step


# ---------------------------------------------------------------------------
# PICMUS pulse (3 Gaussian convolutions) → fractional-delay bank
# ---------------------------------------------------------------------------

@dataclass
class PulseBank:
    bank: np.ndarray   # (Kh, ros+1) — column j = pulse at delay fraction j/ros
    lagt: float        # temporal lag of the pulse [s]
    kh: int
    ros: int


def make_pulse_bank(fs: float, ros: int = 10) -> PulseBank:
    dth = 1.0 / fs / ros

    fe, bwre, ncycles = 5.208e6, 0.67, 2.5
    ae = 4 * math.log(2) / (math.pi * fe * bwre) ** 2
    te = _matlab_colon(-ncycles / fe / 2, dth, ncycles / fe / 2)
    he = np.exp(-te ** 2 / ae) * np.cos(2 * math.pi * fe * te)

    f0, bwr0 = 5.1333e6, 0.65
    a0 = 4 * math.log(2) / (math.pi * f0 * bwr0) ** 2
    t0 = _matlab_colon(-2 / f0, dth, 2 / f0)
    h0 = np.exp(-t0 ** 2 / a0) * np.cos(2 * math.pi * f0 * t0)

    one_way = fftconvolve(h0, he)
    h = fftconvolve(one_way, h0)
    h = h / np.max(np.abs(h))  # MATLAB: h./max(h(:)); max>0 here (positive central peak)

    nth = len(h)
    # MATLAB h(fix(Nth/4) : ceil(Nth/4*3)) — 1-based inclusivo
    h = h[int(nth / 4) - 1 : math.ceil(nth / 4 * 3)]
    nth = len(h)
    lagt = (nth / 2) * dth

    kh = nth // (ros + 1)
    # MATLAB: reshape(h(1:Kh*(ros+1)), ros+1, Kh)'  ≡  numpy reshape(Kh, ros+1)
    bank = h[: kh * (ros + 1)].reshape(kh, ros + 1)
    bank = bank[:, ::-1].copy()  # flip(h, 2)
    return PulseBank(bank=bank, lagt=lagt, kh=kh, ros=ros)


# ---------------------------------------------------------------------------
# RX apodization (replica of tools.apodization, tukey25 window)
# ---------------------------------------------------------------------------

def tukey25(distance: np.ndarray, aperture: np.ndarray) -> np.ndarray:
    roll = 0.25
    d, a = distance, aperture
    flat = (d < (a / 2 * (1 - roll))).astype(np.float64)
    taper = ((d > (a / 2 * (1 - roll))) & (d < (a / 2))) * 0.5 * (
        1 + np.cos(2 * math.pi / roll * (d / a - roll / 2 - 0.5))
    )
    return flat + taper


# ---------------------------------------------------------------------------
# Construction of H and B
# ---------------------------------------------------------------------------

@dataclass
class ModelMatrices:
    H: sp.csr_matrix          # (K*Ne, Npix)
    B: sp.csr_matrix          # (Npix, K*Ne)
    K: int                    # time samples per element
    k_min: int                # first sample with signal (1-based, like MATLAB)
    x_axis: np.ndarray
    z_axis: np.ndarray


def build_h_and_b(
    probe_x: np.ndarray,      # (Ne,) x positions of the elements [m]
    fs: float,
    c: float,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    rx_f_number: float = 1.4,
    ros: int = 10,
    dtype=np.float32,
    pulse: PulseBank | None = None,
) -> ModelMatrices:
    ne = len(probe_x)
    nx, nz = len(x_axis), len(z_axis)
    npix = nx * nz

    if pulse is None:
        pulse = make_pulse_bank(fs, ros)
    bank, kh, lagt = pulse.bank, pulse.kh, pulse.lagt

    # Pixel vectors in the row-major convention p = iz*Nx + ix
    x = np.tile(x_axis, nz)              # (Npix,)
    z = np.repeat(z_axis, nx)            # (Npix,)

    # --- TX delay (plane wave 0°): min distance to the interpolated aperture ---
    # MATLAB: interp1(xe, linspace(1,Ne,4Ne), 'spline') — evaluates xe at
    # fractional indices (a spline of linear data ≡ linear, replicated as is).
    idxi = np.linspace(1, ne, 4 * ne)
    xti = CubicSpline(np.arange(1, ne + 1), probe_x)(idxi)      # (4Ne,)
    # dTX in chunks to avoid materializing the full (Npix, 4Ne) in float64
    dtx = np.empty(npix)
    chunk = 8192
    for s in range(0, npix, chunk):
        e = min(s + chunk, npix)
        dtx[s:e] = np.sqrt((xti[None, :] - x[s:e, None]) ** 2 + z[s:e, None] ** 2).min(axis=1)

    drx = np.sqrt((probe_x[None, :] - x[:, None]) ** 2 + z[:, None] ** 2)   # (Npix, Ne)

    tau = (dtx[:, None] + drx) / c - lagt                                   # (Npix, Ne)
    delay = tau * fs + 1.0                                                  # 1-based, like MATLAB

    k_min = int(np.floor(delay.min()))
    big_k = int(np.ceil(delay.max())) + kh

    # --- RX apodization ---
    rx_aperture = z / rx_f_number                                           # (Npix,)
    apod = tukey25(np.abs(x[:, None] - probe_x[None, :]), rx_aperture[:, None])  # (Npix, Ne)

    # --- Sparse assembly per element ---
    di = np.floor(delay).astype(np.int64)          # integer part (1-based)
    dd = delay - di                                # fraction ∈ [0,1)
    frac_col = np.rint(dd * ros).astype(np.int64)  # bank column, 0..ros

    pix_idx = np.arange(npix, dtype=np.int64)
    rows_parts, cols_parts, hvals_parts, bvals_parts = [], [], [], []
    for e in range(ne):
        base = di[:, e] - 1 + e * big_k                          # 0-based row of the first sample
        rows = (base[:, None] + np.arange(kh)[None, :]).ravel()  # (Npix*Kh,)
        cols = np.repeat(pix_idx, kh)
        vals = bank[:, frac_col[:, e]].T.ravel()                 # (Npix, Kh) → flat pixel-major
        rows_parts.append(rows.astype(np.int64))
        cols_parts.append(cols)
        hvals_parts.append(vals.astype(dtype))
        bvals_parts.append((vals * np.repeat(apod[:, e], kh)).astype(dtype))

    rows = np.concatenate(rows_parts)
    cols = np.concatenate(cols_parts)
    hvals = np.concatenate(hvals_parts)
    bvals = np.concatenate(bvals_parts)
    del rows_parts, cols_parts, hvals_parts, bvals_parts

    shape_h = (big_k * ne, npix)
    H = sp.coo_matrix((hvals, (rows, cols)), shape=shape_h).tocsr()
    B = sp.coo_matrix((bvals, (cols, rows)), shape=(npix, big_k * ne)).tocsr()

    return ModelMatrices(H=H, B=B, K=big_k, k_min=k_min, x_axis=x_axis, z_axis=z_axis)
