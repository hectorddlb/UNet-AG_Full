"""DRUS physical model for the multi-angle Field II geometry (Phase F.1).

Reuses ``build_h_and_b`` from ``physics_picmus`` (same probe: 128 el., pitch
300 µm, fs 20.832 MHz) with two documented adaptations:

1. **Field II pulse bank**: the dataset was generated with
   ``generate_training_data_multiangle.m``:
       excitation  : sin(2π·fc·t), 2.5 cycles @ fc=5.208 MHz (causal burst)
       IR (tx=rx)  : sin(2π·fc·t)·exp(−π²·(0.67·fc)²·t²), t ∈ [−2/fc, 2/fc]
   The two-way h = ir ⊛ exc ⊛ ir is built here analytically at step
   dth = 1/(fs·ros), centred, and cropped to the central half with the same
   procedure (and ``lagt`` convention) as ``make_pulse_bank`` so that the
   delay semantics of ``build_h_and_b`` do not change. The resulting BW
   (~1.8 MHz) differs from the PICMUS bank (~1.0 MHz) — using the PICMUS bank
   here would be a −43 % spectral model mismatch.

2. **Temporal alignment**: Field II returns ``t_start`` per emission
   (angle 0°: ~3.46 µs ≈ 72.0 samples). H assumes y indexed from t=0, so the
   observation is prepended with ``round(t_start·fs)`` zeros plus a global
   correction ``pad_extra`` calibrated empirically against the GT positions
   of the wires (gate of the build script); the residual of the pulse lag
   convention is absorbed there.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import fftconvolve

from .physics_picmus import PulseBank, _matlab_colon


def make_pulse_bank_fieldii(
    fs: float,
    ros: int = 10,
    fc: float = 5.208e6,
    ncycles: float = 2.5,
    frac_bw: float = 0.67,
) -> PulseBank:
    """Two-way pulse bank of the Field II generator, same convention as
    ``make_pulse_bank`` (central-half crop, lagt = half of the cropped pulse)."""
    dth = 1.0 / fs / ros

    # Excitation: 2.5-cycle causal burst, centred for the lagt convention.
    te = _matlab_colon(-ncycles / fc / 2, dth, ncycles / fc / 2)
    he = np.sin(2 * math.pi * fc * (te + ncycles / fc / 2))

    # Transducer IR (identical in TX and RX).
    t0 = _matlab_colon(-2 / fc, dth, 2 / fc)
    h0 = np.sin(2 * math.pi * fc * t0) * np.exp(-(math.pi * frac_bw * fc) ** 2 * t0 ** 2)

    h = fftconvolve(fftconvolve(h0, he), h0)
    h = h / np.max(np.abs(h))

    nth = len(h)
    h = h[int(nth / 4) - 1: math.ceil(nth / 4 * 3)]
    nth = len(h)
    lagt = (nth / 2) * dth

    kh = nth // (ros + 1)
    bank = h[: kh * (ros + 1)].reshape(kh, ros + 1)
    bank = bank[:, ::-1].copy()
    return PulseBank(bank=bank, lagt=lagt, kh=kh, ros=ros)


def align_fieldii_rf(
    rf_0deg: np.ndarray,      # (D, Ne) RF of the 0° angle
    t_start: float,           # t_start of the 0° angle [s]
    fs: float,
    big_k: int,
    pad_extra: int = 0,
) -> np.ndarray:
    """Prepend zeros to bring the RF to H's t=0 convention and crop/pad to
    exactly ``big_k`` samples. ``pad_extra`` is the calibrated correction."""
    n0 = int(round(t_start * fs)) + int(pad_extra)
    d, ne = rf_0deg.shape
    y = np.zeros((big_k, ne), dtype=np.float64)
    if n0 >= 0:
        n_copy = min(d, big_k - n0)
        if n_copy > 0:
            y[n0:n0 + n_copy, :] = rf_0deg[:n_copy, :]
    else:
        n_copy = min(d + n0, big_k)
        if n_copy > 0:
            y[:n_copy, :] = rf_0deg[-n0:-n0 + n_copy, :]
    return y


def compute_by_fieldii(
    B,                        # sp.csr_matrix (Npix, K*Ne)
    rf_0deg: np.ndarray,      # (D, Ne)
    t_start: float,
    fs: float,
    big_k: int,
    k_min: int,
    pad_extra: int = 0,
    tgc_alpha: float = 4.0,
    norm_scale: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror of ``observations.compute_by`` for Field II RF: t_start
    alignment + exponential TGC with a near-field clamp + normalization."""
    y = align_fieldii_rf(rf_0deg, t_start, fs, big_k, pad_extra)

    t = np.arange(big_k) / fs
    tgc = np.exp(tgc_alpha * t / t.max())
    tgc[: k_min + 150] = tgc[k_min + 150]
    y *= tgc[:, None]

    y /= np.max(np.abs(y)) + 1e-30
    y *= norm_scale

    by = B @ y.flatten(order="F")
    return by.astype(np.float64), y
