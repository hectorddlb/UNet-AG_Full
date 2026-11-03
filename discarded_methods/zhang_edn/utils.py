"""
Shared utilities: Hilbert envelope, B-mode conversion, sparse helpers.

All functions operate on PyTorch tensors unless noted otherwise.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Field II t_start compensation
# ---------------------------------------------------------------------------

def apply_t_start_shift(
    rf: np.ndarray,
    t_start: float,
    fs: float,
) -> Tuple[np.ndarray, int]:
    """Align an rf segment so that sample index 0 corresponds to t = 0.

    Field II's calc_scat / calc_scat_multi returns rf_data starting at time
    t = t_start, not at t = 0. The measurement matrix H assumes sample index 0
    corresponds to the transmit firing time (t = 0). Without this alignment,
    the geometric TOF in H is misaligned with sample times, producing an axial
    offset of c · t_start / 2 in the reconstruction.

    Behaviour:
      * t_start > 0  → zero-prepend `shift = round(t_start*fs)` samples.
        Output shape is (D + shift, K), shift > 0.
      * t_start == 0 → no-op. Returns rf, 0.
      * t_start < 0  → drop the first `|shift|` samples (they map to negative
        physical time, which lies outside H's domain). Output shape is
        (D + shift, K) with shift < 0, i.e. shorter than the input.

    Negative t_start arises naturally with steered plane waves whose extreme
    elements transmit before t = 0, when superficial scatterers (small z)
    return echoes ahead of the nominal transmit reference. Magnitudes are
    typically small (≤ 1 µs / ≤ ~25 samples at fs = 20.83 MHz, ≪ 1 mm of axial
    extent), so the truncated samples sit well above the imaging grid
    (z_min ≈ 5 mm) and discarding them is loss-free for the reconstruction.

    PICMUS data is t = 0 aligned natively (initial_time = 0 in HDF5) and does
    not require this shift.

    Convention: rf shape (D, K) where D = samples per channel, K = n_channels.
    """
    if rf.ndim != 2:
        raise ValueError(f"Expected 2D rf array, got shape {rf.shape}")
    D, K = rf.shape
    if D < K:
        raise ValueError(
            f"rf shape ({D}, {K}) suggests (K, D) orientation. "
            f"Expected (D, K) with D > K. Transpose before calling."
        )
    if fs <= 0:
        raise ValueError(f"fs must be > 0, got {fs}")

    shift = int(round(t_start * fs))
    if shift == 0:
        return rf, 0

    if shift > 0:
        pad = np.zeros((shift, K), dtype=rf.dtype)
        rf_shifted = np.concatenate([pad, rf], axis=0)
        return rf_shifted, shift

    # shift < 0: drop first |shift| samples (they correspond to t < 0).
    n_drop = -shift
    if n_drop >= D:
        raise ValueError(
            f"t_start={t_start} would drop {n_drop} samples from rf with D={D}; "
            f"verify fs and t_start units."
        )
    rf_shifted = rf[n_drop:, :]
    return rf_shifted, shift


# ---------------------------------------------------------------------------
# Hilbert transform & envelope
# ---------------------------------------------------------------------------

def hilbert_envelope(x: Tensor, dim: int = -2) -> Tensor:
    """Compute the analytic-signal envelope via the Hilbert transform.

    Parameters
    ----------
    x : Tensor
        Real-valued tensor.  For B-mode images the expected layout is
        ``(batch, 1, Nz, Nx)`` and *dim* should be ``-2`` (axial).
    dim : int
        Axis along which to apply the Hilbert transform (default: axial).

    Returns
    -------
    Tensor
        Non-negative envelope, same shape as *x*.
    """
    N = x.shape[dim]
    X = torch.fft.fft(x, dim=dim)

    # Build the doubling filter h: h[0]=1, h[N/2]=1, h[1..N/2-1]=2, rest=0
    h = torch.zeros(N, device=x.device, dtype=x.dtype)
    if N % 2 == 0:
        h[0] = 1.0
        h[N // 2] = 1.0
        h[1 : N // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (N + 1) // 2] = 2.0

    # Broadcast h to match X shape
    shape = [1] * x.ndim
    shape[dim] = N
    h = h.view(shape)

    analytic = torch.fft.ifft(X * h, dim=dim)
    return analytic.abs()


# ---------------------------------------------------------------------------
# B-mode display
# ---------------------------------------------------------------------------

def to_bmode_db(
    image: Tensor,
    dynamic_range_db: float = 60.0,
    envelope: bool = True,
) -> np.ndarray:
    """Convert an image to log-compressed B-mode in dB.

    Parameters
    ----------
    image : Tensor
        2-D real or complex tensor ``(Nz, Nx)``.
    dynamic_range_db : float
        Dynamic range for clipping.
    envelope : bool
        If *True* and *image* is real-valued, compute the Hilbert envelope
        first.  If *image* is already complex, ``abs()`` is used directly.

    Returns
    -------
    np.ndarray
        Real array in ``[-dynamic_range_db, 0]``.
    """
    with torch.no_grad():
        if torch.is_complex(image):
            env = image.abs()
        elif envelope:
            # Add batch+channel dims for hilbert_envelope, then squeeze
            env = hilbert_envelope(image.unsqueeze(0).unsqueeze(0), dim=-2)
            env = env.squeeze(0).squeeze(0)
        else:
            env = image.abs()

        env_max = env.max()
        if env_max == 0:
            return np.full(image.shape, -dynamic_range_db)

        db = 20.0 * torch.log10(env / env_max + 1e-30)
        db = torch.clamp(db, min=-dynamic_range_db, max=0.0)
        return db.cpu().numpy()


if __name__ == "__main__":
    # Test with (D, K) shape consistent with real data (D=1500, K=128)
    rf = np.ones((1500, 128), dtype=np.float32)
    rf_s, shift = apply_t_start_shift(rf, t_start=1e-6, fs=20.832e6)
    assert shift == 21, f"Expected shift=21 for t_start=1µs, fs=20.832MHz, got {shift}"
    assert rf_s.shape == (1521, 128), f"Expected (1521, 128), got {rf_s.shape}"
    assert (rf_s[:21] == 0).all(), "Leading samples should be zero"
    assert (rf_s[21:] == 1).all(), "Trailing samples should be original"

    rf_z, sz = apply_t_start_shift(rf, t_start=0.0, fs=20.832e6)
    assert sz == 0 and rf_z.shape == rf.shape, "t_start=0 should be a no-op"

    # Negative t_start: drop |shift| samples from the start. This arises with
    # steered plane waves whose extreme elements transmit before t=0.
    rf_neg, sn = apply_t_start_shift(rf, t_start=-0.2e-6, fs=20.832e6)
    assert sn == -4, f"Expected shift=-4 for t_start=-0.2µs, got {sn}"
    assert rf_neg.shape == (1496, 128), f"Expected (1496, 128), got {rf_neg.shape}"

    try:
        apply_t_start_shift(rf, t_start=-1e-3, fs=20.832e6)  # would drop 20832 samples > D
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError when |t_start*fs| >= D")

    try:
        apply_t_start_shift(rf.T, t_start=1e-6, fs=20.832e6)  # (K, D) = (128, 1500)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for (K, D) orientation")

    print("apply_t_start_shift unit test PASSED (positive, zero, negative shifts)")
