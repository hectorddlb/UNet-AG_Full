"""Smoke tests for unet_ag.data.augmentations."""

from __future__ import annotations

import numpy as np
import torch

from unet_ag.data.augmentations import (
    AugConfig,
    add_awgn,
    apply_augmentations,
    gaussian_bandpass_rf,
)
from unet_ag.data.rf_loader import RFSample


def _make_sample() -> RFSample:
    rf = torch.randn(256, 64).float()
    return RFSample(
        rf_1pw=rf,
        rf_all=None,
        angles_deg=np.array([0.0]),
        geometry={
            "pitch_m": 304.8e-6,
            "element_width_m": 270e-6,
            "n_elements": 64,
            "sampling_freq_hz": 20.832e6,
            "speed_of_sound": 1540.0,
            "n_samples": 256,
            "t_start": 0.0,
            "fc": 5.208e6,
        },
        phantom_id="synthetic_0001",
        rf_norm={"mean": 0.0, "std": 1.0},
    )


def test_add_awgn_increases_variance():
    rf = torch.zeros(256, 64) + 1.0
    rng = np.random.default_rng(0)
    out = add_awgn(rf, snr_db=10.0, rng=rng)
    assert out.std().item() > 0.05


def test_apply_augmentations_does_not_mutate_input():
    s = _make_sample()
    rf_before = s.rf_1pw.clone()
    cfg = AugConfig(snr_apply_prob=1.0, t_start_apply_prob=1.0, c_apply_prob=1.0, scatterer_apply_prob=1.0)
    s2 = apply_augmentations(s, cfg, seed=0)
    assert torch.allclose(s.rf_1pw, rf_before), "input RFSample.rf_1pw must not change"
    # Augmentations should produce a different RF
    assert not torch.allclose(s2.rf_1pw, rf_before, atol=1e-7)
    # And touch the geometry for t_start / c
    assert s2.geometry["t_start"] != s.geometry["t_start"]
    assert s2.geometry["speed_of_sound"] != s.geometry["speed_of_sound"]


def test_apply_augmentations_zero_prob_is_passthrough():
    s = _make_sample()
    cfg = AugConfig(snr_apply_prob=0.0, t_start_apply_prob=0.0, c_apply_prob=0.0,
                    scatterer_apply_prob=0.0, pulse_apply_prob=0.0)
    s2 = apply_augmentations(s, cfg, seed=0)
    assert torch.allclose(s.rf_1pw, s2.rf_1pw)
    assert s.geometry == s2.geometry


# ──────────────────────────────────────────────────────────────────
# B1-v5 pulse-shape augmentation
# ──────────────────────────────────────────────────────────────────


def _peak_freq_hz(rf_1d: torch.Tensor, fs: float) -> float:
    """Return the spectral peak frequency of a 1D RF trace."""
    spec = torch.fft.rfft(rf_1d).abs().numpy()
    freqs = torch.fft.rfftfreq(rf_1d.shape[0], 1.0 / fs).numpy()
    return float(freqs[int(np.argmax(spec))])


def _bw_minus_6db_hz(rf_1d: torch.Tensor, fs: float) -> float:
    spec = torch.fft.rfft(rf_1d).abs().numpy()
    freqs = torch.fft.rfftfreq(rf_1d.shape[0], 1.0 / fs).numpy()
    pk = spec.max()
    above = spec >= 0.5 * pk
    if not above.any():
        return float("nan")
    return float(freqs[above][-1] - freqs[above][0])


def test_gaussian_bandpass_narrow_attenuates_variance():
    # White noise → applying a narrow bandpass should drop std significantly.
    torch.manual_seed(0)
    rf = torch.randn(2048, 8).float()
    fs = 20.832e6
    rf_bp = gaussian_bandpass_rf(rf, fs, fc_hz=5.0e6, bw_hz=1.2e6)
    assert rf_bp.std().item() < 0.5 * rf.std().item()


def test_gaussian_bandpass_wide_is_near_identity():
    # σ ≫ Nyquist → flat passband → near identity for any input.
    torch.manual_seed(1)
    rf = torch.randn(2048, 4).float()
    fs = 20.832e6
    rf_id = gaussian_bandpass_rf(rf, fs, fc_hz=5.208e6, bw_hz=100e6)
    # Std preserved within 1%, sample-wise diff small.
    assert abs(rf_id.std().item() / rf.std().item() - 1.0) < 0.01
    assert (rf - rf_id).abs().max().item() < 0.1


def test_gaussian_bandpass_peak_at_fc():
    # Filter white noise; the surviving spectrum should peak near fc.
    torch.manual_seed(2)
    rf = torch.randn(2048, 1).float()
    fs = 20.832e6
    fc = 4.5e6
    rf_bp = gaussian_bandpass_rf(rf, fs, fc_hz=fc, bw_hz=1.0e6)
    pk = _peak_freq_hz(rf_bp[:, 0], fs)
    # Allow some FFT-bin slack (≈ fs/D ≈ 10 kHz here).
    assert abs(pk - fc) < 0.2e6, f"peak {pk:.3e} vs fc {fc:.3e}"


def test_gaussian_bandpass_3d_filters_each_angle():
    # Multi-angle (D, K, NA) tensor: every angle slice gets the same filter.
    torch.manual_seed(3)
    rf = torch.randn(1024, 16, 4).float()
    fs = 20.832e6
    out = gaussian_bandpass_rf(rf, fs, fc_hz=5.0e6, bw_hz=1.0e6)
    assert out.shape == rf.shape
    # Each angle's variance is reduced (narrow filter on white noise).
    for ia in range(4):
        assert out[:, :, ia].std().item() < 0.6 * rf[:, :, ia].std().item()


def test_apply_augmentations_pulse_narrows_bandwidth():
    # Build a broadband synthetic RF with content over ~[1, 9] MHz.
    fs = 20.832e6
    n = 2048
    t = np.arange(n) / fs
    rng = np.random.default_rng(7)
    sig = sum(np.cos(2 * np.pi * f * t + rng.random()) for f in (2e6, 4e6, 5.2e6, 7e6, 8e6))
    sig = sig.astype(np.float32)
    rf = torch.from_numpy(np.tile(sig[:, None], (1, 8))).float()

    s = RFSample(
        rf_1pw=rf, rf_all=None,
        angles_deg=np.array([0.0]),
        geometry={
            "pitch_m": 300e-6, "element_width_m": 270e-6, "n_elements": 8,
            "sampling_freq_hz": fs, "speed_of_sound": 1540.0,
            "n_samples": n, "t_start": 0.0, "fc": 5.208e6,
        },
        phantom_id="synth_bw", rf_norm={"mean": 0.0, "std": 1.0},
    )
    # Force a narrow pulse_bw band, disable all the others.
    cfg = AugConfig(
        snr_apply_prob=0.0, t_start_apply_prob=0.0, c_apply_prob=0.0,
        scatterer_apply_prob=0.0,
        pulse_apply_prob=1.0,
        pulse_bw_range_hz=(0.8e6, 0.8e6),     # deterministic narrow BW
        pulse_fc_shift_hz=0.0,                # no fc shift
    )
    s2 = apply_augmentations(s, cfg, seed=0)
    bw_before = _bw_minus_6db_hz(s.rf_1pw[:, 0], fs)
    bw_after  = _bw_minus_6db_hz(s2.rf_1pw[:, 0], fs)
    assert bw_after < bw_before, f"bw {bw_before:.3e} → {bw_after:.3e}"
    # Peak should be near fc_nominal=5.208 MHz (within bin slack).
    pk_after = _peak_freq_hz(s2.rf_1pw[:, 0], fs)
    assert abs(pk_after - 5.208e6) < 0.3e6


def test_apply_augmentations_pulse_zero_prob_is_passthrough():
    s = _make_sample()
    cfg = AugConfig(snr_apply_prob=0.0, t_start_apply_prob=0.0, c_apply_prob=0.0,
                    scatterer_apply_prob=0.0, pulse_apply_prob=0.0)
    s2 = apply_augmentations(s, cfg, seed=0)
    assert torch.allclose(s.rf_1pw, s2.rf_1pw)
