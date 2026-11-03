"""Smoke + sanity for unet_ag.physics.f_number_map."""

from __future__ import annotations

import numpy as np

from unet_ag.physics.f_number_map import (
    compute_f_number_grid,
    compute_f_number_map_at_fc,
    compute_f_number_map_band_avg,
    compute_spectral_weight,
)


GEOM = dict(
    pitch_m=304.8e-6,
    element_width_m=270e-6,
    n_elements=128,
    sampling_freq_hz=20.832e6,
    speed_of_sound=1540.0,
    n_samples=3500,
    t_start=0.0,
    fc=5.208e6,
)


def test_f_number_grid_shapes_and_finiteness():
    f = np.linspace(2.0e6, 7.0e6, 16)
    z = np.linspace(5e-3, 50e-3, 64)
    F = compute_f_number_grid(f, z, GEOM)
    assert F.shape == (16, 64)
    assert np.isfinite(F).all()
    # Schiffner's F-number is 0 at low p/λ (full aperture, no grating restriction)
    # and grows to F_ub (≈ 1.5) at higher frequencies. We only require non-negativity.
    assert (F >= 0).all()
    # At high frequencies and reasonable depths F must converge to F_ub.
    assert F[-1, -1] > 1.0


def test_f_number_map_at_fc_broadcasts_across_x():
    z = np.linspace(5e-3, 50e-3, 32)
    x = np.linspace(-15e-3, 15e-3, 16)
    F = compute_f_number_map_at_fc(z, x, GEOM)
    assert F.shape == (32, 16)
    # Identical across the lateral axis at every depth.
    for col in range(1, 16):
        assert np.allclose(F[:, 0], F[:, col])


def test_f_number_map_band_avg_has_depth_variation():
    """F2 fix: at fc the map is uniform 1.5 in PICMUS depth range; band-avg
    should reveal depth dependence because low frequencies hit the depth cap."""
    z = np.linspace(5e-3, 50e-3, 64)
    x = np.linspace(-15e-3, 15e-3, 32)
    F_fc = compute_f_number_map_at_fc(z, x, GEOM)
    F_avg = compute_f_number_map_band_avg(z, x, GEOM)
    assert F_avg.shape == (64, 32)
    # Depth profile of band-avg must vary; depth profile at fc was uniform.
    assert F_avg[:, 0].std() > F_fc[:, 0].std()
    # Sanity: shallower depth → lower average F (depth cap engages sooner at low f).
    assert F_avg[0, 0] < F_avg[-1, 0]


def test_spectral_weight_modes():
    f = np.linspace(2.0e6, 7.0e6, 8)
    w_flat = compute_spectral_weight(f, GEOM, mode="flat")
    w_inv = compute_spectral_weight(f, GEOM, mode="inv_F")
    w_inv2 = compute_spectral_weight(f, GEOM, mode="inv_F_squared")
    assert np.allclose(w_flat, 1.0)
    assert (w_inv > 0).all()
    assert np.allclose(w_inv2, w_inv * w_inv, atol=1e-5)
