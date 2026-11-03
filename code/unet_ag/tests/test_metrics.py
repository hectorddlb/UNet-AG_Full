"""TDD specs for unet_ag.metrics — CNR, gCNR, FWHM."""

from __future__ import annotations

import numpy as np
import pytest

from unet_ag.metrics.cnr import compute_cnr
from unet_ag.metrics.gcnr import compute_gcnr
from unet_ag.metrics.fwhm import compute_fwhm_lateral, compute_fwhm_axial


def test_cnr_two_gaussians_known_value():
    # Construct image with two regions of known mean+std.
    img = np.zeros((100, 100), dtype=np.float32)
    rng = np.random.default_rng(0)
    img[20:40, 20:40] = rng.normal(loc=1.0, scale=0.1, size=(20, 20))   # inside
    img[60:80, 60:80] = rng.normal(loc=0.5, scale=0.1, size=(20, 20))   # outside
    mi = np.zeros_like(img, dtype=bool); mi[20:40, 20:40] = True
    mo = np.zeros_like(img, dtype=bool); mo[60:80, 60:80] = True
    cnr = compute_cnr(img, mi, mo, on_envelope=False)
    # |0.5| / sqrt(0.01+0.01) ≈ 3.535 → 20·log10(3.535) ≈ 10.97 dB
    assert 9.5 < cnr < 12.5


def test_gcnr_identical_distributions_is_zero():
    rng = np.random.default_rng(0)
    img = rng.normal(size=(100, 100)).astype(np.float32)
    mi = np.zeros_like(img, dtype=bool); mi[:50, :] = True
    mo = ~mi
    g = compute_gcnr(img, mi, mo, on_envelope=False)
    assert g < 0.15  # finite-sample noise


def test_gcnr_disjoint_distributions_is_one():
    img = np.zeros((100, 100), dtype=np.float32)
    img[:50, :] = 10.0  # totally separated
    mi = np.zeros_like(img, dtype=bool); mi[:50, :] = True
    mo = ~mi
    g = compute_gcnr(img, mi, mo, on_envelope=False)
    assert g > 0.98


def test_fwhm_lateral_on_known_gaussian():
    Nz, Nx = 101, 101
    dx_mm = 0.1
    z, x = np.meshgrid(np.arange(Nz), np.arange(Nx), indexing="ij")
    sigma_px = 5.0
    psf = np.exp(-((x - 50) ** 2) / (2 * sigma_px ** 2)).astype(np.float32)
    fwhm_mm = compute_fwhm_lateral(psf, peak_idx=(50, 50), dx_mm=dx_mm)
    # Theoretical FWHM = 2*sqrt(2*ln2)*sigma ≈ 2.355*sigma = 11.77 px = 1.177 mm
    assert 1.0 < fwhm_mm < 1.35


def test_fwhm_axial_on_known_gaussian():
    Nz, Nx = 101, 101
    dz_mm = 0.1
    z, x = np.meshgrid(np.arange(Nz), np.arange(Nx), indexing="ij")
    sigma_px = 4.0
    psf = np.exp(-((z - 50) ** 2) / (2 * sigma_px ** 2)).astype(np.float32)
    fwhm_mm = compute_fwhm_axial(psf, peak_idx=(50, 50), dz_mm=dz_mm)
    assert 0.8 < fwhm_mm < 1.1


def test_fwhm_returns_nan_when_profile_never_crosses_half_max():
    """Regression: B1 eval 2026-05-21 reported FWHM_lat=38.4 mm = grid extent
    because the lateral profile stayed within -6 dB across the whole image
    (B1 produced near-uniform fields). Old impl returned grid_extent; new
    impl must return NaN so the metric is honest.
    """
    # Flat-ish profile that never drops below -1 dB → never crosses -6 dB.
    Nz, Nx = 11, 101
    img = np.full((Nz, Nx), 1.0, dtype=np.float32)
    img[5, :] = np.linspace(0.95, 1.0, Nx)                # peak ≈ 1.0, dips to 0.95
    fwhm_mm = compute_fwhm_lateral(img, peak_idx=(5, 100), dx_mm=0.1)
    assert np.isnan(fwhm_mm), f"expected NaN, got {fwhm_mm}"


def test_fwhm_aggregator_reports_n_resolved():
    """When a subset of wires is unresolved, aggregate should count them."""
    from unet_ag.metrics.fwhm import aggregate_fwhms_per_wire
    # Layout: two phantom regions on a 201×201 grid, far apart, so each row /
    # column profile sees only one wire at a time (no interference).
    Nz, Nx = 201, 201
    dz_mm = dx_mm = 0.1
    grid_z = np.arange(Nz) * dz_mm
    grid_x = np.arange(Nx) * dx_mm
    z2d, x2d = np.meshgrid(np.arange(Nz), np.arange(Nx), indexing="ij")

    # Wire A: sharp gaussian at (25, 25) — resolves.
    img_a = np.exp(-((z2d - 25) ** 2 + (x2d - 25) ** 2) / (2 * 3.0 ** 2)).astype(np.float32)
    # Wire B: flat smear around (175, 175) — its peak is ~0.55 with a
    # near-uniform skirt that barely dips. Keeping the floor < 0.5 of wire A's
    # peak (=1.0) ensures the smear does not pollute wire A's row profile.
    img_b = (0.45 * np.exp(-((z2d - 175) ** 2 + (x2d - 175) ** 2) / (2 * 100.0 ** 2))).astype(np.float32)
    img = np.maximum(img_a, img_b)

    res = aggregate_fwhms_per_wire(
        img, wires_expected_mm=[(25 * dz_mm, 25 * dx_mm), (175 * dz_mm, 175 * dx_mm)],
        dx_mm=dx_mm, dz_mm=dz_mm,
        grid_z_mm=grid_z, grid_x_mm=grid_x, search_radius_mm=2.0,
    )
    assert res["n_total"] == 2
    # Wire A resolves on both axes; wire B is too flat to cross -6 dB.
    assert res["n_resolved_lat"] == 1
    assert res["n_resolved_ax"] == 1
    assert not np.isnan(res["lat_median"])
    assert not np.isnan(res["ax_median"])
