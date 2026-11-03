"""TDD specs: classical baselines DAS 1PW / DAS 75PW / MV / CF.

All four return (Nz, Nx) float32 images. Concrete numerical checks happen
when the corresponding precompute scripts produce a reference for a PICMUS
phantom — but at scaffold time we assert shape + dtype + finiteness.
"""

from __future__ import annotations

import numpy as np
import pytest

from unet_ag.physics.classical.das_1pw import das_1pw
from unet_ag.physics.classical.das_75pw import das_75pw
from unet_ag.physics.classical.mv_beamformer import mv_1pw
from unet_ag.physics.classical.cf_beamformer import cf_1pw


GEOM = dict(
    pitch_m=304.8e-6,
    element_width_m=270e-6,
    n_elements=128,
    sampling_freq_hz=20.832e6,
    speed_of_sound=1540.0,
    n_samples=3500,
    t_start=0.0,
)


@pytest.fixture
def tiny_grid():
    grid_z = np.linspace(5e-3, 30e-3, 64)
    grid_x = np.linspace(-15e-3, 15e-3, 32)
    return grid_z, grid_x


@pytest.fixture
def tiny_rf():
    rng = np.random.default_rng(0)
    return rng.standard_normal((GEOM["n_samples"], GEOM["n_elements"])).astype(np.float32)


def test_das_1pw_shape_and_dtype(tiny_grid, tiny_rf):
    grid_z, grid_x = tiny_grid
    img = das_1pw(tiny_rf, GEOM, grid_z, grid_x, f_number=1.5)
    assert img.shape == (grid_z.size, grid_x.size)
    assert img.dtype == np.float32
    assert np.isfinite(img).all()


def test_das_75pw_shape(tiny_grid, tiny_rf):
    grid_z, grid_x = tiny_grid
    rf_all = np.stack([tiny_rf] * 5, axis=-1)   # tiny: 5 angles
    angles = np.linspace(-5.0, 5.0, 5) * np.pi / 180.0
    img = das_75pw(rf_all, angles, GEOM, grid_z, grid_x)
    assert img.shape == (grid_z.size, grid_x.size)
    assert np.isfinite(img).all()


def test_mv_1pw_shape(tiny_grid, tiny_rf):
    grid_z, grid_x = tiny_grid
    img = mv_1pw(tiny_rf, GEOM, grid_z, grid_x, subarray_len=16)
    assert img.shape == (grid_z.size, grid_x.size)
    assert np.isfinite(img).all()


def test_cf_1pw_returns_three_maps(tiny_grid, tiny_rf):
    grid_z, grid_x = tiny_grid
    das, cf_map, das_cf = cf_1pw(tiny_rf, GEOM, grid_z, grid_x)
    assert das.shape == cf_map.shape == das_cf.shape == (grid_z.size, grid_x.size)
    assert 0.0 <= np.min(cf_map) and np.max(cf_map) <= 1.0
