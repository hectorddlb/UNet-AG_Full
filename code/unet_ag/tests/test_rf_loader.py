"""TDD specs for unet_ag.data.rf_loader.

Currently failing — implementations are NotImplementedError stubs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from unet_ag.data.rf_loader import (
    FieldIIMultiAngleLoader,
    select_1pw_angle,
    zscore_rf,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIELD_II_ROOT = REPO_ROOT / "datasets" / "Field II" / "field_ii_data_multiangle"


def test_select_1pw_angle_picks_zero_centre():
    angles = np.linspace(-16.0, 16.0, 75)
    assert select_1pw_angle(angles) == 37  # centre angle is nearest 0°


def test_select_1pw_angle_handles_offset_grid():
    angles = np.array([-10.0, -5.0, 1.5, 7.0])
    # 1.5 is nearest 0
    assert select_1pw_angle(angles) == 2


def test_zscore_rf_returns_zero_mean_unit_std():
    rf = torch.randn(3500, 128) * 50.0 + 7.0
    rf_n, norm = zscore_rf(rf)
    assert abs(rf_n.mean().item()) < 1e-5
    assert abs(rf_n.std().item() - 1.0) < 1e-3
    assert "mean" in norm and "std" in norm


@pytest.mark.skipif(not FIELD_II_ROOT.exists(), reason="Field II dataset not on this machine")
def test_field_ii_loader_basic_shape():
    ds = FieldIIMultiAngleLoader(
        data_root=FIELD_II_ROOT,
        phantom_type="points",
        keep_all_angles=True,
    )
    assert len(ds) > 0
    sample = ds[0]
    assert sample.rf_1pw.ndim == 2
    assert sample.rf_1pw.shape[1] == 128
    assert sample.rf_all is not None
    assert sample.rf_all.shape[2] == 75
    assert sample.angles_deg.shape == (75,)
    assert {"pitch_m", "sampling_freq_hz", "n_elements", "speed_of_sound"} <= sample.geometry.keys()
