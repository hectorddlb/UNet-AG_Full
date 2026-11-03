"""TDD spec: H_Fdf wrapper builds operators with PICMUS defaults and the
residual fidelity loss returns a scalar that backprops to ``x_pred`` only.

This test is the *contract* that prevents the Phase 4 1.2 regression: H_Fdf
must NOT have learnable parameters; gradient must NOT flow through the
operator construction.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from unet_ag.physics.h_fdf_wrapper import (
    make_op_for_angle,
    residual_fidelity_loss,
)


PICMUS_GEOMETRY = dict(
    pitch_m=304.8e-6,
    element_width_m=270e-6,
    n_elements=128,
    sampling_freq_hz=20.832e6,
    speed_of_sound=1540.0,
    n_samples=3500,
    t_start=0.0,
)


@pytest.fixture
def grid():
    Nz, Nx = 64, 32  # tiny grid for tests
    grid_z = torch.linspace(5e-3, 30e-3, Nz)
    grid_x = torch.linspace(-15e-3, 15e-3, Nx)
    return grid_z, grid_x


def test_make_op_for_angle_returns_op_with_no_learnable_params(grid):
    grid_z, grid_x = grid
    op = make_op_for_angle(0.0, PICMUS_GEOMETRY, grid_z, grid_x, device="cpu", nf_chunk=2)
    # The operator must have ZERO learnable parameters in the (α) usage.
    n_params = sum(p.numel() for p in op.parameters() if p.requires_grad)
    assert n_params == 0


def test_residual_fidelity_loss_backprops_to_x_pred_only(grid):
    grid_z, grid_x = grid
    Nz, Nx = grid_z.shape[0], grid_x.shape[0]
    op = make_op_for_angle(0.0, PICMUS_GEOMETRY, grid_z, grid_x, device="cpu", nf_chunk=2)

    x_pred = torch.zeros(Nz, Nx, requires_grad=True)
    y_obs = torch.zeros(PICMUS_GEOMETRY["n_samples"], PICMUS_GEOMETRY["n_elements"])

    loss = residual_fidelity_loss(x_pred, y_obs, op)
    assert loss.dim() == 0
    loss.backward()
    assert x_pred.grad is not None
    # No params in op → no other tensor should hold a grad accumulator.
    for p in op.parameters():
        assert not p.requires_grad
