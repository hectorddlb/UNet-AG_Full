"""Smoke tests for proposal A1.

Validates:
- A1ResUNetRF accepts (B, 1, D, K) RF and emits (B, 1, Nz, Nx).
- Gradient flows from output back to the RF input.
- Parameter count is in expected band.
- The fidelity training step (image loss + H_Fdf fidelity) runs end-to-end
  with x_pred carrying gradient through OpHFdf (α contract).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from unet_ag.models.a1_resunet_rf import A1ResUNetRF
from unet_ag.physics.h_fdf_wrapper import make_op_for_angle


PICMUS_GEOMETRY = dict(
    pitch_m=304.8e-6,
    element_width_m=270e-6,
    n_elements=128,
    sampling_freq_hz=20.832e6,
    speed_of_sound=1540.0,
    n_samples=512,           # tiny for CPU
    t_start=0.0,
)


def test_a1_forward_shape_cpu():
    D, K = 512, 32
    Nz, Nx = 32, 16
    model = A1ResUNetRF(n_samples=D, n_elements=K, nz=Nz, nx=Nx, base_ch=8).cpu()
    rf = torch.randn(2, 1, D, K)
    y = model(rf)
    assert y.shape == (2, 1, Nz, Nx)
    assert torch.isfinite(y).all()


def test_a1_backward_flows_to_rf_input():
    D, K = 256, 16
    Nz, Nx = 24, 12
    model = A1ResUNetRF(n_samples=D, n_elements=K, nz=Nz, nx=Nx, base_ch=4).cpu()
    rf = torch.randn(1, 1, D, K, requires_grad=True)
    y = model(rf)
    y.sum().backward()
    assert rf.grad is not None
    assert torch.isfinite(rf.grad).all()


def test_a1_param_count_in_expected_band():
    D, K = 3500, 128
    Nz, Nx = 612, 388
    model = A1ResUNetRF(n_samples=D, n_elements=K, nz=Nz, nx=Nx, base_ch=32)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # RF encoder + 3-level U-Net @ base_ch=32: target band ~1–5M params
    assert 1_000_000 < n < 8_000_000, f"unexpected param count: {n:,}"


def test_a1_full_step_with_h_fdf_fidelity_cpu():
    """End-to-end step: model forward + image loss + H_Fdf fidelity + backward.

    Uses a tiny grid (32×16) and short RF (512) so OpHFdf fits in CPU memory.
    """
    D = PICMUS_GEOMETRY["n_samples"]
    K = PICMUS_GEOMETRY["n_elements"]
    Nz, Nx = 32, 16
    grid_z = torch.linspace(5e-3, 20e-3, Nz)
    grid_x = torch.linspace(-15e-3, 15e-3, Nx)

    model = A1ResUNetRF(n_samples=D, n_elements=K, nz=Nz, nx=Nx, base_ch=4).cpu()
    op = make_op_for_angle(0.0, PICMUS_GEOMETRY, grid_z, grid_x, device="cpu", nf_chunk=1)

    rf = torch.randn(1, 1, D, K) * 0.1
    target = torch.randn(1, 1, Nz, Nx) * 0.1
    rf_obs = rf[0, 0]

    pred = model(rf)
    assert pred.shape == (1, 1, Nz, Nx)

    loss_img = F.mse_loss(pred, target) + 0.1 * F.l1_loss(pred, target)

    y_pred = op.forward(pred[0, 0])
    if y_pred.is_complex():
        y_pred = y_pred.real
    loss_fid = F.mse_loss(y_pred.to(rf_obs.dtype), rf_obs)

    loss = loss_img + 0.1 * loss_fid
    loss.backward()

    # Gradient must reach the conv layers of the model (rf_encoder + decoder).
    grads_present = sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    assert grads_present > 5, f"too few params received finite gradient: {grads_present}"

    # H_Fdf must have NO learnable parameters (α contract verified by wrapper test
    # but also here for A1-specific paranoia).
    assert sum(p.numel() for p in op.parameters() if p.requires_grad) == 0
