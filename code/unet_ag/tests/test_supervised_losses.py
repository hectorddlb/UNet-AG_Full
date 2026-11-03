"""Tests for supervised losses (MSE+L1 and the F3 fix MSE+L1+SSIM)."""

from __future__ import annotations

import torch

from unet_ag.losses.supervised import (
    mse_l1_loss,
    mse_l1_ssim_loss,
    ssim_index,
    ssim_loss,
)


def test_ssim_index_identical_images_is_one():
    x = torch.rand(1, 1, 32, 32)
    s = float(ssim_index(x, x))
    assert s > 0.999


def test_ssim_loss_identical_images_is_zero():
    x = torch.rand(1, 1, 32, 32)
    assert float(ssim_loss(x, x)) < 1e-3


def test_ssim_index_uncorrelated_is_low():
    torch.manual_seed(0)
    x = torch.rand(1, 1, 64, 64)
    y = torch.rand(1, 1, 64, 64)
    assert float(ssim_index(x, y)) < 0.2


def test_mse_l1_ssim_loss_breakdown_keys():
    x = torch.rand(1, 1, 32, 32, requires_grad=True)
    y = torch.rand(1, 1, 32, 32)
    loss, br = mse_l1_ssim_loss(x, y)
    assert set(br) == {"mse", "l1", "ssim_loss", "total"}
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_mse_l1_ssim_loss_weights_sum_to_one_for_unit_inputs():
    """When pred=target the loss should be exactly 0; the breakdown reflects
    each component's pure value (mse=0, l1=0, ssim_loss≈0)."""
    x = torch.rand(2, 1, 32, 32)
    loss, br = mse_l1_ssim_loss(x, x)
    assert br["mse"] < 1e-6
    assert br["l1"] < 1e-6
    assert br["ssim_loss"] < 1e-3
    assert br["total"] < 1e-3


def test_mse_l1_loss_still_works():
    x = torch.rand(1, 1, 16, 16, requires_grad=True)
    y = torch.rand(1, 1, 16, 16)
    loss, br = mse_l1_loss(x, y, l1_weight=0.1)
    loss.backward()
    assert "mse" in br and "l1" in br and "total" in br
