"""Smoke tests for proposal B1.

These exercise the forward pass of the B1 architecture with synthetic inputs
on CPU. Validate:
- Output shape matches input image grid.
- Gradient flows from output to both ``x`` and ``f_map`` paths (through the
  Conditional Attention Gates).
- Parameter count is within an expected band for ``base_ch=32``.
"""

from __future__ import annotations

import torch

from unet_ag.models.b1_unet_attention_fnum import (
    B1UNetAttentionFNum,
    ConditionalAttentionGate,
    FNumberEmbedding,
)


def test_b1_forward_shape_cpu():
    model = B1UNetAttentionFNum(in_ch=1, base_ch=8, embed_ch=4).cpu()
    B, Nz, Nx = 2, 64, 32
    x = torch.randn(B, 1, Nz, Nx)
    f_map = torch.rand(B, 1, Nz, Nx) * 5.0 + 1.5
    y = model(x, f_map)
    assert y.shape == (B, 1, Nz, Nx)
    assert torch.isfinite(y).all()


def test_b1_backward_flows_through_attention_gate():
    model = B1UNetAttentionFNum(in_ch=1, base_ch=8, embed_ch=4).cpu()
    x = torch.randn(1, 1, 64, 32, requires_grad=True)
    f_map = (torch.rand(1, 1, 64, 32) * 5.0 + 1.5).requires_grad_(True)
    y = model(x, f_map)
    y.sum().backward()
    # Gradient must reach both input branches.
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert f_map.grad is not None and torch.isfinite(f_map.grad).all()


def test_b1_param_count_in_expected_band():
    model = B1UNetAttentionFNum(in_ch=1, base_ch=32, embed_ch=16)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # base_ch=32 U-Net with 4 down + 4 up + AGs lands in the low-millions
    # (Down: ~1.2M, bottleneck: ~1.2M, Up: ~0.9M, AGs+head+embed: ~0.3M).
    assert 2_000_000 < n < 8_000_000, f"unexpected param count: {n:,}"


def test_conditional_attention_gate_alpha_in_unit_range():
    ag = ConditionalAttentionGate(x_ch=16, g_ch=16, f_ch=8, inter_ch=8)
    x = torch.randn(1, 16, 32, 32)
    g = torch.randn(1, 16, 32, 32)
    f_emb = torch.randn(1, 8, 32, 32)
    out = ag(x, g, f_emb)
    # AG returns x * α with α = σ(...) ∈ (0, 1). Magnitudes don't exceed |x|.
    assert out.shape == x.shape
    # alpha-like bound: |out| ≤ |x| element-wise.
    assert torch.all(out.abs() <= x.abs() + 1e-5)


def test_f_number_embedding_finite_for_realistic_inputs():
    emb = FNumberEmbedding(embed_ch=8)
    f_map = torch.linspace(1.5, 10.0, 64).reshape(1, 1, 64, 1).repeat(1, 1, 1, 32)
    out = emb(f_map)
    assert out.shape == (1, 8, 64, 32)
    assert torch.isfinite(out).all()
    # tanh output ∈ [-1, 1]
    assert out.min().item() >= -1.0 - 1e-5
    assert out.max().item() <= 1.0 + 1e-5
