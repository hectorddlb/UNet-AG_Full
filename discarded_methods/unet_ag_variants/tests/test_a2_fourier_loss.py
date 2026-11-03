"""TDD for proposal A2 — Fourier-weighted axial loss + model smoke.

Properties verified:
1. ``fourier_weighted_axial_loss(x, x, W) == 0`` (identity).
2. Loss is positive and finite for non-identical inputs.
3. Gradient flows from loss back to ``pred``.
4. Spectral weight ``inv_F_squared`` peaks where Schiffner F-number is small.
5. ``build_axial_spectral_weight`` is normalised to mean 1 (numerically stable
   loss scale across grids).
6. A2 model forward emits ``[0, 1]`` sigmoid (target compatibility).
7. A2 model param count in expected band.
8. End-to-end training step (model → loss → backward) runs and updates params.
"""

from __future__ import annotations

import numpy as np
import torch

from unet_ag.losses.fourier_weighted import (
    axial_freq_to_acoustic,
    build_axial_spectral_weight,
    build_axial_spectral_weight_full,
    build_lateral_spectral_weight,
    build_2d_spectral_weight,
    fourier_weighted_axial_loss,
    fourier_weighted_2d_loss,
    mse_l1_fourier_loss,
)
from unet_ag.models.a2_unet_fourier import A2UNetFourier


PICMUS_GEOMETRY = dict(
    pitch_m=304.8e-6,
    element_width_m=270e-6,
    n_elements=128,
    sampling_freq_hz=20.832e6,
    speed_of_sound=1540.0,
    n_samples=512,
    t_start=0.0,
    fc=5.208e6,
)


# ──────────────────────────────────────────────────────────────────
# Fourier weight
# ──────────────────────────────────────────────────────────────────


def test_axial_freq_mapping_round_trip():
    """k_z = 0 → f = 0; Nyquist k_z → f = c/(4·dz)."""
    nz, dz, c = 256, 1e-4, 1540.0                          # 0.1 mm spacing
    f = axial_freq_to_acoustic(nz, dz, c)
    assert f.shape == (nz // 2 + 1,)
    assert f[0] == 0.0
    np.testing.assert_allclose(f[-1], c / (4.0 * dz), rtol=1e-5)


def test_spectral_weight_mean_one_normalised():
    nz, dz = 256, 1e-4
    W = build_axial_spectral_weight(nz, dz, PICMUS_GEOMETRY, mode="inv_F_squared")
    assert W.shape == (nz // 2 + 1,)
    assert float(W.mean()) == 1.0 or abs(float(W.mean()) - 1.0) < 1e-5


def test_spectral_weight_flat_is_uniform_ones():
    nz, dz = 128, 1e-4
    W = build_axial_spectral_weight(nz, dz, PICMUS_GEOMETRY, mode="flat")
    assert torch.allclose(W, torch.ones_like(W), atol=1e-5)


def test_spectral_weight_inv_F_squared_larger_at_small_F():
    """Mode ``inv_F_squared``: W ∝ 1/F² → larger where F is small.

    For Schiffner GratingAngleLB(60°, 1.5), F = max(F_grating, 1.5). In the
    low-mid band (near fc ≈ 5.2 MHz) F = 1.5; in the high band it grows →
    1/F² drops. Therefore W must be larger in the bins of lower f_acoustic.
    """
    nz, dz = 512, 5e-5                                     # finer to span more freqs
    W = build_axial_spectral_weight(nz, dz, PICMUS_GEOMETRY,
                                    mode="inv_F_squared", fc_offset=False)
    # Drop DC (k_z=0 has f≈0 floored; not informative).
    W_no_dc = W[1:]
    assert float(W_no_dc[0]) >= float(W_no_dc[-1]) - 1e-4


# ──────────────────────────────────────────────────────────────────
# Fourier loss
# ──────────────────────────────────────────────────────────────────


def test_fourier_loss_identity_is_zero():
    B, C, Nz, Nx = 1, 1, 64, 16
    nz, dz = Nz, 1e-4
    W = build_axial_spectral_weight(nz, dz, PICMUS_GEOMETRY)
    x = torch.randn(B, C, Nz, Nx)
    loss = fourier_weighted_axial_loss(x, x, W)
    assert float(loss.item()) < 1e-10


def test_fourier_loss_positive_for_different_inputs():
    Nz, Nx = 64, 16
    W = build_axial_spectral_weight(Nz, 1e-4, PICMUS_GEOMETRY)
    x = torch.randn(1, 1, Nz, Nx)
    y = torch.randn(1, 1, Nz, Nx)
    loss = fourier_weighted_axial_loss(x, y, W)
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0


def test_fourier_loss_gradient_flows_to_pred():
    Nz, Nx = 64, 16
    W = build_axial_spectral_weight(Nz, 1e-4, PICMUS_GEOMETRY)
    pred = torch.randn(1, 1, Nz, Nx, requires_grad=True)
    target = torch.randn(1, 1, Nz, Nx)
    loss = fourier_weighted_axial_loss(pred, target, W)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_mse_l1_fourier_combined_breakdown():
    Nz, Nx = 64, 16
    W = build_axial_spectral_weight(Nz, 1e-4, PICMUS_GEOMETRY)
    pred = torch.rand(1, 1, Nz, Nx, requires_grad=True)
    target = torch.rand(1, 1, Nz, Nx)
    loss, br = mse_l1_fourier_loss(pred, target, W)
    for key in ("mse", "l1", "fourier", "total"):
        assert key in br
        assert np.isfinite(br[key])
    loss.backward()
    assert pred.grad is not None


# ──────────────────────────────────────────────────────────────────
# A2 model
# ──────────────────────────────────────────────────────────────────


def test_a2_forward_in_zero_one():
    D, K = 512, 32
    Nz, Nx = 32, 16
    model = A2UNetFourier(n_samples=D, n_elements=K, nz=Nz, nx=Nx, base_ch=8).cpu()
    rf = torch.randn(2, 1, D, K)
    y = model(rf)
    assert y.shape == (2, 1, Nz, Nx)
    assert torch.isfinite(y).all()
    assert (y >= 0).all() and (y <= 1).all()


def test_a2_backward_flows_to_rf_input():
    D, K = 256, 16
    Nz, Nx = 24, 12
    model = A2UNetFourier(n_samples=D, n_elements=K, nz=Nz, nx=Nx, base_ch=4).cpu()
    rf = torch.randn(1, 1, D, K, requires_grad=True)
    y = model(rf)
    y.sum().backward()
    assert rf.grad is not None
    assert torch.isfinite(rf.grad).all()


def test_a2_param_count_in_expected_band():
    """A2 backbone matches A1: ~1–8M params at base_ch=32 over PICMUS grid."""
    D, K = 3500, 128
    Nz, Nx = 612, 388
    model = A2UNetFourier(n_samples=D, n_elements=K, nz=Nz, nx=Nx, base_ch=32)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 1_000_000 < n < 8_000_000, f"unexpected param count: {n:,}"


# ──────────────────────────────────────────────────────────────────
# A2-v2 — 2-D spectral weight & loss
# ──────────────────────────────────────────────────────────────────


def test_axial_full_weight_shape_and_symmetric():
    """Full-Nz axial weight: shape (Nz,), Hermitian-symmetric |k_z| usage."""
    nz, dz = 64, 1e-4
    W = build_axial_spectral_weight_full(nz, dz, PICMUS_GEOMETRY,
                                          mode="inv_F_squared", fc_offset=False)
    assert W.shape == (nz,)
    # bins k and Nz - k correspond to ±|k_z| → same weight under |k_z| map.
    for k in range(1, nz // 2):
        assert abs(float(W[k]) - float(W[nz - k])) < 1e-5
    assert abs(float(W.mean()) - 1.0) < 1e-5


def test_lateral_weight_flat_is_uniform_and_highpass_is_monotonic():
    nx = 64
    Wf = build_lateral_spectral_weight(nx, mode="flat")
    assert torch.allclose(Wf, torch.ones_like(Wf), atol=1e-5)

    Wl = build_lateral_spectral_weight(nx, mode="highpass_linear", floor=0.1)
    assert Wl.shape == (nx // 2 + 1,)
    assert abs(float(Wl.mean()) - 1.0) < 1e-5
    # Monotonic non-decreasing.
    diffs = Wl[1:] - Wl[:-1]
    assert (diffs >= -1e-5).all()
    # Floor enforced at k=0 (>0).
    assert float(Wl[0]) > 0.0

    Wq = build_lateral_spectral_weight(nx, mode="highpass_quad", floor=0.1)
    # quad shape rises slower at low-k than linear, faster at high-k.
    # Last bin: quad ≥ linear · ratio because both renormalised to mean-1, but
    # the *normalised* value at the top end is strictly larger for quad.
    # Just check positivity + monotonic.
    diffs_q = Wq[1:] - Wq[:-1]
    assert (diffs_q >= -1e-5).all()


def test_2d_weight_shape_separable_and_mean_one():
    nz, dz, nx = 64, 1e-4, 32
    W2 = build_2d_spectral_weight(nz, dz, nx, PICMUS_GEOMETRY,
                                   axial_mode="inv_F_squared",
                                   lateral_mode="highpass_linear",
                                   fc_offset=False)
    assert W2.shape == (nz, nx // 2 + 1)
    assert abs(float(W2.mean()) - 1.0) < 1e-5
    # Separable check: W2 == W_z ⊗ W_x (mean-1 outer of mean-1 factors is also
    # mean-1, so the global renorm in build_2d_spectral_weight is the identity).
    # Direct elementwise equality at high precision — robust to Nyquist bin
    # being ~1e-13 (axial 1/F² collapses there), which would break any
    # ratio-based check that needs an epsilon in the denominator.
    Wz = build_axial_spectral_weight_full(nz, dz, PICMUS_GEOMETRY,
                                           mode="inv_F_squared", fc_offset=False)
    Wx = build_lateral_spectral_weight(nx, mode="highpass_linear")
    outer = Wz.view(-1, 1) * Wx.view(1, -1)
    assert torch.allclose(W2, outer, rtol=1e-5, atol=1e-14)


def test_fourier_2d_loss_identity_zero_and_grad_flows():
    nz, dz, nx = 32, 1e-4, 16
    W2 = build_2d_spectral_weight(nz, dz, nx, PICMUS_GEOMETRY)
    x = torch.randn(2, 1, nz, nx, requires_grad=True)
    # identity
    loss0 = fourier_weighted_2d_loss(x, x.detach(), W2)
    assert float(loss0.item()) < 1e-9
    # positive + gradient
    target = torch.randn(2, 1, nz, nx)
    loss = fourier_weighted_2d_loss(x, target, W2)
    assert float(loss.item()) > 0.0
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_mse_l1_fourier_loss_dispatch_dim2():
    """fourier_dim=2 path runs end-to-end and produces a finite total."""
    nz, dz, nx = 32, 1e-4, 16
    W2 = build_2d_spectral_weight(nz, dz, nx, PICMUS_GEOMETRY)
    pred = torch.rand(1, 1, nz, nx, requires_grad=True)
    target = torch.rand(1, 1, nz, nx)
    loss, br = mse_l1_fourier_loss(pred, target, W2, fourier_dim=2)
    for key in ("mse", "l1", "fourier", "total"):
        assert key in br and np.isfinite(br[key])
    loss.backward()
    assert pred.grad is not None


def test_a2_full_training_step_updates_params():
    """Model forward → combined loss → backward → optimizer step changes params."""
    D, K = 256, 16
    Nz, Nx = 32, 16
    model = A2UNetFourier(n_samples=D, n_elements=K, nz=Nz, nx=Nx, base_ch=4).cpu()
    W = build_axial_spectral_weight(Nz, 1e-4, PICMUS_GEOMETRY)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    rf = torch.randn(1, 1, D, K) * 0.1
    target = torch.rand(1, 1, Nz, Nx)                    # env+log target lives in [0,1]

    p0 = next(model.parameters()).detach().clone()
    pred = model(rf)
    loss, br = mse_l1_fourier_loss(pred, target, W)
    loss.backward()
    opt.step()
    p1 = next(model.parameters()).detach().clone()

    assert not torch.equal(p0, p1), "AdamW step did not update parameters"
    assert np.isfinite(br["total"])
