# -*- coding: utf-8 -*-
"""Tests of the public inference API (unet_ag.inference).

They protect the contract seen by someone who installs the package on another
machine: names, shapes, ranges and actionable error messages.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from unet_ag.inference import (DYN_RANGE_DB, Geometry, UNetAG,
                               _resolve_checkpoint, to_decibels)

GEO = Geometry(n_elements=16, n_samples=256, pitch_m=300e-6,
               sampling_freq_hz=20.832e6, speed_of_sound=1540.0)


def _tiny_model(tmp_path: Path) -> UNetAG:
    """A UNetAG with random weights, independent of the checkpoints."""
    from unet_ag.models.b1_v3_residual import B1V3Residual
    ck = tmp_path / "tiny.pt"
    torch.save(B1V3Residual(base_ch=32, embed_ch=16).state_dict(), ck)
    return UNetAG(ck, device="cpu", nz=48, nx=32)


# ── Geometry contract ───────────────────────────────────────────────────
def test_geometry_exposes_the_keys_the_operator_expects():
    d = GEO.as_dict()
    for k in ("n_elements", "n_samples", "pitch_m", "sampling_freq_hz",
              "speed_of_sound", "element_width_m", "fc", "t_start"):
        assert k in d, f"missing key {k} consumed by make_op_for_angle"


def test_geometry_is_immutable():
    with pytest.raises(Exception):
        GEO.n_elements = 999          # frozen dataclass


# ── reconstruction contract ─────────────────────────────────────────────
def test_reconstruct_returns_normalized_envelope(tmp_path):
    m = _tiny_model(tmp_path)
    rf = np.random.default_rng(0).normal(size=(GEO.n_samples, GEO.n_elements))
    out = m.reconstruct(rf, GEO)
    assert out.shape == (48, 32)
    assert out.dtype == np.float32
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0


def test_reconstruct_can_return_the_initial_solution(tmp_path):
    m = _tiny_model(tmp_path)
    rf = np.random.default_rng(1).normal(size=(GEO.n_samples, GEO.n_elements))
    bmode, adjoint = m.reconstruct(rf, GEO, return_adjoint=True)
    assert bmode.shape == adjoint.shape
    # the adjoint is a reconstruction in its own right, not a constant
    assert float(adjoint.std()) > 0.0


def test_transposed_rf_gives_readable_error(tmp_path):
    m = _tiny_model(tmp_path)
    rf = np.zeros((GEO.n_elements, GEO.n_samples))      # swapped
    with pytest.raises(ValueError, match="transposed"):
        m.reconstruct(rf, GEO)


def test_non_2d_rf_gives_error(tmp_path):
    m = _tiny_model(tmp_path)
    with pytest.raises(ValueError, match="2-D"):
        m.reconstruct(np.zeros((2, GEO.n_samples, GEO.n_elements)), GEO)


def test_accepts_torch_tensor_like_numpy(tmp_path):
    m = _tiny_model(tmp_path)
    rf = np.random.default_rng(2).normal(size=(GEO.n_samples, GEO.n_elements))
    a = m.reconstruct(rf, GEO)
    b = m.reconstruct(torch.from_numpy(rf), GEO)
    np.testing.assert_allclose(a, b, atol=1e-5)


# ── utilities and errors ────────────────────────────────────────────────
def test_grid_has_the_requested_dimensions(tmp_path):
    m = _tiny_model(tmp_path)
    gz, gx = m.grid(GEO)
    assert gz.shape == (48,) and gx.shape == (32,)
    assert gz[0] < gz[-1] and gx[0] < gx[-1]


def test_to_decibels_maps_the_full_range():
    np.testing.assert_allclose(to_decibels(np.array([0.0, 1.0])),
                               np.array([-DYN_RANGE_DB, 0.0]))


def test_missing_checkpoint_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        UNetAG(tmp_path / "does_not_exist.pt", device="cpu")


def test_checkpoint_resolver_honours_the_environment_variable(tmp_path, monkeypatch):
    d = tmp_path / "ckpts"
    d.mkdir()
    (d / "anchor05_s42.pt").write_bytes(b"")
    monkeypatch.setenv("UNETAG_CHECKPOINTS", str(d))
    assert _resolve_checkpoint("s42") == d / "anchor05_s42.pt"


def test_resolver_lists_where_it_searched_on_failure(tmp_path, monkeypatch):
    # Inside a clone of the repo the resolver DOES find the real weights
    # (which is correct), so to test the failure we simulate an installation
    # without them: the candidates are replaced by empty paths.
    import unet_ag.inference as inf
    monkeypatch.setattr(inf, "_candidate_checkpoint_dirs",
                        lambda: [tmp_path / "empty_a", tmp_path / "empty_b"])
    with pytest.raises(FileNotFoundError) as e:
        inf._resolve_checkpoint("s1")
    msg = str(e.value)
    assert "anchor05_s1.pt" in msg
    assert "empty_a" in msg and "empty_b" in msg   # says where it searched
    assert "UNETAG_CHECKPOINTS" in msg             # and how to fix it


def test_from_pretrained_rejects_invalid_seed():
    with pytest.raises(ValueError, match="seed must be one of"):
        UNetAG.from_pretrained(seed="s99")
