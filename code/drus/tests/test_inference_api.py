# -*- coding: utf-8 -*-
"""Tests of the DRUS public API (drus.inference, drus.preparation).

They protect the contract seen by whoever installs the package: thesis
parameters, phantom names, artifact location with actionable errors and the
full inference chain on tiny synthetic artifacts (8 x 8 grid and a toy prior),
without the 34 GB SVD or the 2.2 GB prior.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from drus.inference import DRUS, THESIS_GAMMA, env_log01, to_grid
from drus.observations import PHANTOMS, PicmusRF, phantom_from_filename
from drus.preparation import Artifacts, resolve_artifacts
from drus.prior import PRIOR_FILENAME


# ── thesis parameters ───────────────────────────────────────────────────
def test_thesis_gamma_per_phantom():
    # γ chosen per phantom on the grid {20, 25, 50, 75}
    assert THESIS_GAMMA == {"simu_reso": 75.0, "simu_cont": 50.0,
                            "expe_reso": 75.0, "expe_cont": 25.0}
    assert set(THESIS_GAMMA) == set(PHANTOMS)


@pytest.mark.parametrize("name,key", [
    ("resolution_distorsion_simu_dataset_rf.hdf5", "simu_reso"),
    ("contrast_speckle_simu_dataset_rf.hdf5", "simu_cont"),
    ("resolution_distorsion_expe_dataset_rf.hdf5", "expe_reso"),
    ("contrast_speckle_expe_dataset_rf.hdf5", "expe_cont"),
])
def test_phantom_from_the_file_name(name, key):
    assert phantom_from_filename(Path("/x/y") / name) == key


def test_unknown_phantom_asks_to_give_it():
    with pytest.raises(ValueError, match="give it explicitly"):
        phantom_from_filename("my_data.hdf5")


# ── conversions ─────────────────────────────────────────────────────────
def test_env_log01_is_symmetric_and_normalized():
    x = np.random.default_rng(0).normal(size=(16, 16))
    a, b = env_log01(x), env_log01(-x)
    assert np.array_equal(a, b)                      # uses |x|
    assert a.dtype == np.float32 and 0.0 <= a.min() and a.max() == pytest.approx(1.0)


def test_to_grid_fills_with_zero_outside_the_field():
    img = np.ones((4, 4), np.float32)
    z = np.linspace(0, 1, 4); x = np.linspace(0, 1, 4)
    out = to_grid(img, z, x, np.linspace(-1, 2, 7), np.linspace(0, 1, 4))
    assert out[0].max() == 0.0 and out[-1].max() == 0.0 and out[3].min() == 1.0


# ── artifact location ──────────────────────────────────────────────────
def test_without_artifacts_the_error_says_how_to_build_them(tmp_path):
    with pytest.raises(FileNotFoundError, match="drus prepare"):
        resolve_artifacts(tmp_path / "empty")


def test_status_of_partial_artifacts(tmp_path):
    art = Artifacts(tmp_path)
    (art.operator).mkdir(parents=True)
    (art.operator / "B.npz").write_bytes(b"")
    (art.operator / "meta.json").write_text("{}")
    assert art.missing() == ["svd", "prior"]


# ── full chain on synthetic artifacts ──────────────────────────────────
N_SIDE, NE, K = 8, 4, 200          # K > k_min + 150: the TGC fixes the first samples
NPIX = N_SIDE * N_SIDE


def _fake_artifacts(root: Path) -> Artifacts:
    rng = np.random.default_rng(0)
    art = Artifacts(root)
    art.operator.mkdir(parents=True); art.svd.mkdir(parents=True)
    art.prior.parent.mkdir(parents=True)
    B = sp.random(NPIX, K * NE, density=0.3, random_state=1, format="csr",
                  dtype=np.float32)
    sp.save_npz(art.operator / "B.npz", B)
    np.savez(art.operator / "grid.npz", x_axis=np.linspace(-18e-3, 18e-3, N_SIDE),
             z_axis=np.linspace(9e-3, 48e-3, N_SIDE))
    (art.operator / "meta.json").write_text(json.dumps(
        {"K": K, "k_min": 2, "Ne": NE, "fs": 20.832e6, "c": 1540.0,
         "nx": N_SIDE, "nz": N_SIDE}))
    bh = rng.normal(size=(NPIX, NPIX)).astype(np.float32)
    u, s, vt = np.linalg.svd(bh)
    np.save(art.svd / "U.npy", u.astype(np.float32))
    np.save(art.svd / "S.npy", s.astype(np.float32))
    np.save(art.svd / "Vt.npy", vt.astype(np.float32))
    art.prior.write_bytes(b"")
    return art


class _ZeroPrior(torch.nn.Module):
    """Toy prior: predicts zero noise (6 channels, like learn_sigma)."""

    def forward(self, x, t):
        return torch.zeros(x.shape[0], 6, *x.shape[2:], dtype=x.dtype, device=x.device)


def _rf(seed: int = 0) -> PicmusRF:
    y = np.random.default_rng(seed).normal(size=(K + 200, NE))
    return PicmusRF(y=y, fs=20.832e6, c=1540.0, initial_time=0.0,
                    probe_x=np.linspace(-1e-3, 1e-3, NE), angle_deg=0.0)


def _model(tmp_path: Path) -> DRUS:
    m = DRUS(_fake_artifacts(tmp_path), device="cpu", timesteps=5)
    m._model = _ZeroPrior()
    return m


def test_reconstruct_returns_shape_and_range(tmp_path):
    m = _model(tmp_path)
    res = m.reconstruct(_rf(), phantom="expe_cont", seed=0)
    assert res.x.shape == res.bmode.shape == (N_SIDE, N_SIDE)
    assert res.gamma == THESIS_GAMMA["expe_cont"]
    assert 0.0 <= res.bmode.min() and res.bmode.max() == pytest.approx(1.0)
    assert np.isfinite(res.x).all()


def test_reconstruct_is_deterministic_with_the_seed(tmp_path):
    m = _model(tmp_path)
    a = m.reconstruct(_rf(), phantom="simu_reso", seed=3).x
    b = m.reconstruct(_rf(), phantom="simu_reso", seed=3).x
    assert np.array_equal(a, b)


def test_precomputed_uty_matches_the_one_from_U(tmp_path):
    m = _model(tmp_path)
    by = m.observation(_rf(), "simu_cont")
    via_u = m.spectral_observation(by, phantom=None)          # reads U through mmap
    np.save(m.artifacts.operator / "By_simu_cont.npy", by)
    np.save(m.artifacts.svd / "Uty_simu_cont.npy", via_u + 1.0)  # marks the cached one
    cached = m.spectral_observation(by, phantom="simu_cont")
    assert np.array_equal(cached, via_u + 1.0)                 # used the precomputed one
    other = m.observation(_rf(seed=1), "simu_cont")
    assert np.allclose(m.spectral_observation(other, "simu_cont"),
                       np.load(m.artifacts.svd / "U.npy").T @ other.astype(np.float32),
                       rtol=1e-5, atol=1e-6)                   # different By: recomputes


def test_own_data_require_explicit_parameters(tmp_path):
    m = _model(tmp_path)
    with pytest.raises(ValueError, match="gamma, tgc_alpha and norm_scale"):
        m.reconstruct(_rf())
    res = m.reconstruct(_rf(), gamma=30.0, tgc_alpha=4.0, norm_scale=10.0)
    assert res.phantom is None and res.gamma == 30.0


def test_prior_is_searched_with_the_published_name(tmp_path):
    assert Artifacts(tmp_path).prior.name == PRIOR_FILENAME
