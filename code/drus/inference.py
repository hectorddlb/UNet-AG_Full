# -*- coding: utf-8 -*-
"""Public inference API of DRUS, the compared diffusion method.

Wraps the replica's full chain —``By`` observation with the PICMUS physical
model, spectral projection ``Uᵀ·By``, 50-step DDRM sampling with the diffusion
prior— behind the same interface class as the proposed method
(``unet_ag.inference``).

Minimal usage
-------------
>>> from drus.inference import DRUS
>>> model = DRUS.from_artifacts()             # locates $DRUS_ARTIFACTS
>>> res = model.reconstruct_picmus("contrast_speckle_expe_dataset_rf.hdf5")
>>> res.bmode.shape                           # log envelope [0, 1], 256 x 256
(256, 256)

Thesis conditions: prior without fine-tuning, parameter ``γ`` chosen per
phantom by grid search —as in the original paper— and one sampling seed (0).
Without an explicit ``gamma``, ``reconstruct_picmus`` applies the thesis ``γ``
for the phantom.

Requirements: a GPU for the 552.8 M parameter UNet and ~20 GB of RAM for the
``Vᵀ`` matrix of the spectral decomposition, whose products run on CPU as in
the original code. The artifacts are built once with ``drus prepare`` (see
``drus.preparation``).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp
import torch

from .ddrm_sampler import SpectralOps, ddrm_sample, expand_uty
from .observations import (PHANTOMS, PicmusRF, compute_by, load_picmus_rf_file,
                           phantom_from_filename)
from .preparation import Artifacts, resolve_artifacts
from .prior import load_prior, n_parameters

__all__ = ["DRUS", "DRUSResult", "THESIS_GAMMA", "DYN_RANGE_DB", "env_log01",
           "to_grid"]

DYN_RANGE_DB = 60.0
"""Dynamic range used to compress the envelope, in decibels."""

THESIS_GAMMA = {"simu_reso": 75.0, "simu_cont": 50.0,
                "expe_reso": 75.0, "expe_cont": 25.0}
"""Per-phantom ``γ`` with which the DRUS metrics are reported in the thesis
(grid search over {20, 25, 50, 75}, seed 0, prior without fine-tuning)."""

BETAS = np.linspace(1e-4, 0.02, 1000)
"""Linear noise schedule of the prior (1000 steps)."""


def env_log01(x: np.ndarray, dyn_range_db: float = DYN_RANGE_DB) -> np.ndarray:
    """|x| normalized to its maximum, in dB clipped at ``-dyn_range_db``, in [0, 1].

    It is the conversion with which the thesis evaluation brings the DRUS
    output (signed reflectivity) to the common ``env_log`` domain.
    """
    env = np.abs(np.asarray(x)).astype(np.float64)
    env = env / (env.max() + 1e-30)
    db = np.clip(20.0 * np.log10(env + 1e-30), -dyn_range_db, 0.0)
    return ((db + dyn_range_db) / dyn_range_db).astype(np.float32)


def to_grid(img: np.ndarray, src_z: np.ndarray, src_x: np.ndarray,
            dst_z: np.ndarray, dst_x: np.ndarray) -> np.ndarray:
    """Bilinear interpolation to another grid, with zero outside the field of view.

    It is the adaptation with which the thesis evaluates DRUS on the common
    612 x 388 point grid.
    """
    from scipy.interpolate import RegularGridInterpolator

    f = RegularGridInterpolator((src_z, src_x), np.asarray(img, dtype=np.float64),
                                bounds_error=False, fill_value=0.0)
    zz, xx = np.meshgrid(dst_z, dst_x, indexing="ij")
    return f(np.stack([zz.ravel(), xx.ravel()], axis=-1)) \
        .reshape(len(dst_z), len(dst_x)).astype(np.float32)


@dataclass
class DRUSResult:
    """Output of a reconstruction."""

    x: np.ndarray
    """``(256, 256)`` float32: signed reflectivity in the prior domain
    (mean of the three channels of the last DDRM step)."""
    bmode: np.ndarray
    """``(256, 256)`` float32 in [0, 1]: ``env_log01(x)``."""
    z_axis: np.ndarray
    x_axis: np.ndarray
    phantom: Optional[str]
    gamma: float
    seed: int
    seconds: float


class DRUS:
    """DRUS reconstructor (DDRM with the diffusion prior) from one plane wave.

    Parameters
    ----------
    artifacts : artifacts root (``Artifacts``, a path or ``None`` to search for it).
    device : UNet device: ``"cuda"``, ``"cpu"`` or ``"auto"``.
    timesteps, eta, eta_b : sampler parameters (those of DRUS-v1).
    """

    def __init__(self, artifacts: Union[Artifacts, str, Path, None] = None,
                 device: str = "auto", timesteps: int = 50, eta: float = 0.85,
                 eta_b: float = 1.0):
        self.artifacts = artifacts if isinstance(artifacts, Artifacts) \
            else resolve_artifacts(artifacts)
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") \
            if device == "auto" else device
        self.timesteps, self.eta, self.eta_b = int(timesteps), float(eta), float(eta_b)

        op = self.artifacts.operator
        self.meta = json.loads((op / "meta.json").read_text())
        grid = np.load(op / "grid.npz")
        self.x_axis = grid["x_axis"].astype(np.float64)
        self.z_axis = grid["z_axis"].astype(np.float64)
        self._B: Optional[sp.csr_matrix] = None
        self._ops: Optional[SpectralOps] = None
        self._model: Optional[torch.nn.Module] = None

    # ── constructors ─────────────────────────────────────────────────────
    @classmethod
    def from_artifacts(cls, root: Union[str, Path, None] = None, **kwargs) -> "DRUS":
        """Load the replica from the artifacts root (``$DRUS_ARTIFACTS``)."""
        return cls(resolve_artifacts(root), **kwargs)

    # ── heavy pieces, loaded on demand ──────────────────────────────────
    @property
    def B(self) -> sp.csr_matrix:
        if self._B is None:
            self._B = sp.load_npz(self.artifacts.operator / "B.npz").tocsr()
        return self._B

    @property
    def spectral(self) -> SpectralOps:
        """``S`` and ``Vᵀ`` in RAM (17 GB); the products run on CPU."""
        if self._ops is None:
            t0 = time.time()
            s = np.load(self.artifacts.svd / "S.npy")
            vt = np.load(self.artifacts.svd / "Vt.npy")
            self._ops = SpectralOps(s, vt, spectral_device="cpu")
            print(f"[drus] SVD in RAM in {time.time()-t0:.0f} s", flush=True)
        return self._ops

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            self._model = load_prior(self.artifacts.prior, device=self.device)
        return self._model

    @property
    def n_parameters(self) -> int:
        return n_parameters(self.model)

    # ── inference chain ─────────────────────────────────────────────────
    def observation(self, rf: PicmusRF, phantom: Optional[str] = None,
                    tgc_alpha: Optional[float] = None,
                    norm_scale: Optional[float] = None) -> np.ndarray:
        """``By``: TGC, normalization and apodized beamformer of the paper.

        The TGC and the normalization scale depend on the phantom, as in the
        original code; for your own data they are given explicitly.
        """
        return compute_by(self.B, rf, int(self.meta["K"]), int(self.meta["k_min"]),
                          phantom or "own data", tgc_alpha=tgc_alpha,
                          norm_scale=norm_scale)[0]

    def spectral_observation(self, by: np.ndarray,
                             phantom: Optional[str] = None) -> np.ndarray:
        """``Uᵀ·By``. Reuses the precomputed one if ``By`` is that of the
        canonical phantom (up to floating-point rounding: recomputing ``By`` on
        another machine differs by ~1e-13); otherwise it computes it by reading
        ``U`` (17 GB) through memory mapping.
        """
        sv, op = self.artifacts.svd, self.artifacts.operator
        if phantom is not None:
            p_uty, p_by = sv / f"Uty_{phantom}.npy", op / f"By_{phantom}.npy"
            if p_uty.exists() and p_by.exists():
                ref = np.load(p_by)
                if ref.shape == by.shape and np.allclose(
                        by, ref, rtol=1e-9, atol=1e-12 * float(np.abs(ref).max())):
                    return np.load(p_uty)
        p_u = sv / "U.npy"
        if not p_u.exists():
            raise FileNotFoundError(
                f"An observation other than the precomputed ones needs "
                f"{p_u} (the U matrix of the SVD).")
        u = np.load(p_u, mmap_mode="r")
        return np.asarray(u.T @ by.astype(np.float32), dtype=np.float32)

    def reconstruct(self, rf: PicmusRF, phantom: Optional[str] = None,
                    gamma: Optional[float] = None, seed: int = 0,
                    tgc_alpha: Optional[float] = None,
                    norm_scale: Optional[float] = None) -> DRUSResult:
        """Reconstruct from the RF of the 0° angle (``PicmusRF``).

        ``phantom`` selects the TGC, the scale and the thesis ``γ``; without
        it ``gamma``, ``tgc_alpha`` and ``norm_scale`` must be given.
        """
        if phantom is None and None in (gamma, tgc_alpha, norm_scale):
            raise ValueError("without `phantom`, gamma, tgc_alpha and norm_scale must be given")
        if phantom is not None and phantom not in PHANTOMS:
            raise ValueError(f"phantom must be one of {list(PHANTOMS)}, not {phantom!r}")
        g = THESIS_GAMMA[phantom] if gamma is None else float(gamma)
        by = self.observation(rf, phantom, tgc_alpha, norm_scale)
        uty = self.spectral_observation(by, phantom)
        res = ddrm_sample(self.model, BETAS, self.spectral, expand_uty(uty),
                          gamma=max(g, 0.1), image_size=int(self.meta["nx"]),
                          timesteps=self.timesteps, eta=self.eta,
                          eta_b=self.eta_b, seed=int(seed),
                          model_device=self.device, verbose=False)
        return DRUSResult(x=res.x_mean, bmode=env_log01(res.x_mean),
                          z_axis=self.z_axis, x_axis=self.x_axis,
                          phantom=phantom, gamma=g, seed=int(seed),
                          seconds=res.seconds)

    def reconstruct_picmus(self, hdf5_path: Union[str, Path],
                           phantom: Optional[str] = None,
                           gamma: Optional[float] = None,
                           seed: int = 0) -> DRUSResult:
        """Reconstruct directly from a PICMUS RF file.

        The phantom is inferred from the file name if not given.
        """
        phantom = phantom or phantom_from_filename(hdf5_path)
        rf = load_picmus_rf_file(Path(hdf5_path))
        if abs(rf.fs - float(self.meta["fs"])) >= 1 or abs(rf.c - float(self.meta["c"])) >= 1e-6:
            raise ValueError("the file geometry does not match the operator's")
        return self.reconstruct(rf, phantom=phantom, gamma=gamma, seed=seed)
