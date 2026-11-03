# -*- coding: utf-8 -*-
"""Public inference API of the proposed method (UNet-AG).

Wraps the full chain —Fd-F adjoint, F-number map, network and residual
refinement— behind a stable interface, so the model can be used from another
machine without knowing the package's internal layout or the evaluators'
private functions.

Minimal usage
-------------
>>> from unet_ag.inference import UNetAG
>>> model = UNetAG.from_pretrained()              # seed 42 by default
>>> bmode = model.reconstruct_picmus("resolution_distorsion_expe_dataset_rf.hdf5")
>>> bmode.shape
(612, 388)

From your own RF
----------------
>>> geom = Geometry(pitch_m=300e-6, n_elements=128, sampling_freq_hz=20.832e6,
...                 speed_of_sound=1540.0, n_samples=1527, fc=5.208e6)
>>> bmode = model.reconstruct(rf, geom)           # rf: (n_samples, n_elements)

The output is always the log-compressed envelope normalized to [0, 1] on the
evaluation grid, ready to be displayed with a 60 dB dynamic range.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor

from .models.b1_v3_residual import B1V3Residual
from .physics.f_number_map import compute_f_number_map_band_avg
from .physics.h_fdf_wrapper import make_op_for_angle
from .train.b1_attention_gate import _build_grid, _envelope_log_compress

__all__ = ["Geometry", "UNetAG", "DYN_RANGE_DB", "to_decibels"]

DYN_RANGE_DB = 60.0
"""Dynamic range used to compress the envelope, in decibels."""

_SEEDS = ("s42", "s1", "s2")


_CKPT_SUBPATH = Path("methods") / "05_unet_ag_proposed" / "checkpoints"


def _candidate_checkpoint_dirs() -> list[Path]:
    """Places to look for the weights, in priority order.

    The checkpoints are not packaged inside the distribution (41 MB each, and
    they already live in the repository), so they are located at run time:

    1. ``$UNETAG_CHECKPOINTS`` — explicit control, always wins.
    2. Relative to the package — works with an editable install
       (``pip install -e .``) from a clone of the repository.
    3. Relative to the working directory — works when running from a clone
       without installing anything.
    """
    import os
    out: list[Path] = []
    env = os.environ.get("UNETAG_CHECKPOINTS")
    if env:
        out.append(Path(env).expanduser())
    here = Path(__file__).resolve()
    out.append(here.parents[2] / _CKPT_SUBPATH)      # <repo>/code/unet_ag -> <repo>
    out.append(here.parents[1] / _CKPT_SUBPATH)
    cwd = Path.cwd()
    out += [cwd / _CKPT_SUBPATH, cwd.parent / _CKPT_SUBPATH]
    return out


def _resolve_checkpoint(seed: str) -> Path:
    name = f"anchor05_{seed}.pt"
    tried = []
    for d in _candidate_checkpoint_dirs():
        p = d / name
        tried.append(str(p))
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Checkpoint '{name}' not found. Searched in:\n  "
        + "\n  ".join(tried)
        + "\n\nSolutions:\n"
        "  · run from a clone of the repository, or install it with "
        "`pip install -e .` from it;\n"
        "  · or set UNETAG_CHECKPOINTS=/path/to/the/checkpoints;\n"
        "  · or pass the path directly: UNetAG(checkpoint='...') "
        "/ `unetag reconstruct --ckpt ...`."
    )


@dataclass(frozen=True)
class Geometry:
    """Acquisition geometry needed to build the physical operator.

    All values in SI units. The last three fields default to the PICMUS
    transducer (L11-4v).
    """

    n_elements: int
    n_samples: int
    pitch_m: float
    sampling_freq_hz: float
    speed_of_sound: float = 1540.0
    fc: float = 5.208e6
    element_width_m: float = 270e-6
    t_start: float = 0.0
    """Time of the first RF sample. Zero in PICMUS; non-zero in Field II
    simulations, where ``calc_scat`` returns a delay."""

    def as_dict(self) -> dict:
        return asdict(self)


def to_decibels(bmode: np.ndarray, dyn_range_db: float = DYN_RANGE_DB) -> np.ndarray:
    """Convert the [0, 1] output to decibels in [-dyn_range_db, 0]."""
    return np.asarray(bmode) * dyn_range_db - dyn_range_db


class UNetAG:
    """B-mode reconstructor from a single plane wave.

    Parameters
    ----------
    checkpoint : path to the weights file (``.pt``).
    device : ``"cuda"``, ``"cpu"`` or ``"auto"`` (default: GPU if available).
    nz, nx : evaluation grid dimensions. The defaults (612 x 388) are the
        thesis ones; changing them alters the metrics.
    """

    def __init__(self, checkpoint: Union[str, Path],
                 device: str = "auto", nz: int = 612, nx: int = 388):
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint}\n"
                f"Download it from the thesis repository or give the path with "
                f"UNetAG(checkpoint=...)."
            )
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") \
            if device == "auto" else device
        self.nz, self.nx = int(nz), int(nx)

        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.model = B1V3Residual(base_ch=32, embed_ch=16).to(self.device).eval()
        self.model.load_state_dict(state, strict=True)

    # ── constructors ─────────────────────────────────────────────────────
    @classmethod
    def from_pretrained(cls, seed: str = "s42", **kwargs) -> "UNetAG":
        """Load one of the three trained models of the thesis.

        ``seed`` ∈ {``"s42"``, ``"s1"``, ``"s2"``}. The main model is
        ``s42``; the published numbers are the mean of the three.
        """
        if seed not in _SEEDS:
            raise ValueError(f"seed must be one of {_SEEDS}, not {seed!r}")
        return cls(_resolve_checkpoint(seed), **kwargs)

    # ── properties ───────────────────────────────────────────────────────
    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    # ── inference ────────────────────────────────────────────────────────
    def grid(self, geometry: Geometry) -> tuple[np.ndarray, np.ndarray]:
        """Evaluation grid in metres: ``(z, x)``."""
        gz, gx = _build_grid(geometry.as_dict(), self.nz, self.nx)
        return gz.numpy(), gx.numpy()

    @torch.no_grad()
    def adjoint(self, rf: Union[np.ndarray, Tensor],
                geometry: Geometry) -> np.ndarray:
        """Initial solution: the Fd-F adjoint, as a log envelope in [0, 1].

        It is both the network input and the method's anchor image, and on
        its own it is a usable classical reconstruction.
        """
        rf_t = self._prepare_rf(rf, geometry)
        gz, gx = _build_grid(geometry.as_dict(), self.nz, self.nx)
        op = make_op_for_angle(0.0, geometry.as_dict(), gz, gx,
                               device=self.device, nf_chunk=None)
        x0 = op.adjoint(rf_t.to(self.device).to(op.float_dtype))
        if x0.is_complex():
            x0 = x0.real
        return _envelope_log_compress(x0.float().cpu()).numpy()

    @torch.no_grad()
    def reconstruct(self, rf: Union[np.ndarray, Tensor], geometry: Geometry,
                    return_adjoint: bool = False):
        """Reconstruct the B-mode from single-plane-wave RF.

        Parameters
        ----------
        rf : ``(n_samples, n_elements)``, raw RF of the unsteered angle.
        geometry : acquisition geometry.
        return_adjoint : if ``True`` return ``(bmode, adjoint)``.

        Returns
        -------
        ``(nz, nx)`` float32 in [0, 1]: envelope compressed to 60 dB.
        """
        x0_env = self.adjoint(rf, geometry)
        gz, gx = _build_grid(geometry.as_dict(), self.nz, self.nx)
        f_map = compute_f_number_map_band_avg(gz.numpy(), gx.numpy(),
                                              geometry.as_dict())
        x0_t = torch.from_numpy(x0_env).to(self.device)[None, None]
        f_t = torch.from_numpy(f_map).to(self.device)[None, None]
        pred = self.model(x0_t, f_t)[0, 0].cpu().numpy().astype(np.float32)
        return (pred, x0_env) if return_adjoint else pred

    def reconstruct_picmus(self, hdf5_path: Union[str, Path],
                           angle_index: Optional[int] = None,
                           return_adjoint: bool = False):
        """Reconstruct directly from a PICMUS RF file.

        Selects the unsteered angle (0°) unless another one is given with
        ``angle_index``.
        """
        rf, geom = load_picmus_rf(hdf5_path, angle_index=angle_index)
        return self.reconstruct(rf, geom, return_adjoint=return_adjoint)

    # ── internals ────────────────────────────────────────────────────────
    @staticmethod
    def _prepare_rf(rf: Union[np.ndarray, Tensor], geometry: Geometry) -> Tensor:
        t = torch.as_tensor(np.asarray(rf)) if not isinstance(rf, Tensor) else rf
        t = t.float()
        if t.ndim != 2:
            raise ValueError(f"rf must be 2-D (n_samples, n_elements); "
                             f"got {tuple(t.shape)}")
        if t.shape[1] != geometry.n_elements:
            raise ValueError(
                f"rf has {t.shape[1]} channels but the geometry declares "
                f"n_elements={geometry.n_elements}. Is it transposed?")
        # z-score normalization, as in training and evaluation.
        return (t - t.mean()) / (t.std() + 1e-8)


def load_picmus_rf(hdf5_path: Union[str, Path],
                   angle_index: Optional[int] = None
                   ) -> tuple[np.ndarray, Geometry]:
    """Read a PICMUS RF file and return ``(rf_1pw, Geometry)``.

    Requires ``h5py``. The default angle is the one closest to 0°.
    """
    try:
        import h5py
    except ImportError as exc:                       # pragma: no cover
        raise ImportError("Reading PICMUS requires h5py: pip install h5py") from exc

    with h5py.File(hdf5_path, "r") as f:
        g = f["US"]["US_DATASET0000"]
        data = np.asarray(g["data"]["real"])         # (n_angles, n_el, n_samples)
        angles = np.asarray(g["angles"]).reshape(-1)
        fs = float(np.asarray(g["sampling_frequency"]).reshape(-1)[0])
        c = float(np.asarray(g["sound_speed"]).reshape(-1)[0])
        fc = float(np.asarray(g["modulation_frequency"]).reshape(-1)[0]) or 5.208e6
        probe = np.asarray(g["probe_geometry"])      # (3, n_el) or (n_el, 3)
        t0 = float(np.asarray(g["initial_time"]).reshape(-1)[0])

    if probe.shape[0] == 3:
        xs = probe[0, :]
    else:
        xs = probe[:, 0]
    pitch = float(np.abs(np.diff(xs)).mean())
    idx = int(np.argmin(np.abs(angles))) if angle_index is None else int(angle_index)
    rf = np.asarray(data[idx]).T                     # -> (n_samples, n_elements)

    geom = Geometry(
        n_elements=rf.shape[1], n_samples=rf.shape[0], pitch_m=pitch,
        sampling_freq_hz=fs, speed_of_sound=c, fc=fc, t_start=t0,
    )
    return rf, geom
