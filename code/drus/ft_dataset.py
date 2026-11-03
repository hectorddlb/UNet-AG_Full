"""Fine-tuning dataset of the DDRM prior — strict fairness with anchor05.

The images are EXACTLY the supervision B1/anchor05 saw: the Schiffner_75PW
`compounded.npy` targets (287 points phantoms; cysts have no targets),
processed with the same chain as `b1_v4_anchored.py`:

    raw (native 612×388 or bilinear resize to it) → axial hilbert → envelope
    → log-compress 60 dB → [0,1]

and then brought to the diffusion prior domain:

    bilinear resize to 256×256 → scale to [−1,1] → replicated to 3 channels (RGB).

Training augmentation: horizontal flip only (consistent with the L/R symmetry
of the transducer physics).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import hilbert
from torch.utils.data import Dataset

DYN_RANGE_DB = 60.0
NATIVE_NZ, NATIVE_NX = 612, 388


def envelope_log_compress(arr: np.ndarray, dyn_range_db: float = DYN_RANGE_DB) -> np.ndarray:
    """Exact replica of unet_ag.train.b1_attention_gate._envelope_log_compress."""
    env = np.abs(hilbert(arr, axis=0))
    env_max = float(env.max()) + 1e-12
    db = 20.0 * np.log10(env / env_max + 1e-30)
    db = np.clip(db, -dyn_range_db, 0.0)
    return ((db + dyn_range_db) / dyn_range_db).astype(np.float32)   # [0, 1]


def _resize(arr: np.ndarray, nz: int, nx: int) -> np.ndarray:
    t = torch.from_numpy(arr.astype(np.float32))[None, None]
    t = F.interpolate(t, size=(nz, nx), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def process_target(raw: np.ndarray, size: int = 256) -> np.ndarray:
    """Raw compounded.npy → env_log [0,1] image of size×size (B1 chain + resize)."""
    if raw.shape != (NATIVE_NZ, NATIVE_NX):
        raw = _resize(raw, NATIVE_NZ, NATIVE_NX)
    img01 = envelope_log_compress(raw)
    return _resize(img01, size, size)


def process_target_signed(raw: np.ndarray, size: int = 256) -> np.ndarray:
    """Raw compounded.npy → SIGNED reflectivity in [−1,1] of size×size.

    Correct domain of the DDRM prior: Zhang's x outputs (checked in the repo's
    .mat files, BH50_fineTune) are zero-mean/symmetric — the prior models
    signed reflectivity, consistent with y = Hx. Normalization by the absolute
    maximum per image. Documented caveat: the bilinear 612→256 resize samples
    the axial carrier at ~2.1 px/λ (Nyquist edge).
    """
    signed = _resize(raw.astype(np.float32), size, size)
    return (signed / (np.abs(signed).max() + 1e-30)).astype(np.float32)


def build_ft_dataset(target_root: Path, out_dir: Path, size: int = 256,
                     domain: str = "signed") -> int:
    proc = {"signed": process_target_signed, "envlog": process_target}[domain]
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for sub in sorted(target_root.glob("*/*/compounded.npy")):
        phantom_id = sub.parent.name                     # e.g. points_0001
        raw = np.load(sub)
        np.save(out_dir / f"{phantom_id}.npy", proc(raw, size))
        n += 1
    return n


class SchiffnerFTDataset(Dataset):
    """Yields (img (3,S,S) float32 in [−1,1], {}) — guided-diffusion protocol.

    ``domain='signed'``: the .npy are already in [−1,1] (signed reflectivity).
    ``domain='envlog'``: the .npy are in [0,1] → mapped to [−1,1].
    In signed, the hflip comes with a random global sign flip (physical
    speckle symmetry: x and −x explain the same B-mode).
    """

    def __init__(self, data_dir: Path, hflip: bool = True, domain: str = "signed"):
        self.files = sorted(Path(data_dir).glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"sin .npy en {data_dir}")
        self.hflip = hflip
        self.domain = domain

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        img = np.load(self.files[idx])
        if self.hflip and np.random.rand() < 0.5:
            img = img[:, ::-1].copy()
        if self.domain == "envlog":
            img = img * 2.0 - 1.0
        elif self.domain == "signed" and np.random.rand() < 0.5:
            img = -img
        img3 = np.repeat(img[None, :, :], 3, axis=0).astype(np.float32)
        return torch.from_numpy(img3), {}
