"""
PICMUS dataset loader for PyTorch.

Loads HDF5 files from the PICMUS challenge database, providing:
- RF channel data for each plane wave angle
- Grid definitions (scan points)
- Phantom metadata for metric evaluation

Structure expected:
    data_root/
        database/
            simulation/
                contrast_speckle/   -> *.hdf5
                resolution_distorsion/ -> *.hdf5
            experiments/
                contrast_speckle/   -> *.hdf5
                resolution_distorsion/ -> *.hdf5

PICMUS HDF5 internal structure (UFF v1.96):
    US/
      US_DATASET0000/
        data/             <- Group, NOT a Dataset
          real: (n_angles, n_elements, n_samples) float32
          imag: (n_angles, n_elements, n_samples) float32
        angles: (n_angles,) float32  [radians]
        sampling_frequency: (1,) float32
        sound_speed: (1,) float32
        probe_geometry: (3, n_elements) float32
        modulation_frequency: (1,) float32
        initial_time: (1,) float32
        PRF: (1,) float32
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PICMUSSequence:
    """A single PICMUS acquisition (one phantom, all angles)."""
    name: str
    category: str              # "simulation" or "experiments"
    subcategory: str           # "contrast_speckle" or "resolution_distorsion"
    rf_data: np.ndarray        # (n_angles, n_elements, n_samples)
    angles_deg: np.ndarray     # (n_angles,) steering angles
    sampling_freq_hz: float
    center_freq_hz: float
    sound_speed: float
    n_elements: int
    pitch_m: float
    # Grid info (where to beamform)
    grid_x: Optional[np.ndarray] = None    # lateral positions [m]
    grid_z: Optional[np.ndarray] = None    # axial positions [m]
    # Phantom metadata for evaluation
    phantom_info: Optional[Dict] = None


def load_picmus_hdf5(
    filepath: str | Path,
    data_type: str = "rf",
) -> PICMUSSequence:
    """
    Load a single PICMUS HDF5 file.

    Parameters
    ----------
    filepath : path to .hdf5 file
    data_type : 'rf' or 'iq'

    Returns
    -------
    PICMUSSequence
    """
    filepath = Path(filepath)

    with h5py.File(filepath, "r") as f:
        ds = f

        if "US" in ds:
            ds = ds["US"]
        if "US_DATASET0000" in ds:
            ds = ds["US_DATASET0000"]

        # ── Load RF/IQ data ──────────────────────────────────────
        rf = None

        if "data" in ds:
            data_item = ds["data"]

            if isinstance(data_item, h5py.Group):
                if "real" in data_item:
                    rf = np.array(data_item["real"])
                    if data_type == "iq" and "imag" in data_item:
                        rf_imag = np.array(data_item["imag"])
                        rf = rf + 1j * rf_imag
                else:
                    for key in data_item.keys():
                        if isinstance(data_item[key], h5py.Dataset):
                            rf = np.array(data_item[key])
                            break
            elif isinstance(data_item, h5py.Dataset):
                rf = np.array(data_item)

        if rf is None:
            for key in ds.keys():
                if "data" in key.lower() and isinstance(ds[key], h5py.Dataset):
                    rf = np.array(ds[key])
                    break

        if rf is None:
            raise KeyError(
                f"Cannot find RF data in {filepath}. "
                f"Available keys at navigation level: {list(ds.keys())}"
            )

        # ── Reshape to (n_angles, n_elements, n_samples) ─────────
        if rf.ndim == 3:
            if rf.shape[2] <= 75:  # likely angles dimension
                rf = rf.transpose(2, 1, 0)
            elif rf.shape[0] <= 75:
                pass  # already (n_angles, n_elements, n_samples)
        elif rf.ndim == 2:
            rf = rf[np.newaxis, :, :]  # single angle

        n_elements = rf.shape[1]

        # ── Read metadata ────────────────────────────────────────
        sampling_freq_hz = 20.832e6
        center_freq_hz = 5.208e6
        pitch_m = 300e-6
        sound_speed = 1540.0

        def _read_scalar(group, names):
            for name in names:
                if name in group and isinstance(group[name], h5py.Dataset):
                    return float(np.array(group[name]).flatten()[0])
                if name in group.attrs:
                    val = group.attrs[name]
                    if hasattr(val, '__len__'):
                        return float(np.array(val).flatten()[0])
                    return float(val)
            return None

        val = _read_scalar(ds, ["sampling_frequency", "fs", "Fs"])
        if val is not None:
            sampling_freq_hz = val

        val = _read_scalar(ds, ["modulation_frequency", "center_frequency", "f0"])
        if val is not None:
            center_freq_hz = val

        val = _read_scalar(ds, ["sound_speed", "c"])
        if val is not None:
            sound_speed = val

        if "probe_geometry" in ds and isinstance(ds["probe_geometry"], h5py.Dataset):
            geom = np.array(ds["probe_geometry"])  # (3, n_elements)
            if geom.shape[0] == 3 and geom.shape[1] > 1:
                x_positions = geom[0, :]
                pitch_m = float(np.mean(np.diff(x_positions)))

        # ── Angles ───────────────────────────────────────────────
        n_angles = rf.shape[0]
        if "angles" in ds:
            angles_rad = np.array(ds["angles"], dtype=np.float64).flatten()
            angles_deg = np.degrees(angles_rad)
        else:
            angles_deg = np.linspace(-16, 16, n_angles)

        # ── Category from file path ──────────────────────────────
        category = "simulation" if "simulation" in str(filepath).lower() else "experiments"
        subcategory = "contrast_speckle" if "contrast" in str(filepath).lower() else "resolution_distorsion"

    return PICMUSSequence(
        name=filepath.stem,
        category=category,
        subcategory=subcategory,
        rf_data=rf.astype(np.float32) if np.isrealobj(rf) else rf.real.astype(np.float32),
        angles_deg=angles_deg,
        sampling_freq_hz=sampling_freq_hz,
        center_freq_hz=center_freq_hz,
        sound_speed=sound_speed,
        n_elements=n_elements,
        pitch_m=pitch_m,
    )


class PICMUSDataset(Dataset):
    """
    PyTorch Dataset for PICMUS RF data.

    Each sample is a single plane-wave acquisition:
    - rf_data: (K, D) -- RF channel data for one angle
    - angle_deg: float -- steering angle
    - sequence_name: str -- which phantom
    - fs, fc, c, pitch_m, n_elements: transducer metadata
    """

    def __init__(
        self,
        data_root: str | Path,
        categories: Optional[List[str]] = None,
        subcategories: Optional[List[str]] = None,
        data_type: str = "rf",
        angle_indices: Optional[List[int]] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.data_type = data_type
        self.normalize = normalize

        if categories is None:
            categories = ["simulation", "experiments"]
        if subcategories is None:
            subcategories = ["contrast_speckle", "resolution_distorsion"]

        self.sequences: List[PICMUSSequence] = []
        self.samples: List[Tuple[int, int]] = []  # (seq_idx, angle_idx)

        for cat in categories:
            for subcat in subcategories:
                folder = self.data_root / "database" / cat / subcat
                if not folder.exists():
                    folder = self.data_root / cat / subcat
                if not folder.exists():
                    continue
                for hdf5_file in sorted(folder.glob("*.hdf5")):
                    if "_scan" in hdf5_file.stem or "_phantom" in hdf5_file.stem:
                        continue
                    other_type = "iq" if data_type == "rf" else "rf"
                    if f"_{other_type}" in hdf5_file.stem:
                        continue
                    try:
                        seq = load_picmus_hdf5(hdf5_file, data_type=data_type)
                        seq_idx = len(self.sequences)
                        self.sequences.append(seq)

                        n_angles = seq.rf_data.shape[0]
                        if angle_indices is None:
                            indices = list(range(n_angles))
                        else:
                            indices = [i for i in angle_indices if i < n_angles]
                        for ai in indices:
                            self.samples.append((seq_idx, ai))
                    except Exception as e:
                        print(f"Warning: Could not load {hdf5_file}: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        seq_idx, angle_idx = self.samples[idx]
        seq = self.sequences[seq_idx]

        rf = seq.rf_data[angle_idx]  # (K, D)

        if self.normalize:
            max_val = np.max(np.abs(rf)) + 1e-12
            rf = rf / max_val

        return {
            "rf_data": torch.from_numpy(rf).float(),  # (K, D)
            "angle_deg": torch.tensor(seq.angles_deg[angle_idx]).float(),
            "sequence_name": seq.name,
            "category": seq.category,
            "subcategory": seq.subcategory,
            "fs": seq.sampling_freq_hz,
            "fc": seq.center_freq_hz,
            "c": seq.sound_speed,
            "pitch_m": seq.pitch_m,
            "n_elements": seq.n_elements,
        }
