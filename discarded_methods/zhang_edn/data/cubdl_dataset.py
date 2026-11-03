"""
CUBDL dataset loader for PyTorch.

The CUBDL archive is less uniform than PICMUS: files may differ in the
number of elements, number of transmissions, and sample length. This loader
normalizes the metadata into the same sample dictionary used by PICMUS, but
it expects the caller to select a homogeneous subset when batching is needed.

By default the loader keeps plane-wave acquisitions only, which are the ones
compatible with the current Zhang-style plane-wave forward model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _normalize_label(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).strip().lower()
    return text.replace(" ", "_").replace("-", "_")


def _find_dataset(node: h5py.Group, names: Sequence[str]) -> Optional[h5py.Dataset]:
    for name in names:
        if name in node and isinstance(node[name], h5py.Dataset):
            return node[name]

    for key in node.keys():
        item = node[key]
        if isinstance(item, h5py.Group):
            found = _find_dataset(item, names)
            if found is not None:
                return found

    return None


def _read_text(node: h5py.Group, names: Sequence[str], default: str = "") -> str:
    for name in names:
        if name in node:
            value = node[name]
            if isinstance(value, h5py.Dataset):
                data = np.array(value).reshape(-1)
                if data.size:
                    return _normalize_label(data[0])
            else:
                return _normalize_label(value)
        if name in node.attrs:
            return _normalize_label(node.attrs[name])
    return default


def _read_scalar(node: h5py.Group, names: Sequence[str], default: float) -> float:
    for name in names:
        if name in node and isinstance(node[name], h5py.Dataset):
            data = np.array(node[name]).reshape(-1)
            if data.size:
                return float(data[0])
        if name in node.attrs:
            value = node.attrs[name]
            if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
                value = np.array(value).reshape(-1)[0]
            return float(value)
    return default


def _read_array(node: h5py.Group, names: Sequence[str]) -> Optional[np.ndarray]:
    dataset = _find_dataset(node, names)
    if dataset is None:
        return None
    return np.array(dataset)


@dataclass
class CUBDLSequence:
    """One CUBDL acquisition file with all transmissions."""

    name: str
    path: str
    category: str
    subcategory: str
    transmission_type: str
    data_type: str
    channel_data: np.ndarray
    angles_deg: np.ndarray
    sampling_freq_hz: float
    center_freq_hz: float
    sound_speed: float
    n_elements: int
    pitch_m: float
    element_positions: Optional[np.ndarray] = None
    pixel_positions: Optional[np.ndarray] = None
    beamformed_data: Optional[np.ndarray] = None
    grid_x: Optional[np.ndarray] = None
    grid_z: Optional[np.ndarray] = None


def load_cubdl_hdf5(filepath: str | Path) -> CUBDLSequence:
    """Load one CUBDL HDF5 file into a normalized sequence object."""
    filepath = Path(filepath)

    with h5py.File(filepath, "r") as f:
        channel_data = _read_array(f, ["channel_data", "rf_data", "data"])
        if channel_data is None:
            raise KeyError(f"Cannot find channel_data in {filepath}")

        if channel_data.ndim == 2:
            channel_data = channel_data[np.newaxis, :, :]
        elif channel_data.ndim != 3:
            raise ValueError(
                f"Unsupported channel_data shape {channel_data.shape} in {filepath}"
            )

        n_transmissions = channel_data.shape[0]
        n_elements = channel_data.shape[1]

        angles = _read_array(f, ["angles"])
        if angles is None:
            angles_deg = np.zeros(n_transmissions, dtype=np.float32)
        else:
            angles = np.array(angles).reshape(-1)
            if np.max(np.abs(angles)) <= np.pi + 1e-6:
                angles_deg = np.degrees(angles)
            else:
                angles_deg = angles.astype(np.float32)

        sampling_freq_hz = _read_scalar(
            f,
            ["channel_data_sampling_frequency", "sampling_frequency"],
            default=20.832e6,
        )
        center_freq_hz = _read_scalar(
            f,
            ["modulation_frequency", "center_frequency", "f0"],
            default=5.208e6,
        )
        sound_speed = _read_scalar(f, ["sound_speed", "c"], default=1540.0)

        element_positions = _read_array(f, ["element_positions"])
        pitch_m = _read_scalar(f, ["pitch"], default=float("nan"))
        if not np.isfinite(pitch_m):
            if element_positions is not None:
                flat_positions = np.array(element_positions)
                if flat_positions.ndim == 1 and flat_positions.size > 1:
                    pitch_m = float(np.mean(np.diff(flat_positions)))
                elif flat_positions.ndim == 2 and flat_positions.shape[0] in (2, 3):
                    pitch_m = float(np.mean(np.diff(flat_positions[0, :])))
                else:
                    pitch_m = 300e-6
            else:
                pitch_m = 300e-6

        transmission_type = _read_text(f, ["transmission_type"], default="unknown")
        data_type = _read_text(f, ["data_type"], default="unknown")

        beamformed_data = _read_array(f, ["beamformed_data"])
        pixel_positions = _read_array(f, ["pixel_positions"])

        grid_x = None
        grid_z = None
        if pixel_positions is not None and pixel_positions.ndim == 3 and pixel_positions.shape[0] >= 3:
            grid_x = np.array(pixel_positions[0, 0, :], dtype=np.float32)
            grid_z = np.array(pixel_positions[2, :, 0], dtype=np.float32)

    return CUBDLSequence(
        name=filepath.stem,
        path=str(filepath),
        category=data_type,
        subcategory=transmission_type,
        transmission_type=transmission_type,
        data_type=data_type,
        channel_data=channel_data.astype(np.float32),
        angles_deg=angles_deg.astype(np.float32),
        sampling_freq_hz=float(sampling_freq_hz),
        center_freq_hz=float(center_freq_hz),
        sound_speed=float(sound_speed),
        n_elements=int(n_elements),
        pitch_m=float(pitch_m),
        element_positions=element_positions,
        pixel_positions=pixel_positions,
        beamformed_data=beamformed_data,
        grid_x=grid_x,
        grid_z=grid_z,
    )


class CUBDLDataset(Dataset):
    """
    PyTorch dataset for CUBDL channel data.

    Each sample returns a dict with rf_data (K, D) and transducer metadata.
    """

    def __init__(
        self,
        data_root: str | Path,
        file_glob: str = "**/*.hdf5",
        transmission_types: Optional[Sequence[str]] = ("plane_wave",),
        single_transmission: bool = True,
        normalize: bool = True,
        strict_shape: bool = True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.file_glob = file_glob
        self.normalize = normalize
        self.single_transmission = single_transmission
        self.strict_shape = strict_shape
        self.transmission_types = (
            {_normalize_label(t) for t in transmission_types}
            if transmission_types is not None
            else None
        )

        self.sequences: List[CUBDLSequence] = []
        self.samples: List[Tuple[int, int]] = []
        self.reference_shape: Optional[Tuple[int, int]] = None
        self.skipped_files: List[Tuple[str, str]] = []

        if "**" in file_glob:
            files = sorted(self.data_root.glob(file_glob))
        else:
            files = sorted(self.data_root.rglob(file_glob))

        for hdf5_file in files:
            if not hdf5_file.is_file():
                continue
            try:
                seq = load_cubdl_hdf5(hdf5_file)
            except Exception as exc:
                self.skipped_files.append((str(hdf5_file), str(exc)))
                continue

            if self.transmission_types is not None:
                if _normalize_label(seq.transmission_type) not in self.transmission_types:
                    self.skipped_files.append((str(hdf5_file), f"filtered transmission_type={seq.transmission_type}"))
                    continue

            shape = (seq.n_elements, seq.channel_data.shape[2])
            if self.reference_shape is None:
                self.reference_shape = shape
            elif self.strict_shape and shape != self.reference_shape:
                self.skipped_files.append((str(hdf5_file), f"shape mismatch {shape} != {self.reference_shape}"))
                continue

            seq_idx = len(self.sequences)
            self.sequences.append(seq)

            if self.single_transmission:
                indices = [seq.channel_data.shape[0] // 2]
            else:
                indices = list(range(seq.channel_data.shape[0]))

            for tx_idx in indices:
                self.samples.append((seq_idx, tx_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        seq_idx, tx_idx = self.samples[idx]
        seq = self.sequences[seq_idx]

        rf = seq.channel_data[tx_idx]  # (K, D)
        if self.normalize:
            max_val = np.max(np.abs(rf)) + 1e-12
            rf = rf / max_val

        angle_deg = float(seq.angles_deg[tx_idx]) if tx_idx < len(seq.angles_deg) else 0.0

        return {
            "rf_data": torch.from_numpy(np.asarray(rf)).float(),
            "angle_deg": torch.tensor(angle_deg).float(),
            "sequence_name": seq.name,
            "category": seq.category,
            "subcategory": seq.subcategory,
            "transmission_type": seq.transmission_type,
            "fs": seq.sampling_freq_hz,
            "fc": seq.center_freq_hz,
            "c": seq.sound_speed,
            "pitch_m": seq.pitch_m,
            "n_elements": seq.n_elements,
        }
