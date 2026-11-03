"""
Configuration dataclasses for the Zhang DNN pipeline.

All physical quantities are in SI units (metres, seconds, Hz)
unless the field name explicitly says otherwise (e.g. ``angle_deg``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TransducerConfig:
    """Linear-array transducer parameters (Table 3 / PICMUS defaults)."""

    n_elements: int = 128
    center_freq_hz: float = 5.208e6
    sampling_freq_hz: float = 20.832e6
    pitch_m: float = 0.3e-3
    kerf_m: float = 0.03e-3
    element_width_m: float = 0.27e-3
    elevation_focus_m: float = 20e-3
    speed_of_sound: float = 1540.0

    @property
    def wavelength_m(self) -> float:
        return self.speed_of_sound / self.center_freq_hz


@dataclass
class GridConfig:
    """Reconstruction grid in physical coordinates."""

    n_z: int = 609
    n_x: int = 387
    z_range_m: Tuple[float, float] = (5e-3, 50e-3)
    x_range_m: Tuple[float, float] = (-19.2e-3, 19.2e-3)

    @property
    def n_pixels(self) -> int:
        return self.n_z * self.n_x

    def must_be_divisible_by_4(self) -> "GridConfig":
        """Return a copy with n_z and n_x rounded up to multiples of 4.

        The EDN has two MaxPool(2,2) layers, so input spatial dims must be
        divisible by 4.
        """
        import math

        nz = math.ceil(self.n_z / 4) * 4
        nx = math.ceil(self.n_x / 4) * 4
        return GridConfig(
            n_z=nz,
            n_x=nx,
            z_range_m=self.z_range_m,
            x_range_m=self.x_range_m,
        )


@dataclass
class TrainingConfig:
    """Hyper-parameters for the two-stage training (Table 1, Sec. 8)."""

    batch_size: int = 5
    learning_rate: float = 1e-4
    stage1_epochs: int = 40
    stage2_epochs: int = 160
    lambda_wavelet: float = 0.2
    lambda_smoothness: float = 0.2
    lambda_envelope: float = 0.1

    @property
    def total_epochs(self) -> int:
        return self.stage1_epochs + self.stage2_epochs


@dataclass
class MiniConfig:
    """Reduced-size config for CPU development and quick tests (Sec. 15)."""

    grid: GridConfig = field(
        default_factory=lambda: GridConfig(
            n_z=64,
            n_x=32,
            z_range_m=(5e-3, 45e-3),
            x_range_m=(-20e-3, 20e-3),
        )
    )
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            batch_size=1,
            stage1_epochs=5,
            stage2_epochs=10,
        )
    )
    n_phantoms: int = 5
