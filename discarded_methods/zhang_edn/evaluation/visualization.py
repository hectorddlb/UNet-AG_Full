"""
B-mode visualization utilities for evaluation.

Builds on ``zhang_dnn.utils.to_bmode_db`` to provide comparison plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from zhang_dnn.utils import to_bmode_db


def plot_bmode_comparison(
    images: dict[str, Tensor],
    grid_x_m: np.ndarray,
    grid_z_m: np.ndarray,
    dynamic_range_db: float = 60.0,
    suptitle: str = "",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Side-by-side B-mode comparison of multiple images.

    Parameters
    ----------
    images : {label: tensor} — each tensor is (Nz, Nx) real-valued.
    grid_x_m, grid_z_m : 1-D arrays of grid positions in metres.
    dynamic_range_db : display dynamic range.
    suptitle : figure super-title.
    save_path : if given, save the figure.

    Returns
    -------
    matplotlib Figure.
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 7), squeeze=False)
    axes = axes[0]

    x_mm = grid_x_m * 1e3
    z_mm = grid_z_m * 1e3
    extent = [x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]]

    for i, (label, img) in enumerate(images.items()):
        db = to_bmode_db(img, dynamic_range_db=dynamic_range_db)
        im = axes[i].imshow(
            db, extent=extent, aspect="auto", cmap="gray",
            vmin=-dynamic_range_db, vmax=0,
        )
        axes[i].set_xlabel("Lateral (mm)")
        if i == 0:
            axes[i].set_ylabel("Axial (mm)")
        axes[i].set_title(label)
        plt.colorbar(im, ax=axes[i], label="dB")

    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig
