"""
Initial solution via matched filter / adjoint (Sec. 5 of Zhang et al. 2021).

    x_hat = H^T @ y

This produces a low-quality DAS-like image that serves as input to the EDN.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor


def compute_initial_solution(
    H: sp.spmatrix,
    rf_data: np.ndarray | Tensor,
    n_z: int,
    n_x: int,
) -> Tensor:
    """Compute the adjoint (backprojection) initial solution.

    Parameters
    ----------
    H : sparse matrix of shape (D*K, Nz*Nx).
    rf_data : RF channel data, shape (K, D).
        Will be reordered to the y-vector convention y[d*K + k] = rf[k, d].
    n_z, n_x : spatial dimensions of the output image.

    Returns
    -------
    x_hat : Tensor of shape (Nz, Nx), float32.
    """
    if isinstance(rf_data, Tensor):
        rf_np = rf_data.detach().cpu().numpy()
    else:
        rf_np = np.asarray(rf_data)

    # rf_np is (K, D) -> transpose to (D, K) -> ravel -> y[d*K + k]
    y_vec = rf_np.astype(np.float64).T.ravel()

    H_csc = sp.csc_matrix(H)  # efficient for H^T @ y = (H_csc.T).dot(y)
    x_hat = H_csc.T.dot(y_vec)

    return torch.from_numpy(x_hat.reshape(n_z, n_x).astype(np.float32))
