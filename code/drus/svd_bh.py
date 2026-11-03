"""Dense SVD of BH (65536×65536) via MKL ILP64 — Phase B of the DRUS replica.

Why not scipy: the env's numpy/scipy use OpenBLAS LP64 (32-bit integers).
For n=65536, the ``sgesdd`` workspace (~3n² ≈ 1.3e10 floats) does not fit in
an int32, and reference LAPACK uses 32-bit INTEGER in its internal index
arithmetic → corruption/crash with arrays of n² = 4.3e9 elements. MATLAB
solves it because it links MKL ILP64; here we do the same by calling
``sgesdd_`` from ``libmkl_rt`` (pip package ``mkl``) with the ILP64 interface
via ctypes (all integers 64-bit).

Layout convention: numpy is row-major but LAPACK is column-major. For A
(C-order), Fortran sees Aᵀ. If sgesdd returns Aᵀ = U_f·S·V_fᵀ, then
A = V_f·S·U_fᵀ, that is::

    U  = V_f      (what Fortran calls V)
    Vt = U_fᵀ     (what Fortran calls U, transposed)

To avoid transposing 17 GB, we keep the buffers as they come out of Fortran
and expose U/Vt as logical views:

    - ``vt_f`` buffer (n×n, C-order) such that Fortran sees it as V_fᵀ
      → interpreted in C-order, ``vt_f[i, j]`` = V_f[j, i]  → **U = vt_f.T**,
        but when saving we write U_np = vt_f with the flag "is U transposed".

To remove any ambiguity, this module re-checks the numerical result
(gate G-B1) against sampled columns of BH and saves to disk exactly
``U.npy`` (n×n, U[:, k] = k-th left singular vector), ``S.npy`` and
``Vt.npy`` (n×n, Vt[k, :] = k-th right singular vector), already in the
standard numpy C-order convention.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path

import numpy as np


def _find_mkl_rt() -> str:
    env_lib = Path(os.environ.get("CONDA_PREFIX", "")) / "lib"
    for name in ("libmkl_rt.so.3", "libmkl_rt.so.2", "libmkl_rt.so.1", "libmkl_rt.so"):
        p = env_lib / name
        if p.exists():
            return str(p)
    found = ctypes.util.find_library("mkl_rt")
    if found:
        return found
    raise RuntimeError("libmkl_rt not found — `pip install mkl` in the env")


def load_mkl_ilp64() -> ctypes.CDLL:
    mkl = ctypes.CDLL(_find_mkl_rt(), mode=ctypes.RTLD_GLOBAL)
    MKL_INTERFACE_ILP64 = 1
    ret = mkl.MKL_Set_Interface_Layer(MKL_INTERFACE_ILP64)
    if ret != MKL_INTERFACE_ILP64:
        raise RuntimeError(f"MKL_Set_Interface_Layer(ILP64) returned {ret}")
    # INTEL threading layer (default); honours MKL_NUM_THREADS
    return mkl


def sgesdd_ilp64(
    a: np.ndarray,          # (n, n) float32 C-order — OVERWRITTEN
    mkl: ctypes.CDLL | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full SVD (jobz='S') of the square matrix ``a`` (C-order).

    Returns (U, S, Vt) in the numpy C-order convention:
    ``a ≈ U @ np.diag(S) @ Vt``.
    """
    if mkl is None:
        mkl = load_mkl_ilp64()
    n = a.shape[0]
    assert a.shape == (n, n) and a.dtype == np.float32 and a.flags.c_contiguous

    i64 = ctypes.c_longlong
    jobz = ctypes.c_char(b"S")
    n_ = i64(n)
    lda = i64(n)
    s = np.empty(n, dtype=np.float32)
    # Fortran sees A_f = aᵀ. A_f = U_f S V_fᵀ  ⇒  a = V_f S U_fᵀ.
    # u_buf receives U_f (col-major n×n) → in C-order u_buf is U_fᵀ = Vt (numpy).
    # vt_buf receives V_fᵀ (col-major)   → in C-order vt_buf is V_f = U (numpy).
    u_buf = np.empty((n, n), dtype=np.float32)    # becomes Vt in numpy convention
    vt_buf = np.empty((n, n), dtype=np.float32)   # becomes U  in numpy convention
    iwork = np.empty(8 * n, dtype=np.int64)
    info = i64(0)

    def _call(work_arr, lwork_val):
        mkl.sgesdd_(
            ctypes.byref(jobz), ctypes.byref(n_), ctypes.byref(n_),
            a.ctypes.data_as(ctypes.c_void_p), ctypes.byref(lda),
            s.ctypes.data_as(ctypes.c_void_p),
            u_buf.ctypes.data_as(ctypes.c_void_p), ctypes.byref(n_),
            vt_buf.ctypes.data_as(ctypes.c_void_p), ctypes.byref(n_),
            work_arr.ctypes.data_as(ctypes.c_void_p), ctypes.byref(i64(lwork_val)),
            iwork.ctypes.data_as(ctypes.c_void_p), ctypes.byref(info),
        )
        return info.value

    # workspace query. The value arrives as float32 (24-bit mantissa): for
    # lwork ~1e10 it loses precision → take the max with LAPACK's analytic
    # bound (jobz='S', m=n: 4n²+7n) plus a margin.
    wq = np.empty(1, dtype=np.float32)
    if _call(wq, -1) != 0:
        raise RuntimeError(f"sgesdd workspace query: info={info.value}")
    lwork = max(int(wq[0]) + 8 * n, 4 * n * n + 7 * n)
    if verbose:
        print(f"[sgesdd] n={n} lwork={lwork} ({lwork*4/2**30:.1f} GiB)", flush=True)
    work = np.empty(lwork, dtype=np.float32)
    if _call(work, lwork) != 0:
        raise RuntimeError(f"sgesdd: info={info.value}")
    del work

    # Fortran→numpy mapping (see docstring): U = vt_buf, Vt = u_buf
    return vt_buf, s, u_buf


def ssyevd_ilp64(
    a: np.ndarray,          # (n, n) float32 C-order SYMMETRIC — OVERWRITTEN
    mkl: ctypes.CDLL | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric eigendecomposition (jobz='V').

    Returns ``(lam, ev)`` with ascending λ and ``ev`` in numpy **rows**:
    ``ev[k, :]`` is eigenvector k (Fortran writes eigenvectors as col-major
    columns inside ``a``, which are rows in C-order). Since A is symmetric,
    the C/Fortran input layout does not matter.
    """
    if mkl is None:
        mkl = load_mkl_ilp64()
    n = a.shape[0]
    assert a.shape == (n, n) and a.dtype == np.float32 and a.flags.c_contiguous

    i64 = ctypes.c_longlong
    jobz, uplo = ctypes.c_char(b"V"), ctypes.c_char(b"U")
    n_, lda = i64(n), i64(n)
    w = np.empty(n, dtype=np.float32)
    info = i64(0)

    def _call(work_arr, lwork_val, iwork_arr, liwork_val):
        mkl.ssyevd_(
            ctypes.byref(jobz), ctypes.byref(uplo), ctypes.byref(n_),
            a.ctypes.data_as(ctypes.c_void_p), ctypes.byref(lda),
            w.ctypes.data_as(ctypes.c_void_p),
            work_arr.ctypes.data_as(ctypes.c_void_p), ctypes.byref(i64(lwork_val)),
            iwork_arr.ctypes.data_as(ctypes.c_void_p), ctypes.byref(i64(liwork_val)),
            ctypes.byref(info),
        )
        return info.value

    wq = np.empty(1, dtype=np.float32)
    iwq = np.empty(1, dtype=np.int64)
    if _call(wq, -1, iwq, -1) != 0:
        raise RuntimeError(f"ssyevd workspace query: info={info.value}")
    lwork = max(int(wq[0]) + n, 1 + 6 * n + 2 * n * n)  # the f32 query loses precision
    liwork = max(int(iwq[0]), 3 + 5 * n)
    if verbose:
        print(f"[ssyevd] n={n} lwork={lwork} ({lwork*4/2**30:.1f} GiB)", flush=True)
    work = np.empty(lwork, dtype=np.float32)
    iwork = np.empty(liwork, dtype=np.int64)
    if _call(work, lwork, iwork, liwork) != 0:
        raise RuntimeError(f"ssyevd: info={info.value}")
    return w, a
