# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
_libocas.pyx – thin Cython bindings for libocas binary and multi-class
               linear SVM solvers.

Exported Python functions
-------------------------
train_binary(X, y, C, tol_rel, tol_abs, qp_bound, max_time,
             buf_size, method, fit_intercept)  -> dict
train_msvm  (X, y, nY, C, tol_rel, tol_abs, qp_bound, max_time,
             buf_size, method)                 -> dict
"""

import numpy as np
cimport numpy as cnp

from libc.stdlib  cimport malloc, calloc, free
from libc.string  cimport memset, memcpy
from libc.stdint  cimport uint32_t, uint8_t, int8_t

cnp.import_array()


# ── C declarations ───────────────────────────────────────────────────────────

cdef extern from "libocas.h":
    ctypedef struct ocas_return_value_T:
        uint32_t nIter
        uint32_t nCutPlanes
        uint32_t nNZAlpha
        uint32_t trn_err
        double   Q_P
        double   Q_D
        double   output_time
        double   sort_time
        double   add_time
        double   w_time
        double   qp_solver_time
        double   ocas_time
        double   print_time
        int8_t   qp_exitflag
        int8_t   exitflag


cdef extern from "ocas_wrapper.h":

    ctypedef struct bin_ctx_t:
        double   *X
        uint32_t  nDim
        uint32_t  nData
        double   *data_y
        double    X0
        double   *W
        double   *oldW
        double    W0
        double    oldW0
        double   *new_a
        double   *full_A
        double   *A0

    ctypedef struct msvm_ctx_t:
        double   *X
        uint32_t  nDim
        uint32_t  nData
        uint32_t  nY
        double   *data_y
        double   *W
        double   *oldW
        double   *new_a
        double   *full_A

    ocas_return_value_T train_binary_ocas(
        bin_ctx_t *ctx,
        double C, double TolRel, double TolAbs,
        double QPBound, double MaxTime,
        uint32_t BufSize, uint8_t Method)

    ocas_return_value_T train_msvm_ocas(
        msvm_ctx_t *ctx,
        double C, double TolRel, double TolAbs,
        double QPBound, double MaxTime,
        uint32_t BufSize, uint8_t Method)


# ── helpers ──────────────────────────────────────────────────────────────────

cdef inline dict _retval_to_dict(ocas_return_value_T rv):
    return {
        "n_iter":         int(rv.nIter),
        "n_cut_planes":   int(rv.nCutPlanes),
        "n_nz_alpha":     int(rv.nNZAlpha),
        "trn_err":        int(rv.trn_err),
        "Q_P":            rv.Q_P,
        "Q_D":            rv.Q_D,
        "ocas_time":      rv.ocas_time,
        "qp_exitflag":    int(rv.qp_exitflag),
        "exitflag":       int(rv.exitflag),
    }


# ── Binary SVM ───────────────────────────────────────────────────────────────

def train_binary(
        cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] X,
        cnp.ndarray[cnp.float64_t, ndim=1]           y,
        double   C        = 1.0,
        double   tol_rel  = 1e-3,
        double   tol_abs  = 0.0,
        double   qp_bound = -1e300,
        double   max_time = 1e300,
        uint32_t buf_size = 2000,
        uint8_t  method   = 1,          # 1 = OCAS, 0 = CP
        int      fit_intercept = 1,
    ):
    """
    Train a binary linear SVM.

    Parameters
    ----------
    X            : (nData, nDim) float64 C-contiguous
    y            : (nData,)      float64, values in {-1, +1}
    C            : regularisation constant
    tol_rel      : relative duality-gap tolerance
    tol_abs      : absolute duality-gap tolerance
    qp_bound     : stop if primal ≤ qp_bound
    max_time     : wall-clock budget (seconds)
    buf_size     : maximum number of buffered cutting planes
    method       : 1 = OCAS (default), 0 = standard CP
    fit_intercept: 1 = add bias term, 0 = no bias

    Returns
    -------
    dict with keys:
        W         – weight vector, shape (nDim,)
        W0        – bias scalar
        stats     – solver statistics dict
    """
    cdef uint32_t nData = <uint32_t>X.shape[0]
    cdef uint32_t nDim  = <uint32_t>X.shape[1]

    # Ensure labels are C-contiguous float64
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] y_c = \
        np.ascontiguousarray(y, dtype=np.float64)

    # Clamp buf_size to nData (no point buffering more planes than samples)
    if buf_size > nData:
        buf_size = nData
    if buf_size < 1:
        buf_size = 1

    # Allocate context and all working arrays
    cdef bin_ctx_t ctx
    ctx.nDim  = nDim
    ctx.nData = nData
    ctx.X     = <double *>X.data
    ctx.data_y = <double *>y_c.data
    ctx.X0    = 1.0 if fit_intercept else 0.0
    ctx.W0    = 0.0
    ctx.oldW0 = 0.0

    ctx.W     = <double *>calloc(nDim,          sizeof(double))
    ctx.oldW  = <double *>calloc(nDim,          sizeof(double))
    ctx.new_a = <double *>calloc(nDim,          sizeof(double))
    ctx.full_A = <double *>calloc(<size_t>nDim * buf_size, sizeof(double))
    ctx.A0    = <double *>calloc(buf_size,      sizeof(double))

    if (ctx.W == NULL or ctx.oldW == NULL or ctx.new_a == NULL
            or ctx.full_A == NULL or ctx.A0 == NULL):
        free(ctx.W); free(ctx.oldW); free(ctx.new_a)
        free(ctx.full_A); free(ctx.A0)
        raise MemoryError("Cannot allocate OCAS working arrays")

    cdef ocas_return_value_T rv
    try:
        rv = train_binary_ocas(
            &ctx, C, tol_rel, tol_abs, qp_bound, max_time,
            buf_size, method)
    finally:
        pass   # free below regardless

    # Copy results before freeing
    W  = np.empty(nDim, dtype=np.float64)
    cdef cnp.float64_t[::1] W_view = W
    cdef uint32_t j
    for j in range(nDim):
        W_view[j] = ctx.W[j]
    W0 = ctx.W0

    free(ctx.W); free(ctx.oldW); free(ctx.new_a)
    free(ctx.full_A); free(ctx.A0)

    return {"W": W, "W0": W0, "stats": _retval_to_dict(rv)}


# ── Multi-class SVM ──────────────────────────────────────────────────────────

def train_msvm(
        cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] X,
        cnp.ndarray[cnp.float64_t, ndim=1]           y,
        uint32_t nY,
        double   C        = 1.0,
        double   tol_rel  = 1e-3,
        double   tol_abs  = 0.0,
        double   qp_bound = -1e300,
        double   max_time = 1e300,
        uint32_t buf_size = 2000,
        uint8_t  method   = 1,
    ):
    """
    Train a multi-class linear SVM (Crammer-Singer formulation).

    Parameters
    ----------
    X       : (nData, nDim) float64 C-contiguous
    y       : (nData,)      float64, 1-indexed class labels in {1..nY}
    nY      : number of classes
    (rest as in train_binary)

    Returns
    -------
    dict with keys:
        W     – weight matrix, shape (nY, nDim);  scores = X @ W.T
        stats – solver statistics dict
    """
    cdef uint32_t nData = <uint32_t>X.shape[0]
    cdef uint32_t nDim  = <uint32_t>X.shape[1]
    cdef uint32_t total = nDim * nY

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] y_c = \
        np.ascontiguousarray(y, dtype=np.float64)

    if buf_size > nData:
        buf_size = nData
    if buf_size < 1:
        buf_size = 1

    cdef msvm_ctx_t ctx
    ctx.nDim   = nDim
    ctx.nData  = nData
    ctx.nY     = nY
    ctx.X      = <double *>X.data
    ctx.data_y = <double *>y_c.data

    ctx.W      = <double *>calloc(total,                   sizeof(double))
    ctx.oldW   = <double *>calloc(total,                   sizeof(double))
    ctx.new_a  = <double *>calloc(total,                   sizeof(double))
    ctx.full_A = <double *>calloc(<size_t>total * buf_size, sizeof(double))

    if (ctx.W == NULL or ctx.oldW == NULL or ctx.new_a == NULL
            or ctx.full_A == NULL):
        free(ctx.W); free(ctx.oldW); free(ctx.new_a); free(ctx.full_A)
        raise MemoryError("Cannot allocate MSVM working arrays")

    cdef ocas_return_value_T rv
    try:
        rv = train_msvm_ocas(
            &ctx, C, tol_rel, tol_abs, qp_bound, max_time,
            buf_size, method)
    finally:
        pass

    # W_flat[j + y*nDim] = weight for feature j, class y
    # Reshape to [nY, nDim]: W_matrix[y, j] = W_flat[y*nDim + j]
    W_flat = np.empty(total, dtype=np.float64)
    cdef cnp.float64_t[::1] Wf = W_flat
    cdef uint32_t k
    for k in range(total):
        Wf[k] = ctx.W[k]
    W_matrix = W_flat.reshape(nY, nDim)

    free(ctx.W); free(ctx.oldW); free(ctx.new_a); free(ctx.full_A)

    return {"W": W_matrix, "stats": _retval_to_dict(rv)}
