# cython: language_level=3
"""
Cython wrapper around the libCVM C++ library.

Provides thin Python bindings for svm_train / svm_predict / svm_predict_values
and exposes a LibCVMModel cdef class that owns all allocated C++ memory.
"""

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.string cimport memset

cnp.import_array()

# ─────────────────────────────────────────────────────────────────────────────
# C / C++ declarations from svm.h  (only what the wrapper needs)
# ─────────────────────────────────────────────────────────────────────────────

cdef extern from "svm.h":

    # opaque forward reference ─────────────────────────────────────────────────
    struct SGraphStruct:
        pass

    # sparse feature-vector node: index=-1 terminates a row ───────────────────
    struct svm_node:
        int    index    # INDEX_T = int  by default
        double value    # NODE_T  = double by default

    # training problem ─────────────────────────────────────────────────────────
    struct svm_problem:
        int             l       # number of samples
        int             u       # unlabelled count (always 0 for us)
        double         *y       # label array [l]
        svm_node      **x       # sparse data rows [l]
        SGraphStruct   *graph   # set to NULL

    # hyper-parameters ─────────────────────────────────────────────────────────
    struct svm_parameter:
        int     svm_type
        int     kernel_type
        int     degree
        double  gamma
        double  coef0
        double  cache_size
        double  eps
        double  C
        int     nr_weight
        int    *weight_label
        double *weight
        double  nu
        double  mu
        double  p
        int     mc_type
        int     shrinking
        int     probability
        int     sample_size
        int     num_basis
        int     knn
        int     weight_type

    # trained model ────────────────────────────────────────────────────────────
    struct svm_model:
        svm_parameter  param
        int            nr_class
        int            l
        int            u
        svm_node     **SV
        double       **sv_coef
        double        *rho
        double        *cNorm
        double        *probA
        double        *probB
        int           *label
        int           *nSV
        int            free_sv

    # svm_type constants (anonymous C enum) ────────────────────────────────────
    enum:
        C_SVC        # 0
        NU_SVC       # 1
        ONE_CLASS    # 2
        EPSILON_SVR  # 3
        NU_SVR       # 4
        CVDD         # 5
        CVM          # 6  Core Vector Machine  (squared hinge loss)
        CVM_LS       # 7
        CVR          # 8
        BVM          # 9  Ball Vector Machine

    # kernel_type constants (anonymous C enum) ─────────────────────────────────
    enum:
        LINEAR       # 0
        POLY         # 1
        RBF          # 2
        SIGMOID      # 3
        PRECOMPUTED  # 4
        EXP          # 5  Laplacian: exp(-sqrt(gamma)|u-v|)
        NORMAL_POLY  # 6
        INV_DIST     # 7
        INV_SQDIST   # 8

    # public C API ─────────────────────────────────────────────────────────────
    svm_model  *svm_train(const svm_problem *prob,
                          const svm_parameter *param)
    void        svm_predict_values(const svm_model *model,
                                   const svm_node *x,
                                   double *dec_values)
    double      svm_predict(const svm_model *model, const svm_node *x)
    void        svm_destroy_model(svm_model *model)
    const char *svm_check_parameter(const svm_problem *prob,
                                    const svm_parameter *param)


# ─────────────────────────────────────────────────────────────────────────────
# Public Python-level constants
# ─────────────────────────────────────────────────────────────────────────────

# Mapping from sklearn-style kernel name to libCVM kernel_type integer
KERNEL_MAP = {
    'linear':      0,   # LINEAR
    'poly':        1,   # POLY
    'rbf':         2,   # RBF
    'sigmoid':     3,   # SIGMOID
    'exp':         5,   # EXP  (Laplacian)
    'normal_poly': 6,   # NORMAL_POLY
    'inv_dist':    7,   # INV_DIST
    'inv_sqdist':  8,   # INV_SQDIST
}

# Isotropic kernels supported by BVM
ISOTROPIC_KERNELS = frozenset({'rbf', 'exp', 'normal_poly', 'inv_dist', 'inv_sqdist'})

# Integer svm_type constants exposed for Python code
SVM_TYPE_CVM = CVM
SVM_TYPE_BVM = BVM


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper: fill one svm_node row from a contiguous 1-D double view
# ─────────────────────────────────────────────────────────────────────────────

cdef inline void _fill_row(svm_node *row,
                            const double[::1] x_row,
                            int n) noexcept nogil:
    cdef int j
    for j in range(n):
        row[j].index = j + 1
        row[j].value = x_row[j]
    row[n].index = -1
    row[n].value  = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# LibCVMModel  –  Python-accessible cdef class that owns all C++ memory
# ─────────────────────────────────────────────────────────────────────────────

cdef class LibCVMModel:
    """
    Wrapper around a trained ``svm_model*`` that manages its C++ memory.

    After ``svm_train`` returns, ``model->free_sv == 0`` which means
    ``model->SV[i]`` are *borrowed* pointers into the original training-data
    node buffer.  We keep that buffer alive in ``_node_buf`` for the lifetime
    of this object, freeing it in ``__dealloc__``.
    """

    cdef svm_model *_model
    cdef svm_node  *_node_buf   # flat: m_train × (n_features+1) svm_node items

    # Python-visible read-only integer attributes
    cdef readonly int n_sv      # total support / core vectors (model->l)
    cdef readonly int label0    # model->label[0]  (first class)
    cdef readonly int label1    # model->label[1]  (second class)
    cdef readonly int nr_class  # number of classes (2 for binary CVM/BVM)

    def __cinit__(self):
        self._model    = NULL
        self._node_buf = NULL
        self.n_sv     = 0
        self.label0   = 0
        self.label1   = 0
        self.nr_class = 0

    def __dealloc__(self):
        # free_sv == 0 → svm_destroy_model does NOT free SV node data
        # (model->SV[i] entries point into _node_buf which we own)
        if self._model != NULL:
            svm_destroy_model(self._model)
            self._model = NULL
        if self._node_buf != NULL:
            free(self._node_buf)
            self._node_buf = NULL


# ─────────────────────────────────────────────────────────────────────────────
# train()
# ─────────────────────────────────────────────────────────────────────────────

def train(
    cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] X,
    cnp.ndarray[cnp.float64_t, ndim=1]           y,
    int    svm_type,
    int    kernel_type,
    double C,
    double gamma,
    int    degree,
    double coef0,
    double cache_size,
    double eps,
    int    max_sv,
    int    sample_size,
):
    """
    Train a libCVM / libBVM model and return a :class:`LibCVMModel`.

    Parameters
    ----------
    X          : float64 array (m, n), C-contiguous
    y          : float64 array (m,), values ±1
    svm_type   : CVM (6) or BVM (9)
    kernel_type: 0=LINEAR, 1=POLY, 2=RBF, 3=SIGMOID, 5=EXP, …
    C          : regularisation parameter (default 100 for CVM/BVM)
    gamma      : kernel width (must be > 0 before calling this function)
    degree     : polynomial kernel degree
    coef0      : polynomial / sigmoid kernel offset
    cache_size : kernel cache in MB
    eps        : convergence tolerance (pass -1 for adaptive CVM/BVM default)
    max_sv     : maximum number of core/support vectors (param.num_basis)
    sample_size: probabilistic sampling size for BVM
    """
    cdef int m = X.shape[0]
    cdef int n = X.shape[1]

    # ── allocate node buffer: m rows × (n+1) nodes ────────────────────────────
    cdef svm_node *node_buf = <svm_node *>malloc(
        m * (n + 1) * sizeof(svm_node)
    )
    if node_buf == NULL:
        raise MemoryError("Cannot allocate svm_node buffer")

    # ── allocate row-pointer array ────────────────────────────────────────────
    cdef svm_node **row_ptrs = <svm_node **>malloc(m * sizeof(svm_node *))
    if row_ptrs == NULL:
        free(node_buf)
        raise MemoryError("Cannot allocate row-pointer array")

    # ── allocate label array ──────────────────────────────────────────────────
    cdef double *y_arr = <double *>malloc(m * sizeof(double))
    if y_arr == NULL:
        free(node_buf)
        free(row_ptrs)
        raise MemoryError("Cannot allocate label array")

    # ── fill data ─────────────────────────────────────────────────────────────
    cdef int i
    cdef const double[::1]    y_view = y
    cdef const double[:, ::1] X_view = X
    for i in range(m):
        row_ptrs[i] = node_buf + i * (n + 1)
        _fill_row(row_ptrs[i], X_view[i], n)
        y_arr[i] = y_view[i]

    # ── svm_problem ───────────────────────────────────────────────────────────
    cdef svm_problem prob
    memset(&prob, 0, sizeof(prob))
    prob.l     = m
    prob.u     = 0
    prob.y     = y_arr
    prob.x     = row_ptrs
    prob.graph = NULL

    # ── svm_parameter ─────────────────────────────────────────────────────────
    cdef svm_parameter param
    memset(&param, 0, sizeof(param))
    param.svm_type     = svm_type
    param.kernel_type  = kernel_type
    param.degree       = degree
    param.gamma        = gamma
    param.coef0        = coef0
    param.cache_size   = cache_size
    param.eps          = eps
    param.C            = C
    param.nr_weight    = 0
    param.weight_label = NULL
    param.weight       = NULL
    param.nu           = 0.5    # unused for CVM/BVM, but must be in (0, 1]
    param.mu           = 0.0
    param.p            = 0.1
    param.mc_type      = 0      # ONE_VS_ONE
    param.shrinking    = 1
    param.probability  = 0
    param.sample_size  = sample_size
    param.num_basis    = max_sv
    param.knn          = 0
    param.weight_type  = 0

    # ── validate ──────────────────────────────────────────────────────────────
    cdef const char *err_msg = svm_check_parameter(&prob, &param)
    if err_msg != NULL:
        free(node_buf)
        free(row_ptrs)
        free(y_arr)
        raise ValueError(f"libCVM parameter error: {err_msg.decode()}")

    # ── train ─────────────────────────────────────────────────────────────────
    cdef svm_model *model = svm_train(&prob, &param)

    # y_arr and row_ptrs are no longer needed after training:
    #   model->SV[j] holds a direct pointer into node_buf (not into row_ptrs),
    #   so row_ptrs can be freed while node_buf must stay alive.
    free(y_arr)
    free(row_ptrs)

    if model == NULL:
        free(node_buf)
        raise RuntimeError("svm_train returned NULL (internal libCVM error)")

    # ── build Python result ───────────────────────────────────────────────────
    cdef LibCVMModel result = LibCVMModel.__new__(LibCVMModel)
    result._model     = model
    result._node_buf  = node_buf
    result.n_sv       = model.l
    result.nr_class   = model.nr_class
    result.label0     = model.label[0]
    result.label1     = model.label[1]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# predict_batch()
# ─────────────────────────────────────────────────────────────────────────────

def predict_batch(
    LibCVMModel model,
    cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] X,
):
    """
    Predict class labels (±1.0) for every row of *X*.

    Returns
    -------
    out : float64 ndarray of shape (m,)
    """
    cdef int m = X.shape[0]
    cdef int n = X.shape[1]

    cdef svm_node *row = <svm_node *>malloc((n + 1) * sizeof(svm_node))
    if row == NULL:
        raise MemoryError("Cannot allocate temporary svm_node row")

    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(m, dtype=np.float64)
    cdef const double[:, ::1] Xv = X
    cdef int i

    for i in range(m):
        _fill_row(row, Xv[i], n)
        out[i] = svm_predict(model._model, row)

    free(row)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# decision_function_batch()
# ─────────────────────────────────────────────────────────────────────────────

def decision_function_batch(
    LibCVMModel model,
    cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] X,
):
    """
    Return the raw decision value for each row of *X* (binary classification).

    For binary CVM / BVM (``nr_class == 2``) there is exactly one decision
    value per sample: ``dec_values[0]``.  The sign convention used by libCVM
    is:  ``dec_values[0] > 0 → label[0]``,  ``dec_values[0] ≤ 0 → label[1]``.

    Returns
    -------
    out : float64 ndarray of shape (m,)
    """
    cdef int m = X.shape[0]
    cdef int n = X.shape[1]

    cdef svm_node *row = <svm_node *>malloc((n + 1) * sizeof(svm_node))
    if row == NULL:
        raise MemoryError("Cannot allocate temporary svm_node row")

    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(m, dtype=np.float64)
    cdef const double[:, ::1] Xv = X
    cdef double dec_val
    cdef int i

    for i in range(m):
        _fill_row(row, Xv[i], n)
        svm_predict_values(model._model, row, &dec_val)
        out[i] = dec_val

    free(row)
    return out
