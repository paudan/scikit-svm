# cython: language_level=3
"""
Cython wrapper around the BSVM 2.09 C++ library.

Provides thin Python bindings for svm_train / svm_predict and exposes a
LibBSVMModel cdef class that owns all allocated C++ memory.
"""

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.string cimport memset

cnp.import_array()

# ─────────────────────────────────────────────────────────────────────────────
# C declarations from svm.h  (only what the wrapper needs)
# ─────────────────────────────────────────────────────────────────────────────

cdef extern from "svm.h":

    # sparse feature-vector node: index=-1 terminates a row ───────────────────
    struct svm_node:
        int    index    # 1-based feature index; -1 = end-of-row sentinel
        double value

    # training problem ─────────────────────────────────────────────────────────
    struct svm_problem:
        int             l       # number of samples
        int             n       # number of features
        double         *y       # label array [l]
        svm_node      **x       # sparse data rows [l]

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
        double  p
        int     shrinking
        int     qpsize
        double  Cbegin
        double  Cstep

    # trained model ────────────────────────────────────────────────────────────
    struct svm_model:
        svm_parameter  param
        int            nr_class
        int            l
        svm_node     **SV
        double       **sv_coef
        int           *label
        int           *nSV
        int            free_sv

    # svm_type constants (anonymous C enum) ────────────────────────────────────
    enum:
        C_SVC        # 0
        KBB          # 1
        SPOC         # 2
        EPSILON_SVR  # 3
        SPOC_L2      # 4

    # kernel_type constants (anonymous C enum) ─────────────────────────────────
    enum:
        LINEAR       # 0
        POLY         # 1
        RBF          # 2
        SIGMOID      # 3
        PRECOMPUTED  # 4

    # public C API ─────────────────────────────────────────────────────────────
    svm_model  *svm_train(const svm_problem *prob,
                          const svm_parameter *param)
    double      svm_predict(const svm_model *model, const svm_node *x)
    void        svm_free_and_destroy_model(svm_model **model_ptr_ptr)
    const char *svm_check_parameter(const svm_problem *prob,
                                    const svm_parameter *param)


# ─────────────────────────────────────────────────────────────────────────────
# Public Python-level constants
# ─────────────────────────────────────────────────────────────────────────────

# Mapping from sklearn-style kernel name to BSVM kernel_type integer
KERNEL_MAP = {
    'linear':      0,   # LINEAR
    'poly':        1,   # POLY
    'rbf':         2,   # RBF
    'sigmoid':     3,   # SIGMOID
    'precomputed': 4,   # PRECOMPUTED
}

# Mapping from string svm_type name to integer constant
SVM_TYPE_MAP = {
    'c_svc':       0,   # C_SVC
    'kbb':         1,   # KBB
    'spoc':        2,   # SPOC
    'epsilon_svr': 3,   # EPSILON_SVR
    'spoc_l2':     4,   # SPOC_L2
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper: fill one svm_node row from a contiguous 1-D double view
# ─────────────────────────────────────────────────────────────────────────────

cdef inline void _fill_row(svm_node *row,
                            const double[::1] x_row,
                            int n) noexcept nogil:
    cdef int j
    for j in range(n):
        row[j].index = j + 1       # 1-based feature indices
        row[j].value = x_row[j]
    row[n].index = -1              # end-of-row sentinel
    row[n].value  = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# LibBSVMModel  –  Python-accessible cdef class that owns all C++ memory
# ─────────────────────────────────────────────────────────────────────────────

cdef class LibBSVMModel:
    """
    Wrapper around a trained ``svm_model*`` that manages its C++ memory.

    After ``svm_train`` returns, ``model->free_sv == 0`` which means
    ``model->SV[i]`` are *borrowed* pointers into the original training-data
    node buffer.  We keep that buffer alive in ``_node_buf`` for the lifetime
    of this object, freeing it in ``__dealloc__``.

    The SPOC and SPOC_L2 svm types allocate ``nr_class`` rows in
    ``sv_coef``, but ``svm_free_model_content`` only frees ``nr_class-1``
    rows — leaking the last row.  ``__dealloc__`` patches this by freeing
    the extra row before delegating to ``svm_free_and_destroy_model``.
    """

    cdef svm_model *_model
    cdef svm_node  *_node_buf   # flat: m_train × (n_features+1) svm_node items

    # Python-visible read-only integer attributes
    cdef readonly int nr_class      # number of classes (model->nr_class)
    cdef readonly int n_sv          # total support vectors (model->l)
    cdef readonly int svm_type_int  # svm_type integer from param
    cdef readonly int n_coef_rows   # number of rows in sv_coef

    def __cinit__(self):
        self._model    = NULL
        self._node_buf = NULL
        self.nr_class  = 0
        self.n_sv      = 0
        self.svm_type_int = 0
        self.n_coef_rows  = 0

    def __dealloc__(self):
        # SPOC / SPOC_L2 memory-leak fix:
        #   svm_free_model_content only iterates sv_coef[0..nr_class-2],
        #   missing sv_coef[nr_class-1].  Free it manually and set to NULL
        #   before delegating to svm_free_and_destroy_model.
        if (self._model != NULL
                and self._model.sv_coef != NULL
                and self.nr_class > 1
                and (self.svm_type_int == SPOC
                     or self.svm_type_int == SPOC_L2)):
            free(self._model.sv_coef[self.nr_class - 1])
            self._model.sv_coef[self.nr_class - 1] = NULL

        # free_sv == 0 → svm_free_and_destroy_model does NOT free SV node data
        # (model->SV[i] entries point into _node_buf which we own)
        if self._model != NULL:
            svm_free_and_destroy_model(&self._model)
            self._model = NULL
        if self._node_buf != NULL:
            free(self._node_buf)
            self._node_buf = NULL

    # ──────────────────────────────────────────────────────────────────────────

    def get_labels(self):
        """
        Return the class labels stored in the model.

        Returns
        -------
        labels : int32 ndarray of shape (nr_class,)
        """
        cdef int k = self.nr_class
        cdef cnp.ndarray[cnp.int32_t, ndim=1] out = np.empty(k, dtype=np.int32)
        cdef int i
        for i in range(k):
            out[i] = self._model.label[i]
        return out

    def get_nSV(self):
        """
        Return the per-class support-vector counts.

        Returns
        -------
        nSV : int32 ndarray of shape (nr_class,)
        """
        cdef int k = self.nr_class
        cdef cnp.ndarray[cnp.int32_t, ndim=1] out = np.empty(k, dtype=np.int32)
        cdef int i
        for i in range(k):
            out[i] = self._model.nSV[i]
        return out

    def get_support_vectors(self, int n_features):
        """
        Convert sparse 1-based support vectors to a dense float64 matrix.

        Parameters
        ----------
        n_features : int
            Number of features (columns) in the output array.

        Returns
        -------
        SV : float64 ndarray of shape (n_sv, n_features)
        """
        cdef int l = self.n_sv
        cdef cnp.ndarray[cnp.float64_t, ndim=2] out = np.zeros(
            (l, n_features), dtype=np.float64
        )
        cdef svm_node *node
        cdef int i
        for i in range(l):
            node = self._model.SV[i]
            while node.index != -1:
                # convert from 1-based to 0-based index
                out[i, node.index - 1] = node.value
                node += 1
        return out

    def get_sv_coef(self):
        """
        Return the support-vector coefficient matrix.

        Returns
        -------
        sv_coef : float64 ndarray of shape (n_coef_rows, n_sv)
        """
        cdef int rows = self.n_coef_rows
        cdef int l    = self.n_sv
        cdef cnp.ndarray[cnp.float64_t, ndim=2] out = np.empty(
            (rows, l), dtype=np.float64
        )
        cdef int i, j
        for i in range(rows):
            for j in range(l):
                out[i, j] = self._model.sv_coef[i][j]
        return out


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
    int    shrinking,
    int    qpsize,
    double Cbegin,
    double Cstep,
    double p,
    int    nr_weight,
    weight_label,
    weight,
):
    """
    Train a BSVM model and return a :class:`LibBSVMModel`.

    Parameters
    ----------
    X            : float64 array (m, n), C-contiguous
    y            : float64 array (m,)
    svm_type     : integer svm type (C_SVC=0, KBB=1, SPOC=2, EPSILON_SVR=3, SPOC_L2=4)
    kernel_type  : 0=LINEAR, 1=POLY, 2=RBF, 3=SIGMOID, 4=PRECOMPUTED
    C            : regularisation parameter
    gamma        : kernel coefficient (must be > 0 before calling this function)
    degree       : polynomial kernel degree
    coef0        : polynomial / sigmoid kernel offset
    cache_size   : kernel cache in MB
    eps          : convergence tolerance
    shrinking    : 1 to use shrinking heuristics, 0 otherwise
    qpsize       : QP sub-problem size
    Cbegin       : initial C for linear kernel warm-start
    Cstep        : C step multiplier for linear kernel warm-start
    p            : epsilon-tube half-width for SVR
    nr_weight    : number of class-weight overrides (0 → none)
    weight_label : int32 array of class labels for weight overrides (or None)
    weight       : float64 array of class weights (or None)
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

    # ── weight arrays (keep numpy arrays alive until svm_train returns) ────────
    cdef cnp.ndarray[cnp.int32_t,   ndim=1] wl_arr
    cdef cnp.ndarray[cnp.float64_t, ndim=1] w_arr
    cdef int    *wl_ptr = NULL
    cdef double *w_ptr  = NULL

    if nr_weight > 0 and weight_label is not None and weight is not None:
        wl_arr = np.asarray(weight_label, dtype=np.int32)
        w_arr  = np.asarray(weight,       dtype=np.float64)
        wl_ptr = <int *>    wl_arr.data
        w_ptr  = <double *> w_arr.data

    # ── svm_problem ───────────────────────────────────────────────────────────
    cdef svm_problem prob
    memset(&prob, 0, sizeof(prob))
    prob.l = m
    prob.n = n
    prob.y = y_arr
    prob.x = row_ptrs

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
    param.nr_weight    = nr_weight
    param.weight_label = wl_ptr
    param.weight       = w_ptr
    param.p            = p
    param.shrinking    = shrinking
    param.qpsize       = qpsize
    param.Cbegin       = Cbegin
    param.Cstep        = Cstep

    # ── validate ──────────────────────────────────────────────────────────────
    cdef const char *err_msg = svm_check_parameter(&prob, &param)
    if err_msg != NULL:
        free(node_buf)
        free(row_ptrs)
        free(y_arr)
        raise ValueError(f"libBSVM parameter error: {err_msg.decode()}")

    # ── train ─────────────────────────────────────────────────────────────────
    cdef svm_model *model = svm_train(&prob, &param)

    # y_arr and row_ptrs are no longer needed after training:
    #   model->SV[j] holds a direct pointer into node_buf (not into row_ptrs),
    #   so row_ptrs can be freed while node_buf must stay alive.
    free(y_arr)
    free(row_ptrs)

    if model == NULL:
        free(node_buf)
        raise RuntimeError("svm_train returned NULL (internal libBSVM error)")

    # ── determine n_coef_rows ─────────────────────────────────────────────────
    # EPSILON_SVR  → 1 row
    # SPOC / SPOC_L2 → nr_class rows (library allocates nr_class rows)
    # C_SVC / KBB  → nr_class-1 rows (but at least 1 for degenerate nr_class==1)
    cdef int nr_class   = model.nr_class
    cdef int n_coef_rows
    if svm_type == EPSILON_SVR:
        n_coef_rows = 1
    elif svm_type == SPOC or svm_type == SPOC_L2:
        n_coef_rows = nr_class
    else:
        # C_SVC or KBB
        n_coef_rows = nr_class - 1 if nr_class > 1 else 1

    # ── build Python result ───────────────────────────────────────────────────
    cdef LibBSVMModel result = LibBSVMModel.__new__(LibBSVMModel)
    result._model        = model
    result._node_buf     = node_buf
    result.nr_class      = nr_class
    result.n_sv          = model.l
    result.svm_type_int  = svm_type
    result.n_coef_rows   = n_coef_rows
    return result


# ─────────────────────────────────────────────────────────────────────────────
# predict_batch()
# ─────────────────────────────────────────────────────────────────────────────

def predict_batch(
    LibBSVMModel model,
    cnp.ndarray[cnp.float64_t, ndim=2, mode='c'] X,
):
    """
    Predict labels / regression values for every row of *X*.

    Parameters
    ----------
    model : LibBSVMModel
        Trained model returned by :func:`train`.
    X     : float64 ndarray of shape (m, n), C-contiguous

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
