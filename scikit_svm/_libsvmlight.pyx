# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Cython wrapper for SVM-Light V6.02 by Thorsten Joachims.
Exposes:  train_classification, train_regression, predict_batch
"""

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memset, strcpy

np.import_array()

# ── verbosity helper (avoids C-extern assignment ambiguity) ───────────────────
cdef extern from *:
    """
    extern long verbosity;
    static void _svml_set_verbosity(int v) { verbosity = v; }
    """
    void _svml_set_verbosity(int v)

# ── SVM-Light C declarations ──────────────────────────────────────────────────
cdef extern from "svm_common.h":
    ctypedef long FNUM
    ctypedef float FVAL

    ctypedef struct WORD:
        FNUM wnum
        FVAL weight

    ctypedef struct SVECTOR:
        WORD    *words
        double   twonorm_sq
        char    *userdefined
        long     kernel_id
        SVECTOR *next
        double   factor

    ctypedef struct DOC:
        long    docnum
        long    queryid
        double  costfactor
        long    slackid
        SVECTOR *fvec

    ctypedef struct LEARN_PARM:
        long   type
        double svm_c
        double eps
        double svm_costratio
        double transduction_posratio
        long   biased_hyperplane
        long   sharedslack
        long   svm_maxqpsize
        long   svm_newvarsinqp
        long   kernel_cache_size
        double epsilon_crit
        double epsilon_shrink
        long   svm_iter_to_shrink
        long   maxiter
        long   remove_inconsistent
        long   skip_final_opt_check
        long   compute_loo
        double rho
        long   xa_depth
        char   predfile[200]
        char   alphafile[200]
        double epsilon_const
        double epsilon_a
        double opt_precision
        long   svm_c_steps
        double svm_c_factor
        double svm_costratio_unlab
        double svm_unlabbound
        double *svm_cost
        long   totwords

    ctypedef struct KERNEL_PARM:
        long   kernel_type
        long   poly_degree
        double rbf_gamma
        double coef_lin
        double coef_const
        char   custom[50]

    ctypedef struct MODEL:
        long    sv_num
        long    at_upper_bound
        double  b
        DOC   **supvec
        double *alpha
        long   *index
        long    totwords
        long    totdoc
        KERNEL_PARM kernel_parm
        double  loo_error
        double  loo_recall
        double  loo_precision
        double  xa_error
        double  xa_recall
        double  xa_precision
        double *lin_weights
        double  maxdiff

    SVECTOR *create_svector(WORD *, char *, double)
    DOC     *create_example(long, long, long, double, SVECTOR *)
    void     free_svector(SVECTOR *)
    void     free_example(DOC *, long)
    void     free_model(MODEL *, int)
    double   classify_example(MODEL *, DOC *)
    double   classify_example_linear(MODEL *, DOC *)
    void     add_weight_vector_to_linear_model(MODEL *)
    void    *my_malloc(size_t)

cdef extern from "svm_learn.h":
    ctypedef struct KERNEL_CACHE:
        pass

    KERNEL_CACHE *kernel_cache_init(long, long)
    void          kernel_cache_cleanup(KERNEL_CACHE *)
    void svm_learn_classification(DOC **, double *, long, long,
                                   LEARN_PARM *, KERNEL_PARM *,
                                   KERNEL_CACHE *, MODEL *, double *)
    void svm_learn_regression(DOC **, double *, long, long,
                               LEARN_PARM *, KERNEL_PARM *,
                               KERNEL_CACHE **, MODEL *)

# ── public constants ──────────────────────────────────────────────────────────
LINEAR   = 0
POLY     = 1
RBF      = 2
SIGMOID  = 3

CLASSIFICATION = 1
REGRESSION     = 2

KERNEL_MAP = {'linear': 0, 'poly': 1, 'rbf': 2, 'sigmoid': 3}

# ── internal helpers ──────────────────────────────────────────────────────────
cdef DOC *_row_to_doc(
        const double *row,
        int   n_features,
        long  docnum,
        double costfactor,
        WORD  *wbuf,            # reusable temp buffer (n_features+1 slots)
):
    """Convert a C dense-row pointer to a DOC*.
    create_svector copies wbuf, so wbuf may be reused after this call.
    """
    cdef int    j, wpos = 0
    cdef double v

    for j in range(n_features):
        v = row[j]
        if v != 0.0:
            wbuf[wpos].wnum   = j + 1   # 1-based feature index
            wbuf[wpos].weight = <FVAL>v
            wpos += 1
    wbuf[wpos].wnum   = 0
    wbuf[wpos].weight = 0.0

    return create_example(docnum, 0, 0, costfactor,
                          create_svector(wbuf, "", 1.0))


cdef DOC **_build_docs(
        double[:, :] X,
        double[:]    cost,
        long totdoc,
        int  n_features,
):
    """Allocate and fill DOC** from a C-contiguous 2-D array."""
    cdef DOC  **docs = <DOC **>malloc(totdoc * sizeof(DOC *))
    if docs == NULL:
        raise MemoryError("Cannot allocate docs array")

    cdef WORD *wbuf = <WORD *>malloc((n_features + 1) * sizeof(WORD))
    if wbuf == NULL:
        free(docs)
        raise MemoryError("Cannot allocate word buffer")

    cdef long i
    for i in range(totdoc):
        docs[i] = _row_to_doc(&X[i, 0], n_features, i, cost[i], wbuf)

    free(wbuf)
    return docs


cdef void _default_learn_parm(LEARN_PARM *lp, long svm_type):
    """Fill LEARN_PARM with SVM-Light defaults (from svm_learn_main.c)."""
    memset(lp, 0, sizeof(LEARN_PARM))
    lp.type                  = svm_type
    lp.svm_c                 = 0.0       # 0 → auto
    lp.eps                   = 0.1
    lp.svm_costratio         = 1.0
    lp.transduction_posratio = -1.0
    lp.biased_hyperplane     = 1
    lp.sharedslack           = 0
    lp.svm_maxqpsize         = 10
    lp.svm_newvarsinqp       = 0
    lp.svm_iter_to_shrink    = 100
    lp.maxiter               = 100000
    lp.kernel_cache_size     = 40
    lp.remove_inconsistent   = 0
    lp.skip_final_opt_check  = 0
    lp.compute_loo           = 0
    lp.rho                   = 1.0
    lp.xa_depth              = 0
    lp.epsilon_crit          = 0.001
    lp.epsilon_a             = 1e-15
    lp.opt_precision         = 1e-21
    lp.svm_costratio_unlab   = 1.0
    lp.svm_unlabbound        = 1e-5
    strcpy(lp.predfile, "trans_predictions")
    strcpy(lp.alphafile, "")


cdef void _fill_kernel_parm(KERNEL_PARM *kp, long kernel_type, long poly_degree,
                              double rbf_gamma, double coef_lin, double coef_const):
    memset(kp, 0, sizeof(KERNEL_PARM))
    kp.kernel_type = kernel_type
    kp.poly_degree = poly_degree
    kp.rbf_gamma   = rbf_gamma
    kp.coef_lin    = coef_lin
    kp.coef_const  = coef_const
    strcpy(kp.custom, "empty")


# ── model wrapper ─────────────────────────────────────────────────────────────
cdef class LibSVMLightModel:
    """Owns an SVM-Light MODEL* and its training DOC** array.

    Memory layout after training:
      - model->supvec[1..sv_num-1]  point into docs[0..totdoc-1]
      - free_model(model, 0) frees supvec[], alpha, index, lin_weights, model
      - free_example(docs[i], 1) frees each DOC + SVECTOR + WORD copy
    """
    cdef MODEL *model
    cdef DOC  **docs
    cdef long   totdoc
    cdef double *target_buf

    def __dealloc__(self):
        cdef long i
        if self.model != NULL:
            free_model(self.model, 0)
            self.model = NULL
        if self.docs != NULL:
            for i in range(self.totdoc):
                if self.docs[i] != NULL:
                    free_example(self.docs[i], 1)
            free(self.docs)
            self.docs = NULL
        if self.target_buf != NULL:
            free(self.target_buf)
            self.target_buf = NULL

    # ── read-only properties ──────────────────────────────────────────────
    @property
    def sv_num(self):
        """Total sv_num field (actual SVs = sv_num - 1)."""
        return self.model.sv_num

    @property
    def b(self):
        """Bias threshold: decision = sum(alpha_i * K) - b."""
        return self.model.b

    @property
    def totwords(self):
        return self.model.totwords

    @property
    def kernel_type(self):
        return self.model.kernel_parm.kernel_type

    # ── data extraction ───────────────────────────────────────────────────
    def get_support_vectors(self, int n_features):
        """Return (n_sv, n_features) float64 dense array."""
        cdef long n_sv = self.model.sv_num - 1
        cdef np.ndarray[np.float64_t, ndim=2] out = np.zeros(
            (n_sv, n_features), dtype=np.float64)
        cdef WORD *w
        cdef long  i, fi
        for i in range(n_sv):
            w = self.model.supvec[i + 1].fvec.words
            while w.wnum != 0:
                fi = w.wnum - 1
                if 0 <= fi < n_features:
                    out[i, fi] = w.weight
                w += 1
        return out

    def get_alphas(self):
        """Return (n_sv,) float64 — alpha[1..sv_num-1] (signed: alpha_i * y_i)."""
        cdef long n_sv = self.model.sv_num - 1
        cdef np.ndarray[np.float64_t, ndim=1] out = np.empty(n_sv, dtype=np.float64)
        cdef long i
        for i in range(n_sv):
            out[i] = self.model.alpha[i + 1]
        return out

    def get_sv_docnums(self):
        """Return (n_sv,) int32 — training-set row index for each SV."""
        cdef long n_sv = self.model.sv_num - 1
        cdef np.ndarray[np.int32_t, ndim=1] out = np.empty(n_sv, dtype=np.int32)
        cdef long i
        for i in range(n_sv):
            out[i] = <int>self.model.supvec[i + 1].docnum
        return out


# ── training functions ────────────────────────────────────────────────────────
def train_classification(
        np.ndarray[np.float64_t, ndim=2] X not None,
        np.ndarray[np.float64_t, ndim=1] y not None,
        long   kernel_type,
        double C,
        double rbf_gamma,
        long   poly_degree,
        double coef_lin,
        double coef_const,
        double epsilon_crit,
        long   kernel_cache_size,
        long   svm_maxqpsize,
        long   maxiter,
        long   svm_iter_to_shrink,
        long   biased_hyperplane,
        double svm_costratio,
        bint   verbose,
):
    """Train an SVM-Light binary classifier.  y must be +1 / -1."""
    _svml_set_verbosity(1 if verbose else 0)

    cdef long totdoc     = X.shape[0]
    cdef int  n_features = X.shape[1]

    X = np.ascontiguousarray(X, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cost = np.ones(totdoc, dtype=np.float64)
    cdef DOC **docs = _build_docs(X, cost, totdoc, n_features)

    cdef double *target = <double *>malloc(totdoc * sizeof(double))
    if target == NULL:
        raise MemoryError()
    cdef long i
    for i in range(totdoc):
        target[i] = y[i]

    cdef LEARN_PARM  lp
    cdef KERNEL_PARM kp
    _default_learn_parm(&lp, CLASSIFICATION)
    lp.svm_c             = C
    lp.epsilon_crit      = epsilon_crit
    lp.kernel_cache_size = kernel_cache_size
    lp.svm_maxqpsize     = svm_maxqpsize
    lp.maxiter           = maxiter
    lp.biased_hyperplane = biased_hyperplane
    lp.svm_costratio     = svm_costratio
    lp.svm_iter_to_shrink = (svm_iter_to_shrink if svm_iter_to_shrink > 0
                              else (2 if kernel_type == LINEAR else 100))
    _fill_kernel_parm(&kp, kernel_type, poly_degree, rbf_gamma, coef_lin, coef_const)

    cdef MODEL *model = <MODEL *>my_malloc(sizeof(MODEL))
    memset(model, 0, sizeof(MODEL))

    cdef KERNEL_CACHE *kcache = NULL
    if kernel_type != LINEAR:
        kcache = kernel_cache_init(totdoc, kernel_cache_size)

    svm_learn_classification(docs, target, totdoc, n_features,
                              &lp, &kp, kcache, model, NULL)

    if kcache != NULL:
        kernel_cache_cleanup(kcache)

    if kernel_type == LINEAR:
        add_weight_vector_to_linear_model(model)

    cdef LibSVMLightModel obj = LibSVMLightModel.__new__(LibSVMLightModel)
    obj.model      = model
    obj.docs       = docs
    obj.totdoc     = totdoc
    obj.target_buf = target
    return obj


def train_regression(
        np.ndarray[np.float64_t, ndim=2] X not None,
        np.ndarray[np.float64_t, ndim=1] y not None,
        long   kernel_type,
        double C,
        double rbf_gamma,
        long   poly_degree,
        double coef_lin,
        double coef_const,
        double epsilon,
        double epsilon_crit,
        long   kernel_cache_size,
        long   svm_maxqpsize,
        long   maxiter,
        long   svm_iter_to_shrink,
        bint   verbose,
):
    """Train an SVM-Light epsilon-SVR regression model."""
    _svml_set_verbosity(1 if verbose else 0)

    cdef long totdoc     = X.shape[0]
    cdef int  n_features = X.shape[1]

    X = np.ascontiguousarray(X, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] cost = np.ones(totdoc, dtype=np.float64)
    cdef DOC **docs = _build_docs(X, cost, totdoc, n_features)

    cdef double *target = <double *>malloc(totdoc * sizeof(double))
    if target == NULL:
        raise MemoryError()
    cdef long i
    for i in range(totdoc):
        target[i] = y[i]

    cdef LEARN_PARM  lp
    cdef KERNEL_PARM kp
    _default_learn_parm(&lp, REGRESSION)
    lp.svm_c             = C
    lp.eps               = epsilon
    lp.epsilon_crit      = epsilon_crit
    lp.kernel_cache_size = kernel_cache_size
    lp.svm_maxqpsize     = svm_maxqpsize
    lp.maxiter           = maxiter
    lp.svm_iter_to_shrink = (svm_iter_to_shrink if svm_iter_to_shrink > 0
                              else (2 if kernel_type == LINEAR else 100))
    _fill_kernel_parm(&kp, kernel_type, poly_degree, rbf_gamma, coef_lin, coef_const)

    cdef MODEL *model = <MODEL *>my_malloc(sizeof(MODEL))
    memset(model, 0, sizeof(MODEL))

    # svm_learn_regression takes KERNEL_CACHE** (may reallocate)
    cdef KERNEL_CACHE *kcache = NULL
    cdef KERNEL_CACHE **kcache_pp = &kcache
    if kernel_type != LINEAR:
        kcache = kernel_cache_init(totdoc, kernel_cache_size)

    svm_learn_regression(docs, target, totdoc, n_features,
                          &lp, &kp, kcache_pp, model)

    # Use the (possibly updated) pointer from kcache_pp
    if kcache_pp[0] != NULL:
        kernel_cache_cleanup(kcache_pp[0])

    if kernel_type == LINEAR:
        add_weight_vector_to_linear_model(model)

    cdef LibSVMLightModel obj = LibSVMLightModel.__new__(LibSVMLightModel)
    obj.model      = model
    obj.docs       = docs
    obj.totdoc     = totdoc
    obj.target_buf = target
    return obj


def predict_batch(
        LibSVMLightModel mdl not None,
        np.ndarray[np.float64_t, ndim=2] X not None,
):
    """Evaluate the SVM-Light decision function for all rows of X.

    Returns float64 array of shape (n_samples,).
    Positive values → predicted class +1 (classification) or regression output.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    cdef long m          = X.shape[0]
    cdef int  n_features = X.shape[1]
    cdef np.ndarray[np.float64_t, ndim=1] out = np.empty(m, dtype=np.float64)

    cdef WORD  *wbuf = <WORD *>malloc((n_features + 1) * sizeof(WORD))
    if wbuf == NULL:
        raise MemoryError()

    cdef MODEL *model  = mdl.model
    cdef bint   linear = (model.kernel_parm.kernel_type == LINEAR
                          and model.lin_weights != NULL)
    cdef DOC   *doc
    cdef long   i

    for i in range(m):
        doc = _row_to_doc(&X[i, 0], n_features, -1, 1.0, wbuf)
        if linear:
            out[i] = classify_example_linear(model, doc)
        else:
            out[i] = classify_example(model, doc)
        free_example(doc, 1)

    free(wbuf)
    return out
