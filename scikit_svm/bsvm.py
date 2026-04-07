"""
BSVM  –  Bound-Constrained Support Vector Machine classifiers and regressor.

Cython-backed wrappers around the BSVM 2.09 C++ library by
Mangasarian, Musicant and others (University of Wisconsin-Madison).

Four svm_type values are supported for classification:

    C_SVC    – standard C-SVM (OVO multi-class)
    KBB      – Kernel Bound-constrained SVM
    SPOC     – Support-vector-based Pattern-classifier with Optimum Class
               (native multi-class, OVA-style dual)
    SPOC_L2  – SPOC with L2 loss

One svm_type value is supported for regression:

    EPSILON_SVR – epsilon-insensitive SVR

References
----------
Mangasarian & Musicant. "Successive Overrelaxation for Support Vector
Machines." IEEE Transactions on Neural Networks, 1999.
"""

import contextlib
import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._utils import _suppress_c_stdout


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _import_libbsvm():
    try:
        from . import _libbsvm
        return _libbsvm
    except ImportError as exc:
        raise ImportError(
            "scikit_svm._libbsvm is not built. "
            "Run 'pip install -e .' (or python setup.py build_ext --inplace) "
            "from the project root to compile the Cython extension."
        ) from exc


@contextlib.contextmanager
def _noop_ctx():
    yield


def _resolve_gamma(gamma, X):
    """
    Resolve the ``gamma`` parameter to a concrete positive float.

    Parameters
    ----------
    gamma : None, 'auto', 'scale', or float
        If ``None`` or ``'auto'``: ``1 / n_features``.
        If ``'scale'``:            ``1 / (n_features * X.var())``.
        Otherwise treated as a float value.
    X : ndarray of shape (m, n)
        Training data (used only when ``gamma == 'scale'``).

    Returns
    -------
    float
    """
    n = X.shape[1]
    if gamma is None or gamma == 'auto':
        return 1.0 / n
    if gamma == 'scale':
        var = X.var()
        return 1.0 / (n * var) if var != 0.0 else 1.0 / n
    return float(gamma)


def _kernel_params(kernel, gamma, degree, coef0):
    """
    Build the ``**kwds`` dict accepted by :func:`sklearn.metrics.pairwise.pairwise_kernels`.

    Parameters
    ----------
    kernel : str
    gamma  : float  (already resolved)
    degree : int
    coef0  : float

    Returns
    -------
    dict
    """
    params = {}
    k = kernel.lower()
    if k in ('rbf', 'poly', 'sigmoid', 'laplacian'):
        params['gamma'] = gamma
    if k in ('poly', 'sigmoid'):
        params['coef0'] = coef0
    if k == 'poly':
        params['degree'] = degree
    return params


# ─────────────────────────────────────────────────────────────────────────────
# BSVMClassifier
# ─────────────────────────────────────────────────────────────────────────────

class BSVMClassifier(BaseEstimator, ClassifierMixin):
    """
    Bound-Constrained SVM classifier.

    Wraps the BSVM 2.09 C++ library.  Supports C_SVC, KBB, SPOC and
    SPOC_L2 svm types.  Labels are encoded internally via
    :class:`~sklearn.preprocessing.LabelEncoder`, so arbitrary integer or
    string labels are accepted.

    Parameters
    ----------
    svm_type : str, default='c_svc'
        One of ``'c_svc'``, ``'kbb'``, ``'spoc'``, ``'spoc_l2'``.
    kernel : str, default='rbf'
        Kernel type: ``'linear'``, ``'poly'``, ``'rbf'``, ``'sigmoid'``,
        ``'precomputed'``.
    C : float, default=1.0
        Regularisation parameter.  Must be strictly positive.
    gamma : float, 'auto', 'scale', or None, default=None
        Kernel coefficient.  ``None`` and ``'auto'`` both resolve to
        ``1 / n_features``.  ``'scale'`` uses
        ``1 / (n_features * X.var())``.
    degree : int, default=3
        Degree for polynomial kernel.
    coef0 : float, default=0.0
        Constant term for polynomial and sigmoid kernels.
    cache_size : float, default=100.0
        Kernel cache size in MB.
    tol : float, default=1e-3
        Convergence tolerance (passed as ``eps`` to the library).
    shrinking : bool, default=True
        Whether to use shrinking heuristics.
    qpsize : int, default=10
        QP sub-problem size.
    Cbegin : float, default=1.0
        Initial C for linear-kernel warm-start.
    Cstep : float, default=2.0
        C step multiplier for linear-kernel warm-start.
    class_weight : dict or None, default=None
        Per-class weights in the form ``{label: weight}``.  The labels
        must exist in the training set.  ``None`` means equal weights.
    verbose : bool, default=True
        If ``True``, print BSVM's training progress to stdout.

    Attributes
    ----------
    model_ : LibBSVMModel
        Internal Cython object wrapping the trained ``svm_model*``.
    support_vectors_ : ndarray of shape (n_sv, n_features)
        Support vectors extracted from the model.
    dual_coef_ : ndarray of shape (n_coef_rows, n_sv)
        Coefficients of support vectors in decision functions.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit`` (original, before encoding).
    n_sv_ : int
        Total number of support vectors.
    n_sv_per_class_ : int32 ndarray of shape (n_classes,)
        Number of support vectors per class.
    n_features_in_ : int
        Number of features seen during ``fit``.
    time_ : float
        Wall-clock training time in seconds.
    label_encoder_ : LabelEncoder
        Fitted label encoder used to convert original labels ↔ BSVM integers.
    """

    def __init__(
        self,
        svm_type='c_svc',
        kernel='rbf',
        C=1.0,
        gamma=None,
        degree=3,
        coef0=0.0,
        cache_size=100.0,
        tol=1e-3,
        shrinking=True,
        qpsize=10,
        Cbegin=1.0,
        Cstep=2.0,
        class_weight=None,
        verbose=True,
    ):
        self.svm_type     = svm_type
        self.kernel       = kernel
        self.C            = C
        self.gamma        = gamma
        self.degree       = degree
        self.coef0        = coef0
        self.cache_size   = cache_size
        self.tol          = tol
        self.shrinking    = shrinking
        self.qpsize       = qpsize
        self.Cbegin       = Cbegin
        self.Cstep        = Cstep
        self.class_weight = class_weight
        self.verbose      = verbose

    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Train the BSVM classifier.

        Parameters
        ----------
        X : array-like of shape (m, n)
            Training data.  For ``kernel='precomputed'``, pass the
            (m, m) kernel matrix instead.
        y : array-like of shape (m,)
            Class labels.  Any hashable type is accepted.

        Returns
        -------
        self : BSVMClassifier
        """
        _libbsvm = _import_libbsvm()

        X, y = check_X_y(X, y)
        X = np.ascontiguousarray(X, dtype=np.float64)

        # ── encode labels to consecutive integers 0, 1, 2, … ─────────────────
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y).astype(np.float64)
        self.classes_ = self.label_encoder_.classes_

        # ── resolve svm_type ──────────────────────────────────────────────────
        svm_type_lower = self.svm_type.lower()
        if svm_type_lower not in _libbsvm.SVM_TYPE_MAP:
            raise ValueError(
                f"Unknown svm_type '{self.svm_type}'. "
                f"Choose from: {sorted(_libbsvm.SVM_TYPE_MAP)}."
            )
        svm_type_int = _libbsvm.SVM_TYPE_MAP[svm_type_lower]

        # ── resolve kernel ────────────────────────────────────────────────────
        kernel_lower = self.kernel.lower()
        if kernel_lower not in _libbsvm.KERNEL_MAP:
            raise ValueError(
                f"Unknown kernel '{self.kernel}'. "
                f"Choose from: {sorted(_libbsvm.KERNEL_MAP)}."
            )
        kernel_type = _libbsvm.KERNEL_MAP[kernel_lower]

        m, n = X.shape

        # ── resolve gamma ─────────────────────────────────────────────────────
        self._gamma = _resolve_gamma(self.gamma, X)

        # ── build weight arrays ───────────────────────────────────────────────
        nr_weight    = 0
        weight_label = None
        weight_arr   = None
        if self.class_weight is not None:
            cw = self.class_weight
            # Encode the original labels to BSVM integer labels
            orig_labels = np.array(list(cw.keys()))
            enc_labels  = self.label_encoder_.transform(orig_labels).astype(np.int32)
            weights     = np.array([cw[k] for k in orig_labels], dtype=np.float64)
            nr_weight    = len(enc_labels)
            weight_label = enc_labels
            weight_arr   = weights

        t0 = time.perf_counter()

        ctx = _suppress_c_stdout() if not self.verbose else _noop_ctx()
        with ctx:
            self.model_ = _libbsvm.train(
                X,
                y_enc,
                svm_type     = svm_type_int,
                kernel_type  = kernel_type,
                C            = float(self.C),
                gamma        = self._gamma,
                degree       = int(self.degree),
                coef0        = float(self.coef0),
                cache_size   = float(self.cache_size),
                eps          = float(self.tol),
                shrinking    = int(self.shrinking),
                qpsize       = int(self.qpsize),
                Cbegin       = float(self.Cbegin),
                Cstep        = float(self.Cstep),
                p            = 0.1,   # not used for classification
                nr_weight    = nr_weight,
                weight_label = weight_label,
                weight       = weight_arr,
            )

        self.time_           = time.perf_counter() - t0
        self.n_sv_           = self.model_.n_sv
        self.n_features_in_  = n
        self._labels         = self.model_.get_labels()
        self._nSV            = self.model_.get_nSV()
        self.n_sv_per_class_ = self._nSV
        self.support_vectors_ = self.model_.get_support_vectors(n)
        self.dual_coef_       = self.model_.get_sv_coef()

        if self.verbose:
            print(
                f"BSVM training: {self.time_:.3f}s, "
                f"{self.n_sv_} support vectors."
            )

        return self

    # ──────────────────────────────────────────────────────────────────────────

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (m, n)

        Returns
        -------
        y_pred : ndarray of shape (m,)
            Predicted labels in the original label space.
        """
        check_is_fitted(self)
        _libbsvm = _import_libbsvm()
        X = check_array(X)
        X = np.ascontiguousarray(X, dtype=np.float64)

        raw = _libbsvm.predict_batch(self.model_, X)
        # raw contains encoded integer labels as float64 → cast to int32
        enc = raw.astype(np.int32)
        return self.label_encoder_.inverse_transform(enc)

    # ──────────────────────────────────────────────────────────────────────────

    def decision_function(self, X):
        """
        Compute decision function scores.

        The exact output shape and semantics depend on the svm_type:

        - **C_SVC** binary: 1-D array of shape (m,).
        - **C_SVC** multi-class (OVO): 2-D array of shape (m, K*(K-1)/2)
          following the same ordering as libSVM.
        - **KBB** binary: 1-D array of shape (m,).
        - **KBB** multi-class: 2-D array of shape (m, K) with per-class scores.
        - **SPOC / SPOC_L2**: 2-D array of shape (m, K).

        ``kernel='precomputed'`` is not supported here; use ``predict``
        directly.

        Parameters
        ----------
        X : array-like of shape (m, n)

        Returns
        -------
        scores : ndarray of shape (m,) or (m, n_decision_values)
        """
        check_is_fitted(self)

        if self.kernel.lower() == 'precomputed':
            raise ValueError(
                "decision_function is not supported for kernel='precomputed'. "
                "Use predict() directly."
            )

        X = check_array(X)
        X = np.ascontiguousarray(X, dtype=np.float64)

        svm_type_lower = self.svm_type.lower()
        nr_class       = self.model_.nr_class
        sv             = self.support_vectors_
        coef           = self.dual_coef_     # shape (n_coef_rows, n_sv)
        kp             = _kernel_params(self.kernel, self._gamma,
                                        self.degree, self.coef0)

        # ── compute kernel matrix K: (m, n_sv) ────────────────────────────────
        K = pairwise_kernels(X, sv, metric=self.kernel, **kp)

        # ── augmented kernel K+1 (not for SPOC/SPOC_L2) ───────────────────────
        use_augmented = svm_type_lower not in ('spoc', 'spoc_l2')
        K_aug = (K + 1.0) if use_augmented else K

        # ── SPOC / SPOC_L2: return K @ coef.T, shape (m, nr_class) ───────────
        if svm_type_lower in ('spoc', 'spoc_l2'):
            # coef shape: (nr_class, n_sv) → K @ coef.T = (m, nr_class)
            return K_aug @ coef.T

        # ── SVR / binary C_SVC / binary KBB ───────────────────────────────────
        if nr_class == 2:
            # coef[0] is the single row (n_sv,)
            return K_aug @ coef[0]

        # ── KBB multi-class ───────────────────────────────────────────────────
        # Mirrors the C++ loop in bsvm.cpp svm_predict (KBB branch):
        #   A[j] = sum_{k=0}^{nr_class-2} sv_coef[k][j]
        #   For class i, SV j:
        #     for k < i:        f[k]   -= coef[k][j] * kv
        #     f[i] += A[j] * kv
        #     for k in [i, m):  f[k+1] -= coef[k][j] * kv   (m = nr_class-1)
        if svm_type_lower == 'kbb':
            m_test = X.shape[0]
            f   = np.zeros((m_test, nr_class), dtype=np.float64)
            A   = coef.sum(axis=0)      # shape (n_sv,): A[j] = sum_k coef[k][j]
            nSV = self._nSV
            m_c = nr_class - 1          # number of coef rows
            si  = 0
            for i in range(nr_class):
                ci       = int(nSV[i])
                sv_sl    = slice(si, si + ci)
                K_block  = K_aug[:, sv_sl]  # (m_test, ci)
                # f[i] += A[sv_sl] * kv
                f[:, i] += K_block @ A[sv_sl]
                # for k < i: f[k] -= coef[k][sv_sl] * kv
                for k in range(i):
                    f[:, k] -= K_block @ coef[k, sv_sl]
                # for k in [i, m_c): f[k+1] -= coef[k][sv_sl] * kv
                for k in range(i, m_c):
                    f[:, k + 1] -= K_block @ coef[k, sv_sl]
                si += ci
            return f

        # ── C_SVC multi-class OVO ─────────────────────────────────────────────
        # For each pair (i, j) with i < j:
        #   decision_value = sum_{sv in class i} coef[j-1][sv] * K_aug(x, sv)
        #                  + sum_{sv in class j} coef[i][sv]   * K_aug(x, sv)
        m_test    = X.shape[0]
        n_pairs   = nr_class * (nr_class - 1) // 2
        scores    = np.zeros((m_test, n_pairs), dtype=np.float64)
        nSV       = self._nSV   # shape (nr_class,)
        # precompute cumulative start indices
        sv_start  = np.zeros(nr_class, dtype=np.int32)
        for i in range(1, nr_class):
            sv_start[i] = sv_start[i - 1] + int(nSV[i - 1])

        pair = 0
        for i in range(nr_class):
            for j in range(i + 1, nr_class):
                si = int(sv_start[i])
                ci = int(nSV[i])
                sj = int(sv_start[j])
                cj = int(nSV[j])
                # coef[j-1] for class-i SVs, coef[i] for class-j SVs
                scores[:, pair] = (
                    K_aug[:, si:si+ci] @ coef[j - 1, si:si+ci]
                    + K_aug[:, sj:sj+cj] @ coef[i,     sj:sj+cj]
                )
                pair += 1

        return scores


# ─────────────────────────────────────────────────────────────────────────────
# BSVMRegressor
# ─────────────────────────────────────────────────────────────────────────────

class BSVMRegressor(BaseEstimator, RegressorMixin):
    """
    Bound-Constrained SVM regressor (epsilon-insensitive SVR).

    Wraps the BSVM 2.09 C++ library (svm_type = EPSILON_SVR).

    Parameters
    ----------
    kernel : str, default='rbf'
        Kernel type: ``'linear'``, ``'poly'``, ``'rbf'``, ``'sigmoid'``,
        ``'precomputed'``.
    C : float, default=1.0
        Regularisation parameter.  Must be strictly positive.
    gamma : float, 'auto', 'scale', or None, default=None
        Kernel coefficient.  ``None`` and ``'auto'`` both resolve to
        ``1 / n_features``.  ``'scale'`` uses
        ``1 / (n_features * X.var())``.
    degree : int, default=3
        Degree for polynomial kernel.
    coef0 : float, default=0.0
        Constant term for polynomial and sigmoid kernels.
    epsilon : float, default=0.1
        Epsilon in the epsilon-insensitive loss function (half-tube width).
    cache_size : float, default=100.0
        Kernel cache size in MB.
    tol : float, default=1e-3
        Convergence tolerance (passed as ``eps`` to the library).
    shrinking : bool, default=True
        Whether to use shrinking heuristics.
    qpsize : int, default=10
        QP sub-problem size.
    Cbegin : float, default=1.0
        Initial C for linear-kernel warm-start.
    Cstep : float, default=2.0
        C step multiplier for linear-kernel warm-start.
    verbose : bool, default=True
        If ``True``, print BSVM's training progress to stdout.

    Attributes
    ----------
    model_ : LibBSVMModel
        Internal Cython object wrapping the trained ``svm_model*``.
    support_vectors_ : ndarray of shape (n_sv, n_features)
        Support vectors extracted from the model.
    dual_coef_ : ndarray of shape (1, n_sv)
        Dual coefficients (alpha).
    n_sv_ : int
        Total number of support vectors.
    n_features_in_ : int
        Number of features seen during ``fit``.
    time_ : float
        Wall-clock training time in seconds.
    """

    def __init__(
        self,
        kernel='rbf',
        C=1.0,
        gamma=None,
        degree=3,
        coef0=0.0,
        epsilon=0.1,
        cache_size=100.0,
        tol=1e-3,
        shrinking=True,
        qpsize=10,
        Cbegin=1.0,
        Cstep=2.0,
        verbose=True,
    ):
        self.kernel     = kernel
        self.C          = C
        self.gamma      = gamma
        self.degree     = degree
        self.coef0      = coef0
        self.epsilon    = epsilon
        self.cache_size = cache_size
        self.tol        = tol
        self.shrinking  = shrinking
        self.qpsize     = qpsize
        self.Cbegin     = Cbegin
        self.Cstep      = Cstep
        self.verbose    = verbose

    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Train the BSVM regressor.

        Parameters
        ----------
        X : array-like of shape (m, n)
            Training data.
        y : array-like of shape (m,)
            Target values (real-valued).

        Returns
        -------
        self : BSVMRegressor
        """
        _libbsvm = _import_libbsvm()

        X, y = check_X_y(X, y)
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # ── resolve kernel ────────────────────────────────────────────────────
        kernel_lower = self.kernel.lower()
        if kernel_lower not in _libbsvm.KERNEL_MAP:
            raise ValueError(
                f"Unknown kernel '{self.kernel}'. "
                f"Choose from: {sorted(_libbsvm.KERNEL_MAP)}."
            )
        kernel_type = _libbsvm.KERNEL_MAP[kernel_lower]

        m, n = X.shape

        # ── resolve gamma ─────────────────────────────────────────────────────
        self._gamma = _resolve_gamma(self.gamma, X)

        t0 = time.perf_counter()

        ctx = _suppress_c_stdout() if not self.verbose else _noop_ctx()
        with ctx:
            self.model_ = _libbsvm.train(
                X,
                y,
                svm_type     = _libbsvm.SVM_TYPE_MAP['epsilon_svr'],
                kernel_type  = kernel_type,
                C            = float(self.C),
                gamma        = self._gamma,
                degree       = int(self.degree),
                coef0        = float(self.coef0),
                cache_size   = float(self.cache_size),
                eps          = float(self.tol),
                shrinking    = int(self.shrinking),
                qpsize       = int(self.qpsize),
                Cbegin       = float(self.Cbegin),
                Cstep        = float(self.Cstep),
                p            = float(self.epsilon),
                nr_weight    = 0,
                weight_label = None,
                weight       = None,
            )

        self.time_            = time.perf_counter() - t0
        self.n_sv_            = self.model_.n_sv
        self.n_features_in_   = n
        self.support_vectors_ = self.model_.get_support_vectors(n)
        self.dual_coef_       = self.model_.get_sv_coef()   # shape (1, n_sv)

        if self.verbose:
            print(
                f"BSVM training: {self.time_:.3f}s, "
                f"{self.n_sv_} support vectors."
            )

        return self

    # ──────────────────────────────────────────────────────────────────────────

    def predict(self, X):
        """
        Predict target values.

        Parameters
        ----------
        X : array-like of shape (m, n)

        Returns
        -------
        y_pred : float64 ndarray of shape (m,)
        """
        check_is_fitted(self)
        _libbsvm = _import_libbsvm()
        X = check_array(X)
        X = np.ascontiguousarray(X, dtype=np.float64)
        return _libbsvm.predict_batch(self.model_, X)

    # ──────────────────────────────────────────────────────────────────────────

    def decision_function(self, X):
        """
        Compute the regression prediction via the kernel expansion.

        Uses the augmented kernel ``K(x, sv) + 1`` identical to the C++
        library's prediction path for EPSILON_SVR.

        Parameters
        ----------
        X : array-like of shape (m, n)

        Returns
        -------
        scores : float64 ndarray of shape (m,)
        """
        check_is_fitted(self)

        if self.kernel.lower() == 'precomputed':
            raise ValueError(
                "decision_function is not supported for kernel='precomputed'. "
                "Use predict() directly."
            )

        X = check_array(X)
        X = np.ascontiguousarray(X, dtype=np.float64)

        sv  = self.support_vectors_
        kp  = _kernel_params(self.kernel, self._gamma, self.degree, self.coef0)
        K   = pairwise_kernels(X, sv, metric=self.kernel, **kp)
        # augmented kernel: K + 1  (matches the C++ decision function)
        K_aug = K + 1.0
        # dual_coef_[0] is the single row of shape (n_sv,)
        return K_aug @ self.dual_coef_[0]
