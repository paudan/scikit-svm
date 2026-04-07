"""
CVM  –  Core Vector Machine (binary classifier)

Cython-backed wrapper around the libCVM C++ library by Tsang, Kocsor & Kwok.
The underlying algorithm (svm_type = CVM, type 6) solves:

    min_{w,b} ½‖w‖² + C·Σᵢ max(0, 1 − yᵢ f(xᵢ))²

using a core-set approximation (Minimum Enclosing Ball) that scales to very
large datasets.

Reference
---------
Tsang, Kwok, Cheung. "Core Vector Machines: Fast SVM Training on Very Large
Data Sets." JMLR 6, 2005.
"""

import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._utils import _suppress_c_stdout


def _import_libcvm():
    try:
        from . import _libcvm
        return _libcvm
    except ImportError as exc:
        raise ImportError(
            "scikit_svm._libcvm is not built. "
            "Run 'pip install -e .' (or python setup.py build_ext --inplace) "
            "from the project root to compile the Cython extension."
        ) from exc


class CVM(BaseEstimator, ClassifierMixin):
    """
    Core Vector Machine (CVM) binary classifier.

    Fast large-scale SVM using squared hinge loss and a core-set
    approximation (Minimum Enclosing Ball in kernel feature space).
    Supports the same kernels as libCVM: linear, poly, rbf, sigmoid, exp,
    normal_poly, inv_dist, inv_sqdist.

    Labels must be exactly +1 or -1.

    Parameters
    ----------
    C : float, default=100.0
        Regularisation parameter.  Must be strictly positive.
    kernel : str, default='rbf'
        Kernel type. One of ``'linear'``, ``'poly'``, ``'rbf'``,
        ``'sigmoid'``, ``'exp'``, ``'normal_poly'``, ``'inv_dist'``,
        ``'inv_sqdist'``.
    gamma : float or None, default=None
        Kernel coefficient.  If ``None``, set to ``1 / n_features`` at
        fit time.  Must be > 0.
    degree : int, default=3
        Degree for the polynomial kernel.
    coef0 : float, default=0.0
        Constant term for polynomial and sigmoid kernels.
    cache_size : float, default=200.0
        Size of the kernel cache in MB.
    eps : float, default=-1.0
        Stopping tolerance.  Pass ``-1`` to use the adaptive default that
        libCVM selects based on the approximation bound ``|f(x)−f*(x)|``.
    max_sv : int, default=50000
        Maximum number of core vectors (support vectors).
    verbose : bool, default=True
        If ``True``, print libCVM's training progress to stdout.

    Attributes
    ----------
    model_ : LibCVMModel
        Internal Cython object wrapping the trained ``svm_model*``.
    n_sv_ : int
        Number of support / core vectors after training.
    classes_ : ndarray of shape (2,)
        Always ``[-1.0, 1.0]``.
    n_features_in_ : int
        Number of features seen during ``fit``.
    time_ : float
        Wall-clock training time in seconds.
    """

    def __init__(
        self,
        C=100.0,
        kernel='rbf',
        gamma=None,
        degree=3,
        coef0=0.0,
        cache_size=200.0,
        eps=-1.0,
        max_sv=50000,
        verbose=True,
    ):
        self.C          = C
        self.kernel     = kernel
        self.gamma      = gamma
        self.degree     = degree
        self.coef0      = coef0
        self.cache_size = cache_size
        self.eps        = eps
        self.max_sv     = max_sv
        self.verbose    = verbose

    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Train the CVM model.

        Parameters
        ----------
        X : array-like of shape (m, n)
            Training data.
        y : array-like of shape (m,)
            Class labels; every entry must be exactly +1 or -1.

        Returns
        -------
        self : CVM
        """
        _libcvm = _import_libcvm()

        X, y = check_X_y(X, y)
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if not np.all((y == 1.0) | (y == -1.0)):
            raise ValueError("Labels must be exactly +1 or -1.")

        m, n = X.shape

        # Resolve gamma default
        gamma = 1.0 / n if self.gamma is None else float(self.gamma)
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}.")

        # Map kernel name → integer
        kernel_lower = self.kernel.lower()
        if kernel_lower not in _libcvm.KERNEL_MAP:
            raise ValueError(
                f"Unknown kernel '{self.kernel}'. "
                f"Choose from: {sorted(_libcvm.KERNEL_MAP)}."
            )
        kernel_type = _libcvm.KERNEL_MAP[kernel_lower]

        t0 = time.perf_counter()

        ctx = _suppress_c_stdout() if not self.verbose else _noop_ctx()
        with ctx:
            self.model_ = _libcvm.train(
                X,
                y,
                svm_type   = _libcvm.SVM_TYPE_CVM,
                kernel_type= kernel_type,
                C          = float(self.C),
                gamma      = gamma,
                degree     = int(self.degree),
                coef0      = float(self.coef0),
                cache_size = float(self.cache_size),
                eps        = float(self.eps),
                max_sv     = int(self.max_sv),
                sample_size= 60,   # not used by CVM, but required by param
            )

        self.time_           = time.perf_counter() - t0
        self.n_sv_           = self.model_.n_sv
        self.classes_        = np.array([-1.0, 1.0])
        self.n_features_in_  = n

        return self

    # ──────────────────────────────────────────────────────────────────────────

    def decision_function(self, X):
        """
        Compute raw decision scores.

        Positive values → class +1; negative values → class -1.

        Parameters
        ----------
        X : array-like of shape (m, n)

        Returns
        -------
        scores : ndarray of shape (m,)
        """
        check_is_fitted(self)
        _libcvm = _import_libcvm()
        X = check_array(X)
        X = np.ascontiguousarray(X, dtype=np.float64)
        raw = _libcvm.decision_function_batch(self.model_, X)
        # Normalise sign: libCVM's positive side → label[0].
        # We want positive side → +1 always.
        sign = 1.0 if self.model_.label0 == 1 else -1.0
        return raw * sign

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
            Predicted labels in {-1.0, +1.0}.
        """
        return np.where(self.decision_function(X) > 0, 1.0, -1.0)


# ─────────────────────────────────────────────────────────────────────────────
# tiny helper so the 'verbose=True' path avoids the context-manager overhead
# ─────────────────────────────────────────────────────────────────────────────

import contextlib

@contextlib.contextmanager
def _noop_ctx():
    yield
