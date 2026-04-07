"""
ocas.py – scikit-learn compatible wrappers for the libocas linear SVM solvers.

Two classifiers are provided:

  SVMOCASClassifier   – binary linear SVM (wraps svm_ocas_solver)
  MSVMOCASClassifier  – multi-class linear SVM, Crammer-Singer formulation
                        (wraps msvm_ocas_solver)

Both classifiers are pure linear models: no kernel support.  They scale to
large, high-dimensional datasets (the original design target of libocas).

Reference
---------
Franc, V., & Sonnenburg, S. (2008).  Optimized cutting plane algorithm for
support vector machines.  ICML 2008.
"""

import time
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y, validate_data
from sklearn.utils.multiclass import check_classification_targets, type_of_target

from ._utils import _suppress_c_stdout


def _import_libocas():
    try:
        from . import _libocas
        return _libocas
    except ImportError as exc:
        raise ImportError(
            "scikit_svm._libocas is not built. "
            "Run 'pip install -e .' or 'python setup.py build_ext --inplace'."
        ) from exc


# ── Binary SVM ───────────────────────────────────────────────────────────────

class SVMOCASClassifier(ClassifierMixin, BaseEstimator):
    """Binary linear SVM trained with the OCAS algorithm.

    Minimises the SVM primal objective::

        min_{w, w0}  0.5 ||w||² + C Σ_i max(0, 1 - y_i (w·x_i + w0))

    Parameters
    ----------
    C : float, default=1.0
        Regularisation constant.  Larger → less regularisation.
    method : {'ocas', 'cp'}, default='ocas'
        Optimisation method.  ``'ocas'`` (default) uses the OCAS solver;
        ``'cp'`` uses the standard cutting-plane / BMRM / SVM-Perf variant.
    tol : float, default=1e-3
        Relative duality-gap tolerance: stop when
        ``(Q_P - Q_D) / |Q_P| ≤ tol``.
    buf_size : int, default=2000
        Maximum number of cutting planes buffered by the solver.
    max_time : float, default=inf
        Wall-clock time budget (seconds).
    fit_intercept : bool, default=True
        Whether to fit a bias term.
    verbose : bool, default=False
        Print solver statistics after training.

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_features)
        Weight vector (sklearn convention).
    intercept_ : ndarray of shape (1,)
        Bias term.
    classes_ : ndarray of shape (n_classes,)
        Unique class labels seen during ``fit``.
    n_iter_ : int
        Number of OCAS iterations performed.
    train_time_ : float
        Wall-clock seconds spent in ``fit``.
    n_features_in_ : int
        Number of features seen during ``fit``.
    """

    def __init__(
        self,
        C=1.0,
        method="ocas",
        tol=1e-3,
        buf_size=2000,
        max_time=float("inf"),
        fit_intercept=True,
        verbose=False,
    ):
        self.C = C
        self.method = method
        self.tol = tol
        self.buf_size = buf_size
        self.max_time = max_time
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    # ── validation helpers ─────────────────────────────────────────────────

    def _validate_params(self):
        if self.C <= 0:
            raise ValueError(f"C must be > 0, got {self.C!r}.")
        if self.method not in ("ocas", "cp"):
            raise ValueError(
                f"method must be 'ocas' or 'cp', got {self.method!r}."
            )
        if self.tol <= 0:
            raise ValueError(f"tol must be > 0, got {self.tol!r}.")
        if self.buf_size < 1:
            raise ValueError(f"buf_size must be >= 1, got {self.buf_size!r}.")

    def _encode(self, y_raw):
        """Encode arbitrary two-class labels to ±1."""
        classes = np.unique(y_raw)
        self.classes_ = classes
        if len(classes) < 2:
            raise ValueError(
                f"SVMOCASClassifier requires at least 2 classes; "
                f"got 1 class: {classes}."
            )
        self._le_neg = classes[0]
        self._le_pos = classes[1]
        return np.where(y_raw == self._le_neg, -1.0, 1.0)

    def _decode(self, y_int):
        return np.where(y_int > 0.0, self._le_pos, self._le_neg)

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """Fit the binary OCAS classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Binary class labels (any two values accepted).

        Returns
        -------
        self
        """
        self._validate_params()
        lib = _import_libocas()

        X, y_raw = validate_data(self, X, y)
        check_classification_targets(y_raw)
        y_type = type_of_target(y_raw, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                "Only binary classification is supported. "
                f"The type of the target is {y_type}."
            )

        y_enc = self._encode(y_raw)

        X = np.ascontiguousarray(X, dtype=np.float64)
        y_enc = np.asarray(y_enc, dtype=np.float64)
        method_code = np.uint8(1 if self.method == "ocas" else 0)

        # libocas expects X pre-multiplied by labels column-wise, exactly as
        # the MATLAB MEX interface does before invoking the C solver.
        X_scaled = np.ascontiguousarray(X * y_enc[:, np.newaxis], dtype=np.float64)

        t0 = time.perf_counter()
        ctx = _suppress_c_stdout() if not self.verbose else _noop_ctx()
        with ctx:
            result = lib.train_binary(
                X_scaled, y_enc,
                C=float(self.C),
                tol_rel=float(self.tol),
                tol_abs=0.0,
                qp_bound=-1e300,
                max_time=float(self.max_time),
                buf_size=int(self.buf_size),
                method=method_code,
                fit_intercept=int(self.fit_intercept),
            )
        self.train_time_ = time.perf_counter() - t0

        self.coef_      = result["W"].reshape(1, -1)
        self.intercept_ = np.array([result["W0"]])
        self.n_iter_    = result["stats"]["n_iter"]
        self._stats     = result["stats"]

        if self.verbose:
            st = self._stats
            print(
                f"[SVMOCASClassifier] iters={st['n_iter']}, "
                f"Q_P={st['Q_P']:.6f}, Q_D={st['Q_D']:.6f}, "
                f"exitflag={st['exitflag']}, time={self.train_time_:.3f}s"
            )
        return self

    # ── predict ───────────────────────────────────────────────────────────

    def decision_function(self, X):
        """Signed distance to the decision hyperplane.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = np.ascontiguousarray(X, dtype=np.float64)
        return X @ self.coef_.ravel() + self.intercept_[0]

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        scores = self.decision_function(X)
        return self._decode(np.sign(scores))


# ── Multi-class SVM ──────────────────────────────────────────────────────────

class MSVMOCASClassifier(ClassifierMixin, BaseEstimator):
    """Multi-class linear SVM trained with the OCAS algorithm.

    Uses the Crammer-Singer formulation::

        min_W  0.5 Σ_y ||w_y||² + C Σ_i L_i(W)
        L_i(W) = max_y [ [y ≠ y_i] + (w_y - w_{y_i})·x_i ]

    Decision rule: ``ŷ = argmax_y w_y · x``

    Parameters
    ----------
    C : float, default=1.0
        Regularisation constant.
    method : {'ocas', 'cp'}, default='ocas'
        Optimisation method.
    tol : float, default=1e-3
        Relative duality-gap tolerance.
    buf_size : int, default=2000
        Maximum number of buffered cutting planes.
    max_time : float, default=inf
        Time budget (seconds).
    verbose : bool, default=False

    Attributes
    ----------
    coef_ : ndarray of shape (n_classes, n_features)
        One weight vector per class.
    classes_ : ndarray of shape (n_classes,)
    n_iter_ : int
    train_time_ : float
    n_features_in_ : int
    """

    def __init__(
        self,
        C=1.0,
        method="ocas",
        tol=1e-3,
        buf_size=2000,
        max_time=float("inf"),
        verbose=False,
    ):
        self.C = C
        self.method = method
        self.tol = tol
        self.buf_size = buf_size
        self.max_time = max_time
        self.verbose = verbose

    # ── validation helpers ─────────────────────────────────────────────────

    def _validate_params(self):
        if self.C <= 0:
            raise ValueError(f"C must be > 0, got {self.C!r}.")
        if self.method not in ("ocas", "cp"):
            raise ValueError(
                f"method must be 'ocas' or 'cp', got {self.method!r}."
            )
        if self.tol <= 0:
            raise ValueError(f"tol must be > 0, got {self.tol!r}.")
        if self.buf_size < 1:
            raise ValueError(f"buf_size must be >= 1, got {self.buf_size!r}.")

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """Fit the multi-class OCAS classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        self._validate_params()
        lib = _import_libocas()

        X, y_raw = validate_data(self, X, y)
        check_classification_targets(y_raw)

        # Encode arbitrary labels to 0-based integers, then shift to 1-based
        le = LabelEncoder()
        y_enc_0 = le.fit_transform(y_raw)          # 0-indexed
        self.classes_        = le.classes_
        self._label_encoder  = le
        nY                   = len(self.classes_)

        if nY < 2:
            raise ValueError(
                f"MSVMOCASClassifier requires at least 2 classes; "
                f"got 1 class: {self.classes_}."
            )

        y_1indexed = (y_enc_0 + 1).astype(np.float64)  # 1-indexed for libocas

        X = np.ascontiguousarray(X, dtype=np.float64)
        method_code = np.uint8(1 if self.method == "ocas" else 0)

        t0 = time.perf_counter()
        ctx = _suppress_c_stdout() if not self.verbose else _noop_ctx()
        with ctx:
            result = lib.train_msvm(
                X, y_1indexed,
                np.uint32(nY),
                C=float(self.C),
                tol_rel=float(self.tol),
                tol_abs=0.0,
                qp_bound=-1e300,
                max_time=float(self.max_time),
                buf_size=int(self.buf_size),
                method=method_code,
            )
        self.train_time_ = time.perf_counter() - t0

        self.coef_   = result["W"]   # shape (nY, nDim)
        self.n_iter_ = result["stats"]["n_iter"]
        self._stats  = result["stats"]

        if self.verbose:
            st = self._stats
            print(
                f"[MSVMOCASClassifier] iters={st['n_iter']}, "
                f"Q_P={st['Q_P']:.6f}, Q_D={st['Q_D']:.6f}, "
                f"exitflag={st['exitflag']}, time={self.train_time_:.3f}s"
            )
        return self

    # ── predict ───────────────────────────────────────────────────────────

    def decision_function(self, X):
        """Class scores: scores[i, y] = w_y · x_i.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples, n_classes) or (n_samples,) for binary
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = np.ascontiguousarray(X, dtype=np.float64)
        scores = X @ self.coef_.T        # (nData, nY)
        if scores.shape[1] == 2:
            return scores[:, 1] - scores[:, 0]
        return scores

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = np.ascontiguousarray(X, dtype=np.float64)
        scores = X @ self.coef_.T        # (nData, nY)
        y_idx = np.argmax(scores, axis=1)
        return self.classes_[y_idx]


# ── context manager for verbose suppression ───────────────────────────────────

class _noop_ctx:
    """No-op context manager used when verbose=True."""
    def __enter__(self): return self
    def __exit__(self, *a): return False
