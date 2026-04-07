"""
liblinear.py — scikit-learn compatible wrappers for the liblinear-official package.

Two estimators are provided:

  LibLinearSVC  — linear SVM / logistic-regression classifier (solvers 0–7)
  LibLinearSVR  — linear SVR (solvers 11–13)

Both require the ``liblinear-official`` package::

    pip install liblinear-official

References
----------
Fan, R.-E., Chang, K.-W., Hsieh, C.-J., Wang, X.-R., & Lin, C.-J. (2008).
LIBLINEAR: A library for large linear classification.
Journal of Machine Learning Research, 9, 1871–1874.
"""

import time
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils.multiclass import check_classification_targets


# ── solver metadata ───────────────────────────────────────────────────────────

_CLASSIFICATION_SOLVERS = frozenset({0, 1, 2, 3, 4, 5, 6, 7})
_REGRESSION_SOLVERS = frozenset({11, 12, 13})

_SOLVER_NAMES = {
    0:  "L2R_LR",
    1:  "L2R_L2LOSS_SVC_DUAL",
    2:  "L2R_L2LOSS_SVC",
    3:  "L2R_L1LOSS_SVC_DUAL",
    4:  "MCSVM_CS",
    5:  "L1R_L2LOSS_SVC",
    6:  "L1R_LR",
    7:  "L2R_LR_DUAL",
    11: "L2R_L2LOSS_SVR",
    12: "L2R_L2LOSS_SVR_DUAL",
    13: "L2R_L1LOSS_SVR_DUAL",
}


# ── import helper ─────────────────────────────────────────────────────────────

def _import_liblinear():
    try:
        import liblinear.liblinearutil as lu
        return lu
    except ImportError as exc:
        raise ImportError(
            "liblinear-official is required.  "
            "Install it with: pip install liblinear-official"
        ) from exc


# ── SVC ───────────────────────────────────────────────────────────────────────

class LibLinearSVC(ClassifierMixin, BaseEstimator):
    """Linear SVM / logistic-regression classifier backed by liblinear.

    Thin scikit-learn wrapper around the ``liblinear-official`` Python package.
    Accepts all liblinear classification solvers (0–7).

    Parameters
    ----------
    solver : int, default=1
        Liblinear solver type:

        - 0  : L2-regularised logistic regression (primal)
        - 1  : L2-regularised L2-loss SVM dual  [default]
        - 2  : L2-regularised L2-loss SVM primal
        - 3  : L2-regularised L1-loss SVM dual
        - 4  : Crammer-Singer multi-class SVM
        - 5  : L1-regularised L2-loss SVM
        - 6  : L1-regularised logistic regression
        - 7  : L2-regularised logistic regression (dual)

    C : float, default=1.0
        Regularisation parameter (larger → less regularisation).
    tol : float, default=1e-4
        Stopping tolerance for the solver.
    fit_intercept : bool, default=True
        Whether to fit a bias term (adds an augmented feature with value 1).
    class_weight : dict or ``'balanced'`` or None, default=None
        Per-class weights.  ``'balanced'`` uses
        ``n_samples / (n_classes * class_count)``.  A dict maps original class
        labels to their weights.
    verbose : bool, default=False
        Print liblinear solver output to stdout.

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        Weight vectors.  Binary classification: shape ``(1, n_features)``,
        oriented so that ``decision_function(X) > 0`` predicts ``classes_[1]``.
        Multi-class: shape ``(n_classes, n_features)``.
    intercept_ : ndarray of shape (1,) or (n_classes,)
        Bias terms.
    classes_ : ndarray of shape (n_classes,)
        Unique class labels seen during ``fit``.
    train_time_ : float
        Wall-clock seconds spent inside ``fit``.
    n_features_in_ : int
        Number of input features (set by ``validate_data``).
    """

    def __init__(
        self,
        solver=1,
        C=1.0,
        tol=1e-4,
        fit_intercept=True,
        class_weight=None,
        verbose=False,
    ):
        self.solver = solver
        self.C = C
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.class_weight = class_weight
        self.verbose = verbose

    # ── parameter validation ───────────────────────────────────────────────

    def _validate_params(self):
        if self.solver not in _CLASSIFICATION_SOLVERS:
            raise ValueError(
                f"solver must be one of {sorted(_CLASSIFICATION_SOLVERS)}, "
                f"got {self.solver!r}."
            )
        if self.C <= 0:
            raise ValueError(f"C must be > 0, got {self.C!r}.")
        if self.tol <= 0:
            raise ValueError(f"tol must be > 0, got {self.tol!r}.")

    def _weight_opts(self, y_enc, n_classes):
        """Build -w<label> <weight> option tokens for class weighting."""
        cw = self.class_weight
        if cw is None:
            return ""
        if cw == "balanced":
            n_samples = len(y_enc)
            counts = np.bincount(y_enc, minlength=n_classes).astype(float)
            counts = np.maximum(counts, 1)  # avoid division by zero
            weights = n_samples / (n_classes * counts)
        elif isinstance(cw, dict):
            weights = np.ones(n_classes, dtype=float)
            for orig_label, w in cw.items():
                idx = np.searchsorted(self.classes_, orig_label)
                if idx < n_classes and self.classes_[idx] == orig_label:
                    weights[idx] = w
        else:
            raise ValueError(
                f"class_weight must be None, 'balanced', or a dict; "
                f"got {cw!r}."
            )
        return " ".join(f"-w{i} {weights[i]}" for i in range(n_classes))

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """Fit the liblinear classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Class labels (any comparable values accepted).

        Returns
        -------
        self
        """
        self._validate_params()
        lu = _import_liblinear()

        X, y_raw = validate_data(self, X, y)
        check_classification_targets(y_raw)

        le = LabelEncoder()
        y_enc = le.fit_transform(y_raw)   # integer labels 0 … n_classes-1
        self.classes_ = le.classes_
        self._label_encoder = le
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError(
                f"LibLinearSVC requires at least 2 classes; "
                f"got 1 class: {self.classes_}."
            )

        weight_opts = self._weight_opts(y_enc, n_classes)
        bias_opt = "-B 1" if self.fit_intercept else ""
        quiet_opt = "" if self.verbose else "-q"
        opts = " ".join(
            tok for tok in [
                f"-s {self.solver}",
                f"-c {self.C}",
                f"-e {self.tol}",
                bias_opt,
                weight_opts,
                quiet_opt,
            ] if tok
        )

        prob = lu.problem(y_enc.tolist(), X.tolist())
        t0 = time.perf_counter()
        m = lu.train(prob, opts)
        self.train_time_ = time.perf_counter() - t0

        # Extract weight vectors.
        # m.get_labels() returns liblinear's class ordering, which follows the
        # order of first occurrence in the training data — NOT guaranteed sorted.
        # LabelEncoder always sorts, so we map each liblinear class index back
        # to its position in classes_ (the encoded int label is that position).
        #
        # For binary: coef_ shape (1, n_features), oriented so that
        #   decision_function > 0  ⟺  predict classes_[1]  (sklearn convention).
        # For multi-class: coef_[k] and intercept_[k] belong to classes_[k].
        lib_labels = m.get_labels()   # liblinear class ordering (unsorted)
        n_features = X.shape[1]

        if n_classes == 2:
            # j1: liblinear index for encoded label 1 → classes_[1]
            j1 = lib_labels.index(1)
            coef, bias = m.get_decfun(j1)
            self.coef_ = np.array(coef, dtype=np.float64).reshape(1, -1)
            self.intercept_ = np.array([bias], dtype=np.float64)
        else:
            coef_mat = np.empty((n_classes, n_features), dtype=np.float64)
            intercept = np.empty(n_classes, dtype=np.float64)
            for j, enc_label in enumerate(lib_labels):
                c, b = m.get_decfun(j)
                coef_mat[enc_label] = c
                intercept[enc_label] = b
            self.coef_ = coef_mat
            self.intercept_ = intercept

        return self

    # ── predict ───────────────────────────────────────────────────────────

    def decision_function(self, X):
        """Signed distances to the decision hyperplane(s).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples,) for binary classification,
                 (n_samples, n_classes) for multi-class.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = np.asarray(X, dtype=np.float64)
        if len(self.classes_) == 2:
            return X @ self.coef_[0] + self.intercept_[0]
        return X @ self.coef_.T + self.intercept_

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
        X = np.asarray(X, dtype=np.float64)
        if len(self.classes_) == 2:
            scores = X @ self.coef_[0] + self.intercept_[0]
            return self.classes_[(scores > 0).astype(int)]
        scores = X @ self.coef_.T + self.intercept_
        return self.classes_[np.argmax(scores, axis=1)]


# ── SVR ───────────────────────────────────────────────────────────────────────

class LibLinearSVR(RegressorMixin, BaseEstimator):
    """Linear SVR backed by liblinear.

    Thin scikit-learn wrapper around the ``liblinear-official`` Python package.
    Accepts all liblinear regression solvers (11–13).

    Parameters
    ----------
    solver : int, default=11
        Liblinear solver type:

        - 11 : L2-regularised L2-loss SVR primal  [default]
        - 12 : L2-regularised L2-loss SVR dual
        - 13 : L2-regularised L1-loss SVR dual

    C : float, default=1.0
        Regularisation parameter (larger → less regularisation).
    p : float, default=0.1
        Epsilon in the epsilon-insensitive loss function.
    tol : float, default=1e-4
        Stopping tolerance for the solver.
    fit_intercept : bool, default=True
        Whether to fit a bias term.
    verbose : bool, default=False
        Print liblinear solver output to stdout.

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_features)
        Weight vector.
    intercept_ : ndarray of shape (1,)
        Bias term.
    train_time_ : float
        Wall-clock seconds spent inside ``fit``.
    n_features_in_ : int
        Number of input features (set by ``validate_data``).
    """

    def __init__(
        self,
        solver=11,
        C=1.0,
        p=0.1,
        tol=1e-4,
        fit_intercept=True,
        verbose=False,
    ):
        self.solver = solver
        self.C = C
        self.p = p
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    # ── parameter validation ───────────────────────────────────────────────

    def _validate_params(self):
        if self.solver not in _REGRESSION_SOLVERS:
            raise ValueError(
                f"solver must be one of {sorted(_REGRESSION_SOLVERS)}, "
                f"got {self.solver!r}."
            )
        if self.C <= 0:
            raise ValueError(f"C must be > 0, got {self.C!r}.")
        if self.p < 0:
            raise ValueError(f"p must be >= 0, got {self.p!r}.")
        if self.tol <= 0:
            raise ValueError(f"tol must be > 0, got {self.tol!r}.")

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """Fit the liblinear SVR.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Continuous target values.

        Returns
        -------
        self
        """
        self._validate_params()
        lu = _import_liblinear()

        X, y = validate_data(self, X, y)

        bias_opt = "-B 1" if self.fit_intercept else ""
        quiet_opt = "" if self.verbose else "-q"
        opts = " ".join(
            tok for tok in [
                f"-s {self.solver}",
                f"-c {self.C}",
                f"-p {self.p}",
                f"-e {self.tol}",
                bias_opt,
                quiet_opt,
            ] if tok
        )

        prob = lu.problem(y.tolist(), X.tolist())
        t0 = time.perf_counter()
        m = lu.train(prob, opts)
        self.train_time_ = time.perf_counter() - t0

        coef, bias = m.get_decfun(0)
        self.coef_ = np.array(coef, dtype=np.float64).reshape(1, -1)
        self.intercept_ = np.array([bias], dtype=np.float64)
        return self

    # ── predict ───────────────────────────────────────────────────────────

    def predict(self, X):
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_[0] + self.intercept_[0]
