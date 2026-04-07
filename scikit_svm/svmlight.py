"""
SVM-Light classifiers for scikit-svm.

SVMLightClassifier  – binary SVM classification (SVM-Light V6.02)
SVMLightRegressor   – epsilon-SVR regression     (SVM-Light V6.02)

SVM-Light was written by Thorsten Joachims.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted


# ── helpers ───────────────────────────────────────────────────────────────────
def _import_lib():
    try:
        from . import _libsvmlight
        return _libsvmlight
    except ImportError as exc:
        raise ImportError(
            "scikit_svm._libsvmlight not available. "
            "Rebuild with: pip install -e . --no-build-isolation"
        ) from exc


def _resolve_gamma(gamma, X):
    """Map gamma param to a float following sklearn conventions."""
    if gamma is None or gamma == "auto":
        return 1.0 / X.shape[1]
    if gamma == "scale":
        v = float(X.var())
        return 1.0 / (X.shape[1] * v) if v > 0 else 1.0 / X.shape[1]
    return float(gamma)


def _costratio(class_weight, y_pm):
    """Return svm_costratio = C_positive / C_negative."""
    if class_weight is None:
        return 1.0
    if class_weight == "balanced":
        n_pos = int(np.sum(y_pm > 0))
        n_neg = int(np.sum(y_pm < 0))
        if n_pos == 0 or n_neg == 0:
            return 1.0
        return float(n_neg) / float(n_pos)
    raise ValueError(
        f"class_weight must be None or 'balanced', got {class_weight!r}"
    )


# ── SVMLightClassifier ────────────────────────────────────────────────────────
class SVMLightClassifier(BaseEstimator, ClassifierMixin):
    """Binary SVM classifier backed by SVM-Light V6.02.

    Supports binary classification with arbitrary label types (encoded
    internally to +1 / -1).  For multi-class problems, wrap with
    ``sklearn.multiclass.OneVsRestClassifier`` or ``OneVsOneClassifier``.

    Parameters
    ----------
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, default='rbf'
    C : float, default=1.0
        Regularisation parameter.  Must be > 0.
    gamma : float or {'auto', 'scale'}, default='auto'
        Kernel coefficient for 'rbf', 'poly', 'sigmoid'.
        'auto'  → 1 / n_features
        'scale' → 1 / (n_features * Var(X))
    degree : int, default=3
        Polynomial degree (poly kernel only).
    coef0 : float, default=1.0
        Constant term in poly / sigmoid kernel: K(x,y) = (gamma·x·y + coef0)^d.
    epsilon_crit : float, default=1e-3
        Stopping tolerance for the QP optimiser.
    cache_size : int, default=40
        Kernel cache size in MB.
    max_iter : int, default=100000
        Maximum number of optimisation iterations.
    biased_hyperplane : bool, default=True
        Learn w·x + b = 0 (True) or unbiased w·x = 0 (False).
    shrink_iter : int, default=-1
        Iterations before a variable may be shrunk (-1 → SVM-Light default:
        2 for linear, 100 for non-linear kernels).
    class_weight : None or 'balanced', default=None
        Adjust the cost ratio between positive and negative class.
        'balanced' sets costratio = n_neg / n_pos.
    verbose : bool, default=False
        Print SVM-Light optimisation progress to stdout.

    Attributes
    ----------
    support_vectors_ : ndarray of shape (n_sv, n_features)
    dual_coef_ : ndarray of shape (1, n_sv)
        Signed alphas: alpha_i * y_i.
    intercept_ : ndarray of shape (1,)
        Bias term: intercept_ = -b (sklearn convention).
    support_ : ndarray of int
        Training-set indices of support vectors.
    n_support_ : ndarray of int
        Count of SVs with negative / positive signed alpha.
    classes_ : ndarray
        Class labels in order [negative_class, positive_class].
    """

    def __init__(
        self,
        kernel="rbf",
        C=1.0,
        gamma="auto",
        degree=3,
        coef0=1.0,
        epsilon_crit=1e-3,
        cache_size=40,
        max_iter=100000,
        biased_hyperplane=True,
        shrink_iter=-1,
        class_weight=None,
        verbose=False,
    ):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.epsilon_crit = epsilon_crit
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.biased_hyperplane = biased_hyperplane
        self.shrink_iter = shrink_iter
        self.class_weight = class_weight
        self.verbose = verbose

    def fit(self, X, y):
        """Fit model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Binary class labels (any two distinct values).
        """
        lib = _import_lib()
        X = check_array(X, dtype=np.float64, order="C")
        check_classification_targets(y)

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        self.classes_ = le.classes_
        if len(self.classes_) != 2:
            raise ValueError(
                f"SVMLightClassifier supports binary classification only "
                f"({len(self.classes_)} classes found). "
                "Use sklearn.multiclass wrappers for multi-class."
            )
        self.label_encoder_ = le

        # Map: encoded 0 → -1, encoded 1 → +1
        y_pm = np.where(y_enc == 0, -1.0, 1.0).astype(np.float64)

        kernel_type = lib.KERNEL_MAP.get(self.kernel)
        if kernel_type is None:
            raise ValueError(f"Unknown kernel {self.kernel!r}. "
                             f"Choose from {list(lib.KERNEL_MAP)}")

        gamma = _resolve_gamma(self.gamma, X)
        cost_ratio = _costratio(self.class_weight, y_pm)

        self._model = lib.train_classification(
            X,
            y_pm,
            kernel_type=kernel_type,
            C=float(self.C),
            rbf_gamma=gamma,
            poly_degree=int(self.degree),
            coef_lin=gamma,           # poly/sigmoid: K=(gamma*x·y+coef0)^d
            coef_const=float(self.coef0),
            epsilon_crit=float(self.epsilon_crit),
            kernel_cache_size=int(self.cache_size),
            svm_maxqpsize=10,
            maxiter=int(self.max_iter),
            svm_iter_to_shrink=int(self.shrink_iter),
            biased_hyperplane=int(self.biased_hyperplane),
            svm_costratio=cost_ratio,
            verbose=bool(self.verbose),
        )

        self.n_features_in_ = X.shape[1]

        # sklearn-compatible attributes
        self.support_vectors_ = self._model.get_support_vectors(X.shape[1])
        alphas = self._model.get_alphas()
        self.dual_coef_  = alphas.reshape(1, -1)
        self.intercept_  = np.array([-self._model.b])
        self.support_    = self._model.get_sv_docnums()
        self.n_support_  = np.array(
            [int(np.sum(alphas < 0)), int(np.sum(alphas > 0))],
            dtype=np.int32,
        )
        return self

    def decision_function(self, X):
        """Raw SVM decision scores (positive → class +1 / classes_[1]).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, order="C")
        return _import_lib().predict_batch(self._model, X)

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
        y_enc = (scores > 0).astype(int)          # 0 or 1
        return self.label_encoder_.inverse_transform(y_enc)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# ── SVMLightRegressor ─────────────────────────────────────────────────────────
class SVMLightRegressor(BaseEstimator, RegressorMixin):
    """Epsilon-SVR backed by SVM-Light V6.02.

    Parameters
    ----------
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, default='rbf'
    C : float, default=1.0
    gamma : float or {'auto', 'scale'}, default='auto'
    degree : int, default=3
    coef0 : float, default=1.0
    epsilon : float, default=0.1
        Half-width of the insensitive tube.
    epsilon_crit : float, default=1e-3
    cache_size : int, default=40
    max_iter : int, default=100000
    shrink_iter : int, default=-1
    verbose : bool, default=False

    Attributes
    ----------
    support_vectors_ : ndarray of shape (n_sv, n_features)
    dual_coef_ : ndarray of shape (1, n_sv)
        Signed alphas (alpha_i+ - alpha_i-).
    intercept_ : ndarray of shape (1,)
    support_ : ndarray of int
    """

    def __init__(
        self,
        kernel="rbf",
        C=1.0,
        gamma="auto",
        degree=3,
        coef0=1.0,
        epsilon=0.1,
        epsilon_crit=1e-3,
        cache_size=40,
        max_iter=100000,
        shrink_iter=-1,
        verbose=False,
    ):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.epsilon = epsilon
        self.epsilon_crit = epsilon_crit
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.shrink_iter = shrink_iter
        self.verbose = verbose

    def fit(self, X, y):
        """Fit model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        """
        lib = _import_lib()
        X = check_array(X, dtype=np.float64, order="C")
        y = np.asarray(y, dtype=np.float64).ravel()

        kernel_type = lib.KERNEL_MAP.get(self.kernel)
        if kernel_type is None:
            raise ValueError(f"Unknown kernel {self.kernel!r}. "
                             f"Choose from {list(lib.KERNEL_MAP)}")

        gamma = _resolve_gamma(self.gamma, X)

        self._model = lib.train_regression(
            X,
            y,
            kernel_type=kernel_type,
            C=float(self.C),
            rbf_gamma=gamma,
            poly_degree=int(self.degree),
            coef_lin=gamma,
            coef_const=float(self.coef0),
            epsilon=float(self.epsilon),
            epsilon_crit=float(self.epsilon_crit),
            kernel_cache_size=int(self.cache_size),
            svm_maxqpsize=10,
            maxiter=int(self.max_iter),
            svm_iter_to_shrink=int(self.shrink_iter),
            verbose=bool(self.verbose),
        )

        self.n_features_in_ = X.shape[1]
        self.support_vectors_ = self._model.get_support_vectors(X.shape[1])
        alphas = self._model.get_alphas()
        self.dual_coef_  = alphas.reshape(1, -1)
        self.intercept_  = np.array([-self._model.b])
        self.support_    = self._model.get_sv_docnums()
        return self

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
        X = check_array(X, dtype=np.float64, order="C")
        return _import_lib().predict_batch(self._model, X)

    def score(self, X, y):
        return r2_score(y, self.predict(X))
