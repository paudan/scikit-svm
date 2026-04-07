"""
mySVM classifiers and regressors with a scikit-learn compatible interface.

Wraps the mySVM library (Stefan Rueping, University of Dortmund).

Classes
-------
MySVMClassifier   – C-SVM binary classification
MySVMRegressor    – C-SVM regression (epsilon-SVR)
MySVMNuClassifier – ν-SVM binary classification
MySVMNuRegressor  – ν-SVM regression

All four classes follow the standard sklearn fit / predict /
decision_function / score interface.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted

from ._libmysvm import LibMySVMModel

# ── kernel type codes ──────────────────────────────────────────────────────
_KERNEL_MAP = {"linear": 0, "poly": 1, "rbf": 2, "sigmoid": 3}

# ── SVM type codes ─────────────────────────────────────────────────────────
_C_SVC         = 0
_EPSILON_SVR   = 1
_NU_SVC        = 2
_NU_SVR        = 3


def _resolve_gamma(gamma, X):
    """Return a float gamma value from 'scale', 'auto', or a number."""
    if gamma == "scale":
        return 1.0 / (X.shape[1] * X.var()) if X.var() > 0 else 1.0
    if gamma == "auto":
        return 1.0 / X.shape[1]
    return float(gamma)


def _build_model(svm_type, C, nu, epsilon, kernel, gamma_val, degree,
                 coef0, biased, balance_cost, convergence_epsilon,
                 max_iter, cache_size, verbose):
    """Instantiate and configure a LibMySVMModel."""
    if kernel not in _KERNEL_MAP:
        raise ValueError(
            f"Invalid kernel '{kernel}'. Choose from {list(_KERNEL_MAP)}."
        )
    m = LibMySVMModel()
    m.set_svm_type(svm_type)
    m.set_C(float(C))
    m.set_nu(float(nu))
    m.set_epsilon(float(epsilon))
    m.set_biased(int(biased))
    m.set_verbosity(int(verbose))
    m.set_cache_mb(int(cache_size))
    m.set_max_iter(int(max_iter))
    m.set_convergence_epsilon(float(convergence_epsilon))
    m.set_balance_cost(int(balance_cost))
    m.set_kernel_type(_KERNEL_MAP[kernel])
    m.set_kernel_gamma(float(gamma_val))
    m.set_kernel_degree(int(degree))
    m.set_kernel_coef0(float(coef0))
    return m


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MySVMClassifier – C-SVM binary classification                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class MySVMClassifier(BaseEstimator, ClassifierMixin):
    """C-SVM classifier (binary) wrapping the mySVM library.

    Parameters
    ----------
    C : float, default=1.0
        Regularisation parameter.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, default='rbf'
        Kernel type.
    gamma : float or {'scale', 'auto'}, default='scale'
        Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
    degree : int, default=3
        Degree for polynomial kernel.
    coef0 : float, default=1.0
        Independent term in 'sigmoid' (b in tanh(gamma·x·y + b)).
    biased : bool, default=True
        Use biased hyperplane (w·x + b).
    class_weight : 'balanced' or None, default=None
        Balance class frequencies in the cost term.
    convergence_epsilon : float, default=1e-3
        Convergence criterion.
    max_iter : int, default=100000
        Maximum number of iterations.
    cache_size : int, default=256
        Kernel cache size in MB.
    verbose : bool, default=False
        Print mySVM progress messages.
    """

    def __init__(self, C=1.0, kernel="rbf", gamma="scale", degree=3,
                 coef0=1.0, biased=True, class_weight=None,
                 convergence_epsilon=1e-3, max_iter=100000,
                 cache_size=256, verbose=False):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.biased = biased
        self.class_weight = class_weight
        self.convergence_epsilon = convergence_epsilon
        self.max_iter = max_iter
        self.cache_size = cache_size
        self.verbose = verbose

    # ── fit ───────────────────────────────────────────────────────────────

    def fit(self, X, y):
        X = check_array(X, dtype=np.float64, order="C")
        y = np.asarray(y)

        # Encode arbitrary binary labels to ±1
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        if len(self.le_.classes_) != 2:
            raise ValueError(
                "MySVMClassifier only supports binary classification "
                f"(got {len(self.le_.classes_)} classes)."
            )
        self.classes_ = self.le_.classes_
        y_pm1 = np.where(y_enc == 1, 1.0, -1.0).astype(np.float64)

        balance_cost = 0
        if self.class_weight == "balanced":
            balance_cost = 1
        elif self.class_weight is not None:
            raise ValueError(
                "class_weight must be 'balanced' or None, "
                f"got '{self.class_weight}'."
            )

        gamma_val = _resolve_gamma(self.gamma, X)

        self._model = _build_model(
            svm_type=_C_SVC,
            C=self.C, nu=0.5, epsilon=0.0,
            kernel=self.kernel, gamma_val=gamma_val,
            degree=self.degree, coef0=self.coef0,
            biased=int(self.biased), balance_cost=balance_cost,
            convergence_epsilon=self.convergence_epsilon,
            max_iter=self.max_iter, cache_size=self.cache_size,
            verbose=int(self.verbose),
        )
        self._model.train(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y_pm1, dtype=np.float64),
        )
        self.n_support_ = self._model.n_sv
        self._gamma_val = gamma_val
        return self

    # ── decision_function ─────────────────────────────────────────────────

    def decision_function(self, X):
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, order="C")
        X_c = np.ascontiguousarray(X, dtype=np.float64)
        return self._model.predict(X_c)

    # ── predict ───────────────────────────────────────────────────────────

    def predict(self, X):
        df = self.decision_function(X)
        # df > 0  → class +1 (le_ class index 1)
        # df <= 0 → class -1 (le_ class index 0)
        y_enc = np.where(df > 0, 1, 0)
        return self.le_.inverse_transform(y_enc)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MySVMRegressor – C-SVM regression (epsilon-SVR)                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class MySVMRegressor(BaseEstimator, RegressorMixin):
    """C-SVM regressor (epsilon-SVR) wrapping the mySVM library.

    Parameters
    ----------
    C : float, default=1.0
        Regularisation parameter.
    epsilon : float, default=0.1
        Epsilon-tube width.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, default='rbf'
        Kernel type.
    gamma : float or {'scale', 'auto'}, default='scale'
        Kernel coefficient.
    degree : int, default=3
        Degree for polynomial kernel.
    coef0 : float, default=1.0
        Independent term in 'sigmoid'.
    biased : bool, default=True
        Use biased hyperplane.
    convergence_epsilon : float, default=1e-3
        Convergence criterion.
    max_iter : int, default=100000
        Maximum iterations.
    cache_size : int, default=256
        Kernel cache in MB.
    verbose : bool, default=False
        Print mySVM progress messages.
    """

    def __init__(self, C=1.0, epsilon=0.1, kernel="rbf", gamma="scale",
                 degree=3, coef0=1.0, biased=True,
                 convergence_epsilon=1e-3, max_iter=100000,
                 cache_size=256, verbose=False):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.biased = biased
        self.convergence_epsilon = convergence_epsilon
        self.max_iter = max_iter
        self.cache_size = cache_size
        self.verbose = verbose

    def fit(self, X, y):
        X = check_array(X, dtype=np.float64, order="C")
        y = np.asarray(y, dtype=np.float64)

        gamma_val = _resolve_gamma(self.gamma, X)

        self._model = _build_model(
            svm_type=_EPSILON_SVR,
            C=self.C, nu=0.5, epsilon=self.epsilon,
            kernel=self.kernel, gamma_val=gamma_val,
            degree=self.degree, coef0=self.coef0,
            biased=int(self.biased), balance_cost=0,
            convergence_epsilon=self.convergence_epsilon,
            max_iter=self.max_iter, cache_size=self.cache_size,
            verbose=int(self.verbose),
        )
        self._model.train(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y, dtype=np.float64),
        )
        self.n_support_ = self._model.n_sv
        self._gamma_val = gamma_val
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, order="C")
        X_c = np.ascontiguousarray(X, dtype=np.float64)
        return self._model.predict(X_c)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MySVMNuClassifier – ν-SVM binary classification                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class MySVMNuClassifier(BaseEstimator, ClassifierMixin):
    """ν-SVM classifier (binary) wrapping the mySVM library.

    Parameters
    ----------
    nu : float, default=0.5
        Upper bound on the fraction of margin errors and lower bound
        on the fraction of support vectors.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, default='rbf'
        Kernel type.
    gamma : float or {'scale', 'auto'}, default='scale'
        Kernel coefficient.
    degree : int, default=3
        Degree for polynomial kernel.
    coef0 : float, default=1.0
        Independent term in 'sigmoid'.
    biased : bool, default=True
        Use biased hyperplane.
    convergence_epsilon : float, default=1e-3
        Convergence criterion.
    max_iter : int, default=100000
        Maximum iterations.
    cache_size : int, default=256
        Kernel cache in MB.
    verbose : bool, default=False
        Print mySVM progress messages.
    """

    def __init__(self, nu=0.5, kernel="rbf", gamma="scale", degree=3,
                 coef0=1.0, biased=True, convergence_epsilon=1e-3,
                 max_iter=100000, cache_size=256, verbose=False):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.biased = biased
        self.convergence_epsilon = convergence_epsilon
        self.max_iter = max_iter
        self.cache_size = cache_size
        self.verbose = verbose

    def fit(self, X, y):
        X = check_array(X, dtype=np.float64, order="C")
        y = np.asarray(y)

        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        if len(self.le_.classes_) != 2:
            raise ValueError(
                "MySVMNuClassifier only supports binary classification "
                f"(got {len(self.le_.classes_)} classes)."
            )
        self.classes_ = self.le_.classes_
        y_pm1 = np.where(y_enc == 1, 1.0, -1.0).astype(np.float64)

        gamma_val = _resolve_gamma(self.gamma, X)

        # Use NU_SVR on ±1 labels: the nu-pattern SVM in mySVM has
        # convergence issues; nu-SVR on ±1 targets is equivalent and reliable.
        # mySVM sets alpha_bound = C / n_samples, so scale C by n_samples to
        # get an effective alpha bound of 1.0 regardless of dataset size.
        self._model = _build_model(
            svm_type=_NU_SVC,
            C=float(len(X)), nu=self.nu, epsilon=0.0,
            kernel=self.kernel, gamma_val=gamma_val,
            degree=self.degree, coef0=self.coef0,
            biased=int(self.biased), balance_cost=0,
            convergence_epsilon=self.convergence_epsilon,
            max_iter=self.max_iter, cache_size=self.cache_size,
            verbose=int(self.verbose),
        )
        self._model.train(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y_pm1, dtype=np.float64),
        )
        self.n_support_ = self._model.n_sv
        self._gamma_val = gamma_val
        return self

    def decision_function(self, X):
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, order="C")
        return self._model.predict(np.ascontiguousarray(X, dtype=np.float64))

    def predict(self, X):
        df = self.decision_function(X)
        y_enc = np.where(df > 0, 1, 0)
        return self.le_.inverse_transform(y_enc)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MySVMNuRegressor – ν-SVM regression                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class MySVMNuRegressor(BaseEstimator, RegressorMixin):
    """ν-SVM regressor wrapping the mySVM library.

    Parameters
    ----------
    nu : float, default=0.5
        Upper bound on the fraction of training errors.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, default='rbf'
        Kernel type.
    gamma : float or {'scale', 'auto'}, default='scale'
        Kernel coefficient.
    degree : int, default=3
        Degree for polynomial kernel.
    coef0 : float, default=1.0
        Independent term in 'sigmoid'.
    biased : bool, default=True
        Use biased hyperplane.
    convergence_epsilon : float, default=1e-3
        Convergence criterion.
    max_iter : int, default=100000
        Maximum iterations.
    cache_size : int, default=256
        Kernel cache in MB.
    verbose : bool, default=False
        Print mySVM progress messages.
    """

    def __init__(self, nu=0.5, kernel="rbf", gamma="scale", degree=3,
                 coef0=1.0, biased=True, convergence_epsilon=1e-3,
                 max_iter=100000, cache_size=256, verbose=False):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.biased = biased
        self.convergence_epsilon = convergence_epsilon
        self.max_iter = max_iter
        self.cache_size = cache_size
        self.verbose = verbose

    def fit(self, X, y):
        X = check_array(X, dtype=np.float64, order="C")
        y = np.asarray(y, dtype=np.float64)

        # Standardise y so that alpha bounds (C/n_samples) are proportionate
        # to the target scale.  mySVM sets alpha_bound = C / n_samples; with
        # C = n_samples the effective bound is 1.0 regardless of dataset size.
        self._y_mean_ = y.mean()
        self._y_std_ = float(y.std()) or 1.0
        y_scaled = (y - self._y_mean_) / self._y_std_

        gamma_val = _resolve_gamma(self.gamma, X)

        self._model = _build_model(
            svm_type=_NU_SVR,
            C=float(len(X)), nu=self.nu, epsilon=0.1,
            kernel=self.kernel, gamma_val=gamma_val,
            degree=self.degree, coef0=self.coef0,
            biased=int(self.biased), balance_cost=0,
            convergence_epsilon=self.convergence_epsilon,
            max_iter=self.max_iter, cache_size=self.cache_size,
            verbose=int(self.verbose),
        )
        self._model.train(
            np.ascontiguousarray(X, dtype=np.float64),
            np.ascontiguousarray(y_scaled, dtype=np.float64),
        )
        self.n_support_ = self._model.n_sv
        self._gamma_val = gamma_val
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, order="C")
        y_scaled = self._model.predict(np.ascontiguousarray(X, dtype=np.float64))
        return y_scaled * self._y_std_ + self._y_mean_
