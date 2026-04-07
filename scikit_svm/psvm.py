"""
PSVM and N-PSVM — Proximal Support Vector Machine Classifiers.

Python/NumPy/scikit-learn port of ``psvm_v1.1`` by Glenn Fung,
University of Wisconsin-Madison, 2001–2003.

**Linear PSVM** (``PSVMClassifier``) solves a regularised least-squares
problem in the primal:

    (I/ν + Hᵀ H) v = Hᵀ d,     H = [A  −1]

yielding hyperplane  ``w·x − γ = 0``  where  ``[w; γ] = v``.

**Nonlinear N-PSVM** (``NPSVMClassifier``) maps data through a
reduced RBF kernel (Nyström-style approximation) and then applies the
same linear solver.  The reduced kernel basis is chosen randomly:

    K[i,j] = exp(−μ ‖A_i − B_j‖²),     B = random subset of training rows

References
----------
Fung, G., & Mangasarian, O. L. (2001). Proximal support vector machine
classifiers.  *Proceedings of KDD 2001*, 77–86.
"""

import time

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


# ── Helpers ───────────────────────────────────────────────────────────────────

def _estimate_nu_long(H, d, rng, max_samples=200):
    """Estimate ν by iterative eigenvalue method  (``EstNuLong`` in MATLAB).

    Parameters
    ----------
    H   : ndarray (m, p)
    d   : ndarray (m,) — labels ±1
    rng : numpy RandomState
    """
    m = H.shape[0]
    if m > max_samples:
        idx = rng.choice(m, size=max_samples, replace=False)
        H2, d2 = H[idx], d[idx]
    else:
        H2, d2 = H, d

    # Eigendecomposition of H2 @ H2'
    vals, vecs = np.linalg.eigh(H2 @ H2.T)  # ascending order
    vals = np.maximum(vals, 0.0)             # clip numerical negatives
    yt = d2 @ vecs                           # shape (p,)

    lamda = 1.0
    lamda_old = lamda + 1.0
    cnt = 0

    while abs(lamda_old - lamda) > 1e-3 and cnt < 100:
        cnt += 1
        lamda_old = lamda
        denom = vals + lamda
        nu1 = float(np.sum(lamda / denom))
        pr  = float(np.sum(vals / denom ** 2))
        ee  = float(np.sum(vals * yt ** 2 / denom ** 3))
        waw = float(np.sum(lamda ** 2 * yt ** 2 / denom ** 2))
        if pr * waw > 0.0:
            lamda = nu1 * ee / (pr * waw)
        else:
            break

    return 1.0 if (cnt >= 100 or lamda <= 0.0) else float(lamda)


def _estimate_nu_short(C):
    """Estimate ν via Frobenius norm  (``EstNuShort`` in MATLAB).

    Parameters
    ----------
    C : ndarray (m, p)
    """
    frob2 = float(np.sum(C ** 2))
    p = C.shape[1]
    return float(p) / frob2 if frob2 > 0.0 else 1.0


def _estimate_mu(X, y):
    """Estimate the RBF bandwidth μ  (``EstMu`` in MATLAB).

    μ = 1 / (1 + ‖c⁺ + c⁻‖²),  where c± are class means.
    """
    X_pos = X[y == 1.0]
    X_neg = X[y == -1.0]
    c_pos = X_pos.mean(axis=0) if len(X_pos) > 0 else np.zeros(X.shape[1])
    c_neg = X_neg.mean(axis=0) if len(X_neg) > 0 else np.zeros(X.shape[1])
    x = c_pos + c_neg
    return 1.0 / (1.0 + float(x @ x))


def _rbf_kernel(A, B, mu):
    """RBF kernel matrix  K[i,j] = exp(−μ ‖A_i − B_j‖²).

    Matches ``Rec_Kernel(A, B', mu)`` in MATLAB (MATLAB stores B transposed).

    Parameters
    ----------
    A : ndarray (m, d)
    B : ndarray (k, d)  — reference points (rows)
    mu : float

    Returns
    -------
    K : ndarray (m, k)
    """
    aa = np.sum(A ** 2, axis=1)      # (m,)
    bb = np.sum(B ** 2, axis=1)      # (k,)
    dist2 = aa[:, np.newaxis] + bb[np.newaxis, :] - 2.0 * (A @ B.T)
    return np.exp(-mu * np.maximum(dist2, 0.0))


def _build_HV(X, y, balance):
    """Build augmented matrix H = [X, −1] (optionally class-balanced) and v = Hᵀd.

    Matches ``HV()`` in MATLAB.
    """
    m = X.shape[0]
    H = np.column_stack([X, -np.ones(m)])   # (m, p)

    if balance:
        mm = np.ones(m)
        neg, pos = y == -1.0, y == 1.0
        if neg.any():
            mm[neg] = 1.0 / neg.sum()
        if pos.any():
            mm[pos] = 1.0 / pos.sum()
        mm = np.sqrt(mm)
        H = mm[:, np.newaxis] * H
        v = H.T @ (mm * y)
    else:
        v = H.T @ y

    return H, v


def _psvm_core(H, v, nu):
    """Solve (I/ν + Hᵀ H) sol = v.

    Returns w (shape (p−1,)) and gamma (scalar).
    Matches ``core()`` in both ``psvm.m`` and ``n_psvm.m``.
    """
    p = H.shape[1]
    A_sys = np.eye(p) / nu + H.T @ H
    try:
        sol = linalg.solve(A_sys, v, assume_a="pos")
    except (linalg.LinAlgError, np.linalg.LinAlgError):
        sol = linalg.lstsq(A_sys, v, lapack_driver="gelsd")[0]
    return sol[:-1].copy(), float(sol[-1])


# ── Base class ────────────────────────────────────────────────────────────────

class _PSVMBase(ClassifierMixin, BaseEstimator):
    """Shared sklearn boilerplate for PSVMClassifier and NPSVMClassifier."""

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def _encode(self, y_raw):
        """Map two-class labels to ±1; store inverse mapping."""
        classes = np.unique(y_raw)
        self.classes_ = classes
        if len(classes) < 2:
            raise ValueError(
                f"PSVM requires at least 2 classes; got 1 class: {classes}."
            )
        self._le_neg = classes[0]
        self._le_pos = classes[1]
        y = np.where(y_raw == self._le_neg, -1.0, 1.0)
        return y

    def _decode(self, y_int):
        """Map ±1 back to original label values."""
        return np.where(y_int > 0.0, self._le_pos, self._le_neg)

    def _validate_y(self, y_raw):
        check_classification_targets(y_raw)
        y_type = type_of_target(y_raw, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                "Only binary classification is supported. "
                f"The type of the target is {y_type}."
            )

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array-like (n_test, n_features)

        Returns
        -------
        y_pred : ndarray (n_test,)
        """
        return self._decode(np.sign(self.decision_function(X)))


# ── PSVMClassifier ────────────────────────────────────────────────────────────

class PSVMClassifier(_PSVMBase):
    """Proximal Support Vector Machine (linear, primal).

    Solves the binary classification problem

        min_{w,γ}  ½ (νw·w + νγ² + Σᵢ(yᵢ(w·xᵢ−γ) − 1)²)

    as a single linear system  (I/ν + HᵀH) [w; γ] = Hᵀy,  where
    H = [X, −1].  No QP is needed.

    Parameters
    ----------
    nu : float, default=0
        Regularization parameter.  Special values:

        * ``0``  — estimated automatically via eigenvalue method
          (``EstNuLong`` in MATLAB; slower but more accurate).
        * ``-1`` — estimated via fast Frobenius-norm heuristic
          (``EstNuShort`` in MATLAB).
        * any positive float — used directly.
    balance : bool, default=False
        Weight samples by inverse class frequency.  Useful when classes
        are highly imbalanced.
    random_state : int or None, default=None
        Controls randomness in the ν estimator (subsampling when m > 200).
    verbose : bool, default=False

    Attributes
    ----------
    w_ : ndarray (n_features,)
        Normal vector of the separating hyperplane.
    gamma_ : float
        Bias of the separating hyperplane.
    nu_ : float
        The ν value used (estimated or provided).
    train_time_ : float
        CPU time consumed by the solver.
    classes_ : ndarray (2,)
    n_features_in_ : int

    Notes
    -----
    * Labels can be any two distinct values; they are encoded to ±1.
    * Decision function: ``f(x) = x·w − γ``.  Positive → ``classes_[1]``.
    * Port of ``psvm.m`` v1.1 (Fung & Mangasarian, 2001).

    Examples
    --------
    >>> import numpy as np
    >>> from scikit_svm import PSVMClassifier
    >>> rng = np.random.RandomState(0)
    >>> X = np.vstack([rng.randn(50, 2) + 1, rng.randn(50, 2) - 1])
    >>> y = np.array([1]*50 + [-1]*50)
    >>> clf = PSVMClassifier(nu=0.1).fit(X, y)
    >>> clf.score(X, y)
    1.0
    """

    def __init__(self, nu=0, balance=False, random_state=None, verbose=False):
        self.nu = nu
        self.balance = balance
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the PSVM.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
        y : array-like (n_samples,) — binary labels (any two distinct values)

        Returns
        -------
        self
        """
        X, y_raw = check_X_y(X, y)
        self._validate_y(y_raw)
        y = self._encode(y_raw)
        self.n_features_in_ = X.shape[1]

        rng = np.random.RandomState(self.random_state)
        t0 = time.process_time()

        # Perturb if class centroids are identical (matches MATLAB guard)
        X = X.copy()
        c_pos = X[y == 1.0].mean(axis=0)
        c_neg = X[y == -1.0].mean(axis=0)
        if np.allclose(c_pos, c_neg) and X.shape[0] > 3:
            d3 = X[2] - c_pos
            norm_inf = np.max(np.abs(d3))
            X[2] = X[2] + 0.01 * norm_inf * np.ones(X.shape[1])

        H, v = _build_HV(X, y, self.balance)

        nu = float(self.nu)
        if nu == 0.0:
            nu = _estimate_nu_long(H, y, rng)
        elif nu < 0.0:
            nu = _estimate_nu_short(H)

        w, gamma = _psvm_core(H, v, nu)

        self.w_ = w
        self.gamma_ = gamma
        self.nu_ = nu
        self.train_time_ = time.process_time() - t0

        if self.verbose:
            acc = float(np.mean(np.sign(X @ w - gamma) == y) * 100)
            print(f"Training accuracy: {acc:.2f}%  |  "
                  f"nu={nu:.6g}  |  time={self.train_time_:.4f}s")

        return self

    def decision_function(self, X):
        """Raw decision scores  f(x) = x·w − γ.

        Parameters
        ----------
        X : array-like (n_test, n_features)

        Returns
        -------
        scores : ndarray (n_test,)
        """
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but "
                f"PSVMClassifier is expecting "
                f"{self.n_features_in_} features as input."
            )
        return X @ self.w_ - self.gamma_


# ── NPSVMClassifier ───────────────────────────────────────────────────────────

class NPSVMClassifier(_PSVMBase):
    """Nonlinear Proximal Support Vector Machine (reduced RBF kernel).

    Extends :class:`PSVMClassifier` to nonlinear classification by
    mapping data through a *reduced* RBF kernel (Nyström approximation):

        Φ(x)ⱼ = exp(−μ ‖x − b̄ⱼ‖²),   j = 1 … r

    where the ``r = floor(reduce_ratio · m)`` basis points ``{b̄ⱼ}`` are
    chosen randomly from the training set.  The linear PSVM is then
    solved in this feature space.

    Parameters
    ----------
    nu : float, default=0
        Same as :class:`PSVMClassifier`.  ``0`` → ``EstNuLong``;
        ``-1`` → ``EstNuShort``.
    mu : float, default=0.0
        RBF bandwidth.  ``0`` → estimated from data (``EstMu`` in MATLAB).
    reduce_ratio : float, default=1.0
        Fraction of training points to use as kernel basis (``rr``).
        Values < 1.0 give a *reduced* kernel for scalability.
    balance : bool, default=False
    random_state : int or None, default=None
    verbose : bool, default=False

    Attributes
    ----------
    w_ : ndarray (r,)  — coefficients in kernel feature space
    gamma_ : float
    nu_ : float
    mu_ : float
    X_bar_ : ndarray (r, n_features) — kernel basis (reference) points
    train_time_ : float
    classes_ : ndarray (2,)
    n_features_in_ : int

    Notes
    -----
    * Prediction: ``f(x) = K(x, X̄)·w − γ``  where
      ``K[i,j] = exp(−μ ‖x_i − X̄_j‖²)``.
    * Port of ``n_psvm.m`` v1.1 (Fung & Mangasarian, 2001).

    Examples
    --------
    >>> import numpy as np
    >>> from scikit_svm import NPSVMClassifier
    >>> rng = np.random.RandomState(0)
    >>> X = np.vstack([rng.randn(50, 2) + 1, rng.randn(50, 2) - 1])
    >>> y = np.array([1]*50 + [-1]*50)
    >>> clf = NPSVMClassifier(mu=0.5).fit(X, y)
    >>> clf.score(X, y)
    1.0
    """

    def __init__(self, nu=0, mu=0.0, reduce_ratio=1.0,
                 balance=False, random_state=None, verbose=False):
        self.nu = nu
        self.mu = mu
        self.reduce_ratio = reduce_ratio
        self.balance = balance
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the N-PSVM.

        Parameters
        ----------
        X : array-like (n_samples, n_features)
        y : array-like (n_samples,) — binary labels

        Returns
        -------
        self
        """
        X, y_raw = check_X_y(X, y)
        self._validate_y(y_raw)
        y = self._encode(y_raw)
        self.n_features_in_ = X.shape[1]
        m = X.shape[0]

        rng = np.random.RandomState(self.random_state)
        t0 = time.process_time()

        # Perturb if class centroids are identical
        X = X.copy()
        c_pos = X[y == 1.0].mean(axis=0)
        c_neg = X[y == -1.0].mean(axis=0)
        if np.allclose(c_pos, c_neg) and m > 3:
            d3 = X[2] - c_pos
            norm_inf = np.max(np.abs(d3))
            X[2] = X[2] + 0.01 * norm_inf * np.ones(X.shape[1])

        # Estimate mu if not provided
        mu = float(self.mu)
        if mu <= 0.0:
            mu = _estimate_mu(X, y)
        self.mu_ = mu

        # Estimate nu (on original features, before kernel transformation)
        nu = float(self.nu)
        if nu == 0.0:
            H_raw, _ = _build_HV(X, y, self.balance)
            nu = _estimate_nu_long(H_raw, y, rng)
        elif nu < 0.0:
            nu = _estimate_nu_short(X)
        self.nu_ = nu

        # Build reduced kernel features
        rr = float(self.reduce_ratio)
        r_rows = max(1, int(np.floor(rr * m)))
        idx = rng.choice(m, size=r_rows, replace=False)
        X_bar = X[idx].copy()          # (r, n_features) — basis points
        self.X_bar_ = X_bar

        K = _rbf_kernel(X, X_bar, mu)  # (m, r)

        # Fit linear PSVM in kernel feature space
        H, v = _build_HV(K, y, self.balance)
        w, gamma = _psvm_core(H, v, nu)

        self.w_ = w
        self.gamma_ = gamma
        self.train_time_ = time.process_time() - t0

        if self.verbose:
            acc = float(np.mean(np.sign(K @ w - gamma) == y) * 100)
            print(f"Training accuracy: {acc:.2f}%  |  "
                  f"nu={nu:.6g}  mu={mu:.6g}  r={r_rows}  |  "
                  f"time={self.train_time_:.4f}s")

        return self

    def decision_function(self, X):
        """Raw decision scores  f(x) = K(x, X̄)·w − γ.

        Parameters
        ----------
        X : array-like (n_test, n_features)

        Returns
        -------
        scores : ndarray (n_test,)
        """
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but "
                f"NPSVMClassifier is expecting "
                f"{self.n_features_in_} features as input."
            )
        K = _rbf_kernel(X, self.X_bar_, self.mu_)   # (n_test, r)
        return K @ self.w_ - self.gamma_
