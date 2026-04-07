"""
LSVMK - Lagrangian Support Vector Machine (Kernel)

Precise Python/NumPy port of lsvmk.m by Olvi L. Mangasarian and
David R. Musicant, University of Wisconsin-Madison, 2000.

Copyright (C) 2000 Olvi L. Mangasarian and David R. Musicant.
This software is free for academic and research use only.
For commercial use, contact musicant@cs.wisc.edu.
"""

import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def _pl(x):
    """PLUS function: max{x, 0}  —  exact port of MATLAB pl() helper."""
    return (x + np.abs(x)) / 2.0


class LSVMK(BaseEstimator, ClassifierMixin):
    """
    Lagrangian Support Vector Machine (Kernel).

    Solves a binary (+1 / -1) classification problem using an iterative
    algorithm inspired by an augmented Lagrangian formulation applied to
    a pre-computed (or internally computed) kernel matrix.

    Precise port of ``lsvmk.m`` (Mangasarian & Musicant, 2000).

    Parameters
    ----------
    kernel : str or callable, default='rbf'
        Kernel to use when computing the kernel matrix from raw data.
        Supported strings: ``'linear'``, ``'rbf'``, ``'poly'``,
        ``'sigmoid'``, ``'precomputed'``.  A callable must accept two
        arrays and return a kernel matrix.

        When ``kernel='precomputed'`` the array passed to ``fit`` is
        treated directly as the *m × m* kernel matrix (original MATLAB
        behaviour), and the array passed to ``predict`` /
        ``decision_function`` must be an *m_test × m_train* kernel
        matrix.
    nu : float or None, default=None
        Regularisation parameter. Defaults to ``1/m`` at fit time.
    tol : float, default=1e-5
        Convergence tolerance: stop when ``||u_new - u_old||_2 <= tol``.
    max_iter : int, default=100
        Maximum number of iterations.
    alpha : float or None, default=None
        Step-size parameter. Defaults to ``1.9/nu`` at fit time.
        Convergence requires ``0 < alpha < 2/nu``.
    gamma : float or None, default=None
        Kernel coefficient for ``'rbf'``, ``'poly'``, and ``'sigmoid'``.
        Passed directly to :func:`sklearn.metrics.pairwise.pairwise_kernels`.
        When ``None`` the sklearn default is used (``1 / n_features`` for
        RBF).
    degree : int, default=3
        Degree of the polynomial kernel (``'poly'``).
    coef0 : float, default=1.0
        Zero coefficient for ``'poly'`` and ``'sigmoid'`` kernels.
    verbose : bool, default=True
        Print CPU time, iteration count, and training accuracy after fit,
        matching the MATLAB ``disp`` calls.

    Attributes
    ----------
    dual_coef_ : ndarray of shape (m,)
        Dual-variable vector *u* returned by the algorithm.
    d_ : ndarray of shape (m,)
        Training labels (±1) stored for use in ``predict``.
    X_fit_ : ndarray of shape (m, n) or None
        Training data; ``None`` when ``kernel='precomputed'``.
    n_iter_ : int
        Number of iterations executed.
    opt_cond_ : float
        Value of ``||u_new - u_old||_2`` at termination.
    time_ : float
        CPU time (seconds) consumed by the iterative loop.
    classes_ : ndarray of shape (2,)
        Always ``[-1, 1]``.
    """

    def __init__(
        self,
        kernel="rbf",
        nu=None,
        tol=1e-5,
        max_iter=100,
        alpha=None,
        gamma=None,
        degree=3,
        coef0=1.0,
        verbose=True,
    ):
        self.kernel = kernel
        self.nu = nu
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = alpha
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _kernel_params(self):
        """Collect extra keyword arguments for pairwise_kernels."""
        params = {}
        if self.kernel in ("rbf", "poly", "sigmoid"):
            if self.gamma is not None:
                params["gamma"] = self.gamma
        if self.kernel in ("poly", "sigmoid"):
            params["coef0"] = self.coef0
        if self.kernel == "poly":
            params["degree"] = self.degree
        return params

    def _compute_kernel(self, X, Y=None):
        """Return the kernel matrix K(X, Y)."""
        if callable(self.kernel):
            return self.kernel(X, Y) if Y is not None else self.kernel(X, X)
        return pairwise_kernels(X, Y, metric=self.kernel, **self._kernel_params())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Fit the LSVMK model.

        Parameters
        ----------
        X : array-like of shape (m, n) or (m, m)
            Training data, or a pre-computed kernel matrix when
            ``kernel='precomputed'``.
        y : array-like of shape (m,)
            Class labels; every entry must be exactly +1 or -1.

        Returns
        -------
        self : LSVMK
        """
        # Input validation
        if self.kernel == "precomputed":
            X = check_array(X)
            y = np.asarray(y, dtype=float)
        else:
            X, y = check_X_y(X, y)
            y = y.astype(float)

        # Validate labels (mirrors: checkall = diag(D)==1 | diag(D)==-1)
        if not np.all((y == 1.0) | (y == -1.0)):
            raise ValueError("Error in y: classes must be all 1 or -1.")

        # d replaces the MATLAB diagonal matrix D; operations D*v become d*v
        d = y
        m = X.shape[0]

        # Resolve defaults
        nu = 1.0 / m if self.nu is None else float(self.nu)
        tol = float(self.tol)
        max_iter = int(self.max_iter)
        alpha = 1.9 / nu if self.alpha is None else float(self.alpha)

        # Sanity check on alpha
        if alpha > 2.0 / nu:
            print("Alpha is larger than 2/nu. Algorithm may not converge.")

        # Compute (or adopt) the kernel matrix
        # MATLAB: KM is passed in directly (precomputed case)
        if self.kernel == "precomputed":
            KM = X.astype(float)
            self.X_fit_ = None
        else:
            KM = self._compute_kernel(X).astype(float)
            self.X_fit_ = X

        e = np.ones(m)

        start = time.process_time()

        # Build Q and its inverse
        # MATLAB: Q = I/nu + D*KM*D
        #   D*KM*D  where D = diag(d):  entry (i,j) = d[i]*KM[i,j]*d[j]
        #   In NumPy: d[:, None] * KM * d[None, :]
        Q = np.eye(m) / nu + d[:, np.newaxis] * KM * d[np.newaxis, :]  # (m, m)
        # MATLAB: P = inv(Q)
        P = np.linalg.inv(Q)                                             # (m, m)

        # Initial value
        # MATLAB: u = P*e
        u = P @ e                                                        # (m,)

        # oldu initialised so that ||oldu - u|| > tol on first entry
        # MATLAB: oldu = u + 1
        oldu = u + 1.0

        iter_count = 0

        # MATLAB: while iter < maxIter & norm(oldu-u) > tol
        while iter_count < max_iter and np.linalg.norm(oldu - u) > tol:
            oldu = u
            # MATLAB: u = P*(1 + pl(Q*u - 1 - alpha*u))
            u = P @ (1.0 + _pl(Q @ u - 1.0 - alpha * u))
            iter_count += 1

        elapsed = time.process_time() - start

        # MATLAB: sum(D*KM*D*u > 0) / m
        #   D*KM*D*u  =  d * (KM @ (d*u))  element-wise
        train_acc = float(np.sum((d * (KM @ (d * u))) > 0) / m)

        self.dual_coef_ = u
        self.d_ = d
        self.n_iter_ = iter_count
        self.opt_cond_ = float(np.linalg.norm(u - oldu))
        self.time_ = elapsed
        self.classes_ = np.array([-1, 1])

        if self.verbose:
            print(f"Running time (CPU secs) = {elapsed:g}")
            print(f"Number of iterations = {iter_count:d}")
            print(f"Training accuracy = {train_acc:g}")

        return self

    def decision_function(self, X):
        """
        Compute raw decision scores for samples in *X*.

        For training data the score of sample *i* is
        ``(KM @ (d * u))[i]``; a positive score predicts class +1.

        For new samples the score is ``K(X, X_train) @ (d * u)``.

        Parameters
        ----------
        X : array-like of shape (m_test, n) or (m_test, m_train)
            Test data, or a pre-computed kernel matrix
            (rows = test samples, columns = training samples)
            when ``kernel='precomputed'``.

        Returns
        -------
        scores : ndarray of shape (m_test,)
        """
        check_is_fitted(self)

        if self.kernel == "precomputed":
            # User supplies K(X_test, X_train) directly
            KM_test = check_array(X).astype(float)
        else:
            X = check_array(X).astype(float)
            KM_test = self._compute_kernel(X, self.X_fit_).astype(float)

        # score = K_test @ (d * u)
        # Equivalent to the training formula (KM @ (d*u))[i] for known samples
        return KM_test @ (self.d_ * self.dual_coef_)

    def predict(self, X):
        """
        Predict class labels for samples in *X*.

        Parameters
        ----------
        X : array-like of shape (m_test, n) or (m_test, m_train)
            Test data or pre-computed kernel matrix (see
            ``decision_function``).

        Returns
        -------
        y_pred : ndarray of shape (m_test,)
            Predicted labels in {-1, +1}.
        """
        return np.where(self.decision_function(X) > 0, 1.0, -1.0)
