"""
LSVM - Lagrangian Support Vector Machine (Linear)

Precise Python/NumPy port of lsvm.m by Olvi L. Mangasarian and
David R. Musicant, University of Wisconsin-Madison, 2000.

Copyright (C) 2000 Olvi L. Mangasarian and David R. Musicant.
This software is free for academic and research use only.
For commercial use, contact musicant@cs.wisc.edu.
"""

import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def _pl(x):
    """PLUS function: max{x, 0}  —  exact port of MATLAB pl() helper."""
    return (x + np.abs(x)) / 2.0


class LSVM(BaseEstimator, ClassifierMixin):
    """
    Lagrangian Support Vector Machine (Linear).

    Solves a binary (+1 / -1) classification problem using an iterative
    algorithm inspired by an augmented Lagrangian formulation, with the
    Sherman-Morrison-Woodbury (SMW) formula for efficient matrix inversion.

    Precise port of ``lsvm.m`` (Mangasarian & Musicant, 2000).

    Parameters
    ----------
    nu : float or None, default=None
        Regularisation parameter. Defaults to ``1/m`` at fit time
        (where *m* is the number of training samples).
    tol : float, default=1e-5
        Convergence tolerance: stop when ``||x_new - x_old||_2 <= tol``.
    max_iter : int, default=100
        Maximum number of iterations.
    alpha : float or None, default=None
        Step-size parameter. Defaults to ``1.9/nu`` at fit time.
        Convergence requires ``0 < alpha < 2/nu``.
    perturb : float, default=0.0
        Perturb each entry of *A* by a uniform random value in
        ``[0, perturb)`` before training. Intended for highly degenerate
        problems (e.g. XOR). The random state is seeded at 22 to mirror
        the original ``rand('seed', 22)`` in the MATLAB source.
    normalize : bool, default=False
        Standardise features (zero mean, unit variance) before training
        using column-wise mean and standard deviation (ddof=1, matching
        MATLAB's ``std``). The same transform is applied during
        ``predict`` / ``decision_function``.
    verbose : bool, default=True
        Print CPU time, iteration count, and training accuracy after fit,
        matching the MATLAB ``disp`` calls.

    Attributes
    ----------
    w_ : ndarray of shape (n_features,)
        Coefficient vector of the separating hyperplane.
    gamma_ : float
        Threshold scalar of the separating hyperplane.
    n_iter_ : int
        Number of iterations executed.
    opt_cond_ : float
        Value of ``||x_new - x_old||_2`` at termination.
    time_ : float
        CPU time (seconds) consumed by the iterative loop.
    classes_ : ndarray of shape (2,)
        Always ``[-1, 1]``.
    """

    def __init__(
        self,
        nu=None,
        tol=1e-5,
        max_iter=100,
        alpha=None,
        perturb=0.0,
        normalize=False,
        verbose=True,
    ):
        self.nu = nu
        self.tol = tol
        self.max_iter = max_iter
        self.alpha = alpha
        self.perturb = perturb
        self.normalize = normalize
        self.verbose = verbose

    def fit(self, A, y):
        """
        Fit the LSVM model.

        Parameters
        ----------
        A : array-like of shape (m, n)
            Training data matrix.
        y : array-like of shape (m,)
            Class labels; every entry must be exactly +1 or -1.

        Returns
        -------
        self : LSVM
        """
        A, y = check_X_y(A, y)
        A = A.astype(float, copy=True)
        y = y.astype(float)

        # Validate labels (mirrors: checkall = diag(D)==1 | diag(D)==-1)
        if not np.all((y == 1.0) | (y == -1.0)):
            raise ValueError("Error in y: classes must be all 1 or -1.")

        # d replaces the MATLAB diagonal matrix D; operations D*v become d*v
        d = y
        m = A.shape[0]

        # Resolve defaults (MATLAB sentinel: -1 means "use default")
        nu = 1.0 / m if self.nu is None else float(self.nu)
        tol = float(self.tol)
        max_iter = int(self.max_iter)
        alpha = 1.9 / nu if self.alpha is None else float(self.alpha)

        # Sanity check on alpha
        if alpha > 2.0 / nu:
            print("Alpha is larger than 2/nu. Algorithm may not converge.")

        # Perturb if appropriate
        # MATLAB: rand('seed', 22); A = A + rand(size(A))*perturb
        if self.perturb:
            rng = np.random.RandomState(22)
            A += rng.rand(*A.shape) * self.perturb

        # Normalise if appropriate
        # MATLAB: avg=mean(A); dev=std(A);  (std uses ddof=1 by default)
        #         A = (A - avg(ones(m,1),:)) ./ dev(ones(m,1),:)
        self._avg_ = None
        self._dev_ = None
        if self.normalize:
            avg = np.mean(A, axis=0)
            dev = np.std(A, axis=0, ddof=1)
            if np.all(dev != 0.0):
                A = (A - avg) / dev
                self._avg_ = avg
                self._dev_ = dev
            else:
                print(
                    "Warning: Could not normalize matrix: "
                    "at least one column is constant."
                )

        m, n = A.shape
        e = np.ones(m)

        # H = D * [A, -e]
        # Each row i of [A, -e] is multiplied by the scalar d[i].
        # MATLAB: H = D * [A, -e]  where D is an m×m diagonal matrix.
        H = d[:, np.newaxis] * np.column_stack([A, -e])  # (m, n+1)

        start = time.process_time()

        # SMW intermediate matrix
        # MATLAB: K = H * inv(speye(n+1)/nu + H'*H)
        inner = np.eye(n + 1) / nu + H.T @ H   # (n+1, n+1)
        K = H @ np.linalg.inv(inner)            # (m, n+1)

        # Initial value
        # MATLAB: x = nu*(1 - K*(H'*e))
        # '1' is a scalar broadcast across the m-vector K*(H'*e)
        x = nu * (1.0 - K @ (H.T @ e))         # (m,)

        # y (old x) initialised so that ||y - x|| > tol on first entry
        # MATLAB: y = x + 1
        y_prev = x + 1.0

        iter_count = 0

        # MATLAB: while iter < maxIter & norm(y-x) > tol
        while iter_count < max_iter and np.linalg.norm(y_prev - x) > tol:
            # MATLAB: z = (1 + pl(((x/nu + H*(H'*x)) - alpha*x) - 1))
            z = 1.0 + _pl((x / nu + H @ (H.T @ x) - alpha * x) - 1.0)
            y_prev = x
            # MATLAB: x = nu*(z - K*(H'*z))
            x = nu * (z - K @ (H.T @ z))
            iter_count += 1

        elapsed = time.process_time() - start

        # Derive classifier parameters
        # MATLAB: w = A'*D*x  →  A.T @ (d*x)
        self.w_ = A.T @ (d * x)                 # (n,)
        # MATLAB: gamma = -e'*D*x  →  -(e @ (d*x))
        self.gamma_ = float(-(e @ (d * x)))

        # MATLAB: sum(D*(A*w-gamma)>0)/m
        train_acc = float(np.sum((d * (A @ self.w_ - self.gamma_)) > 0) / m)

        self.n_iter_ = iter_count
        self.opt_cond_ = float(np.linalg.norm(x - y_prev))
        self.time_ = elapsed
        self.classes_ = np.array([-1, 1])

        if self.verbose:
            print(f"Running time (CPU secs) = {elapsed:g}")
            print(f"Number of iterations = {iter_count:d}")
            print(f"Training accuracy = {train_acc:g}")

        return self

    def decision_function(self, A):
        """
        Compute raw decision scores: ``A @ w_ - gamma_``.

        Parameters
        ----------
        A : array-like of shape (m, n)

        Returns
        -------
        scores : ndarray of shape (m,)
            Positive → class +1, negative → class -1.
        """
        check_is_fitted(self)
        A = check_array(A).astype(float)
        if self._avg_ is not None:
            A = (A - self._avg_) / self._dev_
        return A @ self.w_ - self.gamma_

    def predict(self, A):
        """
        Predict class labels for samples in *A*.

        Parameters
        ----------
        A : array-like of shape (m, n)

        Returns
        -------
        y_pred : ndarray of shape (m,)
            Predicted labels in {-1, +1}.
        """
        return np.where(self.decision_function(A) > 0, 1.0, -1.0)
