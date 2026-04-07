"""
SSVM - Smooth Support Vector Machine (Linear)

Precise Python/NumPy port of ssvm.m by Olvi L. Mangasarian and
David R. Musicant, University of Wisconsin-Madison, 2000.

Copyright (C) 2000 Olvi L. Mangasarian and David R. Musicant.
This software is free for academic and research use only.
For commercial use, contact musicant@cs.wisc.edu.
"""

import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._ssvm_base import _core, _est_nu_long, _est_nu_short


class SSVM(BaseEstimator, ClassifierMixin):
    """
    Smooth Support Vector Machine (Linear).

    Minimises the SSVM objective via Newton iterations with an optional
    Armijo line search.  Uses the limit of the smoothed Hessian (as the
    smoothing parameter α → ∞), exploiting sparsity of the active set.

    Precise port of ``ssvm.m`` (Mangasarian & Musicant, 2000).

    Parameters
    ----------
    nu : float, 'easy', or None, default=None
        Regularisation parameter.

        * ``None`` (default) — estimate via eigendecomposition
          (``EstNuLong``, the "hard" method).
        * ``'easy'`` — quick estimate: ``n_features / ||X||_F^2``
          (``EstNuShort``).
        * positive float — use directly.

    use_armijo : bool, default=True
        ``True``  — Armijo backtracking line search (MATLAB default,
        ``step_size=1``).
        ``False`` — pure Newton step (``step_size != 1`` in MATLAB).
    tol : float, default=1e-7
        Convergence threshold.  The loop stops when ``||z||^2 <= tol``
        or when the first-order condition ``||grad||^2 <= tol`` holds.
        Mirrors MATLAB default ``tol = 10e-8 = 1e-7``.
    max_iter : int, default=1000
        Maximum number of Newton iterations.
    random_state : int, RandomState, or None, default=None
        Seed for the data permutation applied before fitting (mirrors
        ``randperm`` in the MATLAB source).
    verbose : bool, default=False
        Print iteration count and elapsed time after fit.

    Attributes
    ----------
    w_ : ndarray of shape (n_features,)
        Normal vector of the separating hyperplane.
    gamma_ : float
        Threshold of the separating hyperplane.
    nu_ : float
        Effective nu value used (estimated or supplied).
    n_iter_ : int
        Number of Newton iterations executed.
    time_ : float
        CPU time (seconds) of the optimisation loop.
    classes_ : ndarray of shape (2,)
        Always ``[-1, 1]``.
    """

    def __init__(
        self,
        nu=None,
        use_armijo=True,
        tol=1e-7,
        max_iter=1000,
        random_state=None,
        verbose=False,
    ):
        self.nu = nu
        self.use_armijo = use_armijo
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fit the SSVM model.

        Parameters
        ----------
        X : array-like of shape (m, n)
            Training data matrix.
        y : array-like of shape (m,)
            Class labels; every entry must be exactly +1 or -1.

        Returns
        -------
        self : SSVM
        """
        X, y = check_X_y(X, y)
        X = X.astype(float, copy=True)
        y = y.astype(float)

        if not np.all((y == 1.0) | (y == -1.0)):
            raise ValueError("Error in y: classes must be all 1 or -1.")

        m, n = X.shape

        # --- random permutation (MATLAB: r=randperm(...); d=d(r,:); C=C(r,:)) ---
        rng = check_random_state(self.random_state)
        perm = rng.permutation(m)
        X, y = X[perm], y[perm]

        # --- nu estimation ------------------------------------------------------
        # MATLAB: if (nargin<4)|(nu==0) → EstNuLong;  elseif nu==-1 → EstNuShort
        if self.nu is None or self.nu == 0:
            nu_eff = _est_nu_long(X, y)
        elif self.nu == 'easy' or self.nu == -1:
            nu_eff = _est_nu_short(X, y)
        else:
            nu_eff = float(self.nu)

        # --- balance check (MATLAB: if c1==c2 → nu=1; perturb C(3,:)) ----------
        # If both class centroids are identical, force nu=1 and perturb row 2.
        pos_mask = y == 1.0
        neg_mask = y == -1.0
        if pos_mask.any() and neg_mask.any():
            c1 = X[pos_mask].mean(axis=0)
            c2 = X[neg_mask].mean(axis=0)
            if np.array_equal(c1, c2):
                nu_eff = 1.0
                # MATLAB: C(3,:) = C(3,:) + .001*norm(C(3,:)-c1,inf)*ones(1,sn)
                row_norm = np.linalg.norm(X[2] - c1, ord=np.inf)
                X[2] += 0.001 * row_norm * np.ones(n)

        # --- initial point ------------------------------------------------------
        w0     = np.zeros(n)
        gamma0 = 0.0

        # --- run core optimisation ----------------------------------------------
        start = time.process_time()
        w, gamma, iteration = _core(
            X, y, nu_eff, w0, gamma0,
            self.use_armijo, float(self.tol), int(self.max_iter),
        )
        elapsed = time.process_time() - start

        self.w_      = w
        self.gamma_  = float(gamma)
        self.nu_     = nu_eff
        self.n_iter_ = iteration
        self.time_   = elapsed
        self.classes_ = np.array([-1.0, 1.0])

        if self.verbose:
            print(f"Number of Iterations: {iteration:d}")
            print(f"Elapse time: {elapsed:10.2f}")

        return self

    def decision_function(self, X):
        """
        Compute raw decision scores: ``X @ w_ - gamma_``.

        Parameters
        ----------
        X : array-like of shape (m, n)

        Returns
        -------
        scores : ndarray of shape (m,)
            Positive → class +1, negative → class -1.
        """
        check_is_fitted(self)
        X = check_array(X).astype(float)
        return X @ self.w_ - self.gamma_

    def predict(self, X):
        """
        Predict class labels for samples in *X*.

        Parameters
        ----------
        X : array-like of shape (m, n)

        Returns
        -------
        y_pred : ndarray of shape (m,)
            Predicted labels in {-1, +1}.
        """
        # MATLAB correctness: p = sign(AA*w - gamma)
        return np.where(self.decision_function(X) > 0, 1.0, -1.0)
