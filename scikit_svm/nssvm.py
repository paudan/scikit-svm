"""
NSSVM - Nonlinear (Kernel) Smooth Support Vector Machine

Precise Python/NumPy port of n_ssvm.m by Olvi L. Mangasarian and
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

from ._ssvm_base import _core, _est_mu, _est_nu_long, _est_nu_short, _rec_kernel


class NSSVM(BaseEstimator, ClassifierMixin):
    """
    Nonlinear (Kernel) Smooth Support Vector Machine.

    Transforms the input via a reduced Gaussian kernel map
    ``K[i,j] = exp(-mu * ||x_i - b_j||^2)`` (where b_j are randomly
    selected basis vectors), then applies the same Newton-based SSVM
    optimisation as ``SSVM``.

    Precise port of ``n_ssvm.m`` (Mangasarian & Musicant, 2000).

    Parameters
    ----------
    nu : float, 'easy', or None, default=None
        Regularisation parameter.

        * ``None`` (default) — estimate via eigendecomposition
          (``EstNuLong``, the "hard" method).
        * ``'easy'`` — quick estimate: ``n_features / ||X||_F^2``.
        * positive float — use directly.

    mu : float or None, default=None
        Gaussian kernel bandwidth.  When ``None``, estimated from data as
        ``1 / (1 + ||mean_pos + mean_neg||^2)`` (``EstMu`` in MATLAB).
    reduce_rate : float, default=1.0
        Fraction of training points to use as kernel basis vectors
        (``rr`` in MATLAB).  ``1.0`` uses all points; ``0.5`` uses half.
    use_armijo : bool, default=True
        ``True`` — Armijo backtracking line search (MATLAB default).
        ``False`` — pure Newton step.
    tol : float, default=1e-5
        Convergence threshold.
        Mirrors MATLAB default ``tol = 10e-6 = 1e-5``.
    max_iter : int, default=30
        Maximum Newton iterations.
    random_state : int, RandomState, or None, default=None
        Controls both the data permutation and the random basis selection.
    verbose : bool, default=False
        Print iteration count and elapsed time after fit.

    Attributes
    ----------
    w_ : ndarray of shape (n_basis,)
        Kernel expansion coefficients.
    gamma_ : float
        Threshold.
    Abar_ : ndarray of shape (n_basis, n_features)
        Basis vectors used to compute the kernel map.
    mu_ : float
        Effective kernel bandwidth used.
    nu_ : float
        Effective regularisation parameter used.
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
        mu=None,
        reduce_rate=1.0,
        use_armijo=True,
        tol=1e-5,
        max_iter=30,
        random_state=None,
        verbose=False,
    ):
        self.nu          = nu
        self.mu          = mu
        self.reduce_rate = reduce_rate
        self.use_armijo  = use_armijo
        self.tol         = tol
        self.max_iter    = max_iter
        self.random_state = random_state
        self.verbose     = verbose

    # ------------------------------------------------------------------
    # Internal: kernel basis selection (calcKer in MATLAB)
    # ------------------------------------------------------------------

    def _calc_ker(self, X, mu, rng):
        """
        Select basis vectors and compute the kernel feature matrix.

        MATLAB calcKer:
            rrows = floor(rr * sm)
            indx  = rand(sm, 1);  [s1,s2] = sort(indx)
            Abar  = A(s2(1:rrows), :)'          % (n_features × rrows)
            A     = Rec_Kernel(A, Abar, mu)     % (sm × rrows)

        Here Abar is stored as (rrows, n_features) for sklearn convention.
        """
        sm = X.shape[0]
        rrows = max(1, int(np.floor(self.reduce_rate * sm)))
        idx   = np.argsort(rng.rand(sm))
        Abar  = X[idx[:rrows], :]                # (rrows, n_features)
        KX    = _rec_kernel(X, Abar, mu)          # (sm, rrows)
        return KX, Abar

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Fit the NSSVM model.

        Parameters
        ----------
        X : array-like of shape (m, n)
            Training data matrix.
        y : array-like of shape (m,)
            Class labels; every entry must be exactly +1 or -1.

        Returns
        -------
        self : NSSVM
        """
        X, y = check_X_y(X, y)
        X = X.astype(float, copy=True)
        y = y.astype(float)

        if not np.all((y == 1.0) | (y == -1.0)):
            raise ValueError("Error in y: classes must be all 1 or -1.")

        m, n = X.shape
        rng = check_random_state(self.random_state)

        # --- random permutation (MATLAB: r=randperm(...)) ---------------
        perm = rng.permutation(m)
        X, y = X[perm], y[perm]

        # --- mu estimation ----------------------------------------------
        # MATLAB: if nargin<6 → mu = EstMu(C,d)
        mu_eff = _est_mu(X, y) if self.mu is None else float(self.mu)

        # --- nu estimation ----------------------------------------------
        # MATLAB: if (nargin<5)|(nu==0) → EstNuLong; elseif nu==-1 → EstNuShort
        if self.nu is None or self.nu == 0:
            nu_eff = _est_nu_long(X, y)
        elif self.nu == 'easy' or self.nu == -1:
            nu_eff = _est_nu_short(X, y)
        else:
            nu_eff = float(self.nu)

        # --- balance check (MATLAB: if c1==c2 → nu=1; perturb C(3,:)) --
        pos_mask = y == 1.0
        neg_mask = y == -1.0
        if pos_mask.any() and neg_mask.any():
            c1 = X[pos_mask].mean(axis=0)
            c2 = X[neg_mask].mean(axis=0)
            if np.array_equal(c1, c2):
                nu_eff = 1.0
                row_norm = np.linalg.norm(X[2] - c1, ord=np.inf)
                X[2] += 0.001 * row_norm * np.ones(n)

        # --- kernel transform (calcKer) ---------------------------------
        # MATLAB (k==1 path):
        #   [kC, Cbar] = calcKer(C, rr, mu, output)
        #   w0 = zeros(size(kC,2), 1)
        KX, Abar = self._calc_ker(X, mu_eff, rng)
        n_basis  = KX.shape[1]
        w0       = np.zeros(n_basis)
        gamma0   = 0.0

        # --- run core optimisation --------------------------------------
        start = time.process_time()
        w, gamma, iteration = _core(
            KX, y, nu_eff, w0, gamma0,
            self.use_armijo, float(self.tol), int(self.max_iter),
        )
        elapsed = time.process_time() - start

        self.w_      = w
        self.gamma_  = float(gamma)
        self.Abar_   = Abar          # (n_basis, n_features) — stored for predict
        self.mu_     = mu_eff
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
        Compute raw decision scores for test samples.

        The kernel map ``K(X, Abar_)`` is applied first, then
        ``scores = K @ w_ - gamma_``.

        This mirrors the MATLAB ``correctness()`` call in ``n_ssvm.m``:
            k = Rec_Kernel(Atest, Abar, mu)
            p = sign(k * w - gamma)

        Parameters
        ----------
        X : array-like of shape (m_test, n_features)

        Returns
        -------
        scores : ndarray of shape (m_test,)
        """
        check_is_fitted(self)
        X = check_array(X).astype(float)
        K = _rec_kernel(X, self.Abar_, self.mu_)   # (m_test, n_basis)
        return K @ self.w_ - self.gamma_

    def predict(self, X):
        """
        Predict class labels for test samples.

        Parameters
        ----------
        X : array-like of shape (m_test, n_features)

        Returns
        -------
        y_pred : ndarray of shape (m_test,)
            Predicted labels in {-1, +1}.
        """
        return np.where(self.decision_function(X) > 0, 1.0, -1.0)
