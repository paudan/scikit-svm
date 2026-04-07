"""
LSSVM - Least Squares Support Vector Machine

Python/NumPy/scikit-learn port of LSSVMlab v1.8 by Suykens et al., KU Leuven (2011).

Reference:
    J.A.K. Suykens, T. Van Gestel, J. De Brabanter, B. De Moor, J. Vandewalle,
    Least Squares Support Vector Machines, World Scientific, Singapore, 2002.
    http://www.esat.kuleuven.be/sista/lssvmlab

Copyright notice: Free for academic/research use. Commercial use requires
contact with the original authors.
"""

import warnings
import numpy as np
from scipy import linalg, optimize, stats
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.utils.multiclass import unique_labels

__all__ = ["LSSVMClassifier", "LSSVMRegressor"]


# ─── Kernel functions ──────────────────────────────────────────────────────────

def _rbf_sq_dists(X, Xt=None):
    """Compute squared Euclidean distances ||x_i - x_j||^2."""
    h = np.sum(X ** 2, axis=1, keepdims=True)
    if Xt is None:
        return h + h.T - 2.0 * (X @ X.T)
    ht = np.sum(Xt ** 2, axis=1, keepdims=True)
    return h + ht.T - 2.0 * (X @ Xt.T)


def _kernel_matrix(X, kernel, kernel_pars, Xt=None):
    """Compute kernel matrix K(X, Xt) [or K(X, X) when Xt is None].

    Mirrors ``kernel_matrix.m`` from LSSVMlab.

    Parameters
    ----------
    X          : (n, d) training points
    kernel     : 'rbf' | 'linear' | 'poly'
    kernel_pars: scalar or array-like
        RBF: sigma2 (bandwidth, sig2 in MATLAB)
        poly: [coef0, degree]  (MATLAB: (x'y + t)^d)
    Xt         : (nt, d) test points, optional

    Returns
    -------
    K : (n, n) or (n, nt) kernel matrix
    """
    if kernel == "rbf":
        sig2 = float(np.asarray(kernel_pars).ravel()[0])
        omega = _rbf_sq_dists(X, Xt)
        return np.exp(-omega / (2.0 * sig2))

    elif kernel == "linear":
        if Xt is None:
            return X @ X.T
        return X @ Xt.T

    elif kernel == "poly":
        kp = np.asarray(kernel_pars).ravel()
        t = float(kp[0])
        d = int(kp[1])
        if Xt is None:
            return (X @ X.T + t) ** d
        return (X @ Xt.T + t) ** d

    else:
        raise ValueError(
            f"Unknown kernel '{kernel}'. Supported: 'rbf', 'linear', 'poly'."
        )


def _kernel_from_precomputed(raw, kernel, kernel_pars):
    """Build kernel from pre-computed raw distance/dot-product matrix.

    For RBF, ``raw`` is the squared-distance matrix (avoids recomputing X@X').
    For linear/poly, ``raw`` is already the inner-product matrix.
    """
    if kernel == "rbf":
        sig2 = float(np.asarray(kernel_pars).ravel()[0])
        return np.exp(-raw / (2.0 * sig2))
    elif kernel == "linear":
        return raw
    elif kernel == "poly":
        kp = np.asarray(kernel_pars).ravel()
        t, d = float(kp[0]), int(kp[1])
        return (raw + t) ** d
    else:
        raise ValueError(f"Unknown kernel '{kernel}'.")


# ─── Kernel PCA ────────────────────────────────────────────────────────────────

def _kpca(X, kernel, kernel_pars):
    """Kernel PCA on the centered kernel matrix.

    Mirrors ``kpca.m`` (etype='eig') from LSSVMlab, including the
    (N-1)-scaling and Rscores normalisation used by the Bayesian framework.

    Returns
    -------
    eigvals   : (k,) positive eigenvalues λ (scaled by N-1), descending
    Rscores   : (n, k) eigenvectors normalised so that R[:,i]' * λ_i * R[:,i] = 1
    peff      : (k,) indices into the full (n,) eigenvalue vector
    """
    n = X.shape[0]
    K = _kernel_matrix(X, kernel, kernel_pars)

    # Centre: K_c = K - 1/N * ones*K - 1/N * K*ones + 1/N² * ones*K*ones
    row_mean = K.mean(axis=1, keepdims=True)   # (n, 1)
    grand_mean = K.mean()
    Kc = K - row_mean - row_mean.T + grand_mean
    Kc = (Kc + Kc.T) * 0.5  # symmetrise

    eigvals, eigvecs = linalg.eigh(Kc)          # ascending order
    idx = np.argsort(eigvals)[::-1]             # descending
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Scale by (N-1) — matches MATLAB: bay.eigvals = bay.eigvals.*(N-1)
    eigvals_sc = eigvals * (n - 1)

    tol = 1000.0 * np.finfo(float).eps
    peff = np.where(eigvals_sc > tol)[0]

    lam = eigvals_sc[peff]
    V = eigvecs[:, peff]

    # Normalise: R[:,i] = V[:,i] / sqrt(V[:,i]' * λ_i * V[:,i])
    Rscores = np.empty_like(V)
    for i in range(len(peff)):
        norm2 = V[:, i] @ (lam[i] * V[:, i])
        Rscores[:, i] = V[:, i] / np.sqrt(max(norm2, 1e-30))

    return lam, Rscores, peff


# ─── Bayesian results container ────────────────────────────────────────────────

class _BayResult:
    """Holds intermediate and final results of Bayesian LS-SVM inference."""
    eigvals = None   # (k,) positive eigenvalues (scaled)
    Rscores = None   # (n, k) normalised eigenvectors
    peff = None      # (k,) indices of positive eigvals
    Neff = 0
    costL1 = None
    Ew = None
    Ed = None
    mu = None
    zeta = None
    Geff = None
    costL2 = None
    DcostL2 = None
    optimal = None
    costL3 = None


# ─── Base class ────────────────────────────────────────────────────────────────

class _LSSVMBase(BaseEstimator):
    """Internal base class for Least Squares SVM (Suykens et al., 2002).

    Implements the core LS-SVM training/prediction together with the
    Bayesian inference framework and cross-validation tuning.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        sigma2: float = 1.0,
        gamma: float = None,
        degree: int = 3,
        coef0: float = 1.0,
        preprocess: bool = True,
    ):
        self.C = C
        self.kernel = kernel
        self.sigma2 = sigma2
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.preprocess = preprocess

    # ── derived kernel properties ──────────────────────────────────────────────

    @property
    def _sig2(self) -> float:
        """Effective RBF bandwidth (sigma² in MATLAB notation)."""
        if self.gamma is not None:
            return 1.0 / (2.0 * float(self.gamma))
        return float(self.sigma2)

    @property
    def _kpars(self) -> np.ndarray:
        """Kernel parameter array passed to ``_kernel_matrix``."""
        if self.kernel == "rbf":
            return np.array([self._sig2])
        if self.kernel == "linear":
            return np.array([])
        if self.kernel == "poly":
            return np.array([float(self.coef0), float(self.degree)])
        return np.array([self._sig2])

    # ── preprocessing ─────────────────────────────────────────────────────────

    def _fit_scale_X(self, X: np.ndarray) -> np.ndarray:
        """Compute and apply zero-mean / unit-std scaling to X."""
        if self.preprocess:
            self._x_mean = X.mean(axis=0)
            self._x_std = X.std(axis=0, ddof=1)
            self._x_std[self._x_std < 1e-12] = 1.0
        else:
            self._x_mean = np.zeros(X.shape[1])
            self._x_std = np.ones(X.shape[1])
        return (X - self._x_mean) / self._x_std

    def _scale_X(self, X: np.ndarray) -> np.ndarray:
        return (X - self._x_mean) / self._x_std

    def _fit_scale_y(self, y: np.ndarray) -> np.ndarray:
        """Regression only: zero-mean / unit-std scaling for y."""
        if self.preprocess:
            self._y_mean = y.mean()
            self._y_std = y.std()
            if self._y_std < 1e-12:
                self._y_std = 1.0
        else:
            self._y_mean = 0.0
            self._y_std = 1.0
        return (y - self._y_mean) / self._y_std

    def _unscale_y(self, y: np.ndarray) -> np.ndarray:
        return y * self._y_std + self._y_mean

    # ── kernel helpers ─────────────────────────────────────────────────────────

    def _K_train(self) -> np.ndarray:
        return _kernel_matrix(self._Xp, self.kernel, self._kpars)

    def _K_test(self, Xp: np.ndarray) -> np.ndarray:
        """K(X_train, X_test), shape (n_train, n_test)."""
        return _kernel_matrix(self._Xp, self.kernel, self._kpars, Xp)

    def _raw_omega(self, Xp: np.ndarray) -> np.ndarray:
        """Pre-compute distance/dot matrix for fast CV (avoids recomputing X@X')."""
        if self.kernel == "rbf":
            return _rbf_sq_dists(Xp)
        return Xp @ Xp.T     # works for linear and poly

    # ── core LS-SVM solve ──────────────────────────────────────────────────────

    def _solve(self, K: np.ndarray, y: np.ndarray):
        """Solve the LS-SVM saddle-point system.

        Solves::

            (K + I/C) α + e·b = y,    e'α = 0

        via two back-solves (mirrors ``lssvmMATLAB.m``).

        Returns
        -------
        alpha : (n,) dual variables
        b     : float bias term
        """
        n = K.shape[0]
        H = K + np.eye(n) / self.C
        e = np.ones(n)
        rhs = np.column_stack([y, e])
        try:
            sol = linalg.solve(H, rhs, assume_a="pos")
        except (linalg.LinAlgError, np.linalg.LinAlgError):
            sol = linalg.lstsq(H, rhs, lapack_driver="gelsd")[0]
        v, nu = sol[:, 0], sol[:, 1]
        s = e @ nu
        b = float((nu @ y) / s)
        alpha = v - nu * b
        return alpha, b

    # ── smoother matrix ────────────────────────────────────────────────────────

    def _smoother(self, Xt_proc: np.ndarray = None) -> np.ndarray:
        """Compute smoother matrix  S  such that  ŷ = S · y_train.

        Mirrors ``smootherlssvm.m``.

        Parameters
        ----------
        Xt_proc : preprocessed test points; if None return the training smoother.

        Returns
        -------
        S : (n, n) or (nt, n)
        """
        K = self._K_train()
        n = K.shape[0]
        Z = np.linalg.pinv(K + np.eye(n) / self.C)
        c = Z.sum()
        J = np.ones((n, n)) / c

        if Xt_proc is None:
            return K @ (Z - Z @ J @ Z) + J @ Z
        Kt_T = self._K_test(Xt_proc).T   # (nt, n)
        J1 = np.ones((Xt_proc.shape[0], n)) / c
        return Kt_T @ (Z - Z @ J @ Z) + J1 @ Z

    # ── fast L-fold cross-validation ──────────────────────────────────────────

    def _cv_cost(
        self,
        C: float,
        kpars: np.ndarray,
        y_proc: np.ndarray,
        omega_raw: np.ndarray,
        n_folds: int,
        cost_fn,
    ) -> float:
        """Fast L-fold CV cost for a candidate (C, kpars) pair.

        Mirrors the Cholesky-based implementation of ``crossvalidatelssvm.m``.
        """
        n = len(y_proc)
        K = _kernel_from_precomputed(omega_raw, self.kernel, kpars)
        Atot = K + np.eye(n) / C
        block = n // n_folds
        e = np.ones(n)

        try:
            R = linalg.cholesky(Atot, lower=False)
            rhs = np.column_stack([y_proc, e])
            # Solve R.T @ R @ q = rhs
            q_all = linalg.cho_solve((R, False), rhs)
            q, p = q_all[:, 0], q_all[:, 1]
            s = float(p.sum())
            bias = float(p @ y_proc) / s
            alpha_full = q - p * bias

            Ri = linalg.solve_triangular(R, np.eye(n), lower=False)
            C_mat = Ri @ Ri.T - (1.0 / s) * np.outer(p, p)
        except linalg.LinAlgError:
            A_aug = np.block(
                [[Atot, e[:, None]], [e[None, :], [[0.0]]]]
            )
            C_aug = np.linalg.pinv(A_aug)
            alpha_full = (C_aug @ np.r_[y_proc, 0.0])[:n]
            C_mat = C_aug[:n, :n]

        fold_costs = []
        for fold in range(n_folds):
            start = block * fold
            end = n if fold == n_folds - 1 else block * (fold + 1)
            val = np.arange(start, end)

            Ckk = C_mat[np.ix_(val, val)]
            try:
                betak = linalg.solve(Ckk, alpha_full[val], assume_a="pos")
            except linalg.LinAlgError:
                betak = np.linalg.lstsq(Ckk, alpha_full[val], rcond=None)[0]

            y_hat = y_proc[val] - betak
            fold_costs.append(cost_fn(y_proc[val], y_hat))

        return float(np.mean(fold_costs))

    # ── Bayesian framework ─────────────────────────────────────────────────────

    def _bay_level1(self, bay: _BayResult = None) -> _BayResult:
        """Bayesian inference level 1: compute Ed, Ew, costL1.

        Mirrors ``lssvm_bayL1`` in ``bay_lssvm.m`` (SVD path).
        """
        check_is_fitted(self, "_Xp")
        n = len(self._Xp)
        y = self._yp

        if bay is None:
            bay = _BayResult()
            bay.eigvals, bay.Rscores, bay.peff = _kpca(
                self._Xp, self.kernel, self._kpars
            )
            bay.Neff = len(bay.peff)

        Ym = y - y.mean()
        lam = bay.eigvals          # (k,)
        YTM = Ym @ bay.Rscores    # (k,)  projections

        # Ew: TvG eq. (4.75/5.73)
        # 0.5 * YTM * diag(λ) * diag((λ + 1/C)^-2) * YTM'
        bay.Ew = 0.5 * float(
            np.sum(YTM ** 2 * lam * (lam + 1.0 / self.C) ** (-2))
        )

        # cost: TvG eq. (4.76)
        # 0.5*C*(Ym'Ym) - 0.5*YTM*diag((1 + 1/(C*λ))^-1 * C)*YTM'
        bay.costL1 = float(
            0.5 * self.C * float(Ym @ Ym)
            - 0.5 * np.sum(YTM ** 2 * self.C * (1.0 + 1.0 / (self.C * lam)) ** (-1))
        )

        bay.Ed = (bay.costL1 - bay.Ew) / self.C
        if bay.costL1 > 0:
            bay.mu = (n - 1) / (2.0 * bay.costL1)
        else:
            bay.mu = 1.0
        bay.zeta = self.C * bay.mu
        return bay

    def _bay_level2(self, bay: _BayResult = None) -> _BayResult:
        """Bayesian inference level 2: costL2 and gradient for C optimisation.

        Mirrors ``lssvm_bayL2`` in ``bay_lssvm.m``.
        """
        check_is_fitted(self, "_Xp")
        n = len(self._Xp)
        bay = self._bay_level1(bay)

        all_lam = np.zeros(n)
        all_lam[bay.peff] = bay.eigvals

        Geff = 1.0 + float(
            np.sum(self.C * all_lam / (1.0 + self.C * all_lam))
        )
        bay.Geff = Geff

        if bay.Ew > 1e-30:
            bay.mu = 0.5 * (Geff - 1.0) / bay.Ew
        if bay.Ed > 1e-30:
            bay.zeta = 0.5 * (n - Geff) / bay.Ed

        denom = bay.Ew + self.C * bay.Ed
        if denom < 1e-30:
            denom = 1e-30

        bay.costL2 = float(
            np.sum(np.log(np.maximum(all_lam + 1.0 / self.C, 1e-300)))
            + (n - 1) * np.log(max(denom, 1e-300))
        )
        bay.DcostL2 = float(
            -np.sum(1.0 / (all_lam * self.C ** 2 + self.C))
            + (n - 1) * bay.Ed / denom
        )

        if bay.Ed > 1e-30 and (Geff - 1.0) > 1e-30:
            bay.optimal = self.C - (n - Geff) / (Geff - 1.0) * bay.Ew / bay.Ed
        else:
            bay.optimal = 0.0

        return bay

    def _bay_level3(self, bay: _BayResult = None) -> _BayResult:
        """Bayesian inference level 3: costL3 for kernel-parameter optimisation.

        Mirrors ``lssvm_bayL3`` in ``bay_lssvm.m``.
        """
        check_is_fitted(self, "_Xp")
        n = len(self._Xp)
        bay = self._bay_level2(bay)

        all_lam = np.zeros(n)
        all_lam[bay.peff] = bay.eigvals

        log_denom = np.log(
            np.maximum(bay.mu + bay.zeta * all_lam, 1e-300)
        )

        bay.costL3 = -float(
            bay.Neff * np.log(max(bay.mu, 1e-300))
            + (n - 1) * np.log(max(bay.zeta, 1e-300))
            - np.log(max(bay.Geff - 1.0, 1e-300))
            - np.log(max(n - bay.Geff, 1e-300))
            - float(np.sum(log_denom))
        )
        return bay

    # ── public Bayesian optimisation ──────────────────────────────────────────

    def bayesian_inference(self, level: int = 1) -> _BayResult:
        """Return Bayesian inference results at the requested level.

        Parameters
        ----------
        level : {1, 2, 3}
            * 1 – compute Ed, Ew, costL1 (requires trained model).
            * 2 – additionally compute Geff, costL2, gradient.
            * 3 – additionally compute costL3.

        Returns
        -------
        bay : _BayResult
        """
        check_is_fitted(self, "_Xp")
        if level == 1:
            return self._bay_level1()
        if level == 2:
            return self._bay_level2()
        if level == 3:
            return self._bay_level3()
        raise ValueError("level must be 1, 2, or 3")

    def tune_bayesian(self, level: int = 2) -> "_LSSVMBase":
        """Optimise hyperparameters using Bayesian inference.

        Mirrors ``bay_optimize.m``.

        Parameters
        ----------
        level : {2, 3}
            * 2 – optimise ``C`` (regularisation).
            * 3 – optimise ``C`` **and** ``sigma2`` (RBF only).

        Returns
        -------
        self : fitted and re-tuned estimator
        """
        check_is_fitted(self, "_Xp")

        if level not in (2, 3):
            raise ValueError("level must be 2 or 3")

        # ── Level 2: optimise C with gradient ──────────────────────────────
        # Use analytical gradient DcostL2 transformed to log(C) space:
        #   d(costL2)/d(log_C) = DcostL2 * C
        def _cost_and_grad_L2(log_C):
            C_try = float(np.exp(np.clip(log_C[0], -15.0, 15.0)))
            self.C = C_try
            K = self._K_train()
            self.alpha_, self.b_ = self._solve(K, self._yp)
            try:
                bay = self._bay_level2()
                cost = bay.costL2
                # Guard against invalid states (negative Ed can destabilise)
                if not np.isfinite(cost) or bay.Ed < 0:
                    return 1e10, np.array([0.0])
                grad = float(bay.DcostL2) * C_try   # chain rule: dL/d(logC) = dL/dC * C
            except Exception:
                cost, grad = 1e10, 0.0
            return cost, np.array([grad])

        res2 = optimize.minimize(
            _cost_and_grad_L2,
            [np.log(self.C)],
            method="L-BFGS-B",
            jac=True,
            bounds=[(-10.0, 10.0)],
            options={"maxiter": 200, "ftol": 1e-6, "gtol": 1e-4},
        )
        self.C = float(np.exp(np.clip(res2.x[0], -10.0, 10.0)))

        # ── Level 3: additionally optimise sigma2 ──────────────────────────
        if level == 3 and self.kernel == "rbf":
            def _cost_L3(log_sig2):
                sig_try = float(np.exp(np.clip(log_sig2[0], -10.0, 10.0)))
                self.sigma2 = sig_try
                K = self._K_train()
                self.alpha_, self.b_ = self._solve(K, self._yp)
                try:
                    cost = self._bay_level3().costL3
                    if not np.isfinite(cost):
                        return 1e10
                except Exception:
                    cost = 1e10
                return cost

            res3 = optimize.minimize(
                _cost_L3,
                [np.log(self.sigma2)],
                method="Nelder-Mead",
                options={"maxiter": 500, "xatol": 1e-3, "fatol": 1e-3},
            )
            self.sigma2 = float(np.exp(np.clip(res3.x[0], -10.0, 10.0)))

            # Re-optimise C at the new sigma2
            res2b = optimize.minimize(
                _cost_and_grad_L2,
                [np.log(self.C)],
                method="L-BFGS-B",
                jac=True,
                bounds=[(-10.0, 10.0)],
                options={"maxiter": 200, "ftol": 1e-6, "gtol": 1e-4},
            )
            self.C = float(np.exp(np.clip(res2b.x[0], -10.0, 10.0)))

        # Final refit with optimal hyperparameters
        K = self._K_train()
        self.alpha_, self.b_ = self._solve(K, self._yp)
        return self

    # ── cross-validation tuning ────────────────────────────────────────────────

    def tune_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 10,
        cost: str = "mse",
        random_state=None,
    ) -> "_LSSVMBase":
        """Tune ``C`` (and ``sigma2`` for RBF) via fast L-fold cross-validation.

        Uses the Cholesky-based shortcut from ``crossvalidatelssvm.m``.

        Parameters
        ----------
        X, y       : training data (will be preprocessed internally)
        n_folds    : number of CV folds
        cost       : 'mse' | 'mae' | 'misclass'
        random_state : int or None — shuffle seed

        Returns
        -------
        self
        """
        X = check_array(X)
        y = np.asarray(y).ravel()
        n = X.shape[0]
        X_orig, y_orig = X.copy(), y.copy()   # keep for final re-fit

        # Shuffle (for CV only; final fit uses original order)
        X_cv, y_cv = X.copy(), y.astype(float).copy()
        if random_state is not None:
            rng = np.random.RandomState(random_state)
            perm = rng.permutation(n)
            X_cv, y_cv = X_cv[perm], y_cv[perm]

        Xp = self._fit_scale_X(X_cv)
        yp = self._preprocess_y_fit(y_cv)

        omega_raw = self._raw_omega(Xp)

        if cost == "mse":
            def cost_fn(yt, yh): return float(np.mean((yt - yh) ** 2))
        elif cost == "mae":
            def cost_fn(yt, yh): return float(np.mean(np.abs(yt - yh)))
        elif cost == "misclass":
            def cost_fn(yt, yh): return float(np.mean(np.sign(yt) != np.sign(yh)))
        else:
            raise ValueError(f"Unknown cost '{cost}'. Use 'mse', 'mae', or 'misclass'.")

        if self.kernel == "rbf":
            def objective(params):
                C = float(np.exp(np.clip(params[0], -50, 50)))
                kp = np.array([np.exp(np.clip(params[1], -20, 20))])
                return self._cv_cost(C, kp, yp, omega_raw, n_folds, cost_fn)

            x0 = np.array([np.log(self.C), np.log(self._sig2)])
        else:
            def objective(params):
                C = float(np.exp(np.clip(params[0], -50, 50)))
                return self._cv_cost(C, self._kpars, yp, omega_raw, n_folds, cost_fn)

            x0 = np.array([np.log(self.C)])

        res = optimize.minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-3, "fatol": 1e-3, "disp": False},
        )

        self.C = float(np.exp(np.clip(res.x[0], -50, 50)))
        if self.kernel == "rbf" and len(res.x) > 1:
            self.sigma2 = float(np.exp(np.clip(res.x[1], -20, 20)))

        # Re-fit with optimal parameters using original (unshuffled) data
        self.fit(X_orig, y_orig)
        return self

    def _preprocess_y_fit(self, y: np.ndarray) -> np.ndarray:
        """Subclass overrides to apply y-scaling."""
        return y

    def _raw_output(self, X: np.ndarray) -> np.ndarray:
        """Compute raw LS-SVM output for (external, unscaled) X."""
        check_is_fitted(self, "alpha_")
        X = check_array(X)
        if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but "
                f"{self.__class__.__name__} is expecting "
                f"{self.n_features_in_} features as input."
            )
        Xp = self._scale_X(X)
        return self._K_test(Xp).T @ self.alpha_ + self.b_


# ─── Classifier ───────────────────────────────────────────────────────────────

class LSSVMClassifier(ClassifierMixin, _LSSVMBase):
    """Least Squares Support Vector Machine — Classifier.

    Binary or multiclass classifier based on the LS-SVM formulation by
    Suykens & Vandewalle (1999), using a straightforward linear-system solve
    instead of the quadratic program of classical SVMs.

    **Bayesian features** (call :meth:`tune_bayesian` or
    :meth:`predict_proba` for full posterior output):

    * Posterior class probabilities via :meth:`predict_proba` (Bayesian
      moderated output, mirrors ``bay_modoutClass.m``).
    * Three-level Bayesian hyper-parameter optimisation via
      :meth:`tune_bayesian` (mirrors ``bay_optimize.m``).
    * Cross-validation tuning via :meth:`tune_cv`.

    Parameters
    ----------
    C : float, default=1.0
        Regularisation parameter (called ``gam`` in the MATLAB toolbox).
        Larger values put more emphasis on fitting the training data.
    kernel : {'rbf', 'linear', 'poly'}, default='rbf'
        Kernel function.
    sigma2 : float, default=1.0
        Bandwidth parameter ``σ²`` for the RBF kernel.
        ``K(x,y) = exp(−‖x−y‖² / (2σ²))``.
        Ignored when ``kernel != 'rbf'``.
    gamma : float or None, default=None
        sklearn-style RBF coefficient (``γ = 1 / (2σ²)``).
        When set, it overrides ``sigma2``.
    degree : int, default=3
        Degree for the polynomial kernel ``K(x,y) = (x·y + coef0)^degree``.
    coef0 : float, default=1.0
        Constant term in the polynomial kernel.
    preprocess : bool, default=True
        Standardise input features (zero mean, unit std) before training.

    Attributes
    ----------
    alpha_ : ndarray of shape (n_train,)
        Dual variables of the trained LS-SVM.
    b_ : float
        Bias term.
    classes_ : ndarray
        Unique class labels encountered during ``fit``.

    Examples
    --------
    >>> from scikit_svm import LSSVMClassifier
    >>> clf = LSSVMClassifier(C=10.0, kernel='rbf', sigma2=0.5)
    >>> clf.fit(X_train, y_train)
    >>> clf.predict(X_test)
    """

    def _preprocess_y_fit(self, y: np.ndarray) -> np.ndarray:
        # No output scaling for classification
        self._y_mean = 0.0
        self._y_std = 1.0
        return y

    def fit(self, X, y):
        """Fit the LS-SVM classifier.

        Parameters
        ----------
        X : array-like of shape (n, d)
        y : array-like of shape (n,)
            Class labels (any dtype; internally encoded as ±1 for binary
            and treated per-class for multiclass).

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        self.n_features_in_ = X.shape[1]

        if len(self.classes_) == 2:
            # Binary: encode as ±1
            self._le_neg = self.classes_[0]
            self._le_pos = self.classes_[1]
            y_enc = np.where(y == self._le_pos, 1.0, -1.0)
            self._multiclass = False
        else:
            # Multiclass via One-vs-One
            self._multiclass = True
            self._ovo_clfs = self._fit_ovo(X, y)
            return self

        Xp = self._fit_scale_X(X)
        self._Xp = Xp
        self._yp = y_enc
        K = self._K_train()
        self.alpha_, self.b_ = self._solve(K, y_enc)
        return self

    def _fit_ovo(self, X, y):
        """Fit One-vs-One binary classifiers for multiclass."""
        from itertools import combinations
        clfs = {}
        for c1, c2 in combinations(self.classes_, 2):
            mask = (y == c1) | (y == c2)
            X_sub = X[mask]
            y_sub = np.where(y[mask] == c1, 1.0, -1.0)
            clf = LSSVMClassifier(**self.get_params())
            clf._multiclass = False
            clf._le_neg = c2
            clf._le_pos = c1
            Xp = clf._fit_scale_X(X_sub)
            clf._Xp = Xp
            clf._yp = y_sub
            clf._multiclass = False
            K = clf._K_train()
            clf.alpha_, clf.b_ = clf._solve(K, y_sub)
            clf.classes_ = np.array([c2, c1])
            clfs[(c1, c2)] = clf
        return clfs

    def decision_function(self, X):
        """Raw LS-SVM output (continuous decision values).

        Parameters
        ----------
        X : array-like of shape (n_test, d)

        Returns
        -------
        scores : ndarray of shape (n_test,) for binary,
                 (n_test, n_classes) for multiclass.
        """
        check_is_fitted(self, "classes_")
        if self._multiclass:
            raise NotImplementedError(
                "decision_function not available for multiclass; use predict()."
            )
        return self._raw_output(X)

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_test, d)

        Returns
        -------
        y_pred : ndarray of shape (n_test,)
        """
        check_is_fitted(self, "classes_")
        if self._multiclass:
            return self._predict_ovo(X)
        raw = self._raw_output(X)
        return np.where(raw >= 0, self._le_pos, self._le_neg)

    def _predict_ovo(self, X):
        """One-vs-One majority vote."""
        X = check_array(X)
        n = X.shape[0]
        votes = {c: np.zeros(n) for c in self.classes_}
        for (c1, c2), clf in self._ovo_clfs.items():
            pred = clf.predict(X)
            for i in range(n):
                votes[pred[i]][i] += 1
        labels = np.array(
            [max(votes, key=lambda c: votes[c][i]) for i in range(n)]
        )
        return labels

    def predict_proba(self, X, prior: float = 0.5):
        """Bayesian posterior class probabilities (binary classifier only).

        Estimates P(class=+1 | x) and P(class=-1 | x) using the Bayesian
        moderated output framework, mirroring ``bay_modoutClass.m``.

        Parameters
        ----------
        X     : array-like of shape (n_test, d)
        prior : float in [0, 1], default=0.5
            Prior probability of the positive class.

        Returns
        -------
        proba : ndarray of shape (n_test, 2)
            Columns are [P(negative), P(positive)].
        """
        check_is_fitted(self, "alpha_")
        if self._multiclass:
            raise NotImplementedError(
                "predict_proba not supported for multiclass LS-SVM."
            )

        X = check_array(X)
        Xt_proc = self._scale_X(X)

        # Raw LS-SVM output (latent variables)
        K_test = self._K_test(Xt_proc)
        Py = K_test.T @ self.alpha_ + self.b_   # (nt,) latent at test points

        # Mean latent variables of each training class
        y_tr = self._yp
        K_train = self._K_train()
        f_sv = K_train @ self.alpha_ + self.b_   # (n,) latent at train points
        Pymp = float(f_sv[y_tr > 0].mean()) if (y_tr > 0).any() else 0.0
        Pymn = float(f_sv[y_tr <= 0].mean()) if (y_tr <= 0).any() else 0.0

        # Bayesian Level 1
        bay = self._bay_level1()
        n = len(self._Xp)
        nt = len(X)

        omega = K_train                            # (n, n)
        theta = K_test                             # (n, nt)

        Zc = np.eye(n) - np.ones((n, n)) / n
        D = 1.0 / bay.mu - 1.0 / (bay.zeta * bay.eigvals + bay.mu)   # (k,)
        Hd = (Zc @ bay.Rscores) * D               # (n, k) broadcast
        Hd = Hd @ bay.Rscores.T @ Zc.T            # (n, n)

        # Self-kernel at test points (diagonal)
        if self.kernel == "rbf":
            kxx = np.ones(nt)  # RBF: k(x,x)=1 for normalised RBF
        else:
            kxx = np.array(
                [float(_kernel_matrix(Xt_proc[[i]], self.kernel, self._kpars))
                 for i in range(nt)]
            )

        # Positive-class variance
        pos_idx = np.where(y_tr > 0)[0]
        Nplus = len(pos_idx)
        Oplus = omega[:, pos_idx]                 # (n, Nplus)
        Oplusplus = omega[np.ix_(pos_idx, pos_idx)]
        mean_Oplus_col = Oplus.sum(axis=1) / Nplus  # (n,)

        var_plus = np.empty(nt)
        for i in range(nt):
            te = theta[:, i]
            tpe = ((te - mean_Oplus_col) @ Zc @ bay.Rscores)  # (k,)
            term1 = kxx[i] - 2.0 / Nplus * theta[pos_idx, i].sum()
            term2 = Oplusplus.sum() / Nplus ** 2
            term3 = float(np.sum(tpe ** 2 * D))
            var_plus[i] = (term1 + term2) / bay.mu - term3

        # Negative-class variance
        neg_idx = np.where(y_tr <= 0)[0]
        Nmin = len(neg_idx)
        Omin = omega[:, neg_idx]
        Ominmin = omega[np.ix_(neg_idx, neg_idx)]
        mean_Omin_col = Omin.sum(axis=1) / Nmin

        var_min = np.empty(nt)
        for i in range(nt):
            te = theta[:, i]
            tme = ((te - mean_Omin_col) @ Zc @ bay.Rscores)
            term1 = kxx[i] - 2.0 / Nmin * theta[neg_idx, i].sum()
            term2 = Ominmin.sum() / Nmin ** 2
            term3 = float(np.sum(tme ** 2 * D))
            var_min[i] = (term1 + term2) / bay.mu - term3

        # Posterior probabilities
        proba = np.empty((nt, 2))
        for i in range(nt):
            std_p = np.sqrt(max(1.0 / bay.zeta + var_plus[i], 1e-30))
            std_n = np.sqrt(max(1.0 / bay.zeta + var_min[i], 1e-30))
            pdf_p = float(prior) * stats.norm.pdf(Py[i], Pymp, std_p)
            pdf_n = (1.0 - float(prior)) * stats.norm.pdf(Py[i], Pymn, std_n)
            total = pdf_p + pdf_n
            if total < 1e-300:
                # Fall back to sign of decision function
                proba[i, 1] = 1.0 if Py[i] >= 0 else 0.0
                proba[i, 0] = 1.0 - proba[i, 1]
            else:
                proba[i, 1] = pdf_p / total    # P(positive)
                proba[i, 0] = pdf_n / total    # P(negative)

        return proba

    def score(self, X, y):
        """Return classification accuracy."""
        return float(np.mean(self.predict(X) == np.asarray(y)))


# ─── Regressor ────────────────────────────────────────────────────────────────

class LSSVMRegressor(RegressorMixin, _LSSVMBase):
    """Least Squares Support Vector Machine — Regressor.

    LS-SVM for function estimation (regression), with optional Bayesian
    error bars and simultaneous / pointwise confidence intervals.

    **Bayesian features**:

    * Predictive uncertainty (error bars) via :meth:`error_bars` (mirrors
      ``bay_errorbar.m``).
    * Three-level Bayesian hyper-parameter optimisation via
      :meth:`tune_bayesian`.
    * Confidence intervals via :meth:`confidence_interval` (mirrors
      ``cilssvm.m``).

    Parameters
    ----------
    C : float, default=1.0
        Regularisation parameter (``gam`` in the MATLAB toolbox).
    kernel : {'rbf', 'linear', 'poly'}, default='rbf'
        Kernel function.
    sigma2 : float, default=1.0
        Bandwidth ``σ²`` for the RBF kernel.
    gamma : float or None, default=None
        sklearn-style coefficient (overrides ``sigma2``).
    degree : int, default=3
        Polynomial kernel degree.
    coef0 : float, default=1.0
        Polynomial kernel constant term.
    preprocess : bool, default=True
        Standardise X (zero mean, unit std) and y (zero mean, unit std).

    Attributes
    ----------
    alpha_ : ndarray of shape (n_train,)
    b_ : float

    Examples
    --------
    >>> from scikit_svm import LSSVMRegressor
    >>> reg = LSSVMRegressor(C=10.0, kernel='rbf', sigma2=0.5)
    >>> reg.fit(X_train, y_train)
    >>> reg.predict(X_test)
    """

    def _preprocess_y_fit(self, y: np.ndarray) -> np.ndarray:
        return self._fit_scale_y(y)

    def fit(self, X, y):
        """Fit the LS-SVM regressor.

        Parameters
        ----------
        X : array-like of shape (n, d)
        y : array-like of shape (n,)

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y, multi_output=False)
        y = y.astype(float)
        self.n_features_in_ = X.shape[1]
        Xp = self._fit_scale_X(X)
        yp = self._fit_scale_y(y)
        self._Xp = Xp
        self._yp = yp
        K = self._K_train()
        self.alpha_, self.b_ = self._solve(K, yp)
        return self

    def predict(self, X):
        """Predict regression values.

        Parameters
        ----------
        X : array-like of shape (n_test, d)

        Returns
        -------
        y_pred : ndarray of shape (n_test,)
        """
        return self._unscale_y(self._raw_output(X))

    def score(self, X, y):
        """Return the coefficient of determination R²."""
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # ── Bayesian error bars ────────────────────────────────────────────────────

    def error_bars(self, X) -> np.ndarray:
        """Bayesian predictive standard deviation at test points.

        Mirrors ``bay_errorbar.m`` / ``bay_confb`` from LSSVMlab.

        Parameters
        ----------
        X : array-like of shape (n_test, d)

        Returns
        -------
        sigma_e : ndarray of shape (n_test,)
            Estimated predictive standard deviation at each test point
            (in the *preprocessed* y space; multiply by ``self._y_std``
            to recover original-scale units).
        """
        check_is_fitted(self, "alpha_")
        X = check_array(X)
        Xt_proc = self._scale_X(X)
        nt = Xt_proc.shape[0]
        n = len(self._Xp)

        bay = self._bay_level1()

        omega = self._K_train()                  # (n, n)
        theta = self._K_test(Xt_proc)            # (n, nt)
        oo = omega.sum(axis=0)                   # (n,) column sums

        # Hd = (Zc @ Rscores) * D * (Zc @ Rscores)'
        Zc = np.eye(n) - np.ones((n, n)) / n
        D = 1.0 / bay.mu - 1.0 / (bay.mu + bay.zeta * bay.eigvals)  # (k,)
        ZR = Zc @ bay.Rscores                   # (n, k)
        Hd = ZR * D @ ZR.T                      # (n, n)

        # Self-kernel at test points
        if self.kernel == "rbf":
            kxx = np.ones(nt)
        else:
            kxx = np.array(
                [float(_kernel_matrix(Xt_proc[[i]], self.kernel, self._kpars))
                 for i in range(nt)]
            )

        # Term 3 (constant over test points)
        term3 = (
            1.0 / (bay.zeta * n)
            + oo.sum() / (bay.mu * n ** 2)
            - float(oo @ Hd @ oo) / n ** 2
        )

        sig_e = np.empty(nt)
        for i in range(nt):
            th = theta[:, i]
            term1 = 1.0 / bay.zeta + kxx[i] / bay.mu - float(th @ Hd @ th)
            term2 = (
                2.0 / n * float(th @ Hd @ oo)
                - 2.0 / (bay.mu * n) * float(th.sum())
            )
            sig_e[i] = term1 + term2 + term3

        return np.sqrt(np.maximum(sig_e, 0.0))

    # ── Confidence intervals ───────────────────────────────────────────────────

    def confidence_interval(
        self,
        alpha: float = 0.05,
        conftype: str = "simultaneous",
    ) -> np.ndarray:
        """Bias-corrected confidence intervals on training points.

        Mirrors ``cilssvm.m`` from LSSVMlab.

        Constructs 100(1−α)% pointwise or simultaneous confidence intervals
        using the central-limit theorem for linear smoothers combined with
        a nonparametric variance estimate and bias correction via a
        second-order kernel (simple Gaussian approximation here).

        Parameters
        ----------
        alpha    : float, default=0.05
            Significance level.
        conftype : {'simultaneous', 'pointwise'}, default='simultaneous'
            Type of confidence interval.

        Returns
        -------
        ci : ndarray of shape (n_train, 2)
            ``ci[:, 0]`` lower bound, ``ci[:, 1]`` upper bound
            (in original y-scale).
        """
        check_is_fitted(self, "alpha_")
        n = len(self._Xp)
        y_raw = self._unscale_y(self._yp)       # original-scale training y

        # Fitted values on training data (in preprocessed space → unscale)
        K = self._K_train()
        y_fit_proc = K @ self.alpha_ + self.b_
        y_fit = self._unscale_y(y_fit_proc)

        # Smoother matrix on training data
        S = self._smoother()                     # (n, n)

        # ── Nonparametric variance estimation ──────────────────────────────
        # Regress squared residuals on X using the same kernel
        resid2 = (y_raw - y_fit) ** 2
        var_model = LSSVMRegressor(C=self.C, kernel=self.kernel,
                                   sigma2=self._sig2, preprocess=False)
        # Work directly in preprocessed X space
        var_model._x_mean = np.zeros(self._Xp.shape[1])
        var_model._x_std = np.ones(self._Xp.shape[1])
        var_model._Xp = self._Xp
        var_model._y_mean = float(resid2.mean())
        var_model._y_std = float(resid2.std())
        if var_model._y_std < 1e-12:
            var_model._y_std = 1.0
        r2_proc = (resid2 - var_model._y_mean) / var_model._y_std
        var_model._yp = r2_proc
        Kv = var_model._K_train()
        var_model.alpha_, var_model.b_ = var_model._solve(Kv, r2_proc)
        var_est_proc = Kv @ var_model.alpha_ + var_model.b_
        var_est = var_model._unscale_y(var_est_proc)
        sigma2_hat = np.maximum(var_est, 0.0)

        # ── Bias correction (simple: use S applied to y) ────────────────────
        # Full bias correction mirrors cilssvm but requires a fourth-order
        # kernel (RBF4). We approximate the bias as (S - I)*y_fit.
        bias_corr = (S - np.eye(n)) @ y_fit

        # ── Covariance matrix ───────────────────────────────────────────────
        Sigma = S @ np.diag(sigma2_hat) @ S.T
        std_vec = np.sqrt(np.maximum(np.diag(Sigma), 1e-30))

        # ── Critical value ──────────────────────────────────────────────────
        if conftype.startswith("s"):
            # Simultaneous: tube formula approximation
            delta = float(np.max(np.abs(bias_corr / std_vec)))
            df = max(int(n - np.trace(S)), 1)
            z = float(stats.t.ppf(1.0 - alpha / 2.0, df)) + delta
        elif conftype.startswith("p"):
            z = float(stats.norm.ppf(1.0 - alpha / 2.0))
            y_fit = y_fit - bias_corr
        else:
            raise ValueError(
                "conftype must be 'simultaneous' or 'pointwise'."
            )

        ci = np.column_stack([y_fit - z * std_vec, y_fit + z * std_vec])
        return ci
