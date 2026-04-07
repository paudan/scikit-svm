"""
LapSVM / LapRLSC — Laplacian Support Vector Machine and Laplacian
Regularized Least-Squares Classifier, trained in the primal with
Newton's method.

Python/NumPy/scikit-learn port of lapsvmp_v02 by Stefano Melacci (2012),
University of Siena, originally based on the primal SVM code of
Olivier Chapelle and the manifold regularization framework of
Vikas Sindhwani and Mikhail Belkin.

Both classifiers implement the same semi-supervised objective:

    min_{α,b}  ½ [ γ_A · αᵀKα  +  γ_I · (Kα)ᵀ L (Kα)  +  Σᵢ loss(yᵢ, fᵢ) ]

where ``f = Kα + b``, unlabeled points contribute only through the
Laplacian term, and *loss* is either hinge (LapSVM) or squared (LapRLSC).

References
----------
Belkin, M., Niyogi, P., & Sindhwani, V. (2006). Manifold regularization:
A geometric framework for learning from labeled and unlabeled examples.
*Journal of Machine Learning Research*, 7, 2399–2434.

Melacci, S., & Belkin, M. (2011). Laplacian support vector machines
trained in the primal. *JMLR*, 12, 1149–1184.
"""

import time

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.utils.multiclass import check_classification_targets


# ── Kernel computation ────────────────────────────────────────────────────────

def _rbf_dist2_sym(X):
    """Return squared Euclidean distance matrix for X vs X."""
    p = np.sum(X ** 2, axis=1)
    d2 = p[:, np.newaxis] + p[np.newaxis, :] - 2.0 * (X @ X.T)
    return np.maximum(d2, 0.0)


def _rbf_dist2(X1, X2):
    """Return squared Euclidean distance matrix (n2 × n1), matching MATLAB calckernel."""
    p1 = np.sum(X1 ** 2, axis=1)   # (n1,)
    p2 = np.sum(X2 ** 2, axis=1)   # (n2,)
    d2 = p2[:, np.newaxis] + p1[np.newaxis, :] - 2.0 * (X2 @ X1.T)
    return np.maximum(d2, 0.0)


def _kernel_sym(X, kernel, kparam):
    """Symmetric Gram matrix K(X, X)  — shape (n, n)."""
    if kernel == "linear":
        return X @ X.T
    if kernel == "poly":
        return (X @ X.T) ** kparam
    if kernel == "rbf":
        return np.exp(-_rbf_dist2_sym(X) / (2.0 * kparam ** 2))
    raise ValueError(f"Unknown kernel '{kernel}'. Use 'linear', 'poly', or 'rbf'.")


def _kernel_cross(X_train, X_test, kernel, kparam):
    """Cross kernel matrix K(X_test, X_train)  — shape (n_test, n_train).

    Matches MATLAB ``calckernel(options, X1, X2)`` with X1=X_train, X2=X_test.
    """
    if kernel == "linear":
        return X_test @ X_train.T
    if kernel == "poly":
        return (X_test @ X_train.T) ** kparam
    if kernel == "rbf":
        return np.exp(-_rbf_dist2(X_train, X_test) / (2.0 * kparam ** 2))
    raise ValueError(f"Unknown kernel '{kernel}'.")


# ── Graph construction ────────────────────────────────────────────────────────

def _euclidean_dist_matrix(A, B):
    """Euclidean distance matrix (n_A × n_B). Matches MATLAB euclidean.m."""
    aa = np.sum(A ** 2, axis=1)
    bb = np.sum(B ** 2, axis=1)
    d2 = aa[:, np.newaxis] + bb[np.newaxis, :] - 2.0 * (A @ B.T)
    return np.sqrt(np.maximum(d2, 0.0))


def _cosine_dist_matrix(A, B):
    """Cosine distance matrix (n_A × n_B). Matches MATLAB cosine.m."""
    na = np.linalg.norm(A, axis=1, keepdims=True)
    nb = np.linalg.norm(B, axis=1, keepdims=True)
    na[na == 0.0] = 1.0
    nb[nb == 0.0] = 1.0
    cos_sim = (A / na) @ (B / nb).T
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return np.maximum(1.0 - cos_sim, 0.0)


def _build_adjacency(X, nn, dist_fn, weights, weight_param):
    """Build sparse adjacency matrix.  Matches MATLAB adjacency.m.

    Parameters
    ----------
    X : ndarray (n, d)
    nn : int — number of nearest neighbors
    dist_fn : str — 'euclidean' or 'cosine'
    weights : str — 'binary', 'distance', or 'heat'
    weight_param : float — heat bandwidth (0 = auto-estimate from data)

    Returns
    -------
    A : scipy.sparse.csr_matrix (n, n)
    """
    n = X.shape[0]
    # Clamp nn to at most n-1 (can't have more neighbors than other points)
    nn_eff = min(nn, max(n - 1, 0))
    block_size = 500

    all_rows, all_cols, all_dists = [], [], []

    if nn_eff == 0:
        # Degenerate: 1 sample — return empty adjacency
        return sparse.csr_matrix((n, n), dtype=float)

    for i1 in range(0, n, block_size):
        i2 = min(i1 + block_size, n)
        Xblk = X[i1:i2]

        if dist_fn == "euclidean":
            D = _euclidean_dist_matrix(Xblk, X)   # (block, n)
        elif dist_fn == "cosine":
            D = _cosine_dist_matrix(Xblk, X)
        else:
            raise ValueError(f"Unknown distance function '{dist_fn}'.")

        # Pick neighbors 2..nn_eff+1 (skip self at position 0)
        idx = np.argsort(D, axis=1)[:, 1: nn_eff + 1]           # (block, nn_eff)
        dv  = np.take_along_axis(D, idx, axis=1).ravel()         # (block*nn_eff,)

        src = np.repeat(np.arange(i1, i2), nn_eff)
        dst = idx.ravel()

        all_rows.append(src)
        all_cols.append(dst)
        all_dists.append(dv)

    rows  = np.concatenate(all_rows)
    cols  = np.concatenate(all_cols)
    dists = np.concatenate(all_dists)

    # Cosine: only positive values (clip zero-distance edges)
    if dist_fn == "cosine":
        mask = dists < 1.0
        rows, cols, dists = rows[mask], cols[mask], dists[mask]

    if weights == "binary":
        vals = np.ones(len(dists))
    elif weights == "distance":
        vals = dists.copy()
    elif weights == "heat":
        t = float(weight_param)
        if t == 0.0:
            nz = dists[dists != 0.0]
            t = float(np.mean(nz)) if len(nz) > 0 else 1.0
        vals = np.exp(-(dists ** 2) / (2.0 * t * t))
    else:
        raise ValueError(f"Unknown graph weights '{weights}'.")

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=float)

    # Symmetrize: A = A + ((A != A') .* A')  — matches MATLAB
    At = A.T.tocsr()
    diff_mask = (A != At)           # boolean sparse matrix
    A = A + diff_mask.multiply(At)
    return A


def build_laplacian(X, nn=6, dist_fn="euclidean", weights="heat",
                    weight_param=0.0, normalize=True, degree=1):
    """Compute the graph Laplacian of the data.  Matches MATLAB laplacian.m.

    Parameters
    ----------
    X : ndarray (n, d) — data matrix
    nn : int — number of nearest neighbors (default 6)
    dist_fn : str — 'euclidean' or 'cosine' (default 'euclidean')
    weights : str — 'heat', 'binary', or 'distance' (default 'heat')
    weight_param : float — heat bandwidth; 0 = auto (default 0)
    normalize : bool — normalized Laplacian (default True)
    degree : int — Laplacian power ≥ 1 (default 1)

    Returns
    -------
    L : scipy.sparse.csr_matrix (n, n)
    """
    W = _build_adjacency(X, nn, dist_fn, weights, weight_param)
    n = W.shape[0]
    d = np.asarray(W.sum(axis=1)).ravel()

    if not normalize:
        D = sparse.diags(d, format="csr")
        L = D - W
    else:
        d_inv_sqrt = np.where(d > 0.0, 1.0 / np.sqrt(d), 0.0)
        Dinvs = sparse.diags(d_inv_sqrt, format="csr")
        W_norm = Dinvs @ W @ Dinvs
        L = sparse.eye(n, format="csr") - W_norm

    if degree > 1:
        L = np.linalg.matrix_power(L.toarray(), degree)
        L = sparse.csr_matrix(L)

    return L


# ── Newton solver ─────────────────────────────────────────────────────────────

def _newton_solve(K, L, Y, gamma_A, gamma_I, hinge, use_bias, max_iter, verbose):
    """Newton's method for primal LapSVM / LapRLSC.

    Matches ``newton()`` inside MATLAB ``lapsvmp.m``.

    Parameters
    ----------
    K : ndarray (n, n)
    L : sparse (n, n)
    Y : ndarray (n,) — labels in {-1, 0, +1}; 0 = unlabeled
    gamma_A, gamma_I : float
    hinge : bool — True → hinge loss (LapSVM); False → squared loss (LapRLSC)
    use_bias : bool
    max_iter : int
    verbose : bool

    Returns
    -------
    alpha : ndarray (n,)
    b : float
    n_iter : int
    train_time : float
    """
    t0 = time.process_time()
    n = len(Y)
    labeled = Y != 0.0
    l = int(labeled.sum())
    oc = 1 if int((Y == -1.0).sum()) == 0 else 0   # one-class flag

    alpha = np.zeros(n)
    b = 0.0
    Kalpha = np.zeros(n)

    # Precompute L*K (dense) once when gamma_I != 0
    LK = None
    L_dense = None
    if gamma_I != 0.0:
        L_dense = L.toarray() if sparse.issparse(L) else np.asarray(L)
        LK = L_dense @ K

    sv = np.zeros(n, dtype=bool)
    sv_prev = np.zeros(n, dtype=bool)

    t = 0
    while True:
        # ── identify support vectors ───────────────────────────────────────
        if hinge:
            sv_prev = sv.copy()
            hloss = np.zeros(n)
            hloss[labeled] = 1.0 - Y[labeled] * (Kalpha[labeled] + b)
            sv = hloss > 0.0
            nsv = int(sv.sum())
        else:                          # squared loss: all labeled are "SV"
            sv_prev = sv.copy()
            sv = labeled.copy()
            nsv = l

        if verbose:
            # objective value for printing
            if not hinge:
                hloss = np.zeros(n)
                hloss[labeled] = 1.0 - Y[labeled] * (Kalpha[labeled] + b)
            if gamma_I != 0.0:
                obj = (gamma_A * float(alpha @ Kalpha) +
                       float(np.sum(hloss[sv] ** 2)) +
                       gamma_I * float(Kalpha @ (L_dense @ Kalpha)) +
                       oc * b) / 2.0
            else:
                obj = (gamma_A * float(alpha @ Kalpha) +
                       float(np.sum(hloss[sv] ** 2)) +
                       oc * b) / 2.0
            print(f"[t={t}] obj={obj:.6f} nev={nsv}")

        # ── stopping conditions ───────────────────────────────────────────
        if t >= max_iter:
            break
        if t > 0 and np.array_equal(sv, sv_prev):
            break

        t += 1

        # ── solve the Newton system ───────────────────────────────────────
        if gamma_I == 0.0:
            # Standard (kernel) SVM — only sv rows/cols enter the system
            sv_idx = np.where(sv)[0]
            K_sv = K[np.ix_(sv_idx, sv_idx)]   # (nsv, nsv)

            if use_bias:
                ones_sv = np.ones(nsv)
                lhs = np.empty((nsv + 1, nsv + 1))
                lhs[0, 0] = 0.0
                lhs[0, 1:] = ones_sv
                lhs[1:, 0] = ones_sv
                lhs[1:, 1:] = gamma_A * np.eye(nsv) + K_sv
                rhs_b = oc / (2.0 * gamma_A) if gamma_A > 0.0 else 0.0
                rhs = np.concatenate([[rhs_b], Y[sv_idx]])
                try:
                    sol = np.linalg.solve(lhs, rhs)
                except np.linalg.LinAlgError:
                    sol = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
                b = float(sol[0])
                alpha_new = np.zeros(n)
                alpha_new[sv_idx] = sol[1:]
            else:
                A_sys = gamma_A * np.eye(nsv) + K_sv
                try:
                    alpha_sv = np.linalg.solve(A_sys, Y[sv_idx])
                except np.linalg.LinAlgError:
                    alpha_sv = np.linalg.lstsq(A_sys, Y[sv_idx], rcond=None)[0]
                alpha_new = np.zeros(n)
                alpha_new[sv_idx] = alpha_sv
                b = 0.0

        else:
            # LapSVM — full n×n system
            IsvK = np.zeros((n, n))
            IsvK[sv, :] = K[sv, :]
            H = gamma_A * np.eye(n) + IsvK + gamma_I * LK   # (n, n)

            IsvY = np.zeros(n)
            IsvY[sv] = Y[sv]

            if use_bias:
                ones_n = np.ones(n)
                sv_col = sv.astype(float)
                lhs = np.empty((n + 1, n + 1))
                lhs[0, 0] = 0.0
                lhs[0, 1:] = ones_n
                lhs[1:, 0] = sv_col
                lhs[1:, 1:] = H
                rhs_b = oc / (2.0 * gamma_A) if gamma_A > 0.0 else 0.0
                rhs = np.concatenate([[rhs_b], IsvY])
                try:
                    sol = np.linalg.solve(lhs, rhs)
                except np.linalg.LinAlgError:
                    sol = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
                b = float(sol[0])
                alpha_new = sol[1:]
            else:
                try:
                    alpha_new = np.linalg.solve(H, IsvY)
                except np.linalg.LinAlgError:
                    alpha_new = np.linalg.lstsq(H, IsvY, rcond=None)[0]
                b = 0.0

        alpha = alpha_new
        Kalpha = K @ alpha

    return alpha, b, t, time.process_time() - t0


# ── Base class ────────────────────────────────────────────────────────────────

class _LapSVMBase(ClassifierMixin, BaseEstimator):
    """Shared logic for LapSVMClassifier and LapRLSCClassifier."""

    # subclasses set this
    _hinge: bool = True

    def __init__(
        self,
        kernel="rbf",
        kernel_param=1.0,
        gamma_A=1e-6,
        gamma_I=1.0,
        nn=6,
        graph_dist="euclidean",
        graph_weights="heat",
        graph_weight_param=0.0,
        laplacian_normalize=True,
        laplacian_degree=1,
        use_bias=False,
        unlabeled_value=None,
        max_iter=200,
        verbose=False,
    ):
        self.kernel = kernel
        self.kernel_param = kernel_param
        self.gamma_A = gamma_A
        self.gamma_I = gamma_I
        self.nn = nn
        self.graph_dist = graph_dist
        self.graph_weights = graph_weights
        self.graph_weight_param = graph_weight_param
        self.laplacian_normalize = laplacian_normalize
        self.laplacian_degree = laplacian_degree
        self.use_bias = use_bias
        self.unlabeled_value = unlabeled_value
        self.max_iter = max_iter
        self.verbose = verbose

    # ── sklearn tags ──────────────────────────────────────────────────────

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    # ── validation helpers ────────────────────────────────────────────────

    def _check_params(self):
        if self.use_bias and self.laplacian_normalize:
            raise ValueError(
                "use_bias=True is not supported together with "
                "laplacian_normalize=True (MATLAB restriction)."
            )

    # ── public API ────────────────────────────────────────────────────────

    def fit(self, X, y):
        """Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Class labels for labeled samples; use ``unlabeled_value``
            (default ``None`` = fully supervised) to mark unlabeled
            samples in the semi-supervised setting.

            When ``unlabeled_value=None`` (default), all samples are
            treated as labeled and the two unique class values are mapped
            to ±1 automatically (sklearn-compatible behaviour).

            When ``unlabeled_value`` is set (e.g. ``0``), samples with
            that value are treated as unlabeled (they contribute only
            through the graph Laplacian).  The two labeled class values
            are mapped to ±1.

        Returns
        -------
        self
        """
        self._check_params()
        X, y = check_X_y(X, y)
        y = np.asarray(y)

        # ── determine unlabeled mask ──────────────────────────────────────
        if self.unlabeled_value is not None:
            unlabeled_mask = y == self.unlabeled_value
        else:
            unlabeled_mask = np.zeros(len(y), dtype=bool)

        labeled_y = y[~unlabeled_mask]
        from sklearn.utils.multiclass import type_of_target
        check_classification_targets(labeled_y)
        y_type = type_of_target(labeled_y, input_name="y", raise_unknown=True)
        if y_type != "binary":
            raise ValueError(
                "Only binary classification is supported. "
                f"The type of the target is {y_type}."
            )
        classes = np.unique(labeled_y)
        if len(classes) < 2:
            raise ValueError(
                f"LapSVM requires at least 2 labeled classes; "
                f"got 1 class: {classes}."
            )

        self.classes_ = classes
        self._le_neg = classes[0]   # maps to -1
        self._le_pos = classes[1]   # maps to +1

        # ── encode labels to ±1 / 0 ──────────────────────────────────────
        y_int = np.zeros(len(y), dtype=float)
        y_int[y == self._le_neg] = -1.0
        y_int[y == self._le_pos] = 1.0
        # unlabeled positions stay 0

        self.n_features_in_ = X.shape[1]

        K = _kernel_sym(X, self.kernel, float(self.kernel_param))
        L = build_laplacian(
            X,
            nn=int(self.nn),
            dist_fn=self.graph_dist,
            weights=self.graph_weights,
            weight_param=float(self.graph_weight_param),
            normalize=bool(self.laplacian_normalize),
            degree=int(self.laplacian_degree),
        )

        alpha, b, n_iter, train_time = _newton_solve(
            K, L, y_int,
            gamma_A=float(self.gamma_A),
            gamma_I=float(self.gamma_I),
            hinge=self._hinge,
            use_bias=bool(self.use_bias),
            max_iter=int(self.max_iter),
            verbose=bool(self.verbose),
        )

        svs = np.where(alpha != 0.0)[0]
        self.alpha_ = alpha[svs]
        self.svs_ = svs
        self.X_train_ = X
        self.b_ = float(b)
        self.n_iter_ = n_iter
        self.train_time_ = train_time

        return self

    def decision_function(self, X):
        """Compute raw decision scores.

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
                f"{self.__class__.__name__} is expecting "
                f"{self.n_features_in_} features as input."
            )
        K_test = _kernel_cross(
            self.X_train_[self.svs_], X,
            self.kernel, float(self.kernel_param),
        )                                               # (n_test, n_svs)
        return K_test @ self.alpha_ + self.b_

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array-like (n_test, n_features)

        Returns
        -------
        y_pred : ndarray (n_test,) — values from ``classes_``
        """
        scores = self.decision_function(X)
        return np.where(scores > 0.0, self._le_pos, self._le_neg)


# ── Public classifiers ────────────────────────────────────────────────────────

class LapSVMClassifier(_LapSVMBase):
    """Laplacian Support Vector Machine (primal, Newton's method).

    Semi-supervised binary classifier that combines a kernel SVM
    (hinge loss) with graph-based manifold regularization.

    Solves in the primal via Newton's method (default; PCG not implemented).
    Unlabeled points contribute only through the Laplacian regularizer,
    encouraging smooth decision boundaries along the data manifold.

    Parameters
    ----------
    kernel : {'linear', 'poly', 'rbf'}, default='rbf'
        Kernel function.
    kernel_param : float, default=1.0
        Kernel parameter: σ for 'rbf', degree *d* for 'poly', ignored
        for 'linear'.
    gamma_A : float, default=1e-6
        Ambient-space regularization (RKHS norm penalty).
    gamma_I : float, default=1.0
        Intrinsic (manifold) regularization.  Set to 0 to recover a
        standard (supervised) kernel SVM.
    nn : int, default=6
        Number of nearest neighbors for the k-NN graph.
    graph_dist : {'euclidean', 'cosine'}, default='euclidean'
        Distance function used to build the k-NN graph.
    graph_weights : {'heat', 'binary', 'distance'}, default='heat'
        Edge-weight scheme.
    graph_weight_param : float, default=0.0
        Heat-kernel bandwidth *t*; 0 = auto (mean edge length).
    laplacian_normalize : bool, default=True
        Use the normalized Laplacian.  Cannot be combined with
        ``use_bias=True``.
    laplacian_degree : int, default=1
        Iterated Laplacian power.
    use_bias : bool, default=False
        Include a bias term *b* in the decision function.
    max_iter : int, default=200
        Maximum Newton iterations.
    verbose : bool, default=False
        Print objective value at each iteration.

    Attributes
    ----------
    alpha_ : ndarray of shape (n_svs,)
        Non-zero coefficients (support vectors in the primal).
    svs_ : ndarray of shape (n_svs,) of int
        Indices of support vectors in the training set.
    X_train_ : ndarray of shape (n_samples, n_features)
        Copy of training data (needed for prediction).
    b_ : float
        Bias term (0 when ``use_bias=False``).
    n_iter_ : int
        Number of Newton iterations performed.
    train_time_ : float
        CPU time (seconds) consumed by the solver.
    classes_ : ndarray — always ``[-1, 1]``
    n_features_in_ : int

    Notes
    -----
    * Labels must be **+1** or **−1**.  Use **0** for unlabeled samples.
    * Equivalent to a standard kernel SVM when ``gamma_I=0`` and all
      samples are labeled.
    * Port of ``lapsvmp.m`` (``lapsvmp_v02``, Melacci 2012).

    Examples
    --------
    >>> import numpy as np
    >>> from scikit_svm import LapSVMClassifier
    >>> rng = np.random.RandomState(0)
    >>> X = rng.randn(50, 2)
    >>> y = np.zeros(50)          # all unlabeled initially
    >>> y[:10] = 1.0              # 10 labeled +1
    >>> y[25:35] = -1.0           # 10 labeled -1
    >>> clf = LapSVMClassifier(kernel_param=0.5, gamma_A=1e-4,
    ...                         gamma_I=1.0, verbose=False)
    >>> clf.fit(X, y)
    LapSVMClassifier(...)
    >>> clf.predict(X[:5])
    array([ 1., ...])
    """

    _hinge = True

    def __init__(
        self,
        kernel="rbf",
        kernel_param=1.0,
        gamma_A=1e-6,
        gamma_I=1.0,
        nn=6,
        graph_dist="euclidean",
        graph_weights="heat",
        graph_weight_param=0.0,
        laplacian_normalize=True,
        laplacian_degree=1,
        use_bias=False,
        unlabeled_value=None,
        max_iter=200,
        verbose=False,
    ):
        super().__init__(
            kernel=kernel,
            kernel_param=kernel_param,
            gamma_A=gamma_A,
            gamma_I=gamma_I,
            nn=nn,
            graph_dist=graph_dist,
            graph_weights=graph_weights,
            graph_weight_param=graph_weight_param,
            laplacian_normalize=laplacian_normalize,
            laplacian_degree=laplacian_degree,
            use_bias=use_bias,
            unlabeled_value=unlabeled_value,
            max_iter=max_iter,
            verbose=verbose,
        )


class LapRLSCClassifier(_LapSVMBase):
    """Laplacian Regularized Least-Squares Classifier (LapRLSC).

    Semi-supervised binary classifier identical to :class:`LapSVMClassifier`
    except that it uses a **squared loss** instead of hinge loss.  This means
    all labeled points always contribute to the gradient (no support-vector
    sparsity) and the Newton step reduces to a single linear system solve.

    Parameters
    ----------
    (same as :class:`LapSVMClassifier`)

    Notes
    -----
    Port of ``laprlsc.m`` / ``lapsvmp.m`` (Hinge=0) from lapsvmp_v02.
    """

    _hinge = False

    def __init__(
        self,
        kernel="rbf",
        kernel_param=1.0,
        gamma_A=1e-6,
        gamma_I=1.0,
        nn=6,
        graph_dist="euclidean",
        graph_weights="heat",
        graph_weight_param=0.0,
        laplacian_normalize=True,
        laplacian_degree=1,
        use_bias=False,
        unlabeled_value=None,
        max_iter=200,
        verbose=False,
    ):
        super().__init__(
            kernel=kernel,
            kernel_param=kernel_param,
            gamma_A=gamma_A,
            gamma_I=gamma_I,
            nn=nn,
            graph_dist=graph_dist,
            graph_weights=graph_weights,
            graph_weight_param=graph_weight_param,
            laplacian_normalize=laplacian_normalize,
            laplacian_degree=laplacian_degree,
            use_bias=use_bias,
            unlabeled_value=unlabeled_value,
            max_iter=max_iter,
            verbose=verbose,
        )
