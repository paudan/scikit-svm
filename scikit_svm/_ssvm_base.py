"""
Shared internals for SSVM and NSSVM.

Precise port of the helper functions that appear (identically) in both
ssvm.m and n_ssvm.m (Mangasarian & Musicant, 2000).
"""

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


# ---------------------------------------------------------------------------
# Nu estimation
# ---------------------------------------------------------------------------

def _est_nu_long(C, d):
    """
    Hard nu estimation (EstNuLong in MATLAB).

    Uses eigendecomposition of H*H' where H=[C,-e] to find lambda
    by a fixed-point iteration.  Subsamples to 200 rows when m > 200.

    MATLAB convergence threshold: abs(lamdaO-lamda) > 10e-4  (= 1e-3).
    Returns 1 if the fixed-point iteration does not converge in 100 steps.
    """
    m, n = C.shape
    e = np.ones(m)
    H = np.column_stack([C, -e])          # (m, n+1)

    if m < 201:
        H2, d2 = H, d
    else:
        rng = np.random.RandomState(0)    # fixed seed for reproducibility
        idx = np.argsort(rng.rand(m))
        H2 = H[idx[:200], :]
        d2 = d[idx[:200]]

    # Eigendecomposition of H2 @ H2.T  (symmetric → eigh)
    u, vu = np.linalg.eigh(H2 @ H2.T)    # u: eigenvalues (p,); vu: eigenvectors (p,p)
    p = len(u)
    yt = d2 @ vu                          # (p,) — projection of labels

    lamda = 1.0
    lamda_old = lamda + 1.0
    cnt = 0

    while abs(lamda_old - lamda) > 1e-3 and cnt < 100:   # 10e-4 = 1e-3
        cnt += 1
        lamda_old = lamda
        denom = u + lamda                                  # (p,)
        nu1 = np.sum(lamda / denom)
        pr   = np.sum(u / denom ** 2)
        ee   = np.sum(u * yt ** 2 / denom ** 3)
        waw  = np.sum(lamda ** 2 * yt ** 2 / denom ** 2)
        lamda = nu1 * ee / (pr * waw)

    return 1.0 if cnt == 100 else lamda


def _est_nu_short(C, d):
    """
    Easy nu estimation (EstNuShort in MATLAB).

    value = 1 / (sum(sum(C.^2)) / size(C,2))
           = n_features / ||C||_F^2
    """
    return C.shape[1] / np.sum(C ** 2)


# ---------------------------------------------------------------------------
# Mu estimation (n_ssvm only)
# ---------------------------------------------------------------------------

def _est_mu(A, d):
    """
    Estimate Gaussian kernel parameter mu (EstMu in MATLAB).

    mu = 1 / (1 + ||mean_pos + mean_neg||^2)
    """
    Aplus  = A[d == 1]
    Aminus = A[d == -1]
    x = Aplus.mean(axis=0) + Aminus.mean(axis=0)
    return 1.0 / (1.0 + x @ x)


# ---------------------------------------------------------------------------
# Kernel computation
# ---------------------------------------------------------------------------

def _rec_kernel(A, B, mu):
    """
    Gaussian kernel matrix (Rec_Kernel.m).

    K[i, j] = exp(-mu * ||A[i] - B[j]||^2)

    Parameters
    ----------
    A : (ma, n_features)  — query points (rows)
    B : (mb, n_features)  — reference points (rows)
    mu : float            — kernel bandwidth

    Returns
    -------
    K : (ma, mb)
    """
    return rbf_kernel(A, B, gamma=mu)


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def _objf(C, d, w, gamma, nu):
    """
    SSVM objective value (objf in MATLAB).

    f(w, gamma) = 0.5 * (nu * ||v||^2 + ||w||^2 + gamma^2)
    where v = max(|d| - (C*w - gamma*d), 0)  and |d| = 1 for ±1 labels.

    C here is the *transformed* matrix D*A (as used inside core()).
    """
    temp = np.abs(d) - (C @ w - gamma * d)
    v = np.maximum(temp, 0.0)
    return 0.5 * (nu * (v @ v) + (w @ w) + gamma ** 2)


# ---------------------------------------------------------------------------
# Armijo line search
# ---------------------------------------------------------------------------

def _armijo(C, d, w, gamma, nu, zd, gap):
    """
    Armijo backtracking line search (armijo in MATLAB).

    Halves the step size while  obj_old - obj_new < -0.05 * step * gap.

    Parameters
    ----------
    C, d    : transformed data (D*A) and label vector used inside core()
    w, gamma: current iterate
    nu      : regularisation parameter
    zd      : Newton direction (n+1,)
    gap     : z' * gradz  (directional derivative)
    """
    n = len(w)
    step = 1.0
    obj1 = _objf(C, d, w, gamma, nu)
    w2     = w     + step * zd[:n]
    gamma2 = gamma + step * zd[n]
    obj2 = _objf(C, d, w2, gamma2, nu)
    diff = obj1 - obj2

    while diff < -0.05 * step * gap:
        step  *= 0.5
        w2     = w     + step * zd[:n]
        gamma2 = gamma + step * zd[n]
        obj2   = _objf(C, d, w2, gamma2, nu)
        diff   = obj1 - obj2

    return step


# ---------------------------------------------------------------------------
# Core Newton / Armijo optimisation loop
# ---------------------------------------------------------------------------

def _core(C, d, nu, w0, gamma0, use_armijo, tol, max_iter):
    """
    Core SSVM optimisation (core() in MATLAB).

    Pre-transforms C and d to the D*A / D*e representation used in the
    SSVM paper, then runs a Newton iteration with optional Armijo step.

    Parameters
    ----------
    C        : (m, n) data matrix (original labels in d)
    d        : (m,) label vector of ±1
    nu       : regularisation parameter
    w0       : (n,) initial weight vector
    gamma0   : float initial bias
    use_armijo: True → Armijo step size; False → pure Newton step
    tol      : convergence threshold (flag = z'z < tol)
    max_iter : maximum iterations

    Returns
    -------
    w        : (n,) weight vector
    gamma    : float bias
    iteration: int  number of iterations
    """
    # --- Pre-transform: C ← D*A,  d ← D*e ----------------------------
    # MATLAB: C=[C(find(d==1),:); -C(find(d==-1),:)]
    #         d=[ones(ma,1); -ones(mb,1)]
    pos_idx = np.where(d == 1)[0]
    neg_idx = np.where(d == -1)[0]
    ma = len(pos_idx)
    mb = len(neg_idx)
    n  = C.shape[1]

    C = np.vstack([C[pos_idx], -C[neg_idx]])          # (ma+mb, n)
    d = np.concatenate([np.ones(ma), -np.ones(mb)])   # (ma+mb,)

    m = ma + mb
    e = np.ones(m)

    w0     = w0.copy().astype(float)
    gamma0 = float(gamma0)

    flag      = 1.0
    iteration = 0

    # MATLAB: while flag > tol & iteration < maxIter
    while flag > tol and iteration < max_iter:
        iteration += 1

        # --- residual ---------------------------------------------------
        # MATLAB: temp = C*w0 - gamma0*d;  rv = e - temp
        temp = C @ w0 - gamma0 * d
        rv   = e - temp

        # --- Hessian (limit of smoothed Hessian as alpha → ∞) ----------
        # MATLAB: H = (e + sign(rv))/2
        #   H[i] = 1   if rv[i] > 0
        #          0.5 if rv[i] = 0
        #          0   if rv[i] < 0
        H  = (e + np.sign(rv)) / 2.0
        Ih = np.where(H != 0)[0]          # nonzero-H indices
        ih = len(Ih)
        Hs = H[Ih]                         # (ih,)

        # MATLAB: SH = C(Ih,:)' * spdiags(Hs,0,T)  →  (n × ih)
        # Each column j of C[Ih,:].T is scaled by Hs[j]
        SH     = C[Ih, :].T * Hs           # (n, ih)
        P      = SH @ C[Ih, :]             # (n, n)
        q      = SH @ d[Ih]               # (n,)
        oneh   = float(np.sum(np.abs(Hs))) # ||Hs||_1  (Hs ≥ 0, so = sum(Hs))

        # --- Hessian matrix Q = I_{n+1} + nu * [[P, -q], [-q', oneh]] --
        # MATLAB: Q = speye(n+1) + nu*[P,(-q); (-q'), oneh]
        Q = np.eye(n + 1) + nu * np.block([
            [P,                  -q[:, np.newaxis]          ],
            [-q[np.newaxis, :],   np.array([[oneh]])        ],
        ])

        # --- gradient ---------------------------------------------------
        # MATLAB: prv = max(rv,0);  gradz = [w0-nu*C'*prv; gamma0+nu*d'*prv]
        prv   = np.maximum(rv, 0.0)
        gradz = np.append(
            w0 - nu * (C.T @ prv),
            gamma0 + nu * (d @ prv),
        )

        # --- check first-order optimality condition ---------------------
        # MATLAB: if gradz'*gradz > tol
        if gradz @ gradz > tol:
            b = -gradz                             # (n+1,)
            z = np.linalg.solve(Q, b)              # Newton direction

            gap = z @ gradz

            if not use_armijo:
                # Pure Newton step (MATLAB: step_size != 1)
                w0     = w0     + z[:n]
                gamma0 = gamma0 + z[n]
            else:
                # Armijo step (MATLAB default: step_size == 1)
                stepsize = _armijo(C, d, w0, gamma0, nu, z, gap)
                w0       = w0     + stepsize * z[:n]
                gamma0   = gamma0 + stepsize * z[n]

            flag = z @ z          # convergence measure: ||z||^2
        else:
            flag = tol            # MATLAB: flag = tol  (force exit)

    return w0, gamma0, iteration
