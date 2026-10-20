"""
als.py
======
Alternating Least Squares (ALS) bilinear factorisation for Poisson-weighted data.

Model
-----
The probability matrix D (shape S x X) is approximated by a rank-K bilinear model:

    D ~ U @ V.T

where U in R^(S x K) and V in R^(X x K).

Loss function
-------------
The fitting objective is a Poisson-weighted squared residual with L2 regularisation:

    L(U, V) = sum_{(s,x) in train} w[s,x] * (D[s,x] - (U @ V.T)[s,x])^2
              + reg * (||U||_F^2 + ||V||_F^2)

where w[s,x] = 1 / sigma[s,x]^2 and sigma is derived from the Poisson noise model.

Normal equations
----------------
ALS alternately fixes V to solve for each row of U (a linear system), then fixes U
to solve for each row of V. Written using einsum notation:

    U update:
        A_U[s,k,l] = sum_x w[s,x] * V[x,k] * V[x,l]  +  reg * delta(k,l)
        b_U[s,k]   = sum_x w[s,x] * D[s,x] * V[x,k]
        u_s        = solve(A_U[s], b_U[s])

    V update:
        A_V[x,k,l] = sum_s w[s,x] * U[s,k] * U[s,l]  +  reg * delta(k,l)
        b_V[x,k]   = sum_s w[s,x] * D[s,x] * U[s,k]
        v_x        = solve(A_V[x], b_V[x])

Initialisation and restarts
---------------------------
The first restart uses an SVD warm start (top-K singular vectors of D). Subsequent
restarts use random Gaussian initialisations. The best restart (lowest training
chi^2/dof) is returned.
"""

import numpy as np
from numpy.linalg import solve

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
ALS_REG     = 1e-6    # L2 regularisation coefficient
MAX_ITER    = 500     # Maximum ALS iterations per restart
TOL         = 1e-7    # Relative convergence tolerance on training loss
N_RESTARTS  = 4       # Number of random restarts (first is SVD warm start)
SIGMA_FLOOR = 1e-4    # Minimum sigma to prevent division by zero


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def poisson_sigma(prob_mat: np.ndarray, n_eff: float) -> np.ndarray:
    """Compute per-entry Poisson standard deviations.

    For a Poisson-distributed count with expected value n_eff * p, the standard
    deviation of the estimated probability p_hat = count / n_eff is:

        sigma = sqrt(p / n_eff)

    A floor is applied to prevent numerical issues for near-zero probabilities.

    Parameters
    ----------
    prob_mat : np.ndarray
        Probability matrix, shape (S, X).
    n_eff : float
        Effective photon count per row.

    Returns
    -------
    np.ndarray
        Standard-deviation matrix, same shape as ``prob_mat``.
    """
    return np.maximum(np.sqrt(np.abs(prob_mat) / n_eff), SIGMA_FLOOR)


def als_fit(
    data:       np.ndarray,
    sigma:      np.ndarray,
    train_mask: np.ndarray,
    K:          int,
    reg:        float = ALS_REG,
    n_restarts: int   = N_RESTARTS,
    seed:       int   = RANDOM_SEED,
):
    """Fit a rank-K bilinear model via Alternating Least Squares.

    Parameters
    ----------
    data : np.ndarray, shape (S, X)
        Observed probability matrix.
    sigma : np.ndarray, shape (S, X)
        Per-entry standard deviations (from :func:`poisson_sigma`).
    train_mask : np.ndarray of bool, shape (S, X)
        True for training entries, False for held-out test entries.
    K : int
        Rank of the bilinear model.
    reg : float
        L2 regularisation coefficient added to the diagonal of the normal
        equation matrices.
    n_restarts : int
        Number of ALS initialisations. The first is an SVD warm start;
        subsequent restarts use random Gaussian initialisations.
    seed : int
        Base random seed for reproducible restarts.

    Returns
    -------
    U : np.ndarray, shape (S, K)
        Left factor of the best-restart solution.
    V : np.ndarray, shape (X, K)
        Right factor of the best-restart solution.
    train_chi2 : float
        Training chi^2/dof for the best restart.
    test_chi2 : float
        Test chi^2/dof evaluated on the held-out entries.
    """
    S, X  = data.shape
    I_K   = np.eye(K)
    w     = np.where(train_mask, 1.0 / sigma ** 2, 0.0)  # zero weight on test entries
    rng   = np.random.default_rng(seed)

    best_U, best_V, best_train = None, None, np.inf

    for restart in range(n_restarts):
        # ---- Initialisation -----------------------------------------------
        if restart == 0:
            # SVD warm start on weighted data
            try:
                from numpy.linalg import svd
                U_svd, s_svd, Vt_svd = svd(data * train_mask, full_matrices=False)
                U = U_svd[:, :K] * s_svd[:K]
                V = Vt_svd[:K, :].T
            except Exception:
                U = rng.standard_normal((S, K)) * 0.01
                V = rng.standard_normal((X, K)) * 0.01
        else:
            U = rng.standard_normal((S, K)) * 0.01
            V = rng.standard_normal((X, K)) * 0.01

        prev_loss = np.inf

        # ---- ALS iterations -----------------------------------------------
        for _ in range(MAX_ITER):
            # U update
            A_U = np.einsum('sx,xk,xl->skl', w, V, V) + reg * I_K[None]
            b_U = np.einsum('sx,sx,xk->sk',  w, data, V)
            try:
                U = solve(A_U, b_U[..., None])[..., 0]
            except np.linalg.LinAlgError:
                break

            # V update
            A_V = np.einsum('sx,sk,sl->xkl', w, U, U) + reg * I_K[None]
            b_V = np.einsum('sx,sx,sk->xk',  w, data, U)
            try:
                V = solve(A_V, b_V[..., None])[..., 0]
            except np.linalg.LinAlgError:
                break

            # Convergence check on training loss
            residual  = data - U @ V.T
            train_loss = float(np.sum(w * residual ** 2))
            if abs(prev_loss - train_loss) / (abs(prev_loss) + 1e-12) < TOL:
                break
            prev_loss = train_loss

        # ---- Evaluate this restart ----------------------------------------
        residual   = data - U @ V.T
        train_chi2 = _chi2_dof(residual, sigma, train_mask)
        if train_chi2 < best_train:
            best_U, best_V, best_train = U.copy(), V.copy(), train_chi2

    # ---- Test chi^2/dof ---------------------------------------------------
    test_mask  = ~train_mask
    residual   = data - best_U @ best_V.T
    test_chi2  = _chi2_dof(residual, sigma, test_mask)

    return best_U, best_V, best_train, test_chi2


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _chi2_dof(residual: np.ndarray, sigma: np.ndarray, mask: np.ndarray) -> float:
    """Compute chi^2/dof over masked entries.

    Parameters
    ----------
    residual : np.ndarray
        Difference matrix D - U @ V.T.
    sigma : np.ndarray
        Per-entry standard deviations.
    mask : np.ndarray of bool
        True for entries to include in the sum.

    Returns
    -------
    float
        Mean squared standardised residual over masked entries.
        Returns 1e6 if no entries are selected.
    """
    n = mask.sum()
    if n == 0:
        return 1e6
    return float(np.sum(mask * (residual / sigma) ** 2) / n)
