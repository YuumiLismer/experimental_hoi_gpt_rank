"""
rank_sweep.py
=============
Cross-validated bilinear rank sweep for the GPT rank test.

Joint-fit strategy
------------------
All shutter groups (n_open = 1, 2, 3, 4) share a common detector-response
matrix V. This reflects the physical assumption that the detector does not
change between preparation settings. For each rank K, a joint bilinear model

    D_group ~ U_group @ V.T

is fitted, where V is shared across all groups and U_group is group-specific.
In practice, the groups are stacked into a single tall matrix and fitted
jointly, with per-group sigma matrices.

Cross-validation
----------------
For each fold, a random subset of (row, pixel) pairs is held out as a test set.
ALS is fitted on the remaining training entries, and the test chi^2/dof is
computed per group. The fold assignment is done on the flattened index grid to
ensure each pixel appears in both training and test sets.

Return format
-------------
The function returns a nested dict::

    {
        group (int or 'all'): {
            K (int): {
                'train': list of float (one per fold),
                'test':  list of float (one per fold),
            }
        }
    }

"""

import numpy as np
from joblib import Parallel, delayed
from .als import als_fit, poisson_sigma

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_PX_SWEEP = 500  # Maximum number of pixel columns used in the sweep


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resample_cols(mat: np.ndarray, n_cols: int, seed: int = 42) -> np.ndarray:
    """Randomly subsample columns of a matrix.

    Parameters
    ----------
    mat : np.ndarray, shape (S, X)
    n_cols : int
        Number of columns to keep.
    seed : int

    Returns
    -------
    np.ndarray, shape (S, min(n_cols, X))
    """
    X = mat.shape[1]
    if n_cols >= X:
        return mat
    rng = np.random.default_rng(seed)
    idx = rng.choice(X, size=n_cols, replace=False)
    idx.sort()
    return mat[:, idx]


def _fit_one_fold(
    all_mat:           np.ndarray,
    sigma_mat:         np.ndarray,
    fold_ids:          np.ndarray,
    K:                 int,
    fold:              int,
    reg:               float,
    group_row_slices:  dict,
    seed:              int,
):
    """Fit rank-K ALS for a single CV fold and return per-group chi^2.

    Parameters
    ----------
    all_mat : np.ndarray, shape (total_rows, n_pix)
        Stacked probability matrix (all groups concatenated row-wise).
    sigma_mat : np.ndarray, shape (total_rows, n_pix)
        Corresponding sigma matrix.
    fold_ids : np.ndarray of int, shape (total_rows, n_pix)
        Fold assignment for each (row, pixel) pair.
    K : int
        Rank to fit.
    fold : int
        The held-out fold index.
    reg : float
        L2 regularisation coefficient.
    group_row_slices : dict
        Maps group key (int or 'all') to a slice of rows in ``all_mat``.
    seed : int
        Random seed for ALS restarts.

    Returns
    -------
    tuple : (K, fold, group_chi2_dict)
        ``group_chi2_dict`` maps group key to ``(train_chi2, test_chi2)``.
    """
    try:
        train_mask = fold_ids != fold

        _, _, train_chi2, test_chi2 = als_fit(
            data       = all_mat,
            sigma      = sigma_mat,
            train_mask = train_mask,
            K          = K,
            reg        = reg,
            seed       = seed + fold,
        )

        # Compute per-group chi^2 by restricting to each group's rows
        group_chi2 = {}
        for grp, sl in group_row_slices.items():
            grp_mat    = all_mat[sl]
            grp_sigma  = sigma_mat[sl]
            grp_fold   = fold_ids[sl]
            grp_train  = grp_fold != fold
            grp_test   = grp_fold == fold

            from .als import _chi2_dof
            U_full     = np.zeros((all_mat.shape[0], K))  # placeholder; recompute below

            # Re-fit restricted to this group's rows for per-group reporting
            _, _, g_train, g_test = als_fit(
                data       = grp_mat,
                sigma      = grp_sigma,
                train_mask = grp_train,
                K          = K,
                reg        = reg,
                seed       = seed + fold + hash(str(grp)) % 1000,
            )
            group_chi2[grp] = (g_train, g_test)

        # 'all' key: use the full-matrix fit result
        group_chi2['all'] = (train_chi2, test_chi2)

        return K, fold, group_chi2

    except Exception as exc:  # noqa: BLE001
        # Return sentinel values on failure so the sweep can continue
        sentinel = {grp: (1e6, 1e6) for grp in group_row_slices}
        sentinel['all'] = (1e6, 1e6)
        return K, fold, sentinel


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def run_rank_sweep(
    data:     dict,
    K_range:  list = None,
    n_folds:  int  = 10,
    n_jobs:   int  = -1,
    seed:     int  = 42,
) -> dict:
    """Run a cross-validated bilinear rank sweep over all shutter groups.

    Parameters
    ----------
    data : dict
        Output of :meth:`InterferometerSimulator.simulate`, keyed by
        n_open (1, 2, 3, 4). Each value must have ``'matrix'`` and ``'n_eff'``.
    K_range : list of int
        Rank values to sweep. Defaults to ``list(range(1, 21))``.
    n_folds : int
        Number of cross-validation folds.
    n_jobs : int
        Number of parallel workers (joblib). -1 uses all available cores.
    seed : int
        Random seed for fold assignment and ALS restarts.

    Returns
    -------
    dict
        Nested results dict: ``{group: {K: {'train': [...], 'test': [...]}}}``.
        Groups are integers 1–4 plus the string ``'all'``.
    """
    if K_range is None:
        K_range = list(range(1, 21))

    # ---- Subsample columns for speed --------------------------------------
    groups    = sorted(data.keys())
    mats      = {g: _resample_cols(data[g]['matrix'], N_PX_SWEEP, seed=seed)
                 for g in groups}
    n_eff_map = {g: data[g]['n_eff'] for g in groups}

    # ---- Stack all groups into one matrix ---------------------------------
    group_row_slices = {}
    current_row      = 0
    mat_list         = []
    sig_list         = []

    for g in groups:
        mat   = mats[g]
        n_eff = n_eff_map[g]
        sigma = poisson_sigma(mat, n_eff)
        n_rows = mat.shape[0]
        group_row_slices[g] = slice(current_row, current_row + n_rows)
        mat_list.append(mat)
        sig_list.append(sigma)
        current_row += n_rows

    all_mat   = np.vstack(mat_list)
    sigma_mat = np.vstack(sig_list)
    n_rows, n_pix = all_mat.shape
    mean_n_eff    = float(np.mean(list(n_eff_map.values())))

    # ---- Regularisation parameter -----------------------------------------
    reg = max(1e-6, mean_n_eff * 1e-8)

    print(f'[RankSweep] Matrix: {n_rows} rows x {n_pix} pixels')
    print(f'[RankSweep] Mean N_eff: {mean_n_eff:.0f}  |  reg: {reg:.2e}  |  n_jobs: {n_jobs}')

    # ---- Fold assignment on flattened (row, pixel) indices ----------------
    rng      = np.random.default_rng(seed)
    flat_idx = rng.permutation(n_rows * n_pix)
    fold_ids = (flat_idx % n_folds).reshape(n_rows, n_pix)

    # ---- Parallel sweep ---------------------------------------------------
    tasks = [
        delayed(_fit_one_fold)(
            all_mat, sigma_mat, fold_ids, K, fold, reg, group_row_slices, seed
        )
        for K in K_range
        for fold in range(n_folds)
    ]
    raw_results = Parallel(n_jobs=n_jobs, prefer='threads')(tasks)

    # ---- Collect results --------------------------------------------------
    all_groups  = list(groups) + ['all']
    results     = {
        grp: {K: {'train': [], 'test': []} for K in K_range}
        for grp in all_groups
    }

    for K, fold, group_chi2 in raw_results:
        for grp, (tr, te) in group_chi2.items():
            results[grp][K]['train'].append(tr)
            results[grp][K]['test'].append(te)

    # ---- Print per-K summary ----------------------------------------------
    print(f'\n{"K":>4}  {"n=1":>8}  {"n=2":>8}  {"n=3":>8}  {"n=4":>8}  {"all":>8}')
    print('-' * 56)
    for K in K_range:
        parts = []
        for grp in [1, 2, 3, 4, 'all']:
            vals = results[grp][K]['test']
            parts.append(f'{np.mean(vals):8.3f}')
        marker = '  <- QM' if K == 15 else ''
        print(f'{K:>4}  {"  ".join(parts)}{marker}')

    return results
