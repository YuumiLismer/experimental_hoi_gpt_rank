"""
interferometer.py
=================
Simulates a 4-slit far-field interference pattern using a Gaussian beam model.

Physical model
--------------
Each slit k contributes a field amplitude:

    psi_k(x) = A_k * exp[-2(x - x_k)^2 / r^2] * exp[i(phi_k + kappa_k * x)]

where:
    A_k  in {0, 1}   -- shutter state (open=1, closed=0)
    x_k              -- transverse slit position
    r                -- Gaussian beam radius
    phi_k            -- applied phase shift
    kappa_k          -- transverse momentum offset (spatial tilt per slit)

The intensity on the detector is:

    I(x) = |psi_1(x) + psi_2(x) + psi_3(x) + psi_4(x)|^2

and the probability distribution is p(x | s) = I(x) / sum_x I(x).

Measurement settings
--------------------
    Shutter configs : 2^4 = 16 binary patterns, grouped by n_open in {1,2,3,4}
    Phase patterns  : {0, pi/4, pi/2}^4 = 81 combinations
    Rows per group  : C(4,1)*81=324, C(4,2)*81=486, C(4,3)*81=324, C(4,4)*81=81

GPT rank prediction (Born rule): rank(D) = n^2 - 1 = 15  for n = 4 slits.
"""

import numpy as np
from itertools import product

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

BEAM_RADIUS  = 1.0
SLIT_X       = np.array([-0.15, -0.05, 0.05, 0.15])
KX_LIST      = np.array([-20.0, -10.0, 10.0, 20.0])
NUM_PIXELS   = 500
X_GRID       = np.linspace(-1.0, 1.0, NUM_PIXELS)
RANDOM_SEED  = 42

# Phase values and the full Cartesian product of phase patterns
PHASE_VALUES   = [0.0, np.pi / 4.0, np.pi / 2.0]
PHASE_PATTERNS = list(product(PHASE_VALUES, repeat=4))  # 81 entries

# All 16 shutter labels as 4-character binary strings (0=closed, 1=open)
SHUTTER_LABELS = [format(i, '04b') for i in range(1, 17)]  # '0001' ... '1111'


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _amplitudes(label: str) -> tuple:
    """Convert a 4-character binary shutter label to a tuple of float amplitudes.

    Parameters
    ----------
    label : str
        4-character binary string, e.g. ``'1010'``.

    Returns
    -------
    tuple of float
        Amplitude for each slit: 1.0 if open, 0.0 if closed.
    """
    return tuple(float(c) for c in label)


def _phase_label(combo: tuple) -> str:
    """Convert a phase-value tuple to a compact string label.

    Parameters
    ----------
    combo : tuple of float
        Four phase values, each in {0, pi/4, pi/2}.

    Returns
    -------
    str
        Human-readable label such as ``'0_p4_p2_0'``.
    """
    mapping = {0.0: '0', np.pi / 4.0: 'p4', np.pi / 2.0: 'p2'}
    return '_'.join(mapping.get(v, f'{v:.3f}') for v in combo)


def _intensity_row(A: np.ndarray, phases: tuple) -> np.ndarray:
    """Compute the normalised probability distribution for one (shutter, phase) setting.

    Parameters
    ----------
    A : np.ndarray, shape (4,)
        Slit amplitudes (0.0 or 1.0).
    phases : tuple of float
        Applied phase shifts for each slit.

    Returns
    -------
    np.ndarray, shape (NUM_PIXELS,)
        Normalised probability vector; sums to 1. If total intensity is zero
        (all shutters closed), returns a uniform distribution.
    """
    # Sum of complex field amplitudes over open slits
    field = np.zeros(NUM_PIXELS, dtype=complex)
    for k in range(4):
        if A[k] == 0.0:
            continue
        envelope = np.exp(-2.0 * (X_GRID - SLIT_X[k]) ** 2 / BEAM_RADIUS ** 2)
        phase    = np.exp(1j * (phases[k] + KX_LIST[k] * X_GRID))
        field   += A[k] * envelope * phase

    intensity = np.abs(field) ** 2
    total     = intensity.sum()
    if total < 1e-12:
        return np.ones(NUM_PIXELS) / NUM_PIXELS
    return intensity / total


# ---------------------------------------------------------------------------
# Main simulator class
# ---------------------------------------------------------------------------

class InterferometerSimulator:
    """Simulate Poisson-noisy probability matrices for the 4-slit interferometer.

    For each group of shutter configurations (grouped by number of open slits
    *n_open*), the simulator builds an exact probability matrix by evaluating
    the Gaussian beam model, then optionally adds Poisson shot noise.

    Parameters
    ----------
    n_eff : float
        Effective photon count per preparation setting. Controls the magnitude
        of Poisson shot noise: higher values give lower noise.
    seed : int
        Random seed for reproducibility of Poisson sampling.
    """

    def __init__(self, n_eff: float = 50_000, seed: int = RANDOM_SEED):
        self.n_eff = float(n_eff)
        self.rng   = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _exact_probs(self, n_open: int):
        """Compute the exact (noiseless) probability matrix for a shutter group.

        Parameters
        ----------
        n_open : int
            Number of open slits (1, 2, 3, or 4).

        Returns
        -------
        matrix : np.ndarray, shape (n_rows, NUM_PIXELS)
            Each row is a normalised probability distribution over detector pixels.
        labels : list of str
            Row label for each preparation setting, formatted as
            ``'<shutter>_ph<phase_label>'``.
        """
        rows   = []
        labels = []

        for label in SHUTTER_LABELS:
            if label.count('1') != n_open:
                continue
            A = np.array(_amplitudes(label))
            for phases in PHASE_PATTERNS:
                row = _intensity_row(A, phases)
                rows.append(row)
                labels.append(f'{label}_ph{_phase_label(phases)}')

        matrix = np.array(rows)  # shape: (n_rows, NUM_PIXELS)
        return matrix, labels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(self, noise: bool = True) -> dict:
        """Simulate probability matrices for all four shutter groups.

        Parameters
        ----------
        noise : bool
            If ``True``, add Poisson shot noise scaled to ``self.n_eff``.
            If ``False``, return exact theoretical probabilities.

        Returns
        -------
        dict
            Keys are integers 1, 2, 3, 4 (= n_open). Each value is a dict::

                {
                    'matrix' : np.ndarray, shape (n_rows, NUM_PIXELS),
                    'exact'  : np.ndarray, shape (n_rows, NUM_PIXELS),
                    'labels' : list of str,
                    'n_eff'  : float,
                }

            ``'matrix'`` is the (possibly noisy) probability matrix;
            ``'exact'`` is always the noiseless theoretical matrix.
        """
        results = {}
        for n_open in range(1, 5):
            exact, labels = self._exact_probs(n_open)

            if noise:
                # Draw Poisson counts and renormalise to probabilities
                counts = self.rng.poisson(exact * self.n_eff).astype(float)
                row_sums = counts.sum(axis=1, keepdims=True)
                # Avoid division by zero for pathological zero rows
                row_sums = np.where(row_sums == 0, 1.0, row_sums)
                noisy = counts / row_sums
            else:
                noisy = exact.copy()

            results[n_open] = {
                'matrix': noisy,
                'exact' : exact,
                'labels': labels,
                'n_eff' : self.n_eff,
            }

        return results
