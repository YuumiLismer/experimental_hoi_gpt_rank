# Scientific Background: GPT Rank Test of a 4-Slit Quantum Interferometer

This document provides a detailed physics background for the analysis implemented in this repository. It covers the physical setup, the GPT theoretical framework, the quantum mechanical prediction, competing theories, the bilinear model and fitting procedure, cross-validation strategy, and the connection to the IQC experimental apparatus.

---

## 1. Physical Setup

### 1.1 Gaussian Beam Field Model

Each of the four slits contributes a far-field amplitude at detector position *x*:

$$\psi_k(x) = A_k \cdot \exp\!\left[-\frac{2(x - x_k)^2}{r^2}\right] \cdot \exp\!\left[i(\varphi_k + \kappa_k x)\right]$$

where:

| Symbol | Value / meaning |
|---|---|
| $A_k \in \{0,1\}$ | Shutter state (1 = open, 0 = closed) |
| $x_k$ | Transverse slit position: $[-0.15, -0.05, 0.05, 0.15]$ |
| $r = 1.0$ | Gaussian beam radius |
| $\varphi_k$ | Applied phase shift (see phase patterns below) |
| $\kappa_k$ | Transverse momentum offset: $[-20, -10, 10, 20]$ |

The intensity on the detector is:

$$I(x) = \left|\sum_{k=1}^{4} \psi_k(x)\right|^2$$

and the conditional probability distribution for a given preparation setting *s* is:

$$p(x \mid s) = \frac{I(x)}{\sum_{x'} I(x')}$$

The detector is modelled as a 500-pixel array over $x \in [-1, 1]$.

### 1.2 Shutter Configurations

There are $2^4 = 16$ binary shutter patterns, grouped by the number of open slits $n_{\mathrm{open}} \in \{1, 2, 3, 4\}$:

| $n_{\mathrm{open}}$ | Configurations | Notation |
|---|---|---|
| 1 | 4 | `0001`, `0010`, `0100`, `1000` |
| 2 | 6 | `0011`, `0101`, `0110`, `1001`, `1010`, `1100` |
| 3 | 4 | `0111`, `1011`, `1101`, `1110` |
| 4 | 1 | `1111` |

### 1.3 Phase Patterns

For each shutter configuration, phases are varied over $\{0, \pi/4, \pi/2\}^4 = 81$ combinations. Each (shutter config, phase pattern) pair is one row of the probability matrix.

### 1.4 Row Count per Group

| Group $n_{\mathrm{open}}$ | Shutter configs | Phase patterns | Total rows |
|---|---|---|---|
| 1 | $\binom{4}{1} = 4$ | 81 | **324** |
| 2 | $\binom{4}{2} = 6$ | 81 | **486** |
| 3 | $\binom{4}{3} = 4$ | 81 | **324** |
| 4 | $\binom{4}{4} = 1$ | 81 | **81** |
| **Total** | | | **1215** |

---

## 2. GPT Framework

### 2.1 State and Effect Duality

In a Generalised Probabilistic Theory (GPT), the primitive objects are:

- **States** $\omega_s$ — preparation procedures labelled by *s*
- **Effects** $e_x$ — measurement outcomes labelled by *x*

The probability of outcome *x* given preparation *s* is given by the bilinear pairing:

$$D[s, x] = \langle e_x, \omega_s \rangle$$

The matrix **D** (rows = preparations, columns = outcomes) is called the *probability matrix* of the experiment.

### 2.2 GPT Rank

The *GPT rank* of a theory is the minimum number of linearly independent states needed to express any state in the theory — equivalently, the minimum rank of **D** over all experiments that are tomographically complete for that theory.

GPT rank is a theory-level invariant: it characterises the dimensionality of the state space and thus distinguishes competing physical theories.

---

## 3. Quantum Mechanical Prediction: rank(D) = n² − 1

### 3.1 Density Matrix Argument

For a quantum system with Hilbert space $\mathbb{C}^n$, any state is described by a density matrix $\rho \in \mathrm{Herm}(n)$ with $\mathrm{Tr}(\rho) = 1$. The space of such matrices is an affine subspace of dimension:

$$\dim(\mathrm{Herm}(n)) - 1 = n^2 - 1$$

This is precisely the dimension of $\mathfrak{su}(n)$, the Lie algebra of traceless Hermitian matrices.

### 3.2 Born Rule Linearity

The Born rule gives $D[s,x] = \mathrm{Tr}(e_x \rho_s)$, which is linear in $\rho_s$. Since every $\rho_s$ can be written as a linear combination of $n^2 - 1$ basis elements of $\mathfrak{su}(n)$ plus the identity, the matrix **D** lies in a subspace of dimension at most $n^2 - 1$:

$$\mathrm{rank}(\mathbf{D}) \leq n^2 - 1$$

For a tomographically complete set of preparations, this bound is achieved with equality:

$$\boxed{\mathrm{rank}(\mathbf{D}) = n^2 - 1 = 15 \quad (n=4)}$$

### 3.3 Competing Theories

| Theory | State space | Predicted rank | Basis |
|---|---|---|---|
| Classical (incoherent) | Probability simplex in $\mathbb{R}^4$ | 4 | No inter-slit coherence |
| Rebit (real QM) | $\mathrm{Sym}(n)$, real density matrices | $n(n+1)/2 - 1 = 9$ | $\mathfrak{so}(n)$ |
| 2nd-order interference | Pairwise coherences only | ≈ 10 | Pairs of slits |
| Quantum Mechanics | $\mathfrak{su}(n)$ | **15** | Full unitary group |

---

## 4. Bilinear Model

### 4.1 Physical Motivation

The probability matrix **D** is modelled as:

$$\mathbf{D} \approx \mathbf{U} \mathbf{V}^\top$$

where:
- $\mathbf{U} \in \mathbb{R}^{S \times K}$ — state representation (rows = preparation settings)
- $\mathbf{V} \in \mathbb{R}^{X \times K}$ — detector response (rows = pixels)

Physically, **V** encodes the detector's mode decomposition and is shared across all shutter groups (since the detector does not change between preparation settings). This joint-fit constraint is the key physical assumption: different shutter groups share the same detector response.

### 4.2 Poisson Noise Weighting

For shot-noise-limited photon counting, the uncertainty in each matrix entry is:

$$\sigma[s,x] = \max\!\left(\sqrt{\frac{|p[s,x]|}{N_{\mathrm{eff}}}},\ \sigma_{\mathrm{floor}}\right)$$

where $\sigma_{\mathrm{floor}} = 10^{-4}$ prevents division by zero for exactly-zero entries.

---

## 5. ALS Fitting

### 5.1 Weighted Loss

The fitting objective is the Poisson-weighted squared residual:

$$\mathcal{L}(\mathbf{U}, \mathbf{V}) = \sum_{(s,x) \in \mathrm{train}} w[s,x] \left(D[s,x] - [\mathbf{U}\mathbf{V}^\top]_{sx}\right)^2 + \lambda \left(\|\mathbf{U}\|_F^2 + \|\mathbf{V}\|_F^2\right)$$

where $w[s,x] = 1/\sigma[s,x]^2$ and $\lambda$ is an L2 regularisation parameter.

### 5.2 Normal Equations

**U update** (fix V, solve for each row $\mathbf{u}_s$):

$$\left[\sum_x w[s,x]\, \mathbf{v}_x \mathbf{v}_x^\top + \lambda \mathbf{I}_K\right] \mathbf{u}_s = \sum_x w[s,x]\, D[s,x]\, \mathbf{v}_x$$

**V update** (fix U, solve for each row $\mathbf{v}_x$):

$$\left[\sum_s w[s,x]\, \mathbf{u}_s \mathbf{u}_s^\top + \lambda \mathbf{I}_K\right] \mathbf{v}_x = \sum_s w[s,x]\, D[s,x]\, \mathbf{u}_s$$

Using Einstein summation notation, these are implemented as:

```python
# U update
A_U = einsum('sx,xk,xl->skl', w, V, V) + reg * I_K[None]
b_U = einsum('sx,sx,xk->sk', w, data, V)
U   = solve(A_U, b_U[..., None])[..., 0]

# V update
A_V = einsum('sx,sk,sl->xkl', w, U, U) + reg * I_K[None]
b_V = einsum('sx,sx,sk->xk', w, data, U)
V   = solve(A_V, b_V[..., None])[..., 0]
```

### 5.3 Warm Start and Restarts

To avoid local minima, ALS is initialised from the top-*K* SVD of the data matrix (warm start) and run with `N_RESTARTS = 4` random initialisations. The restart with lowest training χ²/dof is kept.

Convergence is declared when the relative change in training loss falls below `TOL = 1e-7`, with a maximum of `MAX_ITER = 500` iterations.

---

## 6. Cross-Validation Strategy

### 6.1 Fold Assignment

To avoid overfitting to specific rows or pixels, cross-validation is performed on randomly drawn (row, pixel) pairs:

1. Flatten the $S \times X$ index grid to a 1D array of $S \cdot X$ indices.
2. Randomly permute the indices and assign each to one of $n_{\mathrm{folds}}$ folds.
3. Reshape back to $(S, X)$ to get a fold-ID matrix.

For a given fold *f*, the training mask is `fold_ids != f` and the test mask is `fold_ids == f`.

### 6.2 Test χ²/dof

For held-out test entries, the fit quality is measured by:

$$\frac{\chi^2}{\mathrm{dof}} = \frac{1}{|\mathrm{test}|} \sum_{(s,x) \in \mathrm{test}} \frac{\left(D[s,x] - [\hat{\mathbf{U}}\hat{\mathbf{V}}^\top]_{sx}\right)^2}{\sigma[s,x]^2}$$

where the sum runs over test-set entries only.

---

## 7. Statistical Interpretation

### 7.1 χ²/dof Table

| χ²/dof value | Interpretation |
|---|---|
| $\gg 1$ | Model underfitting — rank K too low |
| $\approx 1$ | Good fit — K matches the true data rank |
| $< 1$ | Possible overfitting (rare for cross-validated test set) |

The optimal rank is identified as:

$$K^* = \arg\min_K \overline{\chi^2_{\mathrm{test}}}(K)$$

where the overline denotes the mean over folds.

### 7.2 Δχ² Test Statistic

To quantify the improvement from K=14 to K=15 (the QM prediction boundary):

$$\Delta\chi^2 = \overline{\chi^2_{\mathrm{test}}}(K=14) - \overline{\chi^2_{\mathrm{test}}}(K=15)$$

A large positive $\Delta\chi^2$ indicates that the transition at K=15 is statistically significant.

---

## 8. IQC Experimental Connection

This simulation framework is designed to match the parameters of the IQC experimental apparatus:

| Parameter | Value |
|---|---|
| Light source | 405 nm Topmode diode laser |
| Nonlinear crystal | PPKTP (type-II SPDC) |
| Detector | emICCD camera |
| Dataset date | 2026-04-09 |
| N_eff scale factor | `EXP_N_EFF_SCALE` (calibrated per dataset) |

The simulation parameter `n_eff` directly models the effective photon count per preparation setting, which in the real experiment is determined by the laser power, SPDC conversion efficiency, and emICCD quantum efficiency.

Systematic effects not modelled here (but relevant for the real experiment) include:
- Phase drift during data acquisition
- Detector dark counts and read noise
- Spatial mode mismatch between SPDC photon pairs
- Alignment stability of the 4-slit aperture

For further details on the experimental setup, see the IQC internal documentation for the 2026-04-09 dataset.
