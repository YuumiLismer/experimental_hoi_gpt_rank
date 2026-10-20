# Experimental Test of Higher-Order Quantum Interference via GPT Rank Analysis

**4-slit interferometer simulation, bilinear rank model selection, and Born rule verification**

---

## Abstract

Generalised Probabilistic Theory (GPT) provides a framework for characterising physical theories by the rank of their probability matrices — a dimension that encodes the complexity of the underlying state space. For a quantum system of *n* slits, the Born rule predicts that the probability matrix **D** (rows = preparation settings, columns = detector pixels) lies in a subspace of rank *n*² − 1, arising from the Hermitian traceless structure of the density matrix. This repository implements a cross-validated bilinear rank sweep on simulated 4-slit interference data to test whether the optimal rank is K = 15, as quantum mechanics predicts. Competing theories — classical wave optics (rank 4), rebit models (rank 9), and second-order-only interference (rank ≈ 10) — predict lower ranks, making this a quantitative, model-selection-based test of the Born rule.

---

## Scientific Background

### GPT Rank and the Born Rule

In any GPT, an experimental scenario consisting of *S* preparation settings and *X* measurement outcomes is fully characterised by a matrix **D** with entries D[s,x] = probability of outcome *x* given preparation *s*. The *GPT rank* of the theory is the minimum rank of **D** over all possible experiments.

For quantum mechanics with *n* distinguishable slits:

- Each preparation ω is a density matrix ρ ∈ Herm(*n*) with Tr(ρ) = 1, so ρ lives in an (*n*²−1)-dimensional affine subspace.
- Each effect e_x is a POVM element, and D[s,x] = Tr(e_x · ρ_s).
- Because **D** is a bilinear pairing over the (*n*²−1)-dimensional space, rank(**D**) = *n*²−1.

For *n* = 4 slits: **QM predicts rank = 4²−1 = 15**.

### Theory Comparison

| Theory | Predicted rank | Physical mechanism |
|---|---|---|
| Classical (incoherent) | 4 | No inter-slit coherence |
| 2nd-order interference only | ≈ 10 | Pairwise coherence only |
| Rebit (real QM) | 9 | Real-valued amplitudes, SO(*n*) |
| Quantum Mechanics | **15** | Full SU(*n*) density matrix |

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.9 and the packages listed in `requirements.txt`.

### Running the Analysis

```bash
# Default: single noise level, N_eff = 50 000 photons
python run_experiment.py

# Sweep three noise levels to test robustness
python run_experiment.py --n_eff 1000 10000 50000

# Noiseless (exact theory probabilities)
python run_experiment.py --no-noise

# Custom sweep parameters
python run_experiment.py --k_max 20 --n_folds 10 --n_jobs 4

# Custom output directory
python run_experiment.py --output results/
```

Results are written to `results/REPORT.md` (human-readable scientific report) and `results/results.json` (machine-readable data).

---

## Repository Structure

```
experimental_hoi_gpt_rank/
├── README.md                   # This file
├── SCIENCE.md                  # Full physics background and derivations
├── requirements.txt            # Python dependencies
├── .gitignore
├── run_experiment.py           # Entry point — CLI for the analysis pipeline
│
├── simulator/
│   ├── __init__.py
│   └── interferometer.py       # Gaussian beam model, shutter/phase configs, Poisson sampling
│
├── model/
│   ├── __init__.py
│   ├── als.py                  # Alternating Least Squares bilinear factorisation
│   └── rank_sweep.py           # Cross-validated rank sweep over K = 1 … k_max
│
├── pipeline/
│   ├── __init__.py
│   ├── experiment.py           # ExperimentPipeline — orchestrates all analysis steps
│   └── reporter.py             # ReportWriter — generates REPORT.md
│
└── results/
    └── .gitkeep                # Placeholder; output files are git-ignored
```

---

## Pipeline Overview

The analysis proceeds through six clearly defined scientific workflow phases:

| Phase | Module | Description |
|---|---|---|
| 1. Hypothesis | `pipeline/experiment.py` | State H₀ (QM: K=15) and competing hypotheses |
| 2. Experiment Design | `pipeline/experiment.py` | Fix noise levels, K range, CV strategy |
| 3. Simulation | `simulator/interferometer.py` | Generate Poisson-noisy probability matrices |
| 4. Model Training | `model/als.py`, `model/rank_sweep.py` | Fit rank-K bilinear models via ALS for each K |
| 5. Analysis | `pipeline/experiment.py` | Select optimal K by cross-validated χ²/dof |
| 6. Reporting | `pipeline/reporter.py` | Write REPORT.md and results.json |

The pipeline is fully reproducible: fix `--seed` and `--n_eff` to get identical output.

---

## CLI Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--n_eff` | float (one or more) | `50000` | Effective photon count(s) — controls shot noise level |
| `--no-noise` | flag | off | Use noiseless exact theory probabilities |
| `--k_max` | int | `20` | Maximum rank K in the sweep |
| `--n_folds` | int | `10` | Number of cross-validation folds |
| `--n_jobs` | int | `-1` | Parallel workers for CV sweep (-1 = all CPU cores) |
| `--output` | str | `results` | Output directory for REPORT.md and results.json |
| `--seed` | int | `42` | Random seed for reproducibility |

---

## Output

### `REPORT.md`

A self-contained scientific report including:
- Abstract and introduction
- Theory background (GPT rank, bilinear model, QM prediction)
- Methods (simulation, ALS fitting, cross-validation)
- Results: per-N_eff χ²/dof tables (mean ± std per fold) with the K = 15 row highlighted
- Summary tables: optimal K, expected K, consistency flag, Δχ²(K=14→15)
- Discussion of hypothesis evaluation, noise robustness, and sanity checks
- Conclusions and future work
- Full pipeline execution log

### `results.json`

Machine-readable JSON containing all numerical results: sweep data, optimal ranks, Δχ², and the full analysis dict — suitable for downstream plotting or archival.

---

## Physics

A brief overview of the physical model is given in the [Physics](#physics) section above. For the full derivation — including the Gaussian beam field formula, GPT state/effect duality, density matrix dimension argument, ALS normal equations, and cross-validation χ²/dof formula — see **[SCIENCE.md](SCIENCE.md)**.

---

## Citation

> Part of ongoing experimental work at IQC on GPT rank tests of quantum foundations.
