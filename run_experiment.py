#!/usr/bin/env python3
"""
run_experiment.py
=================
Run the GPT rank analysis pipeline for the 4-slit quantum interference experiment.

This script simulates probability matrices for a 4-slit interferometer, performs
a cross-validated bilinear rank sweep (K = 1 ... k_max), and writes a full
scientific report to the output directory.

Usage
-----
    python run_experiment.py                           # default: N_eff=50000
    python run_experiment.py --n_eff 1000 10000 50000  # test multiple noise levels
    python run_experiment.py --no-noise                # noiseless theory data
    python run_experiment.py --k_max 20 --n_folds 10   # custom sweep parameters
    python run_experiment.py --output results/         # custom output directory
"""

import argparse
from pipeline.experiment import ExperimentPipeline


def main():
    parser = argparse.ArgumentParser(
        description='GPT Rank Analysis -- 4-slit quantum interference pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--n_eff', type=float, nargs='+', default=[50_000],
                        help='Effective photon counts (one or more values).')
    parser.add_argument('--no-noise', action='store_true',
                        help='Use noiseless (exact) theory data.')
    parser.add_argument('--k_max', type=int, default=20,
                        help='Maximum rank K to sweep.')
    parser.add_argument('--n_folds', type=int, default=10,
                        help='Number of cross-validation folds.')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Parallel jobs for CV sweep (-1 = all cores).')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for REPORT.md and results.json.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    args = parser.parse_args()

    pipeline = ExperimentPipeline(
        n_eff_list = args.n_eff,
        noise      = not args.no_noise,
        k_max      = args.k_max,
        n_folds    = args.n_folds,
        n_jobs     = args.n_jobs,
        output_dir = args.output,
        seed       = args.seed,
    )
    pipeline.run()


if __name__ == '__main__':
    main()
