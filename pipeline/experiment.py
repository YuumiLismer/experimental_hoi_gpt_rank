"""
experiment.py
=============
ExperimentPipeline -- orchestrates the full GPT rank analysis workflow.

This module implements a systematic experimental analysis pipeline that:
  1. Formulates the hypothesis (QM: rank = n^2 - 1 = 15)
  2. Designs the experiment (noise levels, K range, CV strategy)
  3. Simulates probability matrices with Poisson shot noise
  4. Fits the bilinear rank-K model via cross-validated ALS
  5. Analyses results and tests the hypothesis
  6. Writes a full scientific report (REPORT.md + results.json)
"""

import json
import os
import time

import numpy as np

from simulator.interferometer import InterferometerSimulator
from model.rank_sweep import run_rank_sweep
from pipeline.reporter import ReportWriter


class ExperimentPipeline:
    """Orchestrate the GPT rank analysis for the 4-slit interferometer.

    Parameters
    ----------
    n_eff_list : list of float
        Effective photon counts to test. Each value produces an independent
        simulation and rank sweep, enabling noise-robustness analysis.
    noise : bool
        If True, add Poisson shot noise to simulated probability matrices.
    k_max : int
        Maximum rank K in the sweep (K = 1 ... k_max).
    n_folds : int
        Number of cross-validation folds.
    n_jobs : int
        Number of parallel workers for the CV sweep. -1 uses all CPU cores.
    output_dir : str
        Directory for output files (REPORT.md and results.json).
    seed : int
        Random seed for reproducibility across simulation and CV.
    """

    # ---- Theory constants -------------------------------------------------
    QM_RANK        = 15   # n^2 - 1  for n = 4 slits
    CLASSICAL_RANK = 4    # incoherent classical model
    SECOND_ORDER   = 10   # second-order-only interference

    def __init__(
        self,
        n_eff_list: list  = None,
        noise:      bool  = True,
        k_max:      int   = 20,
        n_folds:    int   = 10,
        n_jobs:     int   = -1,
        output_dir: str   = 'results',
        seed:       int   = 42,
    ):
        self.n_eff_list = [float(v) for v in (n_eff_list or [50_000])]
        self.noise      = noise
        self.k_max      = k_max
        self.n_folds    = n_folds
        self.n_jobs     = n_jobs
        self.output_dir = output_dir
        self.seed       = seed
        self._log       = []

    # ---- Logging ----------------------------------------------------------

    def _log_step(self, msg: str):
        """Print and record a pipeline status message."""
        line = f'[Pipeline] {msg}'
        print(line)
        self._log.append(line)

    # ---- Step 1: Hypothesis -----------------------------------------------

    def formulate_hypothesis(self) -> dict:
        """Formulate the competing hypotheses for the GPT rank test.

        Returns
        -------
        dict
            Keys: H0_QM, H1_classical, H2_2nd_order, criterion, test_statistic.
        """
        self._log_step(
            f'Formulating hypothesis: H0 = QM predicts rank = {self.QM_RANK}, '
            f'H1 = classical rank = {self.CLASSICAL_RANK}, '
            f'H2 = 2nd-order rank ~ {self.SECOND_ORDER}'
        )
        return {
            'H0_QM'          : f'rank(D) = {self.QM_RANK}  (quantum mechanics, n=4)',
            'H1_classical'   : f'rank(D) = {self.CLASSICAL_RANK}  (incoherent classical)',
            'H2_2nd_order'   : f'rank(D) ~ {self.SECOND_ORDER}  (2nd-order interference only)',
            'criterion'      : 'cross-validated chi^2/dof minimised at K = K*',
            'test_statistic' : 'Delta_chi2 = mean_test_chi2(K=14) - mean_test_chi2(K=15)',
        }

    # ---- Step 2: Experiment design ----------------------------------------

    def design_experiment(self) -> dict:
        """Record the experimental design parameters.

        Returns
        -------
        dict
            Full specification of the simulation and analysis setup.
        """
        self._log_step(
            f'Designing experiment: n_eff={self.n_eff_list}, '
            f'K_range=1..{self.k_max}, n_folds={self.n_folds}, noise={self.noise}'
        )
        return {
            'n_slits'          : 4,
            'n_phases'         : 3,
            'n_phase_patterns' : 81,
            'shutter_groups'   : [1, 2, 3, 4],
            'n_eff_list'       : self.n_eff_list,
            'K_range'          : list(range(1, self.k_max + 1)),
            'n_folds'          : self.n_folds,
            'noise'            : self.noise,
            'fit_strategy'     : 'joint bilinear ALS with shared V across groups',
            'k_max'            : self.k_max,
        }

    # ---- Step 3: Simulation -----------------------------------------------

    def simulate_data(self) -> dict:
        """Simulate probability matrices for each N_eff value.

        Returns
        -------
        dict
            Keys are N_eff values; values are the simulator output dicts
            (keyed by n_open, each with 'matrix', 'exact', 'labels', 'n_eff').
        """
        datasets = {}
        for n_eff in self.n_eff_list:
            self._log_step(f'Simulating interferometer data  N_eff={n_eff:.0e}  noise={self.noise}')
            sim  = InterferometerSimulator(n_eff=n_eff, seed=self.seed)
            data = sim.simulate(noise=self.noise)
            for n_open, group in data.items():
                shape = group['matrix'].shape
                self._log_step(f'  n_open={n_open}: matrix shape {shape}')
            datasets[n_eff] = data
        return datasets

    # ---- Step 4: Training and evaluation ----------------------------------

    def train_and_evaluate(self, datasets: dict) -> dict:
        """Run the cross-validated rank sweep for each N_eff dataset.

        Parameters
        ----------
        datasets : dict
            Output of :meth:`simulate_data`.

        Returns
        -------
        dict
            Keys are N_eff values; values are rank-sweep result dicts.
        """
        sweep_results = {}
        for n_eff, data in datasets.items():
            self._log_step(
                f'Running rank sweep  N_eff={n_eff:.0e}  '
                f'K=1..{self.k_max}  n_folds={self.n_folds}  n_jobs={self.n_jobs}'
            )
            t0 = time.time()
            results = run_rank_sweep(
                data    = data,
                K_range = list(range(1, self.k_max + 1)),
                n_folds = self.n_folds,
                n_jobs  = self.n_jobs,
                seed    = self.seed,
            )
            elapsed = time.time() - t0
            self._log_step(f'  Sweep completed in {elapsed:.1f}s')
            sweep_results[n_eff] = results
        return sweep_results

    # ---- Step 5: Analysis -------------------------------------------------

    def analyse_results(self, sweep_results: dict, hypothesis: dict) -> dict:
        """Analyse rank sweep results and evaluate the QM hypothesis.

        For n_open=1, the expected rank is 1 (single-slit state is rank-1).
        For n_open=2, 3, 4 and the joint 'all' group, the expected rank is
        QM_RANK = 15.

        Parameters
        ----------
        sweep_results : dict
            Output of :meth:`train_and_evaluate`.
        hypothesis : dict
            Output of :meth:`formulate_hypothesis`.

        Returns
        -------
        dict
            Nested analysis dict: {n_eff: {group: {best_K, delta_chi2,
            consistent_with_QM, expected_rank, mean_test_chi2}}}.
        """
        analysis = {}

        for n_eff, results in sweep_results.items():
            self._log_step(f'Analysing results  N_eff={n_eff:.0e}')
            analysis[n_eff] = {}

            for grp, k_dict in results.items():
                # Determine expected rank for this group
                if grp == 1:
                    expected_rank = 1
                else:
                    expected_rank = self.QM_RANK

                # Mean test chi^2/dof per K
                mean_test = {
                    K: float(np.mean(k_dict[K]['test']))
                    for K in k_dict
                }

                best_K = min(mean_test, key=mean_test.get)

                # Delta chi^2: improvement from K=14 to K=15
                if 14 in mean_test and 15 in mean_test:
                    delta_chi2 = float(mean_test[14] - mean_test[15])
                else:
                    delta_chi2 = float('nan')

                consistent = (best_K == expected_rank)

                self._log_step(
                    f'  group={grp}: best_K={best_K}  expected={expected_rank}  '
                    f'consistent={consistent}  delta_chi2={delta_chi2:.4f}'
                )

                analysis[n_eff][grp] = {
                    'best_K'             : best_K,
                    'expected_rank'      : expected_rank,
                    'consistent_with_QM' : consistent,
                    'delta_chi2'         : delta_chi2,
                    'mean_test_chi2'     : mean_test,
                }

        return analysis

    # ---- Step 6: Report ---------------------------------------------------

    def write_report(
        self,
        hypothesis:    dict,
        design:        dict,
        analysis:      dict,
        sweep_results: dict,
    ):
        """Write REPORT.md and results.json to the output directory.

        Parameters
        ----------
        hypothesis : dict
            From :meth:`formulate_hypothesis`.
        design : dict
            From :meth:`design_experiment`.
        analysis : dict
            From :meth:`analyse_results`.
        sweep_results : dict
            From :meth:`train_and_evaluate`.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        # Write REPORT.md
        report_path = os.path.join(self.output_dir, 'REPORT.md')
        writer = ReportWriter(
            hypothesis    = hypothesis,
            design        = design,
            analysis      = analysis,
            sweep_results = sweep_results,
            pipeline_log  = self._log,
        )
        writer.write(report_path)
        self._log_step(f'Report written to {report_path}')

        # Write results.json
        json_path = os.path.join(self.output_dir, 'results.json')
        payload = {
            'hypothesis'   : hypothesis,
            'design'       : design,
            'analysis'     : _jsonify(analysis),
            'sweep_results': _jsonify(sweep_results),
        }
        with open(json_path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2)
        self._log_step(f'Results JSON written to {json_path}')

    # ---- Main entry point -------------------------------------------------

    def run(self):
        """Execute the full analysis pipeline end-to-end."""
        hypothesis    = self.formulate_hypothesis()
        design        = self.design_experiment()
        datasets      = self.simulate_data()
        sweep_results = self.train_and_evaluate(datasets)
        analysis      = self.analyse_results(sweep_results, hypothesis)
        self.write_report(hypothesis, design, analysis, sweep_results)

        # ---- Print final results table ------------------------------------
        print('\n=== Results ===')
        for n_eff in self.n_eff_list:
            print(f'  N_eff = {n_eff:.1e}:')
            for grp in sorted(
                (k for k in analysis[n_eff] if k != 'all'),
                key=lambda x: (isinstance(x, str), str(x) if isinstance(x, str) else x)
            ) + ['all']:
                grp_info      = analysis[n_eff][grp]
                best_K        = grp_info['best_K']
                expected_rank = grp_info['expected_rank']
                consistent    = grp_info['consistent_with_QM']
                tick          = '\u2713' if consistent else '\u2717'

                if grp == 'all':
                    label = 'joint'
                elif isinstance(grp, int):
                    label = f'n_open={grp}'
                else:
                    label = str(grp)

                if grp == 1:
                    note = f'(single-slit: trivially rank-1, expected=1)'
                else:
                    note = f'(QM: {self.QM_RANK})'

                print(f'    {tick} {label:10s}: optimal K = {best_K:<3d} {note}')

        print(f'\nPipeline complete. Results in {self.output_dir}/REPORT.md')


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _jsonify(obj):
    """Recursively convert numpy types and non-string dict keys for JSON output."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj
