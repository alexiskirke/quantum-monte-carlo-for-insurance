#!/usr/bin/env python3
"""Project Fairy Queen – main pipeline entry point.

Runs all six experiments and writes results to ./results/.

Usage:
    python run_pipeline.py                   # default MEDIUM verbosity
    python run_pipeline.py --verbosity HIGH  # full debug output
    python run_pipeline.py --verbosity LOW   # quiet mode
"""

from __future__ import annotations

import argparse
import time
import numpy as np

from fairy_queen.logging_config import setup_logging, Verbosity
from fairy_queen.data_pipeline import get_loss_data, get_synthetic_loss_data, get_real_loss_data
from fairy_queen.experiment1 import run_experiment1
from fairy_queen.experiment2 import run_experiment2
from fairy_queen.experiment3 import run_experiment3
from fairy_queen.experiment4 import run_experiment4
from fairy_queen.experiment5 import run_experiment5
from fairy_queen.experiment6 import run_experiment6
from fairy_queen.experiment7 import run_experiment7
from fairy_queen.results import save_all

GLOBAL_SEED = 42
N_QUBITS = 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Project Fairy Queen pipeline")
    parser.add_argument(
        "--verbosity", type=str, default="MEDIUM",
        choices=["LOW", "MEDIUM", "HIGH"],
        help="Logging verbosity level",
    )
    parser.add_argument("--n-qubits", type=int, default=N_QUBITS,
                        help="Qubits for distribution encoding (default 3 → 8 bins)")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--skip-real", action="store_true",
                        help="Skip Experiment 4 (real NOAA data)")
    args = parser.parse_args()

    log = setup_logging(args.verbosity)
    np.random.seed(args.seed)

    log.warning("=" * 60)
    log.warning("  Project Fairy Queen – Quantum Tail-Risk Pricing Pipeline")
    log.warning("=" * 60)

    t_start = time.time()

    # --- Synthetic data (Experiments 1–3, 5–6) ---
    log.info("Step 1/8: Generating synthetic loss data ...")
    synth_losses, synth_params = get_synthetic_loss_data()
    log.info("  %d synthetic records, params: shape=%.4f loc=%.4f scale=%.4f",
             len(synth_losses), *synth_params)

    log.info("Step 2/8: Experiment 1 – Noiseless Convergence Scaling ...")
    exp1 = run_experiment1(synth_params, synth_losses,
                           n_qubits=args.n_qubits, seed=args.seed)

    log.info("Step 3/8: Experiment 2 – NISQ Noise Model ...")
    exp2 = run_experiment2(synth_params, synth_losses,
                           n_qubits=args.n_qubits, seed=args.seed)

    log.info("Step 4/8: Experiment 3 – Tail-Specific Excess Loss ...")
    exp3 = run_experiment3(synth_params, synth_losses,
                           n_qubits=args.n_qubits, seed=args.seed)

    # --- Real data (Experiment 4) ---
    exp4 = None
    if not args.skip_real:
        log.info("Step 5/8: Experiment 4 – Real NOAA Data Validation ...")
        try:
            real_losses, real_params = get_real_loss_data()
            log.info("  %d real records, params: shape=%.4f loc=%.4f scale=%.4f",
                     len(real_losses), *real_params)
            exp4 = run_experiment4(real_params, real_losses,
                                   n_qubits=args.n_qubits, seed=args.seed)
        except Exception as exc:
            log.warning("Experiment 4 failed (%s); skipping real-data section.", exc)
    else:
        log.info("Step 5/8: Skipped (--skip-real)")

    # --- Fair comparison (Experiment 5 on synthetic) ---
    log.info("Step 6/8: Experiment 5 – Fair Budget-Matched Comparison (synthetic) ...")
    exp5_synth = run_experiment5(synth_params, synth_losses,
                                n_qubits=args.n_qubits, seed=args.seed,
                                label="synthetic")

    # --- Fair comparison (Experiment 5 on real, if available) ---
    exp5_real = None
    if exp4 is not None:
        log.info("Step 6b/8: Experiment 5 – Fair Budget-Matched Comparison (real) ...")
        exp5_real = run_experiment5(real_params, real_losses,
                                   n_qubits=args.n_qubits, seed=args.seed,
                                   label="real")

    # --- Qubit sweep (Experiment 6) ---
    log.info("Step 7/9: Experiment 6 – Qubit Scaling Sweep (synthetic) ...")
    exp6 = run_experiment6(synth_params, synth_losses, seed=args.seed,
                           label="synthetic")

    # --- Empirical PMF (Experiment 7, on real NOAA data) ---
    exp7 = None
    if exp4 is not None:
        log.info("Step 8/9: Experiment 7 – Empirical PMF Comparison (NOAA) ...")
        exp7 = run_experiment7(real_losses, n_qubits=args.n_qubits,
                               seed=args.seed)
    else:
        log.info("Step 8/9: Skipped (no real data)")

    # --- Save ---
    log.info("Step 9/9: Saving results ...")
    save_all(exp1, exp2, exp3, exp4, exp5_synth, exp5_real, exp6, exp7)

    elapsed = time.time() - t_start
    log.warning("Pipeline complete in %.1f s.  Results in ./results/", elapsed)


if __name__ == "__main__":
    main()
