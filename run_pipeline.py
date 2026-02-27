#!/usr/bin/env python3
"""Project Fairy Queen – main pipeline entry point.

Runs all three experiments and writes results to ./results/.

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
from fairy_queen.data_pipeline import get_loss_data
from fairy_queen.experiment1 import run_experiment1
from fairy_queen.experiment2 import run_experiment2
from fairy_queen.experiment3 import run_experiment3
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
    args = parser.parse_args()

    log = setup_logging(args.verbosity)
    np.random.seed(args.seed)

    log.warning("=" * 60)
    log.warning("  Project Fairy Queen – Quantum Tail-Risk Pricing Pipeline")
    log.warning("=" * 60)

    t_start = time.time()

    # --- Data ---
    log.info("Step 1/4: Acquiring loss data and fitting distribution ...")
    losses, dist_params = get_loss_data()
    log.info("  %d loss records, lognormal params: shape=%.4f loc=%.4f scale=%.4f",
             len(losses), *dist_params)

    # --- Experiment 1 ---
    log.info("Step 2/4: Experiment 1 – Noiseless Convergence Scaling ...")
    exp1 = run_experiment1(dist_params, losses, n_qubits=args.n_qubits,
                           seed=args.seed)

    # --- Experiment 2 ---
    log.info("Step 3/4: Experiment 2 – NISQ Noise Model ...")
    exp2 = run_experiment2(dist_params, losses, n_qubits=args.n_qubits,
                           seed=args.seed)

    # --- Experiment 3 ---
    log.info("Step 4/4: Experiment 3 – Tail-Specific Excess Loss ...")
    exp3 = run_experiment3(dist_params, losses, n_qubits=args.n_qubits,
                           seed=args.seed)

    # --- Save ---
    save_all(exp1, exp2, exp3)

    elapsed = time.time() - t_start
    log.warning("Pipeline complete in %.1f s.  Results in ./results/", elapsed)


if __name__ == "__main__":
    main()
