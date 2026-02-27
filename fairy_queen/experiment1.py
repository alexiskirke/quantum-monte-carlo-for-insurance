"""Experiment 1 – Noiseless Convergence Scaling (Oracle-Call Budget).

Demonstrates the quadratic speedup of Grover-amplified QAE over Classical
Monte Carlo for estimating tail excess loss.

Design:
  Fix a 95th-percentile catastrophe threshold so P(|1⟩) ≈ 0.012,
  giving k_max = 6 safe Grover iterations.

  Quantum: fix shots = 1000, sweep k = [0, 1, 2, 3, 4, 5, 6].
    Total oracle queries per point = shots × (2k+1).
  Classical: sweep N = matching oracle budgets.

  Plot RMSE vs total oracle queries on log-log.  Classical follows
  O(1/√N); the quantum curve falls faster, demonstrating the advantage.
"""

from __future__ import annotations

import time
import numpy as np
from scipy import stats
from typing import Dict, Tuple, List

from fairy_queen.logging_config import get_logger
from fairy_queen.quantum_circuits import (
    discretise_distribution,
    build_oracle_A,
    exact_amplitude_readout,
    grover_boosted_estimate,
    max_safe_k,
)

K_VALUES = [0, 1, 2, 3, 4, 5, 6]
QUANTUM_SHOTS = 1000
N_REPETITIONS = 30


def _exact_excess_loss(midpoints: np.ndarray, probs: np.ndarray,
                       threshold: float) -> float:
    return float(np.dot(probs, np.maximum(0.0, midpoints - threshold)))


def classical_mc_excess_loss(
    shape: float, loc: float, scale: float,
    threshold: float, n_samples: int, rng: np.random.Generator,
) -> float:
    samples = stats.lognorm.rvs(shape, loc, scale, size=n_samples,
                                random_state=rng.integers(2**31))
    return float(np.mean(np.maximum(0.0, samples - threshold)))


def run_experiment1(
    dist_params: Tuple[float, float, float],
    losses: np.ndarray,
    n_qubits: int = 3,
    k_values: List[int] | None = None,
    quantum_shots: int = QUANTUM_SHOTS,
    n_reps: int = N_REPETITIONS,
    seed: int = 42,
) -> Dict:
    """Run oracle-call convergence experiment at a tail threshold.

    Returns dict with quantum and classical RMSE data keyed by total
    oracle queries, plus metadata.
    """
    log = get_logger()
    log.info("=== Experiment 1: Noiseless Convergence Scaling ===")
    t0 = time.time()

    if k_values is None:
        k_values = K_VALUES

    shape, loc, scale = dist_params
    midpoints, probs = discretise_distribution(shape, loc, scale, n_qubits)

    # 95th-percentile threshold — gives small P(|1⟩) for safe amplification
    threshold = float(np.percentile(losses, 95))
    exact_excess = _exact_excess_loss(midpoints, probs, threshold)
    log.info("  Threshold (95th pctl) = $%.0f", threshold)
    log.info("  Exact discretised E[excess] = $%.2f", exact_excess)

    A_circuit, obj_qubit, rescale = build_oracle_A(
        probs, midpoints, catastrophe_threshold=threshold
    )

    exact_prob = exact_amplitude_readout(A_circuit, obj_qubit)
    k_safe = max_safe_k(exact_prob)
    log.info("  P(|1⟩) = %.6f, θ = %.4f rad, k_max = %d",
             exact_prob, np.arcsin(np.sqrt(exact_prob)), k_safe)

    # Trim k_values to safe range
    k_values = [k for k in k_values if k <= k_safe]

    rng = np.random.default_rng(seed)
    quantum_results = []
    classical_results = []

    for k in k_values:
        total_queries = quantum_shots * (2 * k + 1)

        # Quantum: fixed shots, k Grover iterations
        q_errors = []
        for _ in range(n_reps):
            est_prob, info = grover_boosted_estimate(
                A_circuit, obj_qubit, k_iters=k, shots=quantum_shots
            )
            est_loss = est_prob * rescale
            q_errors.append((est_loss - exact_excess) ** 2)
        q_rmse = float(np.sqrt(np.mean(q_errors)))

        quantum_results.append({
            "k": k,
            "total_oracle_queries": total_queries,
            "rmse": q_rmse,
            "shots": quantum_shots,
            "circuit_depth": info.get("circuit_depth", 0),
            "gate_count": info.get("gate_count", 0),
        })
        log.info("  Quantum k=%d  queries=%5d  RMSE=$%.2f", k, total_queries, q_rmse)

        # Classical: same total budget as oracle queries
        c_errors = []
        for _ in range(n_reps):
            est = classical_mc_excess_loss(
                shape, loc, scale, threshold, total_queries, rng
            )
            c_errors.append((est - exact_excess) ** 2)
        c_rmse = float(np.sqrt(np.mean(c_errors)))

        classical_results.append({
            "n_samples": total_queries,
            "total_oracle_queries": total_queries,
            "rmse": c_rmse,
        })
        log.info("  Classical  N=%5d             RMSE=$%.2f", total_queries, c_rmse)

    runtime = time.time() - t0
    log.info("Experiment 1 completed in %.1f s", runtime)

    return {
        "quantum_results": quantum_results,
        "classical_results": classical_results,
        "exact_excess_loss": exact_excess,
        "threshold": threshold,
        "readout_prob": exact_prob,
        "k_safe": k_safe,
        "n_qubits": n_qubits,
        "n_reps": n_reps,
        "quantum_shots": quantum_shots,
        "rescale": rescale,
        "runtime_seconds": runtime,
    }
