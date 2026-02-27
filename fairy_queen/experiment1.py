"""Experiment 1 – Noiseless Convergence Scaling.

Compare Classical Monte Carlo vs Quantum Amplitude Estimation on a
noiseless simulator.

For each oracle-query budget N:
  - Classical MC draws N samples from the fitted loss distribution.
  - Quantum (k=0) uses N shot-based measurements on the oracle circuit.
  - Quantum (exact) uses a single statevector readout (idealised QAE limit).

Metric: RMSE of the estimated expected loss against the discretised exact value.

Expected outcome:
  - Classical MC: RMSE ∝ O(1/√N)  (central limit theorem)
  - Quantum k=0: RMSE ∝ O(1/√N)  (same shot-noise regime)
  - Quantum exact: RMSE = 0        (the O(1/N) idealised limit)
  - A reference O(1/N) line shows the theoretical QAE advantage.
"""

from __future__ import annotations

import time
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple

from fairy_queen.logging_config import get_logger
from fairy_queen.quantum_circuits import (
    discretise_distribution,
    build_oracle_A,
    exact_amplitude_readout,
    grover_boosted_estimate,
)

SAMPLE_COUNTS = [100, 500, 1000, 5000, 10000]
N_REPETITIONS = 30


def _discretised_exact_expected_loss(
    midpoints: np.ndarray, probs: np.ndarray
) -> float:
    return float(np.dot(midpoints, probs))


def classical_mc_estimate(
    shape: float, loc: float, scale: float,
    n_samples: int, rng: np.random.Generator,
) -> float:
    samples = stats.lognorm.rvs(shape, loc, scale, size=n_samples,
                                random_state=rng.integers(2**31))
    return float(np.mean(samples))


def run_experiment1(
    dist_params: Tuple[float, float, float],
    n_qubits: int = 3,
    sample_counts: List[int] | None = None,
    n_reps: int = N_REPETITIONS,
    seed: int = 42,
) -> Dict:
    """Run convergence-scaling experiment.

    Returns dict with classical_rmse, quantum_k0_rmse, quantum_exact_loss,
    sample_counts, and circuit metadata.
    """
    log = get_logger()
    log.info("=== Experiment 1: Noiseless Convergence Scaling ===")
    t0 = time.time()

    if sample_counts is None:
        sample_counts = SAMPLE_COUNTS

    shape, loc, scale = dist_params
    midpoints, probs = discretise_distribution(shape, loc, scale, n_qubits)
    exact_loss = _discretised_exact_expected_loss(midpoints, probs)
    log.info("Discretised exact E[Loss] = $%.2f  (%d bins)", exact_loss, len(probs))

    # Build oracle once
    A_circuit, obj_qubit, rescale = build_oracle_A(
        probs, midpoints, catastrophe_threshold=0.0
    )

    # Quantum exact (statevector) — O(1) oracle calls, idealised QAE limit
    exact_prob = exact_amplitude_readout(A_circuit, obj_qubit)
    quantum_exact_loss = exact_prob * rescale
    log.info("  Quantum exact (statevector): $%.2f  (error=$%.4f)",
             quantum_exact_loss, abs(quantum_exact_loss - exact_loss))

    # Classical MC sweep
    classical_rmse = []
    rng = np.random.default_rng(seed)
    for N in sample_counts:
        errors = [(classical_mc_estimate(shape, loc, scale, N, rng) - exact_loss) ** 2
                  for _ in range(n_reps)]
        rmse = float(np.sqrt(np.mean(errors)))
        classical_rmse.append(rmse)
        log.info("  Classical N=%6d  RMSE=$%.2f", N, rmse)

    # Quantum k=0 (shot-based, no Grover) — same oracle-call budget as classical
    quantum_k0_rmse = []
    circuit_info_list = []
    for N in sample_counts:
        errors = []
        last_info = {}
        for _ in range(n_reps):
            est_prob, info = grover_boosted_estimate(
                A_circuit, obj_qubit, k_iters=0, shots=N
            )
            est_loss = est_prob * rescale
            errors.append((est_loss - exact_loss) ** 2)
            last_info = info
        rmse = float(np.sqrt(np.mean(errors)))
        quantum_k0_rmse.append(rmse)
        circuit_info_list.append(last_info)
        log.info("  Quantum  k=0  shots=%6d  RMSE=$%.2f", N, rmse)

    runtime = time.time() - t0
    log.info("Experiment 1 completed in %.1f s", runtime)

    return {
        "sample_counts": sample_counts,
        "classical_rmse": classical_rmse,
        "quantum_k0_rmse": quantum_k0_rmse,
        "quantum_exact_loss": quantum_exact_loss,
        "exact_expected_loss": exact_loss,
        "n_qubits": n_qubits,
        "n_reps": n_reps,
        "circuit_info": circuit_info_list,
        "runtime_seconds": runtime,
    }
