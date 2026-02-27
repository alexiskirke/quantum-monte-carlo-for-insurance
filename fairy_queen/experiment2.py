"""Experiment 2 – Realistic NISQ Noise Model.

Re-run QAE under depolarising noise at three severity levels (low / medium /
high) and compare accuracy degradation against the noiseless baseline.

Establishes the fault-tolerance requirements for production deployment of
quantum tail-risk pricing.
"""

from __future__ import annotations

import time
import numpy as np
from typing import Dict, Tuple

from fairy_queen.logging_config import get_logger
from fairy_queen.quantum_circuits import (
    discretise_distribution,
    build_estimation_circuit,
    exact_amplitude_readout,
    grover_boosted_estimate,
    get_noisy_backend,
    NOISE_PRESETS,
)


def run_experiment2(
    dist_params: Tuple[float, float, float],
    n_qubits: int = 3,
    shots: int = 8192,
    n_reps: int = 20,
    seed: int = 42,
) -> Dict:
    """Run QAE under multiple noise levels.

    Uses a tail threshold (95th percentile) where the readout probability
    is small enough for safe Grover amplification (k=1), then compares
    the noise degradation against the noiseless baseline.

    Returns a dict keyed by noise level with estimated errors and circuit info.
    """
    log = get_logger()
    log.info("=== Experiment 2: NISQ Noise Model ===")
    t0 = time.time()

    shape, loc, scale = dist_params
    midpoints, probs = discretise_distribution(shape, loc, scale, n_qubits)

    # Use expected loss (threshold=0) with k=0 for safe comparison
    exact_loss = float(np.dot(midpoints, probs))
    A_circuit, obj_qubit, rescale = build_estimation_circuit(
        probs, midpoints, catastrophe_threshold=0.0
    )

    # Determine safe k from the exact probability
    exact_prob = exact_amplitude_readout(A_circuit, obj_qubit)
    theta = np.arcsin(np.sqrt(exact_prob))
    max_safe_k = max(0, int(np.pi / (2 * theta) - 1) // 2) if theta > 1e-10 else 0
    k_iters = min(max_safe_k, 2)
    log.info("  P(|1⟩)=%.4f, θ=%.4f, max safe k=%d, using k=%d",
             exact_prob, theta, max_safe_k, k_iters)

    results_by_level = {}

    for level_name in ["noiseless", "low", "medium", "high"]:
        errors = []
        raw_estimates = []

        for rep in range(n_reps):
            if level_name == "noiseless":
                est_prob, info = grover_boosted_estimate(
                    A_circuit, obj_qubit, k_iters=k_iters, shots=shots,
                    backend=None,
                )
            else:
                backend, nm = get_noisy_backend(level_name)
                est_prob, info = grover_boosted_estimate(
                    A_circuit, obj_qubit, k_iters=k_iters, shots=shots,
                    backend=backend,
                )
            est_loss = est_prob * rescale
            errors.append((est_loss - exact_loss) ** 2)
            raw_estimates.append(est_loss)

        rmse = float(np.sqrt(np.mean(errors)))
        mean_est = float(np.mean(raw_estimates))
        std_est = float(np.std(raw_estimates))

        noise_params = NOISE_PRESETS.get(level_name, {"p1q": 0, "p2q": 0, "p_ro": 0})

        results_by_level[level_name] = {
            "rmse": rmse,
            "mean_estimate": mean_est,
            "std_estimate": std_est,
            "exact_loss": exact_loss,
            "n_reps": n_reps,
            "noise_params": noise_params,
            "circuit_depth": info.get("circuit_depth", 0),
            "gate_count": info.get("gate_count", 0),
        }
        log.info("  Noise=%-10s  RMSE=$%.2f  mean=$%.2f  std=$%.2f",
                 level_name, rmse, mean_est, std_est)

    runtime = time.time() - t0
    log.info("Experiment 2 completed in %.1f s", runtime)

    results_by_level["_meta"] = {
        "k_iters": k_iters,
        "shots": shots,
        "n_qubits": n_qubits,
        "runtime_seconds": runtime,
    }
    return results_by_level
