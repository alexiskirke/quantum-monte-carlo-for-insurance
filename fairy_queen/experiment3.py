"""Experiment 3 – Tail-Specific Excess Loss (the Insurance Metric).

Compute  E[max(0, X − M)]  for catastrophe thresholds M at the 90th, 95th,
and 99th percentiles of the fitted loss distribution.

For deep-tail thresholds the QAE probability P(|1⟩) is small, so Grover
amplification (k ≥ 1) is safe (no angle aliasing) and provides genuine
quadratic speedup over shot noise.  Classical MC struggles because so few
samples exceed M.
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
)

TAIL_PERCENTILES = [0.90, 0.95, 0.99]


def _exact_excess_loss_discretised(
    midpoints: np.ndarray,
    probs: np.ndarray,
    catastrophe_threshold: float,
) -> float:
    excess = np.maximum(0.0, midpoints - catastrophe_threshold)
    return float(np.dot(excess, probs))


def _max_safe_k(prob: float) -> int:
    """Largest Grover iteration count k such that (2k+1)θ < π/2."""
    if prob <= 0 or prob >= 1:
        return 0
    theta = np.arcsin(np.sqrt(prob))
    if theta < 1e-10:
        return 100
    max_2k1 = np.pi / (2 * theta)
    return max(0, int(max_2k1 - 1) // 2)


def classical_mc_excess_loss(
    shape: float, loc: float, scale: float,
    catastrophe_threshold: float,
    n_samples: int, rng: np.random.Generator,
) -> float:
    samples = stats.lognorm.rvs(shape, loc, scale, size=n_samples,
                                random_state=rng.integers(2**31))
    excess = np.maximum(0.0, samples - catastrophe_threshold)
    return float(np.mean(excess))


def run_experiment3(
    dist_params: Tuple[float, float, float],
    losses: np.ndarray,
    n_qubits: int = 3,
    percentiles: List[float] | None = None,
    n_classical_samples: int = 500,
    shots: int = 8192,
    n_reps: int = 30,
    seed: int = 42,
) -> Dict:
    """Run excess-loss comparison across tail percentiles.

    For each percentile, automatically determines the maximum safe Grover
    iteration count to avoid angle aliasing, then compares:
      - Classical MC (N samples)
      - Quantum k=0 (shot-based, no Grover)
      - Quantum k=k_safe (Grover-amplified, if k_safe > 0)
      - Quantum exact (statevector readout)
    """
    log = get_logger()
    log.info("=== Experiment 3: Tail-Specific Excess Loss ===")
    t0 = time.time()

    if percentiles is None:
        percentiles = TAIL_PERCENTILES

    shape, loc, scale = dist_params
    midpoints, probs = discretise_distribution(shape, loc, scale, n_qubits)
    rng = np.random.default_rng(seed)

    results_by_percentile = {}

    for pct in percentiles:
        catastrophe_threshold = float(np.percentile(losses, pct * 100))
        log.info("  Percentile %.0f%% → threshold M = $%.0f", pct * 100,
                 catastrophe_threshold)

        exact_excess = _exact_excess_loss_discretised(
            midpoints, probs, catastrophe_threshold
        )
        log.info("    Exact discretised E[excess] = $%.4f", exact_excess)

        # Build oracle for this threshold
        A_circuit, obj_qubit, rescale = build_oracle_A(
            probs, midpoints, catastrophe_threshold
        )

        # Quantum exact (statevector)
        exact_prob = exact_amplitude_readout(A_circuit, obj_qubit)
        quantum_exact_est = exact_prob * rescale
        k_safe = _max_safe_k(exact_prob)
        log.info("    P(|1⟩) = %.6f → max safe k = %d", exact_prob, k_safe)

        # Classical MC
        classical_errors = []
        classical_estimates = []
        for _ in range(n_reps):
            est = classical_mc_excess_loss(
                shape, loc, scale, catastrophe_threshold,
                n_classical_samples, rng,
            )
            classical_errors.append((est - exact_excess) ** 2)
            classical_estimates.append(est)
        classical_rmse = float(np.sqrt(np.mean(classical_errors)))
        classical_var = float(np.var(classical_estimates))

        # Quantum k=0 (shot-based, no Grover amplification)
        q_k0_errors = []
        q_k0_estimates = []
        for _ in range(n_reps):
            est_prob, _ = grover_boosted_estimate(
                A_circuit, obj_qubit, k_iters=0, shots=shots
            )
            q_k0_estimates.append(est_prob * rescale)
            q_k0_errors.append((est_prob * rescale - exact_excess) ** 2)
        q_k0_rmse = float(np.sqrt(np.mean(q_k0_errors)))
        q_k0_var = float(np.var(q_k0_estimates))

        # Quantum k=k_safe (Grover-amplified) — same total oracle queries
        k_use = min(k_safe, 3)
        q_grover_errors = []
        q_grover_estimates = []
        grover_shots = max(100, shots // (2 * k_use + 1)) if k_use > 0 else shots
        last_info = {}
        for _ in range(n_reps):
            est_prob, info = grover_boosted_estimate(
                A_circuit, obj_qubit, k_iters=k_use, shots=grover_shots
            )
            q_grover_estimates.append(est_prob * rescale)
            q_grover_errors.append((est_prob * rescale - exact_excess) ** 2)
            last_info = info
        q_grover_rmse = float(np.sqrt(np.mean(q_grover_errors)))
        q_grover_var = float(np.var(q_grover_estimates))

        total_queries_k0 = shots
        total_queries_grover = grover_shots * (2 * k_use + 1)

        results_by_percentile[str(pct)] = {
            "percentile": pct,
            "catastrophe_threshold": catastrophe_threshold,
            "exact_excess_loss": exact_excess,
            "quantum_exact_estimate": quantum_exact_est,
            "readout_prob": exact_prob,
            "max_safe_k": k_safe,
            "k_used": k_use,
            "classical_rmse": classical_rmse,
            "classical_variance": classical_var,
            "classical_mean_estimate": float(np.mean(classical_estimates)),
            "quantum_k0_rmse": q_k0_rmse,
            "quantum_k0_variance": q_k0_var,
            "quantum_k0_mean_estimate": float(np.mean(q_k0_estimates)),
            "quantum_grover_rmse": q_grover_rmse,
            "quantum_grover_variance": q_grover_var,
            "quantum_grover_mean_estimate": float(np.mean(q_grover_estimates)),
            "total_queries_k0": total_queries_k0,
            "total_queries_grover": total_queries_grover,
            "n_classical_samples": n_classical_samples,
            "circuit_depth": last_info.get("circuit_depth", 0),
            "gate_count": last_info.get("gate_count", 0),
        }
        log.info("    Classical      RMSE=$%.4f  var=$%.4f", classical_rmse, classical_var)
        log.info("    Quantum k=0    RMSE=$%.4f  var=$%.4f", q_k0_rmse, q_k0_var)
        log.info("    Quantum k=%d    RMSE=$%.4f  var=$%.4f", k_use, q_grover_rmse, q_grover_var)

    runtime = time.time() - t0
    log.info("Experiment 3 completed in %.1f s", runtime)

    return {
        "percentile_results": results_by_percentile,
        "n_qubits": n_qubits,
        "n_reps": n_reps,
        "runtime_seconds": runtime,
    }
