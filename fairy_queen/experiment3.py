"""Experiment 3 – Tail-Specific Excess Loss (the Insurance Metric).

Compute  E[max(0, X − M)]  for catastrophe thresholds M at the 90th, 95th,
and 99th percentiles of the fitted loss distribution.

With amplitude encoding, P(|1⟩) is genuinely small for tail events, so
Grover amplification (k ≥ 1) is safe.  The deeper the tail, the smaller
P(|1⟩), the more Grover iterations are possible, and the larger the
quantum advantage — precisely the paper's thesis.
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
from fairy_queen.experiment5 import analytic_lognormal_excess

TAIL_PERCENTILES = [0.90, 0.95, 0.97]


def _exact_excess_loss_discretised(
    midpoints: np.ndarray,
    probs: np.ndarray,
    catastrophe_threshold: float,
) -> float:
    excess = np.maximum(0.0, midpoints - catastrophe_threshold)
    return float(np.dot(excess, probs))


def classical_mc_excess_loss(
    shape: float, loc: float, scale: float,
    catastrophe_threshold: float,
    n_samples: int, rng: np.random.Generator,
) -> float:
    samples = stats.lognorm.rvs(shape, loc, scale, size=n_samples,
                                random_state=rng.integers(2**31))
    excess = np.maximum(0.0, samples - catastrophe_threshold)
    return float(np.mean(excess))


def _classical_mc_on_bins(midpoints: np.ndarray, probs: np.ndarray,
                          threshold: float, n_samples: int,
                          rng: np.random.Generator) -> float:
    indices = rng.choice(len(midpoints), size=n_samples, p=probs)
    return float(np.mean(np.maximum(0.0, midpoints[indices] - threshold)))


def run_experiment3(
    dist_params: Tuple[float, float, float],
    losses: np.ndarray,
    n_qubits: int = 3,
    percentiles: List[float] | None = None,
    shots: int = 8192,
    n_reps: int = 30,
    seed: int = 42,
) -> Dict:
    """Run excess-loss comparison across tail percentiles with Grover amplification.

    All methods receive the same total query/sample budget for fair comparison.
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
        exact_bins = _exact_excess_loss_discretised(
            midpoints, probs, catastrophe_threshold
        )
        analytic = analytic_lognormal_excess(shape, loc, scale,
                                             catastrophe_threshold)
        disc_err = abs(exact_bins - analytic)
        log.info("  Pctl %.0f%%  M=$%.0f  analytic=$%.2f  bins=$%.2f  "
                 "disc_err=$%.2f",
                 pct * 100, catastrophe_threshold, analytic, exact_bins,
                 disc_err)

        A_circuit, obj_qubit, rescale = build_oracle_A(
            probs, midpoints, catastrophe_threshold
        )

        exact_prob = exact_amplitude_readout(A_circuit, obj_qubit)
        quantum_exact_est = exact_prob * rescale
        k_safe = max_safe_k(exact_prob)
        log.info("    P(|1⟩) = %.6f → max safe k = %d", exact_prob, k_safe)

        # Quantum k=0 (shot-based, no Grover amplification)
        total_queries_k0 = shots
        q_k0_errors = []
        q_k0_estimates = []
        for _ in range(n_reps):
            est_prob, _ = grover_boosted_estimate(
                A_circuit, obj_qubit, k_iters=0, shots=shots
            )
            q_k0_estimates.append(est_prob * rescale)
            q_k0_errors.append((est_prob * rescale - exact_bins) ** 2)
        q_k0_rmse = float(np.sqrt(np.mean(q_k0_errors)))

        # Quantum k=k_safe (Grover-amplified), budget-matched to k=0
        k_use = min(k_safe, 6)
        grover_shots = max(100, shots // (2 * k_use + 1)) if k_use > 0 else shots
        total_queries_grover = grover_shots * (2 * k_use + 1)
        qg_errs_bins = []
        q_grover_estimates = []
        last_info = {}
        for _ in range(n_reps):
            est_prob, info = grover_boosted_estimate(
                A_circuit, obj_qubit, k_iters=k_use, shots=grover_shots
            )
            est = est_prob * rescale
            q_grover_estimates.append(est)
            qg_errs_bins.append((est - exact_bins) ** 2)
            last_info = info
        qg_rmse_bins = float(np.sqrt(np.mean(qg_errs_bins)))

        # Classical MC (continuous) — budget-matched
        n_classical_samples = total_queries_grover
        c_errs_anal = []
        classical_estimates = []
        for _ in range(n_reps):
            est = classical_mc_excess_loss(
                shape, loc, scale, catastrophe_threshold,
                n_classical_samples, rng,
            )
            c_errs_anal.append((est - analytic) ** 2)
            classical_estimates.append(est)
        c_rmse_anal = float(np.sqrt(np.mean(c_errs_anal)))

        # Classical MC on same bins — budget-matched
        cb_errs = []
        for _ in range(n_reps):
            est = _classical_mc_on_bins(midpoints, probs,
                                        catastrophe_threshold,
                                        n_classical_samples, rng)
            cb_errs.append((est - exact_bins) ** 2)
        cb_rmse = float(np.sqrt(np.mean(cb_errs)))

        results_by_percentile[str(pct)] = {
            "percentile": pct,
            "catastrophe_threshold": catastrophe_threshold,
            "exact_excess_loss": exact_bins,
            "analytic_excess_loss": analytic,
            "disc_error": disc_err,
            "quantum_exact_estimate": quantum_exact_est,
            "readout_prob": exact_prob,
            "max_safe_k": k_safe,
            "k_used": k_use,
            "classical_rmse": cb_rmse,
            "classical_cont_rmse": c_rmse_anal,
            "classical_bins_rmse": cb_rmse,
            "classical_mean_estimate": float(np.mean(classical_estimates)),
            "quantum_k0_rmse": q_k0_rmse,
            "quantum_k0_mean_estimate": float(np.mean(q_k0_estimates)),
            "quantum_grover_rmse": qg_rmse_bins,
            "quantum_grover_mean_estimate": float(np.mean(q_grover_estimates)),
            "total_queries_k0": total_queries_k0,
            "total_queries_grover": total_queries_grover,
            "n_classical_samples": n_classical_samples,
            "grover_shots": grover_shots,
            "circuit_depth": last_info.get("circuit_depth", 0),
            "gate_count": last_info.get("gate_count", 0),
        }
        log.info("    Q(bins)=$%.2f  C_bins(bins)=$%.2f  C_cont(anal)=$%.2f  "
                 "Q/C_bins=%.1fx",
                 qg_rmse_bins, cb_rmse, c_rmse_anal,
                 cb_rmse / qg_rmse_bins if qg_rmse_bins > 0 else float('inf'))

    runtime = time.time() - t0
    log.info("Experiment 3 completed in %.1f s", runtime)

    return {
        "percentile_results": results_by_percentile,
        "n_qubits": n_qubits,
        "n_reps": n_reps,
        "runtime_seconds": runtime,
    }
