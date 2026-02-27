"""Experiment 4 – Validation on Real NOAA Catastrophe Data.

Repeats the convergence-scaling and tail-sweep experiments on real US Storm
Events property-damage data (2020–2024) downloaded from NOAA/NCEI.

Part A: Oracle-call convergence at 95th percentile (like Experiment 1).
Part B: Tail-specific excess loss at 90th/95th/97th percentiles (like Experiment 3).

The real data has a much heavier tail (lognormal σ ≈ 1.9 vs. 0.67 for synthetic),
providing a more challenging and realistic test of the quantum advantage.
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

K_VALUES_CONV = [0, 1, 2, 3, 4, 5, 6]
QUANTUM_SHOTS_CONV = 1000
TAIL_PERCENTILES = [0.90, 0.95, 0.97]
N_REPETITIONS = 30


def _exact_excess_loss(midpoints: np.ndarray, probs: np.ndarray,
                       threshold: float) -> float:
    return float(np.dot(probs, np.maximum(0.0, midpoints - threshold)))


def _classical_mc_excess(shape: float, loc: float, scale: float,
                         threshold: float, n_samples: int,
                         rng: np.random.Generator) -> float:
    samples = stats.lognorm.rvs(shape, loc, scale, size=n_samples,
                                random_state=rng.integers(2**31))
    return float(np.mean(np.maximum(0.0, samples - threshold)))


def _run_convergence(
    dist_params: Tuple[float, float, float],
    losses: np.ndarray,
    n_qubits: int,
    k_values: List[int],
    quantum_shots: int,
    n_reps: int,
    seed: int,
) -> Dict:
    """Part A: convergence scaling at 95th percentile."""
    log = get_logger()
    shape, loc, scale = dist_params
    midpoints, probs = discretise_distribution(shape, loc, scale, n_qubits)

    threshold = float(np.percentile(losses, 95))
    exact_excess = _exact_excess_loss(midpoints, probs, threshold)
    log.info("  [4A] Threshold (95th pctl) = $%.0f, exact E[excess] = $%.2f",
             threshold, exact_excess)

    A_circuit, obj_qubit, rescale = build_oracle_A(
        probs, midpoints, catastrophe_threshold=threshold
    )
    exact_prob = exact_amplitude_readout(A_circuit, obj_qubit)
    k_safe = max_safe_k(exact_prob)
    log.info("  [4A] P(|1⟩) = %.6f, k_max = %d", exact_prob, k_safe)

    k_values = [k for k in k_values if k <= k_safe]
    rng = np.random.default_rng(seed)

    quantum_results = []
    classical_results = []

    for k in k_values:
        total_queries = quantum_shots * (2 * k + 1)

        q_errors = []
        for _ in range(n_reps):
            est_prob, info = grover_boosted_estimate(
                A_circuit, obj_qubit, k_iters=k, shots=quantum_shots
            )
            q_errors.append((est_prob * rescale - exact_excess) ** 2)
        q_rmse = float(np.sqrt(np.mean(q_errors)))

        quantum_results.append({
            "k": k, "total_oracle_queries": total_queries,
            "rmse": q_rmse, "shots": quantum_shots,
            "circuit_depth": info.get("circuit_depth", 0),
            "gate_count": info.get("gate_count", 0),
        })

        c_errors = []
        for _ in range(n_reps):
            est = _classical_mc_excess(shape, loc, scale, threshold,
                                       total_queries, rng)
            c_errors.append((est - exact_excess) ** 2)
        c_rmse = float(np.sqrt(np.mean(c_errors)))

        classical_results.append({
            "n_samples": total_queries,
            "total_oracle_queries": total_queries, "rmse": c_rmse,
        })
        log.info("  [4A] k=%d  queries=%5d  Q_RMSE=$%.2f  C_RMSE=$%.2f",
                 k, total_queries, q_rmse, c_rmse)

    return {
        "quantum_results": quantum_results,
        "classical_results": classical_results,
        "exact_excess_loss": exact_excess,
        "threshold": threshold,
        "readout_prob": exact_prob,
        "k_safe": k_safe,
    }


def _run_tail_sweep(
    dist_params: Tuple[float, float, float],
    losses: np.ndarray,
    n_qubits: int,
    percentiles: List[float],
    shots: int,
    n_reps: int,
    seed: int,
) -> Dict:
    """Part B: tail-specific excess loss at multiple percentiles."""
    log = get_logger()
    shape, loc, scale = dist_params
    midpoints, probs = discretise_distribution(shape, loc, scale, n_qubits)
    rng = np.random.default_rng(seed)

    results_by_percentile = {}

    for pct in percentiles:
        threshold = float(np.percentile(losses, pct * 100))
        exact_excess = _exact_excess_loss(midpoints, probs, threshold)
        log.info("  [4B] Pctl %.0f%%: M=$%.0f, exact E[excess]=$%.2f",
                 pct * 100, threshold, exact_excess)

        A_circuit, obj_qubit, rescale = build_oracle_A(
            probs, midpoints, threshold
        )
        exact_prob = exact_amplitude_readout(A_circuit, obj_qubit)
        k_safe = max_safe_k(exact_prob)
        log.info("       P(|1⟩)=%.6f, k_max=%d", exact_prob, k_safe)

        # Classical MC (500 samples)
        n_cl = 500
        c_errors = []
        for _ in range(n_reps):
            est = _classical_mc_excess(shape, loc, scale, threshold, n_cl, rng)
            c_errors.append((est - exact_excess) ** 2)
        c_rmse = float(np.sqrt(np.mean(c_errors)))

        # Quantum k=0
        q0_errors = []
        for _ in range(n_reps):
            est_prob, _ = grover_boosted_estimate(
                A_circuit, obj_qubit, k_iters=0, shots=shots
            )
            q0_errors.append((est_prob * rescale - exact_excess) ** 2)
        q0_rmse = float(np.sqrt(np.mean(q0_errors)))

        # Quantum Grover (budget-matched)
        k_use = min(k_safe, 6)
        g_shots = max(100, shots // (2 * k_use + 1)) if k_use > 0 else shots
        qg_errors = []
        last_info = {}
        for _ in range(n_reps):
            est_prob, info = grover_boosted_estimate(
                A_circuit, obj_qubit, k_iters=k_use, shots=g_shots
            )
            qg_errors.append((est_prob * rescale - exact_excess) ** 2)
            last_info = info
        qg_rmse = float(np.sqrt(np.mean(qg_errors)))

        results_by_percentile[str(pct)] = {
            "percentile": pct,
            "catastrophe_threshold": threshold,
            "exact_excess_loss": exact_excess,
            "readout_prob": exact_prob,
            "max_safe_k": k_safe,
            "k_used": k_use,
            "classical_rmse": c_rmse,
            "quantum_k0_rmse": q0_rmse,
            "quantum_grover_rmse": qg_rmse,
            "total_queries_k0": shots,
            "total_queries_grover": g_shots * (2 * k_use + 1),
            "n_classical_samples": n_cl,
            "grover_shots": g_shots,
            "circuit_depth": last_info.get("circuit_depth", 0),
            "gate_count": last_info.get("gate_count", 0),
        }
        log.info("       C_RMSE=$%.2f  Q_k0=$%.2f  Q_grover(k=%d)=$%.2f",
                 c_rmse, q0_rmse, k_use, qg_rmse)

    return {"percentile_results": results_by_percentile}


def run_experiment4(
    real_dist_params: Tuple[float, float, float],
    real_losses: np.ndarray,
    n_qubits: int = 3,
    n_reps: int = N_REPETITIONS,
    seed: int = 42,
) -> Dict:
    """Run convergence and tail-sweep experiments on real NOAA data.

    Returns dict with 'convergence' and 'tail_sweep' sub-dicts plus metadata.
    """
    log = get_logger()
    log.info("=== Experiment 4: Real NOAA Data Validation ===")
    t0 = time.time()

    shape, loc, scale = real_dist_params
    log.info("  Real data: sigma=%.4f, mu=%.4f (median=$%.0f), %d records",
             shape, np.log(scale), np.exp(np.log(scale)), len(real_losses))

    conv = _run_convergence(
        real_dist_params, real_losses, n_qubits,
        k_values=K_VALUES_CONV,
        quantum_shots=QUANTUM_SHOTS_CONV,
        n_reps=n_reps, seed=seed,
    )

    tail = _run_tail_sweep(
        real_dist_params, real_losses, n_qubits,
        percentiles=TAIL_PERCENTILES,
        shots=8192, n_reps=n_reps, seed=seed,
    )

    runtime = time.time() - t0
    log.info("Experiment 4 completed in %.1f s", runtime)

    return {
        "convergence": conv,
        "tail_sweep": tail,
        "dist_params": {"shape": shape, "loc": loc, "scale": scale},
        "n_records": len(real_losses),
        "n_qubits": n_qubits,
        "n_reps": n_reps,
        "runtime_seconds": runtime,
    }
