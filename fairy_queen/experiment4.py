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
from fairy_queen.experiment5 import analytic_lognormal_excess

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


def _classical_mc_on_bins(midpoints: np.ndarray, probs: np.ndarray,
                          threshold: float, n_samples: int,
                          rng: np.random.Generator) -> float:
    """Classical MC sampling from the same discretised distribution as quantum."""
    indices = rng.choice(len(midpoints), size=n_samples, p=probs)
    return float(np.mean(np.maximum(0.0, midpoints[indices] - threshold)))


def _run_convergence(
    dist_params: Tuple[float, float, float],
    losses: np.ndarray,
    n_qubits: int,
    k_values: List[int],
    quantum_shots: int,
    n_reps: int,
    seed: int,
) -> Dict:
    """Part A: convergence scaling at 95th percentile.

    Reports RMSE vs both analytic (continuous) and exact-on-bins truths.
    """
    log = get_logger()
    shape, loc, scale = dist_params
    midpoints, probs = discretise_distribution(shape, loc, scale, n_qubits)

    threshold = float(np.percentile(losses, 95))
    exact_bins = _exact_excess_loss(midpoints, probs, threshold)
    analytic = analytic_lognormal_excess(shape, loc, scale, threshold)
    disc_err = abs(exact_bins - analytic)
    log.info("  [4A] Threshold (95th pctl) = $%.0f", threshold)
    log.info("  [4A] analytic=$%.2f  exact_bins=$%.2f  disc_err=$%.2f",
             analytic, exact_bins, disc_err)

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
    classical_bins_results = []

    for k in k_values:
        total_queries = quantum_shots * (2 * k + 1)

        # Quantum AE
        q_errs_bins, q_errs_analytic = [], []
        for _ in range(n_reps):
            est_prob, info = grover_boosted_estimate(
                A_circuit, obj_qubit, k_iters=k, shots=quantum_shots
            )
            est = est_prob * rescale
            q_errs_bins.append((est - exact_bins) ** 2)
            q_errs_analytic.append((est - analytic) ** 2)
        q_rmse_bins = float(np.sqrt(np.mean(q_errs_bins)))
        q_rmse_analytic = float(np.sqrt(np.mean(q_errs_analytic)))

        quantum_results.append({
            "k": k, "total_oracle_queries": total_queries,
            "rmse_vs_bins": q_rmse_bins,
            "rmse_vs_analytic": q_rmse_analytic,
            "rmse": q_rmse_bins,
            "shots": quantum_shots,
            "circuit_depth": info.get("circuit_depth", 0),
            "gate_count": info.get("gate_count", 0),
        })

        # Classical MC (continuous lognormal)
        c_errs_bins, c_errs_analytic = [], []
        for _ in range(n_reps):
            est = _classical_mc_excess(shape, loc, scale, threshold,
                                       total_queries, rng)
            c_errs_bins.append((est - exact_bins) ** 2)
            c_errs_analytic.append((est - analytic) ** 2)
        c_rmse_bins = float(np.sqrt(np.mean(c_errs_bins)))
        c_rmse_analytic = float(np.sqrt(np.mean(c_errs_analytic)))

        classical_results.append({
            "n_samples": total_queries,
            "total_oracle_queries": total_queries,
            "rmse_vs_bins": c_rmse_bins,
            "rmse_vs_analytic": c_rmse_analytic,
            "rmse": c_rmse_analytic,
        })

        # Classical MC on bins (same oracle as quantum)
        cb_errs = []
        for _ in range(n_reps):
            est = _classical_mc_on_bins(midpoints, probs, threshold,
                                        total_queries, rng)
            cb_errs.append((est - exact_bins) ** 2)
        cb_rmse = float(np.sqrt(np.mean(cb_errs)))

        classical_bins_results.append({
            "n_samples": total_queries,
            "total_oracle_queries": total_queries, "rmse_vs_bins": cb_rmse,
        })

        log.info("  [4A] k=%d  queries=%5d  Q(bins)=$%.2f  Q(anal)=$%.2f  "
                 "C_cont(anal)=$%.2f  C_bins(bins)=$%.2f",
                 k, total_queries, q_rmse_bins, q_rmse_analytic,
                 c_rmse_analytic, cb_rmse)

    return {
        "quantum_results": quantum_results,
        "classical_results": classical_results,
        "classical_bins_results": classical_bins_results,
        "exact_excess_loss": exact_bins,
        "analytic_excess_loss": analytic,
        "disc_error": disc_err,
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
    """Part B: tail-specific excess loss at multiple percentiles.

    Reports RMSE vs both analytic (continuous) and exact-on-bins truths,
    plus classical MC on the same bins for apples-to-apples comparison.
    """
    log = get_logger()
    shape, loc, scale = dist_params
    midpoints, probs = discretise_distribution(shape, loc, scale, n_qubits)
    rng = np.random.default_rng(seed)

    results_by_percentile = {}

    for pct in percentiles:
        threshold = float(np.percentile(losses, pct * 100))
        exact_bins = _exact_excess_loss(midpoints, probs, threshold)
        analytic = analytic_lognormal_excess(shape, loc, scale, threshold)
        disc_err = abs(exact_bins - analytic)
        log.info("  [4B] Pctl %.0f%%: M=$%.0f  analytic=$%.2f  "
                 "exact_bins=$%.2f  disc_err=$%.2f",
                 pct * 100, threshold, analytic, exact_bins, disc_err)

        A_circuit, obj_qubit, rescale = build_oracle_A(
            probs, midpoints, threshold
        )
        exact_prob = exact_amplitude_readout(A_circuit, obj_qubit)
        k_safe = max_safe_k(exact_prob)
        log.info("       P(|1⟩)=%.6f, k_max=%d", exact_prob, k_safe)

        # Quantum k=0
        q0_errs_bins, q0_errs_anal = [], []
        for _ in range(n_reps):
            est_prob, _ = grover_boosted_estimate(
                A_circuit, obj_qubit, k_iters=0, shots=shots
            )
            est = est_prob * rescale
            q0_errs_bins.append((est - exact_bins) ** 2)
            q0_errs_anal.append((est - analytic) ** 2)
        q0_rmse_bins = float(np.sqrt(np.mean(q0_errs_bins)))

        # Quantum Grover (budget-matched to k=0)
        k_use = min(k_safe, 6)
        g_shots = max(100, shots // (2 * k_use + 1)) if k_use > 0 else shots
        total_queries_grover = g_shots * (2 * k_use + 1)
        qg_errs_bins, qg_errs_anal = [], []
        last_info = {}
        for _ in range(n_reps):
            est_prob, info = grover_boosted_estimate(
                A_circuit, obj_qubit, k_iters=k_use, shots=g_shots
            )
            est = est_prob * rescale
            qg_errs_bins.append((est - exact_bins) ** 2)
            qg_errs_anal.append((est - analytic) ** 2)
            last_info = info
        qg_rmse_bins = float(np.sqrt(np.mean(qg_errs_bins)))
        qg_rmse_anal = float(np.sqrt(np.mean(qg_errs_anal)))

        # Classical MC (continuous lognormal) — budget-matched
        n_cl = total_queries_grover
        c_errs_bins, c_errs_anal = [], []
        for _ in range(n_reps):
            est = _classical_mc_excess(shape, loc, scale, threshold, n_cl, rng)
            c_errs_bins.append((est - exact_bins) ** 2)
            c_errs_anal.append((est - analytic) ** 2)
        c_rmse_bins = float(np.sqrt(np.mean(c_errs_bins)))
        c_rmse_anal = float(np.sqrt(np.mean(c_errs_anal)))

        # Classical MC on same discretised bins — budget-matched
        cb_errs = []
        for _ in range(n_reps):
            est = _classical_mc_on_bins(midpoints, probs, threshold,
                                        n_cl, rng)
            cb_errs.append((est - exact_bins) ** 2)
        cb_rmse = float(np.sqrt(np.mean(cb_errs)))

        results_by_percentile[str(pct)] = {
            "percentile": pct,
            "catastrophe_threshold": threshold,
            "exact_excess_loss": exact_bins,
            "analytic_excess_loss": analytic,
            "disc_error": disc_err,
            "readout_prob": exact_prob,
            "max_safe_k": k_safe,
            "k_used": k_use,
            "classical_rmse_vs_bins": c_rmse_bins,
            "classical_rmse_vs_analytic": c_rmse_anal,
            "classical_bins_rmse": cb_rmse,
            "quantum_k0_rmse": q0_rmse_bins,
            "quantum_grover_rmse_vs_bins": qg_rmse_bins,
            "quantum_grover_rmse_vs_analytic": qg_rmse_anal,
            "total_queries_k0": shots,
            "total_queries_grover": total_queries_grover,
            "n_classical_samples": n_cl,
            "grover_shots": g_shots,
            "circuit_depth": last_info.get("circuit_depth", 0),
            "gate_count": last_info.get("gate_count", 0),
            # Legacy keys for results.py compatibility (apples-to-apples: all vs bins)
            "classical_rmse": cb_rmse,
            "quantum_grover_rmse": qg_rmse_bins,
            "classical_cont_rmse_vs_analytic": c_rmse_anal,
        }
        log.info("       vs analytic: C_cont=$%.2f  Q_grover=$%.2f",
                 c_rmse_anal, qg_rmse_anal)
        log.info("       vs bins:     C_bins=$%.2f  Q_grover=$%.2f  "
                 "Q/C_bins=%.1fx",
                 cb_rmse, qg_rmse_bins,
                 cb_rmse / qg_rmse_bins if qg_rmse_bins > 0 else float('inf'))

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
