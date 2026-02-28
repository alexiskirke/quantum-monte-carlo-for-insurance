"""Experiment 6 – Qubit / Bin Scaling Sweep.

Separates discretisation error from estimation error by sweeping n = 3..7
qubits (8..128 bins).  For each n, reports:

  1. Discretisation error:  |exact_discrete − analytic_truth|
  2. Quantum estimation error: RMSE(quantum_estimate, exact_discrete)
  3. Classical-on-bins estimation error: RMSE(classical_discrete, exact_discrete)

Also records transpiled circuit metrics (two-qubit gate count, depth) for
each n to quantify state-preparation overhead.
"""

from __future__ import annotations

import time
import numpy as np
from scipy.stats import norm
from typing import Dict, List, Tuple

from qiskit import transpile as qk_transpile

from fairy_queen.logging_config import get_logger
from fairy_queen.quantum_circuits import (
    discretise_distribution,
    build_oracle_A,
    build_state_prep,
    exact_amplitude_readout,
    grover_boosted_estimate,
    max_safe_k,
    get_qasm_backend,
)
from fairy_queen.experiment5 import analytic_lognormal_excess

QUBIT_RANGE = [3, 4, 5, 6, 7]
BUDGET = 4_000
N_REPS = 50
TARGET_PERCENTILE = 0.95


def _circuit_stats(qc) -> Dict:
    """Transpile to {cx, rz, sx, x} and extract hardware-relevant metrics."""
    tc = qk_transpile(qc, basis_gates=["cx", "rz", "sx", "x"],
                       optimization_level=1)
    ops = tc.count_ops()
    cx_count = ops.get("cx", 0)
    return {
        "total_gates": tc.size(),
        "two_qubit_gates": cx_count,
        "depth": tc.depth(),
        "ops_breakdown": {k: int(v) for k, v in ops.items()},
    }


def _classical_mc_discrete(midpoints: np.ndarray, probs: np.ndarray,
                           threshold: float, n_samples: int,
                           rng: np.random.Generator) -> float:
    indices = rng.choice(len(midpoints), size=n_samples, p=probs)
    X = midpoints[indices]
    return float(np.mean(np.maximum(0.0, X - threshold)))


def run_experiment6(
    dist_params: Tuple[float, float, float],
    losses: np.ndarray,
    qubit_range: List[int] | None = None,
    budget: int = BUDGET,
    percentile: float = TARGET_PERCENTILE,
    n_reps: int = N_REPS,
    seed: int = 42,
    label: str = "",
) -> Dict:
    """Run the qubit/bin scaling sweep."""
    log = get_logger()
    log.info("=== Experiment 6: Qubit Scaling Sweep %s ===", label)
    t0 = time.time()

    if qubit_range is None:
        qubit_range = QUBIT_RANGE

    shape, loc, scale = dist_params
    threshold = float(np.percentile(losses, percentile * 100))
    gt_analytic = analytic_lognormal_excess(shape, loc, scale, threshold)
    rng = np.random.default_rng(seed)

    log.info("  Threshold (%.0f%% pctl) = $%.0f, analytic E[excess] = $%.4f",
             percentile * 100, threshold, gt_analytic)

    sweep_results: List[Dict] = []

    for n in qubit_range:
        n_bins = 2 ** n
        midpoints, probs = discretise_distribution(shape, loc, scale, n)
        gt_discrete = float(np.dot(probs, np.maximum(0.0, midpoints - threshold)))
        disc_error = abs(gt_discrete - gt_analytic)

        # Circuit metrics
        sp_circuit = build_state_prep(probs, n)
        sp_stats = _circuit_stats(sp_circuit)

        A_circuit, obj_qubit, rescale = build_oracle_A(probs, midpoints, threshold)
        exact_prob = exact_amplitude_readout(A_circuit, obj_qubit)
        k_safe = max_safe_k(exact_prob)

        a_stats = _circuit_stats(A_circuit)

        # Quantum estimation at budget
        min_shots = 100
        k_use = min(k_safe, max(0, (budget // min_shots - 1) // 2))
        shots = max(min_shots, budget // (2 * k_use + 1))

        q_errors = []
        full_info: dict = {}
        for _ in range(n_reps):
            est_prob, info = grover_boosted_estimate(
                A_circuit, obj_qubit, k_iters=k_use, shots=shots
            )
            q_errors.append((est_prob * rescale - gt_discrete) ** 2)
            full_info = info
        q_rmse = float(np.sqrt(np.mean(q_errors)))

        # Full circuit stats (with Grover iterations)
        full_stats = {
            "total_gates": full_info.get("gate_count", 0),
            "depth": full_info.get("circuit_depth", 0),
        }

        # Classical MC on same bins at same budget
        c_errors = []
        for _ in range(n_reps):
            est = _classical_mc_discrete(midpoints, probs, threshold, budget, rng)
            c_errors.append((est - gt_discrete) ** 2)
        c_rmse = float(np.sqrt(np.mean(c_errors)))

        entry = {
            "n_qubits": n,
            "n_bins": n_bins,
            "analytic_truth": gt_analytic,
            "discrete_truth": gt_discrete,
            "discretisation_error": disc_error,
            "disc_error_relative": disc_error / max(abs(gt_analytic), 1e-10),
            "readout_prob": exact_prob,
            "k_safe": k_safe,
            "k_used": k_use,
            "shots": shots,
            "quantum_rmse_vs_discrete": q_rmse,
            "classical_rmse_vs_discrete": c_rmse,
            "quantum_total_rmse": float(np.sqrt(disc_error ** 2 + q_rmse ** 2)),
            "classical_total_rmse": float(np.sqrt(disc_error ** 2 + c_rmse ** 2)),
            "state_prep_stats": sp_stats,
            "oracle_A_stats": a_stats,
            "full_circuit_stats": full_stats,
            "effective_work_quantum": shots * (2 * k_use + 1) * full_stats["total_gates"],
        }
        sweep_results.append(entry)

        log.info("  n=%d (%3d bins): disc_err=$%.2f  Q_RMSE=$%.2f  C_RMSE=$%.2f  "
                 "k=%d  SP_2q=%d  depth=%d",
                 n, n_bins, disc_error, q_rmse, c_rmse,
                 k_use, sp_stats["two_qubit_gates"], a_stats["depth"])

    runtime = time.time() - t0
    log.info("Experiment 6 %s completed in %.1f s", label, runtime)

    return {
        "sweep": sweep_results,
        "threshold": threshold,
        "percentile": percentile,
        "analytic_truth": gt_analytic,
        "budget": budget,
        "n_reps": n_reps,
        "dist_params": {"shape": shape, "loc": loc, "scale": scale},
        "runtime_seconds": runtime,
        "label": label,
    }
