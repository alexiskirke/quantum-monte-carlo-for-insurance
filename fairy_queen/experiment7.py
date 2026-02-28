"""Experiment 7 – Empirical PMF Comparison (no parametric fit).

Encodes the NOAA loss data directly as an empirical discrete PMF
(quantile-binned histogram) into the quantum oracle, bypassing the
lognormal fit entirely.  This removes the "CT/IS only win because
lognormal is analytically easy" escape hatch.

Ground truth is the exact sum over bins (a non-sampling baseline).
All methods receive the same query budget B.

Methods compared:
  1. Exact-on-bins (one-line sum — the non-sampling ceiling)
  2. Classical MC on bins (sample from the empirical PMF)
  3. Quantum AE on the same PMF
  4. Naive MC from the raw loss array (resample with replacement)
"""

from __future__ import annotations

import time
import numpy as np
from typing import Dict, List, Tuple

from fairy_queen.logging_config import get_logger
from fairy_queen.quantum_circuits import (
    build_oracle_A,
    exact_amplitude_readout,
    grover_boosted_estimate,
    max_safe_k,
)

BUDGETS = [500, 2_000, 8_000]
PERCENTILES = [0.90, 0.95, 0.97]
N_REPS = 50


def build_empirical_pmf(
    losses: np.ndarray,
    n_qubits: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build an empirical PMF using quantile-based bins.

    Unlike the lognormal discretisation (equal-width, 0.1-99.9th pctl),
    this uses quantile bin edges so each bin has roughly equal probability
    mass, extending to the data maximum for full tail coverage.
    """
    n_bins = 2 ** n_qubits
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(losses, quantiles)
    edges[-1] = losses.max() + 1.0

    probs = np.zeros(n_bins)
    midpoints = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (losses >= edges[i]) & (losses < edges[i + 1])
        count = mask.sum()
        probs[i] = count / len(losses)
        midpoints[i] = losses[mask].mean() if count > 0 else (edges[i] + edges[i + 1]) / 2

    probs = probs / probs.sum()
    return midpoints, probs


def _exact_on_bins(midpoints: np.ndarray, probs: np.ndarray,
                   threshold: float) -> float:
    return float(np.dot(probs, np.maximum(0.0, midpoints - threshold)))


def _classical_mc_bins(midpoints: np.ndarray, probs: np.ndarray,
                       threshold: float, n_samples: int,
                       rng: np.random.Generator) -> float:
    indices = rng.choice(len(midpoints), size=n_samples, p=probs)
    X = midpoints[indices]
    return float(np.mean(np.maximum(0.0, X - threshold)))


def _naive_mc_resample(losses: np.ndarray, threshold: float,
                       n_samples: int, rng: np.random.Generator) -> float:
    X = rng.choice(losses, size=n_samples, replace=True)
    return float(np.mean(np.maximum(0.0, X - threshold)))


def _quantum_at_budget(A_circuit, obj_qubit, rescale, budget: int,
                       k_safe: int) -> Tuple[float, int, int, dict]:
    min_shots = 100
    k = min(k_safe, max(0, (budget // min_shots - 1) // 2))
    shots = max(min_shots, budget // (2 * k + 1))
    est_prob, info = grover_boosted_estimate(
        A_circuit, obj_qubit, k_iters=k, shots=shots
    )
    return est_prob * rescale, k, shots, info


def run_experiment7(
    losses: np.ndarray,
    n_qubits: int = 3,
    budgets: List[int] | None = None,
    percentiles: List[float] | None = None,
    n_reps: int = N_REPS,
    seed: int = 42,
) -> Dict:
    """Run the empirical PMF comparison on raw loss data."""
    log = get_logger()
    log.info("=== Experiment 7: Empirical PMF Comparison ===")
    t0 = time.time()

    if budgets is None:
        budgets = BUDGETS
    if percentiles is None:
        percentiles = PERCENTILES

    midpoints, probs = build_empirical_pmf(losses, n_qubits)
    rng = np.random.default_rng(seed)

    log.info("  Empirical PMF: %d bins (quantile-based), %d raw records",
             len(midpoints), len(losses))
    for i, (m, p) in enumerate(zip(midpoints, probs)):
        log.info("    bin %d: midpoint=$%.0f  prob=%.4f", i, m, p)

    empirical_mean = float(np.mean(losses))
    log.info("  Raw data mean=$%.0f, PMF mean=$%.0f",
             empirical_mean, float(np.dot(probs, midpoints)))

    results_by_pctl: Dict[str, Dict] = {}

    for pct in percentiles:
        threshold = float(np.percentile(losses, pct * 100))
        gt_exact = _exact_on_bins(midpoints, probs, threshold)
        gt_resample = float(np.mean(np.maximum(0.0, losses - threshold)))

        A_circuit, obj_qubit, rescale = build_oracle_A(
            probs, midpoints, threshold
        )
        exact_prob = exact_amplitude_readout(A_circuit, obj_qubit)
        k_safe = max_safe_k(exact_prob)

        log.info("  Pctl %.0f%%: M=$%.0f  exact_bins=$%.2f  "
                 "resample_truth=$%.2f  P(|1>)=%.6f  k_max=%d",
                 pct * 100, threshold, gt_exact, gt_resample,
                 exact_prob, k_safe)

        budget_results: Dict[str, Dict] = {}
        for B in budgets:
            naive_ests, disc_ests, q_ests = [], [], []
            q_k_used, q_shots_used = 0, 0
            q_info_last: dict = {}

            for _ in range(n_reps):
                naive_ests.append(_naive_mc_resample(losses, threshold, B, rng))
                disc_ests.append(_classical_mc_bins(midpoints, probs, threshold, B, rng))
                q_val, qk, qs, qinf = _quantum_at_budget(
                    A_circuit, obj_qubit, rescale, B, k_safe
                )
                q_ests.append(q_val)
                q_k_used, q_shots_used, q_info_last = qk, qs, qinf

            naive_a = np.array(naive_ests)
            disc_a = np.array(disc_ests)
            q_a = np.array(q_ests)

            def _rmse(arr, truth):
                return float(np.sqrt(np.mean((arr - truth) ** 2)))

            entry = {
                "budget": B,
                "exact_on_bins": gt_exact,
                "naive_mc_resample": {
                    "rmse_vs_exact_bins": _rmse(naive_a, gt_exact),
                    "rmse_vs_resample_truth": _rmse(naive_a, gt_resample),
                    "mean": float(np.mean(naive_a)),
                },
                "classical_bins": {
                    "rmse_vs_exact_bins": _rmse(disc_a, gt_exact),
                    "mean": float(np.mean(disc_a)),
                },
                "quantum": {
                    "rmse_vs_exact_bins": _rmse(q_a, gt_exact),
                    "mean": float(np.mean(q_a)),
                    "k_used": q_k_used,
                    "shots": q_shots_used,
                    "circuit_depth": q_info_last.get("circuit_depth", 0),
                    "gate_count": q_info_last.get("gate_count", 0),
                },
            }
            budget_results[str(B)] = entry

            log.info("    B=%5d  NaiveMC=$%.1f  ClassBins=$%.1f  "
                     "QAE(k=%d)=$%.1f  exact=$%.2f  [RMSE vs exact-on-bins]",
                     B,
                     entry["naive_mc_resample"]["rmse_vs_exact_bins"],
                     entry["classical_bins"]["rmse_vs_exact_bins"],
                     q_k_used,
                     entry["quantum"]["rmse_vs_exact_bins"],
                     gt_exact)

        results_by_pctl[str(pct)] = {
            "percentile": pct,
            "threshold": threshold,
            "exact_on_bins": gt_exact,
            "resample_truth": gt_resample,
            "readout_prob": exact_prob,
            "k_safe": k_safe,
            "budgets": budget_results,
        }

    runtime = time.time() - t0
    log.info("Experiment 7 completed in %.1f s", runtime)

    return {
        "percentile_results": results_by_pctl,
        "n_qubits": n_qubits,
        "n_records": len(losses),
        "n_reps": n_reps,
        "budgets": budgets,
        "runtime_seconds": runtime,
        "bin_midpoints": midpoints.tolist(),
        "bin_probs": probs.tolist(),
    }
