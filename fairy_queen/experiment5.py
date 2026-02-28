"""Experiment 5 – Fair Budget-Matched Comparison with Strong Classical Baselines.

Addresses three key reviewer concerns:
  (a) Ground truth is computed analytically (closed-form lognormal excess loss)
      *and* as the exact discretised sum, so both references are airtight.
  (b) ALL methods receive the same oracle-query / sample budget B.
  (c) Two variance-reduced classical baselines are included:
      - Conditional Tail MC (sample X|X>M, multiply by P(X>M))
      - Importance Sampling MC (exponential tilt toward the threshold)

For each (percentile, budget) pair the experiment reports RMSE of five
estimators vs the analytic ground truth:
  1. Naive classical MC  (continuous lognormal)
  2. Conditional Tail MC (continuous lognormal, variance-reduced)
  3. Importance Sampling MC (continuous lognormal, variance-reduced)
  4. Classical MC on discretised bins (same oracle as quantum)
  5. Quantum AE with Grover amplification
"""

from __future__ import annotations

import time
import numpy as np
from scipy.stats import norm, lognorm
from typing import Dict, List, Tuple

from fairy_queen.logging_config import get_logger
from fairy_queen.quantum_circuits import (
    discretise_distribution,
    build_oracle_A,
    exact_amplitude_readout,
    grover_boosted_estimate,
    max_safe_k,
)

BUDGETS = [500, 2_000, 8_000]
PERCENTILES = [0.90, 0.95, 0.97]
N_REPS = 50


def analytic_lognormal_excess(shape: float, loc: float, scale: float,
                              threshold: float) -> float:
    """Closed-form E[(X-M)+] for X ~ Lognormal(shape, loc=0, scale).

    E[(X-M)+] = exp(mu+sigma^2/2) Phi(d1) - M [1-Phi(d2)]
    where mu=log(scale), sigma=shape, d1=(mu+sigma^2-log(M))/sigma,
    d2=(log(M)-mu)/sigma.
    """
    mu = np.log(scale)
    sigma = shape
    M = threshold
    if M <= 0:
        return np.exp(mu + sigma ** 2 / 2) - M
    d1 = (mu + sigma ** 2 - np.log(M)) / sigma
    d2 = (np.log(M) - mu) / sigma
    return float(np.exp(mu + sigma ** 2 / 2) * norm.cdf(d1) - M * (1 - norm.cdf(d2)))


def exact_discretised_excess(midpoints: np.ndarray, probs: np.ndarray,
                             threshold: float) -> float:
    """Sum over bins: sum(p_i * max(0, x_i - M))."""
    return float(np.dot(probs, np.maximum(0.0, midpoints - threshold)))


def _naive_mc(shape: float, loc: float, scale: float, threshold: float,
              n_samples: int, rng: np.random.Generator) -> float:
    X = lognorm.rvs(shape, loc, scale, size=n_samples,
                    random_state=rng.integers(2 ** 31))
    return float(np.mean(np.maximum(0.0, X - threshold)))


def _conditional_tail_mc(shape: float, loc: float, scale: float,
                         threshold: float, n_samples: int,
                         rng: np.random.Generator) -> float:
    """Sample X|X>M via inverse CDF of truncated normal, multiply by P(X>M)."""
    mu = np.log(scale)
    sigma = shape
    M = threshold
    p_tail = 1.0 - norm.cdf((np.log(M) - mu) / sigma)
    if p_tail < 1e-15:
        return 0.0
    u_low = norm.cdf((np.log(M) - mu) / sigma)
    U = rng.uniform(u_low, 1.0, size=n_samples)
    U = np.clip(U, u_low + 1e-15, 1.0 - 1e-15)
    Z = mu + sigma * norm.ppf(U)
    X = np.exp(Z)
    return float(p_tail * np.mean(X - M))


def _importance_sampling_mc(shape: float, loc: float, scale: float,
                            threshold: float, n_samples: int,
                            rng: np.random.Generator) -> float:
    """IS with exponential tilt so the proposal mean equals M."""
    mu = np.log(scale)
    sigma = shape
    M = threshold
    delta = max(0.0, np.log(M) - mu - sigma ** 2 / 2)
    if delta < 1e-10:
        return _naive_mc(shape, loc, scale, threshold, n_samples, rng)
    mu_q = mu + delta
    Z = rng.normal(mu_q, sigma, size=n_samples)
    X = np.exp(Z)
    log_w = delta * (delta - 2.0 * (Z - mu)) / (2.0 * sigma ** 2)
    log_w = np.clip(log_w, -50, 50)
    w = np.exp(log_w)
    return float(np.mean(np.maximum(0.0, X - M) * w))


def _classical_mc_discrete(midpoints: np.ndarray, probs: np.ndarray,
                           threshold: float, n_samples: int,
                           rng: np.random.Generator) -> float:
    """Sample from the discretised distribution (same oracle as quantum)."""
    indices = rng.choice(len(midpoints), size=n_samples, p=probs)
    X = midpoints[indices]
    return float(np.mean(np.maximum(0.0, X - threshold)))


def _quantum_at_budget(A_circuit, obj_qubit, rescale, budget: int,
                       k_safe: int) -> Tuple[float, int, int, dict]:
    """Run quantum AE with exactly *budget* oracle calls."""
    min_shots = 100
    k = min(k_safe, max(0, (budget // min_shots - 1) // 2))
    shots = max(min_shots, budget // (2 * k + 1))
    est_prob, info = grover_boosted_estimate(
        A_circuit, obj_qubit, k_iters=k, shots=shots
    )
    return est_prob * rescale, k, shots, info


def run_experiment5(
    dist_params: Tuple[float, float, float],
    losses: np.ndarray,
    n_qubits: int = 3,
    budgets: List[int] | None = None,
    percentiles: List[float] | None = None,
    n_reps: int = N_REPS,
    seed: int = 42,
    label: str = "",
) -> Dict:
    """Run the fair budget-matched comparison."""
    log = get_logger()
    log.info("=== Experiment 5: Fair Budget-Matched Comparison %s ===", label)
    t0 = time.time()

    if budgets is None:
        budgets = BUDGETS
    if percentiles is None:
        percentiles = PERCENTILES

    shape, loc, scale = dist_params
    midpoints, probs = discretise_distribution(shape, loc, scale, n_qubits)
    rng = np.random.default_rng(seed)

    results_by_pctl: Dict[str, Dict] = {}

    for pct in percentiles:
        threshold = float(np.percentile(losses, pct * 100))
        gt_analytic = analytic_lognormal_excess(shape, loc, scale, threshold)
        gt_discrete = exact_discretised_excess(midpoints, probs, threshold)
        disc_error = abs(gt_discrete - gt_analytic)

        A_circuit, obj_qubit, rescale = build_oracle_A(
            probs, midpoints, threshold
        )
        exact_prob = exact_amplitude_readout(A_circuit, obj_qubit)
        k_safe = max_safe_k(exact_prob)

        log.info("  Pctl %.0f%%: M=$%.0f  analytic=$%.2f  discrete=$%.2f  "
                 "disc_err=$%.2f  P(|1>)=%.6f  k_max=%d",
                 pct * 100, threshold, gt_analytic, gt_discrete,
                 disc_error, exact_prob, k_safe)

        budget_results: Dict[str, Dict] = {}
        for B in budgets:
            naive_ests, ct_ests, is_ests, disc_ests, q_ests = [], [], [], [], []
            q_k_used, q_shots_used = 0, 0
            q_info_last: dict = {}

            for _ in range(n_reps):
                naive_ests.append(_naive_mc(shape, loc, scale, threshold, B, rng))
                ct_ests.append(_conditional_tail_mc(shape, loc, scale, threshold, B, rng))
                is_ests.append(_importance_sampling_mc(shape, loc, scale, threshold, B, rng))
                disc_ests.append(_classical_mc_discrete(midpoints, probs, threshold, B, rng))
                q_val, qk, qs, qinf = _quantum_at_budget(
                    A_circuit, obj_qubit, rescale, B, k_safe
                )
                q_ests.append(q_val)
                q_k_used, q_shots_used, q_info_last = qk, qs, qinf

            naive_a = np.array(naive_ests)
            ct_a = np.array(ct_ests)
            is_a = np.array(is_ests)
            disc_a = np.array(disc_ests)
            q_a = np.array(q_ests)

            def _rmse(arr, truth):
                return float(np.sqrt(np.mean((arr - truth) ** 2)))

            entry = {
                "budget": B,
                "naive_mc": {
                    "rmse_vs_analytic": _rmse(naive_a, gt_analytic),
                    "rmse_vs_discrete": _rmse(naive_a, gt_discrete),
                    "mean": float(np.mean(naive_a)),
                    "std": float(np.std(naive_a)),
                },
                "conditional_tail": {
                    "rmse_vs_analytic": _rmse(ct_a, gt_analytic),
                    "rmse_vs_discrete": _rmse(ct_a, gt_discrete),
                    "mean": float(np.mean(ct_a)),
                    "std": float(np.std(ct_a)),
                },
                "importance_sampling": {
                    "rmse_vs_analytic": _rmse(is_a, gt_analytic),
                    "rmse_vs_discrete": _rmse(is_a, gt_discrete),
                    "mean": float(np.mean(is_a)),
                    "std": float(np.std(is_a)),
                },
                "classical_discrete": {
                    "rmse_vs_analytic": _rmse(disc_a, gt_analytic),
                    "rmse_vs_discrete": _rmse(disc_a, gt_discrete),
                    "mean": float(np.mean(disc_a)),
                    "std": float(np.std(disc_a)),
                },
                "quantum": {
                    "rmse_vs_analytic": _rmse(q_a, gt_analytic),
                    "rmse_vs_discrete": _rmse(q_a, gt_discrete),
                    "mean": float(np.mean(q_a)),
                    "std": float(np.std(q_a)),
                    "k_used": q_k_used,
                    "shots": q_shots_used,
                    "circuit_depth": q_info_last.get("circuit_depth", 0),
                    "gate_count": q_info_last.get("gate_count", 0),
                },
            }
            budget_results[str(B)] = entry

            log.info("    B=%5d  NaiveMC=$%.1f  CT=$%.1f  IS=$%.1f  "
                     "Disc=$%.1f  QAE(k=%d)=$%.1f   [RMSE vs analytic]",
                     B,
                     entry["naive_mc"]["rmse_vs_analytic"],
                     entry["conditional_tail"]["rmse_vs_analytic"],
                     entry["importance_sampling"]["rmse_vs_analytic"],
                     entry["classical_discrete"]["rmse_vs_analytic"],
                     q_k_used,
                     entry["quantum"]["rmse_vs_analytic"])

        results_by_pctl[str(pct)] = {
            "percentile": pct,
            "threshold": threshold,
            "analytic_truth": gt_analytic,
            "discrete_truth": gt_discrete,
            "discretisation_error": disc_error,
            "readout_prob": exact_prob,
            "k_safe": k_safe,
            "budgets": budget_results,
        }

    runtime = time.time() - t0
    log.info("Experiment 5 %s completed in %.1f s", label, runtime)

    return {
        "percentile_results": results_by_pctl,
        "dist_params": {"shape": shape, "loc": loc, "scale": scale},
        "n_qubits": n_qubits,
        "n_reps": n_reps,
        "budgets": budgets,
        "runtime_seconds": runtime,
        "label": label,
    }
