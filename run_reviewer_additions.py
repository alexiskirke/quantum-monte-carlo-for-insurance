"""Reviewer-requested additions:
1. Fitted log-log convergence slopes with bootstrap CIs (Exps 1, 4A)
2. Experiment 6 variant with log-spaced bins
3. Quasi-Monte Carlo baseline for Experiment 5
"""
import json
import numpy as np
from scipy import stats as sp_stats
from scipy.stats import lognorm, norm

from fairy_queen.data_pipeline import get_synthetic_loss_data, get_real_loss_data
from fairy_queen.quantum_circuits import (
    discretise_distribution, build_oracle_A, exact_amplitude_readout,
    grover_boosted_estimate, max_safe_k, build_state_prep,
)
from fairy_queen.experiment5 import analytic_lognormal_excess
from fairy_queen.experiment1 import _exact_excess_loss, _classical_mc_on_bins
from qiskit import transpile as qk_transpile

SEED = 42
N_BOOTSTRAP = 1000

# ============================================================
# 1. Log-log slope fitting with bootstrap CIs
# ============================================================

def fit_loglog_slope(queries, rmses, n_bootstrap=N_BOOTSTRAP):
    """Fit slope of log(RMSE) vs log(queries) with bootstrap CIs."""
    log_q = np.log(np.array(queries, dtype=float))
    log_r = np.log(np.array(rmses, dtype=float))
    n = len(log_q)

    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(log_q, log_r)

    rng = np.random.default_rng(SEED)
    boot_slopes = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        s, _, _, _, _ = sp_stats.linregress(log_q[idx], log_r[idx])
        boot_slopes.append(s)
    boot_slopes = np.array(boot_slopes)
    ci_lo, ci_hi = np.percentile(boot_slopes, [2.5, 97.5])

    return {
        "slope": slope, "intercept": intercept, "r_squared": r_value**2,
        "std_err": std_err, "ci_95": (ci_lo, ci_hi),
    }


def run_slope_analysis():
    """Compute slopes for Exp 1 (synthetic) and Exp 4A (NOAA)."""
    print("=" * 70)
    print("PART 1: Log-log convergence slope fitting")
    print("=" * 70)

    # Exp 1: synthetic
    losses_s, dp_s = get_synthetic_loss_data()
    shape, loc, scale = dp_s
    midpoints, probs = discretise_distribution(shape, loc, scale, 3)
    threshold_s = float(np.percentile(losses_s, 95))
    exact_bins_s = _exact_excess_loss(midpoints, probs, threshold_s)

    A, oq, rsc = build_oracle_A(probs, midpoints, catastrophe_threshold=threshold_s)
    ep = exact_amplitude_readout(A, oq)
    ks = max_safe_k(ep)
    rng = np.random.default_rng(SEED)

    q_queries_s, q_rmses_s = [], []
    cb_queries_s, cb_rmses_s = [], []
    for k in range(min(ks, 6) + 1):
        tq = 1000 * (2 * k + 1)
        qe = []
        for _ in range(30):
            ep2, _ = grover_boosted_estimate(A, oq, k_iters=k, shots=1000)
            qe.append((ep2 * rsc - exact_bins_s) ** 2)
        q_queries_s.append(tq)
        q_rmses_s.append(float(np.sqrt(np.mean(qe))))

        cbe = []
        for _ in range(30):
            est = _classical_mc_on_bins(midpoints, probs, threshold_s, tq, rng)
            cbe.append((est - exact_bins_s) ** 2)
        cb_queries_s.append(tq)
        cb_rmses_s.append(float(np.sqrt(np.mean(cbe))))

    q_fit = fit_loglog_slope(q_queries_s, q_rmses_s)
    c_fit = fit_loglog_slope(cb_queries_s, cb_rmses_s)

    print(f"\nExp 1 (synthetic, 95th pctl):")
    print(f"  Quantum AE slope = {q_fit['slope']:.3f}  95% CI [{q_fit['ci_95'][0]:.3f}, {q_fit['ci_95'][1]:.3f}]  R²={q_fit['r_squared']:.3f}")
    print(f"  Classical bins slope = {c_fit['slope']:.3f}  95% CI [{c_fit['ci_95'][0]:.3f}, {c_fit['ci_95'][1]:.3f}]  R²={c_fit['r_squared']:.3f}")
    print(f"  Reference: O(1/√N) slope = -0.500, O(1/N) slope = -1.000")

    # Exp 4A: NOAA
    losses_r, dp_r = get_real_loss_data()
    shape_r, loc_r, scale_r = dp_r
    mp_r, pr_r = discretise_distribution(shape_r, loc_r, scale_r, 3)
    threshold_r = float(np.percentile(losses_r, 95))
    exact_bins_r = _exact_excess_loss(mp_r, pr_r, threshold_r)

    A_r, oq_r, rsc_r = build_oracle_A(pr_r, mp_r, catastrophe_threshold=threshold_r)
    ep_r = exact_amplitude_readout(A_r, oq_r)
    ks_r = max_safe_k(ep_r)

    q_queries_r, q_rmses_r = [], []
    cb_queries_r, cb_rmses_r = [], []
    for k in range(min(ks_r, 6) + 1):
        tq = 1000 * (2 * k + 1)
        qe = []
        for _ in range(30):
            ep2, _ = grover_boosted_estimate(A_r, oq_r, k_iters=k, shots=1000)
            qe.append((ep2 * rsc_r - exact_bins_r) ** 2)
        q_queries_r.append(tq)
        q_rmses_r.append(float(np.sqrt(np.mean(qe))))

        cbe = []
        for _ in range(30):
            est = _classical_mc_on_bins(mp_r, pr_r, threshold_r, tq, rng)
            cbe.append((est - exact_bins_r) ** 2)
        cb_queries_r.append(tq)
        cb_rmses_r.append(float(np.sqrt(np.mean(cbe))))

    q_fit_r = fit_loglog_slope(q_queries_r, q_rmses_r)
    c_fit_r = fit_loglog_slope(cb_queries_r, cb_rmses_r)

    print(f"\nExp 4A (NOAA, 95th pctl):")
    print(f"  Quantum AE slope = {q_fit_r['slope']:.3f}  95% CI [{q_fit_r['ci_95'][0]:.3f}, {q_fit_r['ci_95'][1]:.3f}]  R²={q_fit_r['r_squared']:.3f}")
    print(f"  Classical bins slope = {c_fit_r['slope']:.3f}  95% CI [{c_fit_r['ci_95'][0]:.3f}, {c_fit_r['ci_95'][1]:.3f}]  R²={c_fit_r['r_squared']:.3f}")

    return {
        "exp1": {"quantum": q_fit, "classical": c_fit,
                 "queries": q_queries_s, "q_rmse": q_rmses_s, "c_rmse": cb_rmses_s},
        "exp4a": {"quantum": q_fit_r, "classical": c_fit_r,
                  "queries": q_queries_r, "q_rmse": q_rmses_r, "c_rmse": cb_rmses_r},
    }


# ============================================================
# 2. Experiment 6 with log-spaced bins
# ============================================================

def discretise_logspaced(shape, loc, scale, n_qubits):
    """Log-spaced bins spanning 0.1st to 99.99th percentile (extended tail)."""
    n_bins = 2 ** n_qubits
    low = lognorm.ppf(0.001, shape, loc, scale)
    high = lognorm.ppf(0.9999, shape, loc, scale)
    edges = np.geomspace(low, high, n_bins + 1)
    probs = np.diff(lognorm.cdf(edges, shape, loc, scale))
    probs = probs / probs.sum()
    midpoints = np.sqrt(edges[:-1] * edges[1:])  # geometric midpoint
    return midpoints, probs


def run_logspaced_experiment6():
    print("\n" + "=" * 70)
    print("PART 2: Experiment 6 with log-spaced bins (extended to 99.99th pctl)")
    print("=" * 70)

    losses, dp = get_synthetic_loss_data()
    shape, loc, scale = dp
    threshold = float(np.percentile(losses, 95))
    gt_analytic = analytic_lognormal_excess(shape, loc, scale, threshold)
    rng = np.random.default_rng(SEED)
    budget = 4000
    n_reps = 50

    print(f"  Threshold = ${threshold:.0f}, analytic E[excess] = ${gt_analytic:.2f}")
    print(f"\n  {'n':>3} {'bins':>5} {'disc_err':>10} {'Q_RMSE':>10} {'C_RMSE':>10} {'Q/C':>6} {'k':>3} {'k_max':>5} {'P(1)':>10}")

    results = []
    for n in [3, 4, 5, 6, 7]:
        n_bins = 2 ** n
        midpoints, probs = discretise_logspaced(shape, loc, scale, n)
        gt_discrete = float(np.dot(probs, np.maximum(0.0, midpoints - threshold)))
        disc_error = abs(gt_discrete - gt_analytic)

        A, oq, rsc = build_oracle_A(probs, midpoints, threshold)
        ep = exact_amplitude_readout(A, oq)
        ksafe = max_safe_k(ep)
        k_use = min(ksafe, max(0, (budget // 100 - 1) // 2))
        shots = max(100, budget // (2 * k_use + 1))

        q_errors = []
        for _ in range(n_reps):
            ep2, _ = grover_boosted_estimate(A, oq, k_iters=k_use, shots=shots)
            q_errors.append((ep2 * rsc - gt_discrete) ** 2)
        q_rmse = float(np.sqrt(np.mean(q_errors)))

        c_errors = []
        for _ in range(n_reps):
            est = float(np.mean(np.maximum(0.0,
                midpoints[rng.choice(len(midpoints), size=budget, p=probs)] - threshold)))
            c_errors.append((est - gt_discrete) ** 2)
        c_rmse = float(np.sqrt(np.mean(c_errors)))

        ratio = c_rmse / q_rmse if q_rmse > 0 else 0
        print(f"  {n:3d} {n_bins:5d} {disc_error:10.0f} {q_rmse:10.0f} {c_rmse:10.0f} "
              f"{ratio:5.1f}x {k_use:3d} {ksafe:5d} {ep:10.6f}")
        results.append({
            "n": n, "bins": n_bins, "disc_err": disc_error,
            "q_rmse": q_rmse, "c_rmse": c_rmse, "ratio": ratio,
            "k_use": k_use, "k_safe": ksafe, "p_one": ep,
        })

    return results


# ============================================================
# 3. Quasi-Monte Carlo baseline
# ============================================================

def _qmc_excess(shape, loc, scale, threshold, n_samples):
    """QMC with Sobol sequence for lognormal excess loss."""
    from scipy.stats.qmc import Sobol
    sampler = Sobol(d=1, scramble=True)
    U = sampler.random(n_samples).flatten()
    U = np.clip(U, 1e-10, 1 - 1e-10)
    X = lognorm.ppf(U, shape, loc, scale)
    return float(np.mean(np.maximum(0.0, X - threshold)))


def run_qmc_comparison():
    print("\n" + "=" * 70)
    print("PART 3: Quasi-Monte Carlo baseline for Experiment 5")
    print("=" * 70)

    losses, dp = get_synthetic_loss_data()
    shape, loc, scale = dp
    rng = np.random.default_rng(SEED)

    budgets = [500, 2000, 8000]
    percentiles = [0.90, 0.95, 0.97]

    print(f"\n  {'Pctl':>5} {'B':>6} {'QMC':>10} {'NaiveMC':>10} {'CT':>10} {'IS':>10}")

    results = {}
    for pct in percentiles:
        threshold = float(np.percentile(losses, pct * 100))
        gt = analytic_lognormal_excess(shape, loc, scale, threshold)

        for B in budgets:
            qmc_ests = []
            naive_ests = []
            from fairy_queen.experiment5 import (
                _naive_mc, _conditional_tail_mc, _importance_sampling_mc,
            )
            ct_ests, is_ests = [], []
            for _ in range(50):
                qmc_ests.append(_qmc_excess(shape, loc, scale, threshold, B))
                naive_ests.append(_naive_mc(shape, loc, scale, threshold, B, rng))
                ct_ests.append(_conditional_tail_mc(shape, loc, scale, threshold, B, rng))
                is_ests.append(_importance_sampling_mc(shape, loc, scale, threshold, B, rng))

            qmc_rmse = float(np.sqrt(np.mean((np.array(qmc_ests) - gt) ** 2)))
            naive_rmse = float(np.sqrt(np.mean((np.array(naive_ests) - gt) ** 2)))
            ct_rmse = float(np.sqrt(np.mean((np.array(ct_ests) - gt) ** 2)))
            is_rmse = float(np.sqrt(np.mean((np.array(is_ests) - gt) ** 2)))

            key = f"{pct}_{B}"
            results[key] = {"qmc": qmc_rmse, "naive": naive_rmse,
                           "ct": ct_rmse, "is": is_rmse}

            if B == 8000:
                print(f"  {pct*100:4.0f}% {B:6d} {qmc_rmse:10.1f} {naive_rmse:10.1f} "
                      f"{ct_rmse:10.1f} {is_rmse:10.1f}")

    return results


if __name__ == "__main__":
    slopes = run_slope_analysis()
    logsp = run_logspaced_experiment6()
    qmc = run_qmc_comparison()

    with open("results/reviewer_additions.json", "w") as f:
        json.dump({"slopes": {k: {kk: str(vv) for kk, vv in v.items()}
                              for k, v in slopes.items()},
                   "logspaced": [str(r) for r in logsp],
                   "qmc": {k: str(v) for k, v in qmc.items()}}, f, indent=2)
    print("\nDone. Results saved to results/reviewer_additions.json")
