"""Part 1: Log-log slopes with bootstrap CIs for Exps 1 and 4A."""
import numpy as np
from scipy import stats as sp_stats

from fairy_queen.data_pipeline import get_synthetic_loss_data, get_real_loss_data
from fairy_queen.quantum_circuits import (
    discretise_distribution, build_oracle_A, exact_amplitude_readout,
    grover_boosted_estimate, max_safe_k,
)
from fairy_queen.experiment5 import analytic_lognormal_excess

SEED = 42
N_BOOTSTRAP = 2000


def _exact_excess(midpoints, probs, threshold):
    return float(np.dot(probs, np.maximum(0.0, midpoints - threshold)))


def _classical_mc_on_bins(midpoints, probs, threshold, n, rng):
    idx = rng.choice(len(midpoints), size=n, p=probs)
    return float(np.mean(np.maximum(0.0, midpoints[idx] - threshold)))


def fit_loglog_slope(queries, rmses, n_bootstrap=N_BOOTSTRAP):
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
    return slope, ci_lo, ci_hi, r_value**2


def run_slopes(label, losses, dp, n_qubits=3, shots_per_k=1000, n_reps=30):
    shape, loc, scale = dp
    midpoints, probs = discretise_distribution(shape, loc, scale, n_qubits)
    threshold = float(np.percentile(losses, 95))
    exact_bins = _exact_excess(midpoints, probs, threshold)
    A, oq, rsc = build_oracle_A(probs, midpoints, catastrophe_threshold=threshold)
    ep = exact_amplitude_readout(A, oq)
    ks = max_safe_k(ep)
    rng = np.random.default_rng(SEED)

    print(f"\n{label}: threshold=${threshold:.0f}, exact_bins=${exact_bins:.2f}, "
          f"P(|1>)={ep:.6f}, k_max={ks}")

    q_queries, q_rmses = [], []
    c_queries, c_rmses = [], []

    for k in range(min(ks, 6) + 1):
        tq = shots_per_k * (2 * k + 1)
        qe = []
        for _ in range(n_reps):
            ep2, _ = grover_boosted_estimate(A, oq, k_iters=k, shots=shots_per_k)
            qe.append((ep2 * rsc - exact_bins) ** 2)
        q_queries.append(tq)
        q_rmses.append(float(np.sqrt(np.mean(qe))))

        cbe = []
        for _ in range(n_reps):
            est = _classical_mc_on_bins(midpoints, probs, threshold, tq, rng)
            cbe.append((est - exact_bins) ** 2)
        c_queries.append(tq)
        c_rmses.append(float(np.sqrt(np.mean(cbe))))

        print(f"  k={k}: queries={tq:5d}  Q_RMSE=${q_rmses[-1]:.1f}  C_RMSE=${c_rmses[-1]:.1f}")

    q_slope, q_lo, q_hi, q_r2 = fit_loglog_slope(q_queries, q_rmses)
    c_slope, c_lo, c_hi, c_r2 = fit_loglog_slope(c_queries, c_rmses)

    print(f"\n  Quantum slope = {q_slope:.3f}  95% CI [{q_lo:.3f}, {q_hi:.3f}]  R²={q_r2:.3f}")
    print(f"  Classical slope = {c_slope:.3f}  95% CI [{c_lo:.3f}, {c_hi:.3f}]  R²={c_r2:.3f}")
    print(f"  Reference: O(1/√N) = -0.500, O(1/N) = -1.000")

    return {"q_slope": q_slope, "q_ci": (q_lo, q_hi), "c_slope": c_slope, "c_ci": (c_lo, c_hi)}


if __name__ == "__main__":
    print("=" * 60)
    print("Exp 1 (synthetic)")
    print("=" * 60)
    losses_s, dp_s = get_synthetic_loss_data()
    r1 = run_slopes("Exp 1 synthetic", losses_s, dp_s)

    print("\n" + "=" * 60)
    print("Exp 4A (NOAA)")
    print("=" * 60)
    losses_r, dp_r = get_real_loss_data()
    r4 = run_slopes("Exp 4A NOAA", losses_r, dp_r)
