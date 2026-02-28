"""Extended qubit experiments: n=8,9 for Exp 6 sweep + wider-range slope fitting."""
import time
import numpy as np
from scipy import stats as sp_stats
from scipy.stats import lognorm

from fairy_queen.data_pipeline import get_synthetic_loss_data, get_real_loss_data
from fairy_queen.quantum_circuits import (
    discretise_distribution, build_oracle_A, exact_amplitude_readout,
    grover_boosted_estimate, max_safe_k, build_state_prep,
)
from fairy_queen.experiment5 import analytic_lognormal_excess
from qiskit import transpile as qk_transpile

SEED = 42
N_BOOTSTRAP = 2000


def _exact_excess(midpoints, probs, threshold):
    return float(np.dot(probs, np.maximum(0.0, midpoints - threshold)))


def _classical_mc_on_bins(midpoints, probs, threshold, n_samples, rng):
    idx = rng.choice(len(midpoints), size=n_samples, p=probs)
    return float(np.mean(np.maximum(0.0, midpoints[idx] - threshold)))


def _circuit_stats(qc):
    tc = qk_transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1)
    ops = tc.count_ops()
    return {"two_qubit_gates": ops.get("cx", 0), "depth": tc.depth(), "total": tc.size()}


def fit_loglog_slope(queries, rmses):
    log_q = np.log(np.array(queries, dtype=float))
    log_r = np.log(np.array(rmses, dtype=float))
    n = len(log_q)
    slope, intercept, r_value, _, std_err = sp_stats.linregress(log_q, log_r)
    rng = np.random.default_rng(SEED)
    boot_slopes = [sp_stats.linregress(log_q[rng.choice(n, size=n, replace=True)],
                                        log_r[rng.choice(n, size=n, replace=True)])[0]
                   for _ in range(N_BOOTSTRAP)]
    ci_lo, ci_hi = np.percentile(boot_slopes, [2.5, 97.5])
    return slope, ci_lo, ci_hi, r_value**2


def run_exp6_extended(label, losses, dp, qubit_range, budget=4000, n_reps=50):
    """Experiment 6 sweep at extended qubit counts."""
    shape, loc, scale = dp
    threshold = float(np.percentile(losses, 95))
    gt_analytic = analytic_lognormal_excess(shape, loc, scale, threshold)
    rng = np.random.default_rng(SEED)

    print(f"\n{label}: threshold=${threshold:.0f}, analytic=${gt_analytic:.2f}")
    print(f"{'n':>3} {'bins':>5} {'disc_err':>10} {'Q_RMSE':>10} {'C_RMSE':>10} "
          f"{'Q/C':>6} {'k':>3} {'k_max':>5} {'P(1)':>10} {'SP_2Q':>8} {'depth':>8} {'time':>6}")

    results = []
    for n in qubit_range:
        t0 = time.time()
        midpoints, probs = discretise_distribution(shape, loc, scale, n)
        gt_discrete = _exact_excess(midpoints, probs, threshold)
        disc_error = abs(gt_discrete - gt_analytic)

        A, oq, rsc = build_oracle_A(probs, midpoints, threshold)
        ep = exact_amplitude_readout(A, oq)
        ksafe = max_safe_k(ep)
        k_use = min(ksafe, max(0, (budget // 100 - 1) // 2))
        shots = max(100, budget // (2 * k_use + 1))

        sp = build_state_prep(probs, n)
        sp_stats_d = _circuit_stats(sp)
        a_stats = _circuit_stats(A)

        q_errors = []
        for _ in range(n_reps):
            ep2, _ = grover_boosted_estimate(A, oq, k_iters=k_use, shots=shots)
            q_errors.append((ep2 * rsc - gt_discrete) ** 2)
        q_rmse = float(np.sqrt(np.mean(q_errors)))

        c_errors = []
        for _ in range(n_reps):
            est = _classical_mc_on_bins(midpoints, probs, threshold, budget, rng)
            c_errors.append((est - gt_discrete) ** 2)
        c_rmse = float(np.sqrt(np.mean(c_errors)))

        elapsed = time.time() - t0
        ratio = c_rmse / q_rmse if q_rmse > 0 else 0

        print(f"{n:3d} {2**n:5d} {disc_error:10.0f} {q_rmse:10.0f} {c_rmse:10.0f} "
              f"{ratio:5.1f}x {k_use:3d} {ksafe:5d} {ep:10.6f} "
              f"{sp_stats_d['two_qubit_gates']:8d} {a_stats['depth']:8d} {elapsed:5.0f}s")

        results.append({
            "n": n, "bins": 2**n, "disc_err": disc_error,
            "q_rmse": q_rmse, "c_rmse": c_rmse, "ratio": ratio,
            "k_use": k_use, "k_safe": ksafe, "p_one": ep,
            "sp_2q": sp_stats_d["two_qubit_gates"], "oracle_depth": a_stats["depth"],
            "elapsed": elapsed,
        })
    return results


def run_extended_slopes(label, losses, dp, qubit_range, shots_per_k=1000, n_reps=30):
    """Slope fitting across multiple qubit counts for wider query range."""
    shape, loc, scale = dp
    rng = np.random.default_rng(SEED)

    q_queries, q_rmses = [], []
    c_queries, c_rmses = [], []

    for n in qubit_range:
        midpoints, probs = discretise_distribution(shape, loc, scale, n)
        threshold = float(np.percentile(losses, 95))
        exact_bins = _exact_excess(midpoints, probs, threshold)

        A, oq, rsc = build_oracle_A(probs, midpoints, catastrophe_threshold=threshold)
        ep = exact_amplitude_readout(A, oq)
        ks = max_safe_k(ep)

        for k in range(min(ks, 6) + 1):
            tq = shots_per_k * (2 * k + 1)
            t0 = time.time()
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

            elapsed = time.time() - t0
            print(f"  n={n} k={k}: queries={tq:5d}  Q=${q_rmses[-1]:.1f}  "
                  f"C=${c_rmses[-1]:.1f}  ({elapsed:.1f}s)")

    q_slope, q_lo, q_hi, q_r2 = fit_loglog_slope(q_queries, q_rmses)
    c_slope, c_lo, c_hi, c_r2 = fit_loglog_slope(c_queries, c_rmses)

    print(f"\n  {label}:")
    print(f"  Quantum slope = {q_slope:.3f}  95% CI [{q_lo:.3f}, {q_hi:.3f}]  R²={q_r2:.3f}")
    print(f"  Classical slope = {c_slope:.3f}  95% CI [{c_lo:.3f}, {c_hi:.3f}]  R²={c_r2:.3f}")
    print(f"  Data points: {len(q_queries)} (n={qubit_range}, k=0..6)")
    print(f"  Query range: {min(q_queries)}--{max(q_queries)}")

    return {
        "q_slope": q_slope, "q_ci": (q_lo, q_hi), "q_r2": q_r2,
        "c_slope": c_slope, "c_ci": (c_lo, c_hi), "c_r2": c_r2,
        "n_points": len(q_queries), "query_range": (min(q_queries), max(q_queries)),
    }


if __name__ == "__main__":
    losses_s, dp_s = get_synthetic_loss_data()

    # First: quick timing test at n=8
    print("=" * 70)
    print("TIMING TEST: single n=8 run")
    print("=" * 70)
    t0 = time.time()
    shape, loc, scale = dp_s
    mp8, pr8 = discretise_distribution(shape, loc, scale, 8)
    threshold = float(np.percentile(losses_s, 95))
    A8, oq8, rsc8 = build_oracle_A(pr8, mp8, threshold)
    ep8 = exact_amplitude_readout(A8, oq8)
    ks8 = max_safe_k(ep8)
    print(f"  n=8: P(|1>)={ep8:.6f}, k_max={ks8}")
    print(f"  Build time: {time.time()-t0:.1f}s")

    t1 = time.time()
    est, info = grover_boosted_estimate(A8, oq8, k_iters=min(ks8, 6), shots=100)
    print(f"  Single Grover run (k={min(ks8,6)}, 100 shots): {time.time()-t1:.1f}s")
    print(f"  Circuit depth: {info.get('circuit_depth', '?')}, gates: {info.get('gate_count', '?')}")

    # Exp 6 extended sweep
    print("\n" + "=" * 70)
    print("EXPERIMENT 6 EXTENDED (n=3..8)")
    print("=" * 70)
    r6 = run_exp6_extended("Synthetic", losses_s, dp_s, [3, 4, 5, 6, 7, 8])

    # Extended slopes on synthetic (n=3 and n=8 for wider range)
    print("\n" + "=" * 70)
    print("EXTENDED SLOPES (synthetic, n=3 + n=8)")
    print("=" * 70)
    rs = run_extended_slopes("Synthetic extended", losses_s, dp_s, [3, 8])

    # Compare with original n=3 only
    print("\n" + "=" * 70)
    print("ORIGINAL SLOPES (synthetic, n=3 only, for comparison)")
    print("=" * 70)
    ro = run_extended_slopes("Synthetic n=3 only", losses_s, dp_s, [3])

    # NOAA
    print("\n" + "=" * 70)
    print("EXTENDED SLOPES (NOAA, n=3 + n=8)")
    print("=" * 70)
    losses_r, dp_r = get_real_loss_data()
    rn = run_extended_slopes("NOAA extended", losses_r, dp_r, [3, 8])
