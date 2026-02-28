"""Targeted n=8 experiment: Exp 6 row + slope data points at k=0,3,6."""
import time
import numpy as np
from scipy import stats as sp_stats

from fairy_queen.data_pipeline import get_synthetic_loss_data, get_real_loss_data
from fairy_queen.quantum_circuits import (
    discretise_distribution, build_oracle_A, exact_amplitude_readout,
    grover_boosted_estimate, max_safe_k, build_state_prep,
)
from fairy_queen.experiment5 import analytic_lognormal_excess
from qiskit import transpile as qk_transpile

SEED = 42
N_REPS = 20  # fewer reps to keep runtime manageable


def _exact_excess(midpoints, probs, threshold):
    return float(np.dot(probs, np.maximum(0.0, midpoints - threshold)))


def _classical_mc_on_bins(midpoints, probs, threshold, n_samples, rng):
    idx = rng.choice(len(midpoints), size=n_samples, p=probs)
    return float(np.mean(np.maximum(0.0, midpoints[idx] - threshold)))


def _circuit_stats(qc):
    tc = qk_transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1)
    ops = tc.count_ops()
    return {"two_qubit_gates": ops.get("cx", 0), "depth": tc.depth(), "total": tc.size()}


def run_single_n(label, losses, dp, n, budget=4000, k_values=None):
    shape, loc, scale = dp
    threshold = float(np.percentile(losses, 95))
    gt_analytic = analytic_lognormal_excess(shape, loc, scale, threshold)
    mp, pr = discretise_distribution(shape, loc, scale, n)
    gt_discrete = _exact_excess(mp, pr, threshold)
    disc_error = abs(gt_discrete - gt_analytic)
    rng = np.random.default_rng(SEED)

    print(f"\n{label} n={n} ({2**n} bins):")
    print(f"  threshold=${threshold:.0f}, analytic=${gt_analytic:.2f}, "
          f"exact_bins=${gt_discrete:.2f}, disc_err=${disc_error:.0f}")

    A, oq, rsc = build_oracle_A(pr, mp, threshold)
    ep = exact_amplitude_readout(A, oq)
    ks = max_safe_k(ep)
    print(f"  P(|1>)={ep:.6f}, k_max={ks}")

    # Circuit stats
    t0 = time.time()
    sp = build_state_prep(pr, n)
    sp_stats = _circuit_stats(sp)
    a_stats = _circuit_stats(A)
    print(f"  SP 2Q gates: {sp_stats['two_qubit_gates']}, Oracle depth: {a_stats['depth']}")
    print(f"  Transpile time: {time.time()-t0:.1f}s")

    # Exp 6 row: quantum + classical at fixed budget
    k_use = min(ks, max(0, (budget // 100 - 1) // 2))
    shots_budget = max(100, budget // (2 * k_use + 1))

    t1 = time.time()
    q_errors = []
    for i in range(N_REPS):
        ep2, info = grover_boosted_estimate(A, oq, k_iters=k_use, shots=shots_budget)
        q_errors.append((ep2 * rsc - gt_discrete) ** 2)
        if i == 0:
            print(f"  Full circuit (k={k_use}): depth={info.get('circuit_depth','?')}, "
                  f"gates={info.get('gate_count','?')}")
    q_rmse = float(np.sqrt(np.mean(q_errors)))
    print(f"  Quantum RMSE (k={k_use}, B={budget}, {N_REPS} reps): ${q_rmse:.0f} "
          f"({time.time()-t1:.1f}s)")

    c_errors = []
    for _ in range(N_REPS):
        est = _classical_mc_on_bins(mp, pr, threshold, budget, rng)
        c_errors.append((est - gt_discrete) ** 2)
    c_rmse = float(np.sqrt(np.mean(c_errors)))
    ratio = c_rmse / q_rmse if q_rmse > 0 else 0
    print(f"  Classical RMSE (B={budget}, {N_REPS} reps): ${c_rmse:.0f}")
    print(f"  Q/C ratio: {ratio:.1f}x")

    # Slope data: a few k values for convergence
    if k_values is None:
        k_values = [0, 2, 4, 6]
    k_values = [k for k in k_values if k <= ks]

    shots_per_k = 500
    q_queries, q_rmses = [], []
    c_queries, c_rmses = [], []

    print(f"\n  Convergence data (slope fitting, {N_REPS} reps, {shots_per_k} shots/k):")
    for k in k_values:
        tq = shots_per_k * (2 * k + 1)
        t2 = time.time()

        qe = []
        for _ in range(N_REPS):
            ep2, _ = grover_boosted_estimate(A, oq, k_iters=k, shots=shots_per_k)
            qe.append((ep2 * rsc - gt_discrete) ** 2)
        q_queries.append(tq)
        q_rmses.append(float(np.sqrt(np.mean(qe))))

        ce = []
        for _ in range(N_REPS):
            est = _classical_mc_on_bins(mp, pr, threshold, tq, rng)
            ce.append((est - gt_discrete) ** 2)
        c_queries.append(tq)
        c_rmses.append(float(np.sqrt(np.mean(ce))))

        print(f"    k={k}: queries={tq:5d}  Q=${q_rmses[-1]:.1f}  C=${c_rmses[-1]:.1f}  "
              f"({time.time()-t2:.1f}s)")

    return {
        "disc_err": disc_error, "q_rmse_budget": q_rmse, "c_rmse_budget": c_rmse,
        "ratio": ratio, "k_use": k_use, "k_safe": ks, "p_one": ep,
        "sp_2q": sp_stats["two_qubit_gates"], "oracle_depth": a_stats["depth"],
        "q_queries": q_queries, "q_rmses": q_rmses,
        "c_queries": c_queries, "c_rmses": c_rmses,
    }


def combined_slope(r3_data, r8_data):
    """Combine n=3 and n=8 data for slope fitting."""
    q_queries = r3_data["q_queries"] + r8_data["q_queries"]
    q_rmses = r3_data["q_rmses"] + r8_data["q_rmses"]
    c_queries = r3_data["c_queries"] + r8_data["c_queries"]
    c_rmses = r3_data["c_rmses"] + r8_data["c_rmses"]

    def fit(queries, rmses):
        lq = np.log(np.array(queries, dtype=float))
        lr = np.log(np.array(rmses, dtype=float))
        m = len(lq)
        slope, _, r_value, _, _ = sp_stats.linregress(lq, lr)
        rng = np.random.default_rng(SEED)
        boots = [sp_stats.linregress(lq[rng.choice(m, m, True)],
                                      lr[rng.choice(m, m, True)])[0]
                 for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return slope, lo, hi, r_value**2

    qs, qlo, qhi, qr2 = fit(q_queries, q_rmses)
    cs, clo, chi, cr2 = fit(c_queries, c_rmses)

    print(f"\n  Combined n=3+n=8 slopes ({len(q_queries)} points, "
          f"query range {min(q_queries)}--{max(q_queries)}):")
    print(f"    Quantum: {qs:.3f} [{qlo:.3f}, {qhi:.3f}] R²={qr2:.3f}")
    print(f"    Classical: {cs:.3f} [{clo:.3f}, {chi:.3f}] R²={cr2:.3f}")
    return {"q": (qs, qlo, qhi, qr2), "c": (cs, clo, chi, cr2)}


if __name__ == "__main__":
    # Synthetic
    losses_s, dp_s = get_synthetic_loss_data()

    print("=" * 70)
    print("SYNTHETIC DATA")
    print("=" * 70)

    # n=3 for baseline slope data
    r3s = run_single_n("Synthetic", losses_s, dp_s, 3, k_values=[0, 1, 2, 3, 4, 5, 6])
    # n=8 for extended data
    r8s = run_single_n("Synthetic", losses_s, dp_s, 8, k_values=[0, 2, 4, 6])

    print("\n--- SYNTHETIC COMBINED SLOPES ---")
    combined_slope(r3s, r8s)

    # NOAA
    losses_r, dp_r = get_real_loss_data()

    print("\n" + "=" * 70)
    print("NOAA DATA")
    print("=" * 70)

    r3n = run_single_n("NOAA", losses_r, dp_r, 3, k_values=[0, 1, 2, 3, 4, 5, 6])
    r8n = run_single_n("NOAA", losses_r, dp_r, 8, k_values=[0, 2, 4, 6])

    print("\n--- NOAA COMBINED SLOPES ---")
    combined_slope(r3n, r8n)
