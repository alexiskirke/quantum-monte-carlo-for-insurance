"""Part 2: Experiment 6 with log-spaced bins (extended tail)."""
import numpy as np
from scipy.stats import lognorm

from fairy_queen.data_pipeline import get_synthetic_loss_data
from fairy_queen.quantum_circuits import (
    build_oracle_A, exact_amplitude_readout,
    grover_boosted_estimate, max_safe_k,
)
from fairy_queen.experiment5 import analytic_lognormal_excess

SEED = 42


def discretise_logspaced(shape, loc, scale, n_qubits):
    n_bins = 2 ** n_qubits
    low = lognorm.ppf(0.001, shape, loc, scale)
    high = lognorm.ppf(0.9999, shape, loc, scale)
    edges = np.geomspace(max(low, 1e-3), high, n_bins + 1)
    probs = np.diff(lognorm.cdf(edges, shape, loc, scale))
    probs = probs / probs.sum()
    midpoints = np.sqrt(edges[:-1] * edges[1:])
    return midpoints, probs


if __name__ == "__main__":
    losses, dp = get_synthetic_loss_data()
    shape, loc, scale = dp
    threshold = float(np.percentile(losses, 95))
    gt_analytic = analytic_lognormal_excess(shape, loc, scale, threshold)
    rng = np.random.default_rng(SEED)
    budget = 4000
    n_reps = 30

    print("Log-spaced bins experiment (extended to 99.99th pctl)")
    print(f"Threshold = ${threshold:.0f}, analytic E[excess] = ${gt_analytic:.2f}")

    # Also compare with equal-width from standard experiment 6
    from fairy_queen.quantum_circuits import discretise_distribution

    print(f"\n{'Binning':<12} {'n':>3} {'bins':>5} {'disc_err':>10} {'Q_RMSE':>10} "
          f"{'C_RMSE':>10} {'Q/C':>6} {'k':>3} {'k_max':>5} {'P(1)':>10}")

    for n in [3, 4, 5]:
        for binning, disc_fn in [("equal_w", lambda s, l, sc, nq: discretise_distribution(s, l, sc, nq)),
                                   ("log_sp", lambda s, l, sc, nq: discretise_logspaced(s, l, sc, nq))]:
            midpoints, probs = disc_fn(shape, loc, scale, n)
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
                idx = rng.choice(len(midpoints), size=budget, p=probs)
                est = float(np.mean(np.maximum(0.0, midpoints[idx] - threshold)))
                c_errors.append((est - gt_discrete) ** 2)
            c_rmse = float(np.sqrt(np.mean(c_errors)))

            ratio = c_rmse / q_rmse if q_rmse > 0 else 0
            print(f"{binning:<12} {n:3d} {2**n:5d} {disc_error:10.0f} {q_rmse:10.0f} "
                  f"{c_rmse:10.0f} {ratio:5.1f}x {k_use:3d} {ksafe:5d} {ep:10.6f}")
