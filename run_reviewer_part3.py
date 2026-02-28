"""Part 3: Quasi-Monte Carlo baseline comparison."""
import numpy as np
from scipy.stats import lognorm
from scipy.stats.qmc import Sobol

from fairy_queen.data_pipeline import get_synthetic_loss_data
from fairy_queen.experiment5 import (
    analytic_lognormal_excess, _naive_mc, _conditional_tail_mc,
    _importance_sampling_mc, _classical_mc_discrete,
)
from fairy_queen.quantum_circuits import discretise_distribution

SEED = 42


def _qmc_excess(shape, loc, scale, threshold, n_samples, seed):
    sampler = Sobol(d=1, scramble=True, seed=seed)
    n_pow2 = 2 ** int(np.ceil(np.log2(n_samples)))
    U = sampler.random(n_pow2).flatten()[:n_samples]
    U = np.clip(U, 1e-10, 1 - 1e-10)
    X = lognorm.ppf(U, shape, loc, scale)
    return float(np.mean(np.maximum(0.0, X - threshold)))


if __name__ == "__main__":
    losses, dp = get_synthetic_loss_data()
    shape, loc, scale = dp
    rng = np.random.default_rng(SEED)

    budgets = [512, 2048, 8192]
    percentiles = [0.90, 0.95, 0.97]
    n_reps = 50
    n_qubits = 3

    midpoints, probs = discretise_distribution(shape, loc, scale, n_qubits)

    print("Quasi-Monte Carlo vs classical baselines (RMSE vs analytic truth)")
    print(f"\n{'Pctl':>5} {'B':>6} {'QMC':>10} {'NaiveMC':>10} {'CT':>10} "
          f"{'IS':>10} {'Disc_MC':>10} {'QMC/Naive':>10}")

    for pct in percentiles:
        threshold = float(np.percentile(losses, pct * 100))
        gt = analytic_lognormal_excess(shape, loc, scale, threshold)

        for B in budgets:
            qmc_ests, naive_ests, ct_ests, is_ests, disc_ests = [], [], [], [], []
            for i in range(n_reps):
                qmc_ests.append(_qmc_excess(shape, loc, scale, threshold, B, seed=SEED+i))
                naive_ests.append(_naive_mc(shape, loc, scale, threshold, B, rng))
                ct_ests.append(_conditional_tail_mc(shape, loc, scale, threshold, B, rng))
                is_ests.append(_importance_sampling_mc(shape, loc, scale, threshold, B, rng))
                disc_ests.append(_classical_mc_discrete(midpoints, probs, threshold, B, rng))

            qmc_rmse = float(np.sqrt(np.mean((np.array(qmc_ests) - gt) ** 2)))
            naive_rmse = float(np.sqrt(np.mean((np.array(naive_ests) - gt) ** 2)))
            ct_rmse = float(np.sqrt(np.mean((np.array(ct_ests) - gt) ** 2)))
            is_rmse = float(np.sqrt(np.mean((np.array(is_ests) - gt) ** 2)))
            disc_rmse = float(np.sqrt(np.mean((np.array(disc_ests) - gt) ** 2)))

            qmc_ratio = naive_rmse / qmc_rmse if qmc_rmse > 0 else 0

            print(f"  {pct*100:4.0f}% {B:6d} {qmc_rmse:10.1f} {naive_rmse:10.1f} "
                  f"{ct_rmse:10.1f} {is_rmse:10.1f} {disc_rmse:10.1f} {qmc_ratio:9.1f}x")
