"""Compare equal-width vs quantile binning for Experiments 1 and 3 (synthetic data).

Outputs a summary table showing that speedup ratios are consistent across
binning schemes, confirming that the oracle-model advantage is not an
artefact of the binning choice.
"""
import numpy as np
from scipy import stats

from fairy_queen.data_pipeline import get_synthetic_loss_data
from fairy_queen.quantum_circuits import discretise_distribution
from fairy_queen.experiment5 import analytic_lognormal_excess
from fairy_queen.experiment1 import (
    _exact_excess_loss, classical_mc_excess_loss, _classical_mc_on_bins,
)
from fairy_queen.quantum_circuits import (
    build_oracle_A, exact_amplitude_readout, grover_boosted_estimate, max_safe_k,
)

SEED = 42
N_REPS = 30
SHOTS = 1000
K_VALUES = [0, 1, 2, 3, 4, 5, 6]
PERCENTILES = [0.90, 0.95, 0.97]
EXP3_SHOTS = 8192


def run_exp1_variant(dist_params, losses, binning):
    shape, loc, scale = dist_params
    midpoints, probs = discretise_distribution(shape, loc, scale, 3, binning=binning)
    threshold = float(np.percentile(losses, 95))
    exact_bins = _exact_excess_loss(midpoints, probs, threshold)
    analytic = analytic_lognormal_excess(shape, loc, scale, threshold)
    disc_err = abs(exact_bins - analytic)

    A_circuit, obj_qubit, rescale = build_oracle_A(probs, midpoints, catastrophe_threshold=threshold)
    exact_prob = exact_amplitude_readout(A_circuit, obj_qubit)
    k_safe = max_safe_k(exact_prob)
    k_vals = [k for k in K_VALUES if k <= k_safe]
    rng = np.random.default_rng(SEED)

    rows = []
    for k in k_vals:
        total_queries = SHOTS * (2 * k + 1)
        q_errs = []
        for _ in range(N_REPS):
            est_prob, _ = grover_boosted_estimate(A_circuit, obj_qubit, k_iters=k, shots=SHOTS)
            q_errs.append((est_prob * rescale - exact_bins) ** 2)
        q_rmse = float(np.sqrt(np.mean(q_errs)))

        cb_errs = []
        for _ in range(N_REPS):
            est = _classical_mc_on_bins(midpoints, probs, threshold, total_queries, rng)
            cb_errs.append((est - exact_bins) ** 2)
        cb_rmse = float(np.sqrt(np.mean(cb_errs)))

        ratio = cb_rmse / q_rmse if q_rmse > 0 else 0
        rows.append((k, total_queries, q_rmse, cb_rmse, ratio))

    return rows, disc_err, analytic, exact_bins, k_safe


def run_exp3_variant(dist_params, losses, binning):
    shape, loc, scale = dist_params
    midpoints, probs = discretise_distribution(shape, loc, scale, 3, binning=binning)
    rng = np.random.default_rng(SEED)

    rows = []
    for pct in PERCENTILES:
        threshold = float(np.percentile(losses, pct * 100))
        exact_bins = _exact_excess_loss(midpoints, probs, threshold)
        analytic = analytic_lognormal_excess(shape, loc, scale, threshold)
        disc_err = abs(exact_bins - analytic)

        A_circuit, obj_qubit, rescale = build_oracle_A(probs, midpoints, catastrophe_threshold=threshold)
        exact_prob = exact_amplitude_readout(A_circuit, obj_qubit)
        k_safe = max_safe_k(exact_prob)
        k_use = min(k_safe, 6)
        grover_shots = max(100, EXP3_SHOTS // (2 * k_use + 1)) if k_use > 0 else EXP3_SHOTS
        total_queries = grover_shots * (2 * k_use + 1)

        qg_errs = []
        for _ in range(N_REPS):
            est_prob, _ = grover_boosted_estimate(A_circuit, obj_qubit, k_iters=k_use, shots=grover_shots)
            qg_errs.append((est_prob * rescale - exact_bins) ** 2)
        qg_rmse = float(np.sqrt(np.mean(qg_errs)))

        cb_errs = []
        for _ in range(N_REPS):
            est = _classical_mc_on_bins(midpoints, probs, threshold, total_queries, rng)
            cb_errs.append((est - exact_bins) ** 2)
        cb_rmse = float(np.sqrt(np.mean(cb_errs)))

        ratio = cb_rmse / qg_rmse if qg_rmse > 0 else 0
        rows.append((pct, threshold, disc_err, k_use, k_safe, qg_rmse, cb_rmse, ratio))

    return rows


def main():
    losses, dist_params = get_synthetic_loss_data()
    print(f"Synthetic: shape={dist_params[0]:.4f}, loc={dist_params[1]:.4f}, scale={dist_params[2]:.4f}")
    print(f"{'='*90}")

    for binning in ["equal_width", "quantile"]:
        print(f"\n--- Experiment 1 ({binning} binning) ---")
        rows, disc_err, analytic, exact_bins, k_safe = run_exp1_variant(dist_params, losses, binning)
        print(f"  analytic=${analytic:.0f}  exact_bins=${exact_bins:.0f}  disc_err=${disc_err:.0f}  k_safe={k_safe}")
        print(f"  {'k':>3}  {'Queries':>7}  {'Q(bins)':>10}  {'Cb(bins)':>10}  {'Q/Cb':>6}")
        for k, q, qr, cr, ratio in rows:
            print(f"  {k:3d}  {q:7d}  {qr:10.0f}  {cr:10.0f}  {ratio:5.1f}x")

    for binning in ["equal_width", "quantile"]:
        print(f"\n--- Experiment 3 ({binning} binning) ---")
        rows = run_exp3_variant(dist_params, losses, binning)
        print(f"  {'Pctl':>5}  {'Disc.err':>10}  {'k':>3}  {'k_max':>5}  {'Q(bins)':>10}  {'Cb(bins)':>10}  {'Q/Cb':>6}")
        for pct, thr, de, ku, km, qr, cr, ratio in rows:
            print(f"  {pct*100:4.0f}%  {de:10.0f}  {ku:3d}  {km:5d}  {qr:10.0f}  {cr:10.0f}  {ratio:5.1f}x")


if __name__ == "__main__":
    main()
