"""Automated result generation: metrics.json, summary.csv, and plots.

All outputs land in ./results/ and are structured for easy LLM ingestion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fairy_queen.logging_config import get_logger

RESULTS_DIR = Path("results")


def save_all(exp1: Dict, exp2: Dict, exp3: Dict,
             exp4: Optional[Dict] = None,
             exp5_synth: Optional[Dict] = None,
             exp5_real: Optional[Dict] = None,
             exp6: Optional[Dict] = None,
             exp7: Optional[Dict] = None) -> None:
    log = get_logger()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _save_metrics_json(exp1, exp2, exp3, exp4, exp5_synth, exp5_real, exp6, exp7)
    _save_summary_csv(exp1, exp2, exp3, exp4, exp5_synth, exp5_real, exp6, exp7)
    _plot_exp1_convergence(exp1)
    _plot_exp2_noise(exp2)
    _plot_exp3_tail(exp3)
    if exp4 is not None:
        _plot_exp4_convergence(exp4)
        _plot_exp4_tail(exp4)
    if exp5_synth is not None:
        _plot_exp5_budget_sweep(exp5_synth, "synth")
        _plot_exp5_method_bars(exp5_synth, "synth")
    if exp5_real is not None:
        _plot_exp5_budget_sweep(exp5_real, "real")
        _plot_exp5_method_bars(exp5_real, "real")
    if exp6 is not None:
        _plot_exp6_scaling(exp6)
        _plot_exp6_gates(exp6)
    if exp7 is not None:
        _plot_exp7_empirical(exp7)

    log.warning("All results saved to %s/", RESULTS_DIR)


# ------------------------------------------------------------------
# metrics.json
# ------------------------------------------------------------------

def _save_metrics_json(exp1, exp2, exp3, exp4, exp5s, exp5r, exp6, exp7=None):
    path = RESULTS_DIR / "metrics.json"

    def _clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        return obj

    payload = {
        "experiment1_convergence": _clean(exp1),
        "experiment2_noise": _clean(exp2),
        "experiment3_tail": _clean(exp3),
    }
    if exp4 is not None:
        payload["experiment4_real_data"] = _clean(exp4)
    if exp5s is not None:
        payload["experiment5_fair_synth"] = _clean(exp5s)
    if exp5r is not None:
        payload["experiment5_fair_real"] = _clean(exp5r)
    if exp6 is not None:
        payload["experiment6_scaling"] = _clean(exp6)
    if exp7 is not None:
        payload["experiment7_empirical_pmf"] = _clean(exp7)

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    get_logger().info("Wrote %s", path)


# ------------------------------------------------------------------
# summary.csv
# ------------------------------------------------------------------

def _save_summary_csv(exp1, exp2, exp3, exp4, exp5s, exp5r, exp6, exp7=None):
    rows = []

    for qr, cr in zip(exp1["quantum_results"], exp1["classical_results"]):
        rows.append({
            "experiment": "1_convergence",
            "parameter": f"k={qr['k']}",
            "total_oracle_queries": qr["total_oracle_queries"],
            "classical_rmse": cr["rmse"],
            "quantum_rmse": qr["rmse"],
            "exact_value": exp1["exact_excess_loss"],
        })

    for level, data in exp2.items():
        if level.startswith("_"):
            continue
        rows.append({
            "experiment": "2_noise",
            "parameter": level,
            "classical_rmse": None,
            "quantum_rmse": data["rmse"],
            "exact_value": data["exact_loss"],
        })

    for key, data in exp3["percentile_results"].items():
        rows.append({
            "experiment": "3_tail",
            "parameter": f"p={data['percentile']}",
            "total_oracle_queries": data.get("total_queries_grover"),
            "classical_rmse": data["classical_rmse"],
            "quantum_rmse": data["quantum_grover_rmse"],
            "exact_value": data["exact_excess_loss"],
        })

    if exp4 is not None:
        for qr, cr in zip(exp4["convergence"]["quantum_results"],
                          exp4["convergence"]["classical_results"]):
            rows.append({
                "experiment": "4A_real_convergence",
                "parameter": f"k={qr['k']}",
                "total_oracle_queries": qr["total_oracle_queries"],
                "classical_rmse": cr["rmse"],
                "quantum_rmse": qr["rmse"],
                "exact_value": exp4["convergence"]["exact_excess_loss"],
            })
        for key, data in exp4["tail_sweep"]["percentile_results"].items():
            rows.append({
                "experiment": "4B_real_tail",
                "parameter": f"p={data['percentile']}",
                "total_oracle_queries": data.get("total_queries_grover"),
                "classical_rmse": data["classical_rmse"],
                "quantum_rmse": data["quantum_grover_rmse"],
                "exact_value": data["exact_excess_loss"],
            })

    for exp5, tag in [(exp5s, "5_fair_synth"), (exp5r, "5_fair_real")]:
        if exp5 is None:
            continue
        for pkey, pdata in exp5["percentile_results"].items():
            for bkey, bdata in pdata["budgets"].items():
                for method in ["naive_mc", "conditional_tail",
                               "importance_sampling", "classical_discrete",
                               "quantum"]:
                    rows.append({
                        "experiment": tag,
                        "parameter": f"p={pdata['percentile']}_B={bdata['budget']}_{method}",
                        "total_oracle_queries": bdata["budget"],
                        "classical_rmse": None,
                        "quantum_rmse": bdata[method]["rmse_vs_analytic"],
                        "exact_value": pdata["analytic_truth"],
                    })

    if exp6 is not None:
        for entry in exp6["sweep"]:
            rows.append({
                "experiment": "6_scaling",
                "parameter": f"n={entry['n_qubits']}",
                "total_oracle_queries": exp6["budget"],
                "classical_rmse": entry["classical_rmse_vs_discrete"],
                "quantum_rmse": entry["quantum_rmse_vs_discrete"],
                "exact_value": entry["discrete_truth"],
            })

    if exp7 is not None:
        for pkey, pdata in exp7["percentile_results"].items():
            for bkey, bdata in pdata["budgets"].items():
                for method in ["naive_mc_resample", "classical_bins", "quantum"]:
                    rows.append({
                        "experiment": "7_empirical_pmf",
                        "parameter": f"p={pdata['percentile']}_B={bdata['budget']}_{method}",
                        "total_oracle_queries": bdata["budget"],
                        "classical_rmse": None,
                        "quantum_rmse": bdata[method]["rmse_vs_exact_bins"],
                        "exact_value": pdata["exact_on_bins"],
                    })

    df = pd.DataFrame(rows)
    path = RESULTS_DIR / "summary.csv"
    df.to_csv(path, index=False)
    get_logger().info("Wrote %s", path)


# ------------------------------------------------------------------
# Plots — Experiments 1–3 (synthetic data, unchanged)
# ------------------------------------------------------------------

def _plot_exp1_convergence(exp1: Dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    q_queries = [r["total_oracle_queries"] for r in exp1["quantum_results"]]
    q_rmse_bins = [r.get("rmse_vs_bins", r["rmse"])
                   for r in exp1["quantum_results"]]
    q_k = [r["k"] for r in exp1["quantum_results"]]
    c_queries = [r["total_oracle_queries"] for r in exp1["classical_results"]]
    c_rmse_anal = [r.get("rmse_vs_analytic", r["rmse"])
                   for r in exp1["classical_results"]]
    cb_queries = [r["total_oracle_queries"]
                  for r in exp1.get("classical_bins_results", [])]
    cb_rmse = [r["rmse_vs_bins"]
               for r in exp1.get("classical_bins_results", [])]
    disc_err = exp1.get("disc_error", 0)

    # Left: estimation error (all vs exact-on-bins)
    ax1.loglog(q_queries, q_rmse_bins, "s-",
               label="Quantum AE (vs bins)", color="#1f77b4",
               linewidth=2, markersize=7)
    if cb_queries:
        ax1.loglog(cb_queries, cb_rmse, "^-",
                   label="Classical MC on bins", color="#ff7f0e",
                   linewidth=2, markersize=7)
    for x_pt, y_pt, k in zip(q_queries, q_rmse_bins, q_k):
        ax1.annotate(f"k={k}", (x_pt, y_pt), textcoords="offset points",
                     xytext=(5, 5), fontsize=7, color="#1f77b4")
    if cb_queries:
        x_ref = np.linspace(min(cb_queries), max(cb_queries), 200)
        c0 = cb_rmse[0] * np.sqrt(cb_queries[0])
        ax1.loglog(x_ref, c0 / np.sqrt(x_ref), "--", alpha=0.3,
                   color="#ff7f0e", label=r"$O(1/\sqrt{N})$ ref")
        q0 = q_rmse_bins[-1] * q_queries[-1]
        ax1.loglog(x_ref, q0 / x_ref, "--", alpha=0.3,
                   color="#2ca02c", label=r"$O(1/N)$ ref")
    ax1.set_xlabel("Total Oracle Queries", fontsize=11)
    ax1.set_ylabel("RMSE ($)", fontsize=11)
    ax1.set_title("Estimation Error (vs exact-on-bins)", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", ls=":", alpha=0.3)

    # Right: end-to-end (classical cont. vs analytic, quantum vs analytic)
    q_rmse_anal = [r.get("rmse_vs_analytic", r["rmse"])
                   for r in exp1["quantum_results"]]
    ax2.loglog(c_queries, c_rmse_anal, "o-",
               label="Classical MC cont. (vs analytic)", color="#d62728",
               linewidth=2, markersize=7)
    ax2.loglog(q_queries, q_rmse_anal, "s-",
               label="Quantum AE (vs analytic)", color="#1f77b4",
               linewidth=2, markersize=7)
    if disc_err > 0:
        ax2.axhline(disc_err, ls="--", color="grey", alpha=0.6,
                     label=f"Disc. error = ${disc_err:,.0f}")
    ax2.set_xlabel("Total Oracle Queries", fontsize=11)
    ax2.set_ylabel("RMSE ($)", fontsize=11)
    ax2.set_title("End-to-end Error (vs analytic truth)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", ls=":", alpha=0.3)

    fig.suptitle(
        f"Experiment 1: Convergence — Synthetic Data "
        f"(P(|1⟩)={exp1['readout_prob']:.4f})", fontsize=11, y=1.02)
    fig.tight_layout()

    path = RESULTS_DIR / "plot_exp1_convergence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    get_logger().info("Wrote %s", path)


def _plot_exp2_noise(exp2: Dict) -> None:
    levels = [l for l in exp2 if not l.startswith("_")]
    rmses = [exp2[l]["rmse"] for l in levels]
    meta = exp2.get("_meta", {})

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"noiseless": "#2ca02c", "low": "#1f77b4",
              "medium": "#ff7f0e", "high": "#d62728"}
    x = np.arange(len(levels))
    bars = ax.bar(x, rmses,
                  color=[colors.get(l, "#999999") for l in levels],
                  edgecolor="black", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel("RMSE ($)")
    ax.set_title(f"Experiment 2: QAE Accuracy Under NISQ Noise "
                 f"(k={meta.get('k_iters', '?')}, "
                 f"P(|1⟩)={meta.get('readout_prob', 0):.4f})")

    for bar, r in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"${r:.0f}", ha="center", va="bottom", fontsize=9)

    ax.grid(axis="y", ls=":", alpha=0.3)
    fig.tight_layout()

    path = RESULTS_DIR / "plot_exp2_noise.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    get_logger().info("Wrote %s", path)


def _plot_exp3_tail(exp3: Dict) -> None:
    pdata = exp3["percentile_results"]
    pcts = sorted(pdata.keys(), key=float)

    labels = [f"{float(p)*100:.0f}th %" for p in pcts]
    cb_rmse = [pdata[p].get("classical_bins_rmse",
                            pdata[p]["classical_rmse"]) for p in pcts]
    q_gr_rmse = [pdata[p]["quantum_grover_rmse"] for p in pcts]
    c_cont = [pdata[p].get("classical_cont_rmse", 0) for p in pcts]
    disc_err = [pdata[p].get("disc_error", 0) for p in pcts]
    k_used = [pdata[p]["k_used"] for p in pcts]
    k_max = [pdata[p]["max_safe_k"] for p in pcts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(pcts))
    w = 0.3

    # Left: estimation error (all vs exact-on-bins — apples-to-apples)
    ax1.bar(x - w/2, cb_rmse, w, label="Classical MC on bins",
            color="#ff7f0e", alpha=0.85)
    ax1.bar(x + w/2, q_gr_rmse, w, label="Quantum Grover",
            color="#2ca02c", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"{l}\nk={ku} (max {km})"
         for l, ku, km in zip(labels, k_used, k_max)])
    ax1.set_ylabel("RMSE vs exact-on-bins ($)")
    ax1.set_title("Estimation Error (same discretisation)", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", ls=":", alpha=0.3)
    for i in range(len(pcts)):
        ratio = cb_rmse[i] / q_gr_rmse[i] if q_gr_rmse[i] > 0 else 0
        ax1.annotate(f"{ratio:.1f}×", (x[i] + w/2, q_gr_rmse[i]),
                     textcoords="offset points", xytext=(0, 5),
                     fontsize=8, color="#2ca02c", fontweight="bold",
                     ha="center")

    # Right: discretisation error context
    ax2.bar(x - w/2, disc_err, w, label="Discretisation error",
            color="#9467bd", alpha=0.85)
    ax2.bar(x + w/2, c_cont, w,
            label="Classical cont. RMSE (vs analytic)",
            color="#d62728", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Error ($)")
    ax2.set_title("Discretisation Error vs Classical Accuracy", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", ls=":", alpha=0.3)

    fig.suptitle("Experiment 3: Tail Sweep — Synthetic Data (budget-matched ≈8192)",
                 fontsize=11, y=1.02)
    fig.tight_layout()

    path = RESULTS_DIR / "plot_exp3_tail.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    get_logger().info("Wrote %s", path)


# ------------------------------------------------------------------
# Plots — Experiment 4 (real data)
# ------------------------------------------------------------------

def _plot_exp4_convergence(exp4: Dict) -> None:
    conv = exp4["convergence"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    q_queries = [r["total_oracle_queries"] for r in conv["quantum_results"]]
    q_rmse_bins = [r["rmse_vs_bins"] for r in conv["quantum_results"]]
    q_k = [r["k"] for r in conv["quantum_results"]]
    c_queries = [r["total_oracle_queries"] for r in conv["classical_results"]]
    c_rmse_anal = [r["rmse_vs_analytic"] for r in conv["classical_results"]]
    cb_queries = [r["total_oracle_queries"]
                  for r in conv.get("classical_bins_results", [])]
    cb_rmse = [r["rmse_vs_bins"]
               for r in conv.get("classical_bins_results", [])]

    disc_err = conv.get("disc_error", 0)

    # Left panel: estimation error (all vs exact-on-bins)
    ax1.loglog(q_queries, q_rmse_bins, "s-",
               label="Quantum AE (vs bins)", color="#1f77b4",
               linewidth=2, markersize=7)
    if cb_queries:
        ax1.loglog(cb_queries, cb_rmse, "^-",
                   label="Classical MC on bins (vs bins)", color="#ff7f0e",
                   linewidth=2, markersize=7)
    for x_pt, y_pt, k in zip(q_queries, q_rmse_bins, q_k):
        ax1.annotate(f"k={k}", (x_pt, y_pt), textcoords="offset points",
                     xytext=(5, 5), fontsize=7, color="#1f77b4")
    ax1.set_xlabel("Total Oracle Queries", fontsize=11)
    ax1.set_ylabel("RMSE ($)", fontsize=11)
    ax1.set_title("Estimation Error (vs exact-on-bins)", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", ls=":", alpha=0.3)

    # Right panel: end-to-end (vs analytic)
    q_rmse_anal = [r["rmse_vs_analytic"] for r in conv["quantum_results"]]
    ax2.loglog(c_queries, c_rmse_anal, "o-",
               label="Classical MC cont. (vs analytic)", color="#d62728",
               linewidth=2, markersize=7)
    ax2.loglog(q_queries, q_rmse_anal, "s-",
               label="Quantum AE (vs analytic)", color="#1f77b4",
               linewidth=2, markersize=7)
    if disc_err > 0:
        ax2.axhline(disc_err, ls="--", color="grey", alpha=0.6,
                     label=f"Disc. error = ${disc_err:,.0f}")
    ax2.set_xlabel("Total Oracle Queries", fontsize=11)
    ax2.set_ylabel("RMSE ($)", fontsize=11)
    ax2.set_title("End-to-end Error (vs analytic truth)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", ls=":", alpha=0.3)

    dp = exp4.get("dist_params", {})
    fig.suptitle(
        f"Experiment 4A: Convergence on Real NOAA Data "
        f"(σ={dp.get('shape',0):.2f}, {exp4.get('n_records',0):,} records)",
        fontsize=11, y=1.02)
    fig.tight_layout()

    path = RESULTS_DIR / "plot_exp4_convergence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    get_logger().info("Wrote %s", path)


def _plot_exp4_tail(exp4: Dict) -> None:
    pdata = exp4["tail_sweep"]["percentile_results"]
    pcts = sorted(pdata.keys(), key=float)

    labels = [f"{float(p)*100:.0f}th %" for p in pcts]
    cb_rmse = [pdata[p]["classical_bins_rmse"] for p in pcts]
    q_gr_rmse = [pdata[p]["quantum_grover_rmse_vs_bins"] for p in pcts]
    disc_err = [pdata[p]["disc_error"] for p in pcts]
    c_cont = [pdata[p]["classical_cont_rmse_vs_analytic"] for p in pcts]
    k_used = [pdata[p]["k_used"] for p in pcts]
    k_max_vals = [pdata[p]["max_safe_k"] for p in pcts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(pcts))
    w = 0.3

    # Left: estimation error (all vs exact-on-bins — apples-to-apples)
    ax1.bar(x - w/2, cb_rmse, w, label="Classical MC on bins",
            color="#ff7f0e", alpha=0.85)
    ax1.bar(x + w/2, q_gr_rmse, w, label="Quantum Grover",
            color="#2ca02c", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"{l}\nk={ku} (max {km})"
         for l, ku, km in zip(labels, k_used, k_max_vals)])
    ax1.set_ylabel("RMSE vs exact-on-bins ($)")
    ax1.set_title("Estimation Error (same discretisation)", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", ls=":", alpha=0.3)
    for i in range(len(pcts)):
        ratio = cb_rmse[i] / q_gr_rmse[i] if q_gr_rmse[i] > 0 else 0
        lbl = f"{ratio:.1f}×" if ratio >= 1 else f"Q worse"
        ax1.annotate(lbl, (x[i] + w/2, q_gr_rmse[i]),
                     textcoords="offset points", xytext=(0, 5),
                     fontsize=8, color="#2ca02c", fontweight="bold",
                     ha="center")

    # Right: discretisation error context
    ax2.bar(x - w/2, disc_err, w, label="Discretisation error",
            color="#9467bd", alpha=0.85)
    ax2.bar(x + w/2, c_cont, w,
            label="Classical cont. RMSE (vs analytic)",
            color="#d62728", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Error ($)")
    ax2.set_title("Discretisation Error vs Classical Accuracy", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", ls=":", alpha=0.3)

    dp = exp4.get("dist_params", {})
    fig.suptitle(
        f"Experiment 4B: Tail Sweep — Real NOAA Data "
        f"(σ={dp.get('shape',0):.2f}, {exp4.get('n_records',0):,} records, "
        f"≈8192 queries)",
        fontsize=11, y=1.02)
    fig.tight_layout()

    path = RESULTS_DIR / "plot_exp4_tail.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    get_logger().info("Wrote %s", path)


# ------------------------------------------------------------------
# Plots — Experiment 5 (fair budget-matched comparison)
# ------------------------------------------------------------------

_METHOD_COLORS = {
    "naive_mc": "#d62728",
    "conditional_tail": "#ff7f0e",
    "importance_sampling": "#9467bd",
    "classical_discrete": "#8c564b",
    "quantum": "#2ca02c",
}
_METHOD_LABELS = {
    "naive_mc": "Naive MC",
    "conditional_tail": "Conditional Tail MC",
    "importance_sampling": "Importance Sampling",
    "classical_discrete": "Classical (same bins)",
    "quantum": "Quantum AE",
}


def _plot_exp5_budget_sweep(exp5: Dict, tag: str) -> None:
    """RMSE vs budget at 95th percentile, all methods on one log-log plot."""
    pdata = exp5["percentile_results"]
    pct_key = "0.95"
    if pct_key not in pdata:
        pct_key = list(pdata.keys())[len(pdata) // 2]
    pentry = pdata[pct_key]
    budgets_data = pentry["budgets"]
    budgets = sorted(budgets_data.keys(), key=int)

    fig, ax = plt.subplots(figsize=(9, 6))
    methods = ["naive_mc", "conditional_tail", "importance_sampling",
               "classical_discrete", "quantum"]

    for method in methods:
        bvals = [int(b) for b in budgets]
        rmses = [budgets_data[b][method]["rmse_vs_analytic"] for b in budgets]
        ax.loglog(bvals, rmses, "o-", label=_METHOD_LABELS[method],
                  color=_METHOD_COLORS[method], linewidth=2, markersize=7)

    ax.set_xlabel("Budget (oracle calls / samples)", fontsize=11)
    ax.set_ylabel("RMSE vs Analytic Truth ($)", fontsize=11)
    ax.set_title(
        f"Experiment 5: Budget-Matched RMSE ({tag} data, "
        f"{pentry['percentile']*100:.0f}th pctl, M=${pentry['threshold']:,.0f})\n"
        f"Analytic E[excess]=${pentry['analytic_truth']:,.2f}",
        fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.3)
    fig.tight_layout()

    path = RESULTS_DIR / f"plot_exp5_budget_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    get_logger().info("Wrote %s", path)


def _plot_exp5_method_bars(exp5: Dict, tag: str) -> None:
    """Bar chart: RMSE at largest budget, all percentiles × all methods."""
    pdata = exp5["percentile_results"]
    pcts = sorted(pdata.keys(), key=float)
    max_budget_key = str(max(exp5["budgets"]))

    methods = ["naive_mc", "conditional_tail", "importance_sampling",
               "classical_discrete", "quantum"]
    n_methods = len(methods)
    x = np.arange(len(pcts))
    w = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(12, 6))

    for j, method in enumerate(methods):
        rmses = [pdata[p]["budgets"][max_budget_key][method]["rmse_vs_analytic"]
                 for p in pcts]
        ax.bar(x + j * w - 0.4 + w / 2, rmses, w,
               label=_METHOD_LABELS[method],
               color=_METHOD_COLORS[method], alpha=0.85, edgecolor="black",
               linewidth=0.5)

    labels = []
    for p in pcts:
        pe = pdata[p]
        labels.append(
            f"{pe['percentile']*100:.0f}th %\n"
            f"M=${pe['threshold']:,.0f}\n"
            f"disc_err=${pe['discretisation_error']:,.1f}"
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("RMSE vs Analytic Truth ($)", fontsize=11)
    ax.set_title(
        f"Experiment 5: All Methods at Budget={max_budget_key} ({tag} data)\n"
        f"Ground truth = closed-form lognormal E[(X−M)+]",
        fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", ls=":", alpha=0.3)
    fig.tight_layout()

    path = RESULTS_DIR / f"plot_exp5_methods_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    get_logger().info("Wrote %s", path)


# ------------------------------------------------------------------
# Plots — Experiment 6 (qubit scaling sweep)
# ------------------------------------------------------------------

def _plot_exp6_scaling(exp6: Dict) -> None:
    """Discretisation error and estimation error vs n_qubits."""
    sweep = exp6["sweep"]
    ns = [e["n_qubits"] for e in sweep]
    disc_errs = [e["discretisation_error"] for e in sweep]
    q_est_errs = [e["quantum_rmse_vs_discrete"] for e in sweep]
    c_est_errs = [e["classical_rmse_vs_discrete"] for e in sweep]
    q_total = [e["quantum_total_rmse"] for e in sweep]
    c_total = [e["classical_total_rmse"] for e in sweep]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.semilogy(ns, disc_errs, "D-", label="Discretisation error",
                 color="#e377c2", linewidth=2, markersize=8)
    ax1.semilogy(ns, q_est_errs, "s-", label="Quantum estimation RMSE",
                 color="#2ca02c", linewidth=2, markersize=7)
    ax1.semilogy(ns, c_est_errs, "o-", label="Classical (bins) estimation RMSE",
                 color="#d62728", linewidth=2, markersize=7)
    ax1.set_xlabel("Qubits (n)", fontsize=11)
    ax1.set_ylabel("Error ($)", fontsize=11)
    ax1.set_title("Error Decomposition: Discretisation vs Estimation", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", ls=":", alpha=0.3)
    ax1.set_xticks(ns)
    ax1.set_xticklabels([f"{n}\n({2**n} bins)" for n in ns])

    ax2.semilogy(ns, q_total, "s-", label="Quantum total RMSE",
                 color="#2ca02c", linewidth=2, markersize=7)
    ax2.semilogy(ns, c_total, "o-", label="Classical (bins) total RMSE",
                 color="#d62728", linewidth=2, markersize=7)
    ax2.set_xlabel("Qubits (n)", fontsize=11)
    ax2.set_ylabel(r"Total RMSE = $\sqrt{\mathrm{disc}^2 + \mathrm{est}^2}$  ($)",
                   fontsize=11)
    ax2.set_title(f"Total RMSE vs Analytic Truth (budget={exp6['budget']})",
                  fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", ls=":", alpha=0.3)
    ax2.set_xticks(ns)
    ax2.set_xticklabels([f"{n}\n({2**n} bins)" for n in ns])

    fig.suptitle(
        f"Experiment 6: Qubit Scaling Sweep "
        f"({exp6['percentile']*100:.0f}th pctl, "
        f"analytic E[excess]=${exp6['analytic_truth']:,.2f})",
        fontsize=12, y=1.02)
    fig.tight_layout()

    path = RESULTS_DIR / "plot_exp6_scaling.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    get_logger().info("Wrote %s", path)


def _plot_exp6_gates(exp6: Dict) -> None:
    """Circuit cost: two-qubit gates and depth vs n_qubits."""
    sweep = exp6["sweep"]
    ns = [e["n_qubits"] for e in sweep]
    sp_2q = [e["state_prep_stats"]["two_qubit_gates"] for e in sweep]
    sp_depth = [e["state_prep_stats"]["depth"] for e in sweep]
    oracle_2q = [e["oracle_A_stats"]["two_qubit_gates"] for e in sweep]
    oracle_depth = [e["oracle_A_stats"]["depth"] for e in sweep]
    full_gates = [e["full_circuit_stats"]["total_gates"] for e in sweep]
    full_depth = [e["full_circuit_stats"]["depth"] for e in sweep]
    ks = [e["k_used"] for e in sweep]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.semilogy(ns, sp_2q, "^-", label="State prep (2Q gates)",
                 color="#1f77b4", linewidth=2, markersize=7)
    ax1.semilogy(ns, oracle_2q, "s-", label="Oracle A (2Q gates)",
                 color="#ff7f0e", linewidth=2, markersize=7)
    ax1.set_xlabel("Qubits (n)", fontsize=11)
    ax1.set_ylabel("Two-Qubit Gate Count", fontsize=11)
    ax1.set_title("State-Preparation and Oracle Cost", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", ls=":", alpha=0.3)
    ax1.set_xticks(ns)
    ax1.set_xticklabels([f"{n}\n({2**n} bins)" for n in ns])

    ax2.semilogy(ns, full_depth, "D-", label="Full circuit depth",
                 color="#9467bd", linewidth=2, markersize=7)
    ax2.semilogy(ns, full_gates, "o-", label="Full circuit gate count",
                 color="#8c564b", linewidth=2, markersize=7)
    for i, k in enumerate(ks):
        ax2.annotate(f"k={k}", (ns[i], full_gates[i]),
                     textcoords="offset points", xytext=(5, 5),
                     fontsize=8, color="#8c564b")
    ax2.set_xlabel("Qubits (n)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Full Algorithm Circuit Cost (with Grover iterations)",
                  fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", ls=":", alpha=0.3)
    ax2.set_xticks(ns)
    ax2.set_xticklabels([f"{n}\n({2**n} bins)" for n in ns])

    fig.suptitle(
        f"Experiment 6: Circuit Resource Scaling (budget={exp6['budget']})",
        fontsize=12, y=1.02)
    fig.tight_layout()

    path = RESULTS_DIR / "plot_exp6_gates.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    get_logger().info("Wrote %s", path)


# ------------------------------------------------------------------
# Plots — Experiment 7 (empirical PMF)
# ------------------------------------------------------------------

def _plot_exp7_empirical(exp7: Dict) -> None:
    """Bar chart: RMSE at largest budget, all percentiles, with exact-on-bins line."""
    pdata = exp7["percentile_results"]
    pcts = sorted(pdata.keys(), key=float)
    max_budget_key = str(max(exp7["budgets"]))

    methods = ["naive_mc_resample", "classical_bins", "quantum"]
    labels_map = {
        "naive_mc_resample": "Naive MC (resample)",
        "classical_bins": "Classical (same bins)",
        "quantum": "Quantum AE",
    }
    colors_map = {
        "naive_mc_resample": "#d62728",
        "classical_bins": "#8c564b",
        "quantum": "#2ca02c",
    }

    n_methods = len(methods)
    x = np.arange(len(pcts))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for j, method in enumerate(methods):
        rmses = [pdata[p]["budgets"][max_budget_key][method]["rmse_vs_exact_bins"]
                 for p in pcts]
        bars = ax.bar(x + j * w - w, rmses, w,
                      label=labels_map[method],
                      color=colors_map[method], alpha=0.85,
                      edgecolor="black", linewidth=0.5)
        if method == "quantum":
            for i, (bar, rmse) in enumerate(zip(bars, rmses)):
                cb = pdata[pcts[i]]["budgets"][max_budget_key]["classical_bins"]["rmse_vs_exact_bins"]
                if rmse > 0:
                    speedup = cb / rmse
                    ax.annotate(f"{speedup:.1f}x",
                                (bar.get_x() + bar.get_width() / 2, rmse),
                                textcoords="offset points", xytext=(0, 5),
                                fontsize=8, color="#2ca02c",
                                fontweight="bold", ha="center")

    xlabels = []
    for p in pcts:
        pe = pdata[p]
        xlabels.append(
            f"{pe['percentile']*100:.0f}th %\n"
            f"M=${pe['threshold']:,.0f}\n"
            f"exact=${pe['exact_on_bins']:,.0f}"
        )
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("RMSE vs Exact-on-Bins ($)", fontsize=11)
    ax.set_title(
        f"Experiment 7: Empirical NOAA PMF — No Parametric Fit\n"
        f"Budget={max_budget_key}, {exp7['n_records']:,} records, "
        f"quantile-binned into {2**exp7['n_qubits']} bins",
        fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", ls=":", alpha=0.3)
    fig.tight_layout()

    path = RESULTS_DIR / "plot_exp7_empirical.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    get_logger().info("Wrote %s", path)
