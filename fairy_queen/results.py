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
             exp4: Optional[Dict] = None) -> None:
    log = get_logger()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _save_metrics_json(exp1, exp2, exp3, exp4)
    _save_summary_csv(exp1, exp2, exp3, exp4)
    _plot_exp1_convergence(exp1)
    _plot_exp2_noise(exp2)
    _plot_exp3_tail(exp3)
    if exp4 is not None:
        _plot_exp4_convergence(exp4)
        _plot_exp4_tail(exp4)

    log.warning("All results saved to %s/", RESULTS_DIR)


# ------------------------------------------------------------------
# metrics.json
# ------------------------------------------------------------------

def _save_metrics_json(exp1: Dict, exp2: Dict, exp3: Dict,
                       exp4: Optional[Dict] = None) -> None:
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

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    get_logger().info("Wrote %s", path)


# ------------------------------------------------------------------
# summary.csv
# ------------------------------------------------------------------

def _save_summary_csv(exp1: Dict, exp2: Dict, exp3: Dict,
                      exp4: Optional[Dict] = None) -> None:
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
            "total_oracle_queries": None,
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

    df = pd.DataFrame(rows)
    path = RESULTS_DIR / "summary.csv"
    df.to_csv(path, index=False)
    get_logger().info("Wrote %s", path)


# ------------------------------------------------------------------
# Plots — Experiments 1–3 (synthetic data)
# ------------------------------------------------------------------

def _plot_exp1_convergence(exp1: Dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))

    q_queries = [r["total_oracle_queries"] for r in exp1["quantum_results"]]
    q_rmse = [r["rmse"] for r in exp1["quantum_results"]]
    q_k = [r["k"] for r in exp1["quantum_results"]]

    c_queries = [r["total_oracle_queries"] for r in exp1["classical_results"]]
    c_rmse = [r["rmse"] for r in exp1["classical_results"]]

    ax.loglog(c_queries, c_rmse, "o-", label="Classical MC",
              color="#d62728", linewidth=2, markersize=7)
    ax.loglog(q_queries, q_rmse, "s-", label="Quantum AE (Grover-amplified)",
              color="#1f77b4", linewidth=2, markersize=7)

    for x, y, k in zip(q_queries, q_rmse, q_k):
        ax.annotate(f"k={k}", (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, color="#1f77b4")

    x_ref = np.linspace(min(c_queries), max(c_queries), 200)
    c0 = c_rmse[0] * np.sqrt(c_queries[0])
    ax.loglog(x_ref, c0 / np.sqrt(x_ref), "--", alpha=0.3,
              color="#d62728", label=r"$O(1/\sqrt{N})$ reference")
    q0 = q_rmse[-1] * q_queries[-1]
    ax.loglog(x_ref, q0 / x_ref, "--", alpha=0.3,
              color="#2ca02c", label=r"$O(1/N)$ reference (ideal QAE)")

    ax.set_xlabel("Total Oracle Queries", fontsize=11)
    ax.set_ylabel("RMSE ($)", fontsize=11)
    ax.set_title("Experiment 1: Convergence — Classical MC vs Grover-Amplified QAE\n"
                 f"(Synthetic data, 95th-pctl threshold, P(|1⟩)={exp1['readout_prob']:.4f})",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.3)
    fig.tight_layout()

    path = RESULTS_DIR / "plot_exp1_convergence.png"
    fig.savefig(path, dpi=150)
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
    c_rmse = [pdata[p]["classical_rmse"] for p in pcts]
    q_k0_rmse = [pdata[p]["quantum_k0_rmse"] for p in pcts]
    q_gr_rmse = [pdata[p]["quantum_grover_rmse"] for p in pcts]
    k_used = [pdata[p]["k_used"] for p in pcts]
    k_max = [pdata[p]["max_safe_k"] for p in pcts]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(pcts))
    w = 0.25

    axes[0].bar(x - w, c_rmse, w, label="Classical MC",
                color="#d62728", alpha=0.85)
    axes[0].bar(x, q_k0_rmse, w, label="Quantum k=0",
                color="#1f77b4", alpha=0.85)
    axes[0].bar(x + w, q_gr_rmse, w, label="Quantum Grover",
                color="#2ca02c", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        [f"{l}\nk={ku} (max {km})" for l, ku, km in zip(labels, k_used, k_max)]
    )
    axes[0].set_ylabel("RMSE ($)")
    axes[0].set_title("Excess-Loss RMSE by Tail Percentile (Synthetic)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", ls=":", alpha=0.3)

    q_k0_queries = [pdata[p]["total_queries_k0"] for p in pcts]
    q_gr_queries = [pdata[p]["total_queries_grover"] for p in pcts]
    c_samples = [pdata[p]["n_classical_samples"] for p in pcts]

    c_eff = [r / np.sqrt(n) if n > 0 else 0 for r, n in zip(c_rmse, c_samples)]
    q0_eff = [r / np.sqrt(n) if n > 0 else 0 for r, n in zip(q_k0_rmse, q_k0_queries)]
    qg_eff = [r / np.sqrt(n) if n > 0 else 0 for r, n in zip(q_gr_rmse, q_gr_queries)]

    axes[1].bar(x - w, c_eff, w, label="Classical MC", color="#d62728", alpha=0.85)
    axes[1].bar(x, q0_eff, w, label="Quantum k=0", color="#1f77b4", alpha=0.85)
    axes[1].bar(x + w, qg_eff, w, label="Quantum Grover", color="#2ca02c", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel(r"RMSE $\times \sqrt{N}$  (normalised inefficiency)")
    axes[1].set_title("Query Efficiency (lower = better)")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", ls=":", alpha=0.3)

    fig.suptitle("Experiment 3: Classical vs Quantum at Catastrophe Tail Thresholds",
                 fontsize=13, y=1.02)
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
    fig, ax = plt.subplots(figsize=(8, 5.5))

    q_queries = [r["total_oracle_queries"] for r in conv["quantum_results"]]
    q_rmse = [r["rmse"] for r in conv["quantum_results"]]
    q_k = [r["k"] for r in conv["quantum_results"]]

    c_queries = [r["total_oracle_queries"] for r in conv["classical_results"]]
    c_rmse = [r["rmse"] for r in conv["classical_results"]]

    ax.loglog(c_queries, c_rmse, "o-", label="Classical MC",
              color="#d62728", linewidth=2, markersize=7)
    ax.loglog(q_queries, q_rmse, "s-", label="Quantum AE (Grover-amplified)",
              color="#1f77b4", linewidth=2, markersize=7)

    for x_pt, y_pt, k in zip(q_queries, q_rmse, q_k):
        ax.annotate(f"k={k}", (x_pt, y_pt), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, color="#1f77b4")

    if len(c_queries) >= 2:
        x_ref = np.linspace(min(c_queries), max(c_queries), 200)
        c0 = c_rmse[0] * np.sqrt(c_queries[0])
        ax.loglog(x_ref, c0 / np.sqrt(x_ref), "--", alpha=0.3,
                  color="#d62728", label=r"$O(1/\sqrt{N})$ reference")
        q0 = q_rmse[-1] * q_queries[-1]
        ax.loglog(x_ref, q0 / x_ref, "--", alpha=0.3,
                  color="#2ca02c", label=r"$O(1/N)$ reference")

    dp = exp4.get("dist_params", {})
    ax.set_xlabel("Total Oracle Queries", fontsize=11)
    ax.set_ylabel("RMSE ($)", fontsize=11)
    ax.set_title(
        f"Experiment 4A: Convergence on Real NOAA Data\n"
        f"(σ={dp.get('shape',0):.2f}, {exp4.get('n_records',0):,} records, "
        f"P(|1⟩)={conv.get('readout_prob',0):.4f}, k_max={conv.get('k_safe',0)})",
        fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.3)
    fig.tight_layout()

    path = RESULTS_DIR / "plot_exp4_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    get_logger().info("Wrote %s", path)


def _plot_exp4_tail(exp4: Dict) -> None:
    pdata = exp4["tail_sweep"]["percentile_results"]
    pcts = sorted(pdata.keys(), key=float)

    labels = [f"{float(p)*100:.0f}th %" for p in pcts]
    c_rmse = [pdata[p]["classical_rmse"] for p in pcts]
    q_k0_rmse = [pdata[p]["quantum_k0_rmse"] for p in pcts]
    q_gr_rmse = [pdata[p]["quantum_grover_rmse"] for p in pcts]
    k_used = [pdata[p]["k_used"] for p in pcts]
    k_max_vals = [pdata[p]["max_safe_k"] for p in pcts]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    x = np.arange(len(pcts))
    w = 0.25
    ax.bar(x - w, c_rmse, w, label="Classical MC", color="#d62728", alpha=0.85)
    ax.bar(x, q_k0_rmse, w, label="Quantum k=0", color="#1f77b4", alpha=0.85)
    ax.bar(x + w, q_gr_rmse, w, label="Quantum Grover", color="#2ca02c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{l}\nk={ku} (max {km})" for l, ku, km in zip(labels, k_used, k_max_vals)]
    )
    ax.set_ylabel("RMSE ($)")

    dp = exp4.get("dist_params", {})
    ax.set_title(
        f"Experiment 4B: Tail Excess-Loss RMSE — Real NOAA Data\n"
        f"(σ={dp.get('shape',0):.2f}, {exp4.get('n_records',0):,} records)",
        fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", ls=":", alpha=0.3)

    for i, p in enumerate(pcts):
        speedup = c_rmse[i] / q_gr_rmse[i] if q_gr_rmse[i] > 0 else float("inf")
        ax.annotate(f"{speedup:.1f}x",
                    (x[i] + w, q_gr_rmse[i]),
                    textcoords="offset points", xytext=(0, 5),
                    fontsize=8, color="#2ca02c", fontweight="bold",
                    ha="center")

    fig.tight_layout()
    path = RESULTS_DIR / "plot_exp4_tail.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    get_logger().info("Wrote %s", path)
