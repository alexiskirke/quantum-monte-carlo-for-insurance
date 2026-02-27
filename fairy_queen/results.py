"""Automated result generation: metrics.json, summary.csv, and plots.

All outputs land in ./results/ and are structured for easy LLM ingestion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fairy_queen.logging_config import get_logger

RESULTS_DIR = Path("results")


def save_all(exp1: Dict, exp2: Dict, exp3: Dict) -> None:
    log = get_logger()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _save_metrics_json(exp1, exp2, exp3)
    _save_summary_csv(exp1, exp2, exp3)
    _plot_exp1_convergence(exp1)
    _plot_exp2_noise(exp2)
    _plot_exp3_tail(exp3)

    log.warning("All results saved to %s/", RESULTS_DIR)


# ------------------------------------------------------------------
# metrics.json
# ------------------------------------------------------------------

def _save_metrics_json(exp1: Dict, exp2: Dict, exp3: Dict) -> None:
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
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    get_logger().info("Wrote %s", path)


# ------------------------------------------------------------------
# summary.csv
# ------------------------------------------------------------------

def _save_summary_csv(exp1: Dict, exp2: Dict, exp3: Dict) -> None:
    rows = []

    for i, N in enumerate(exp1["sample_counts"]):
        rows.append({
            "experiment": "1_convergence",
            "parameter": f"N={N}",
            "classical_rmse": exp1["classical_rmse"][i],
            "quantum_rmse": exp1["quantum_k0_rmse"][i],
            "oracle_queries": N,
            "exact_value": exp1["exact_expected_loss"],
        })

    for level, data in exp2.items():
        if level.startswith("_"):
            continue
        rows.append({
            "experiment": "2_noise",
            "parameter": level,
            "classical_rmse": None,
            "quantum_rmse": data["rmse"],
            "oracle_queries": None,
            "exact_value": data["exact_loss"],
        })

    for key, data in exp3["percentile_results"].items():
        rows.append({
            "experiment": "3_tail",
            "parameter": f"p={data['percentile']}",
            "classical_rmse": data["classical_rmse"],
            "quantum_rmse": data["quantum_grover_rmse"],
            "oracle_queries": data.get("total_queries_grover"),
            "exact_value": data["exact_excess_loss"],
        })

    df = pd.DataFrame(rows)
    path = RESULTS_DIR / "summary.csv"
    df.to_csv(path, index=False)
    get_logger().info("Wrote %s", path)


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

def _plot_exp1_convergence(exp1: Dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    N = np.array(exp1["sample_counts"], dtype=float)

    ax.loglog(N, exp1["classical_rmse"], "o-", label="Classical MC",
              color="#d62728", linewidth=2, markersize=7)
    ax.loglog(N, exp1["quantum_k0_rmse"], "s-", label="Quantum AE (k=0, shot-based)",
              color="#1f77b4", linewidth=2, markersize=7)

    # Reference slopes
    x_ref = np.linspace(N.min(), N.max(), 100)
    c_scale = exp1["classical_rmse"][0] * np.sqrt(N[0])
    ax.loglog(x_ref, c_scale / np.sqrt(x_ref), "--", alpha=0.35,
              color="#d62728", label=r"$O(1/\sqrt{N})$ reference")
    q_scale = exp1["classical_rmse"][0] * N[0]
    ax.loglog(x_ref, q_scale / x_ref, "--", alpha=0.35,
              color="#2ca02c", label=r"$O(1/N)$ (ideal QAE)")

    # Mark the exact (statevector) result at N=1
    ax.plot(1, 0, marker="*", color="#ff7f0e", markersize=15, zorder=5,
            label=f"Quantum exact = ${exp1['quantum_exact_loss']:.0f}  (RMSE=0)")

    ax.set_xlabel("Number of Oracle Queries / Samples")
    ax.set_ylabel("RMSE ($)")
    ax.set_title("Experiment 1: Convergence Scaling — Classical MC vs Quantum AE")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.3)
    fig.tight_layout()

    path = RESULTS_DIR / "plot_exp1_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    get_logger().info("Wrote %s", path)


def _plot_exp2_noise(exp2: Dict) -> None:
    levels = [l for l in exp2 if not l.startswith("_")]
    rmses = [exp2[l]["rmse"] for l in levels]

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
    ax.set_title("Experiment 2: QAE Accuracy Degradation Under NISQ Noise")

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
    c_var = [pdata[p]["classical_variance"] for p in pcts]
    q_k0_var = [pdata[p]["quantum_k0_variance"] for p in pcts]
    q_gr_var = [pdata[p]["quantum_grover_variance"] for p in pcts]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(pcts))
    w = 0.25

    # Panel A: RMSE comparison
    axes[0].bar(x - w, c_rmse, w, label="Classical MC", color="#d62728", alpha=0.85)
    axes[0].bar(x, q_k0_rmse, w, label="Quantum k=0", color="#1f77b4", alpha=0.85)
    axes[0].bar(x + w, q_gr_rmse, w, label="Quantum Grover", color="#2ca02c", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("RMSE ($)")
    axes[0].set_title("Excess-Loss RMSE by Tail Percentile")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", ls=":", alpha=0.3)

    # Panel B: Variance comparison
    axes[1].bar(x - w, c_var, w, label="Classical MC", color="#d62728", alpha=0.85)
    axes[1].bar(x, q_k0_var, w, label="Quantum k=0", color="#1f77b4", alpha=0.85)
    axes[1].bar(x + w, q_gr_var, w, label="Quantum Grover", color="#2ca02c", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Variance ($²)")
    axes[1].set_title("Estimator Variance by Tail Percentile")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", ls=":", alpha=0.3)

    fig.suptitle("Experiment 3: Classical vs Quantum at Catastrophe Tail Thresholds",
                 fontsize=13, y=1.02)
    fig.tight_layout()

    path = RESULTS_DIR / "plot_exp3_tail.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    get_logger().info("Wrote %s", path)
