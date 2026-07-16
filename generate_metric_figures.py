# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.figure

"""
Generate ALL definitive publication-quality figures (Figs 3--8) from
*_final2.csv result files.

Methods (7 total):
    Average, OLS, GCV, L-curve, Emp. Bayes, SURE(lambda), SURE(kernel)

All figures saved as PDF + PNG in paper/figures/.
Style: IEEE, serif fonts, 9 pt, 300 DPI, double-column width = 7.2 in.

Usage:
    python paper/generate_final2_figures.py
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

mpl.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Style (IEEE, serif, 9 pt)
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.size": 9,
        "font.family": "serif",
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

# ---------------------------------------------------------------------------
# Method definitions
# ---------------------------------------------------------------------------
METHOD_ORDER = [
    "Average",
    "OLS",
    "GCV",
    "L-curve",
    "EmpBayes",
    "SURE(lambda)",
    "SURE(kernel)",
]

METHOD_LABELS = {
    "Average": "Average",
    "OLS": "OLS",
    "GCV": "GCV",
    "L-curve": "L-curve",
    "EmpBayes": "Emp. Bayes",
    "SURE(lambda)": r"SURE $\lambda$",
    "SURE(kernel)": "SURE kernel",
}

COLORS = {
    "Average": "#888888",
    "OLS": "#444444",
    "GCV": "#1f77b4",
    "L-curve": "#9467bd",
    "EmpBayes": "#ff7f0e",
    "SURE(lambda)": "#d62728",
    "SURE(kernel)": "#000000",
}

MARKERS = {
    "Average": "s",
    "OLS": "v",
    "GCV": "^",
    "L-curve": "<",
    "EmpBayes": ">",
    "SURE(lambda)": "o",
    "SURE(kernel)": "*",
}

LINESTYLES = {
    "Average": ":",
    "OLS": ":",
    "GCV": "-.",
    "L-curve": "-.",
    "EmpBayes": "--",
    "SURE(lambda)": "-",
    "SURE(kernel)": "-",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_method(name: str) -> str:
    return name.strip()


def load_csv(fname: str) -> pd.DataFrame:
    path = RESULTS_DIR / fname
    if not path.exists():
        print(f"  WARNING: {fname} not found, skipping.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "method" in df.columns:
        df["method"] = df["method"].apply(normalize_method)
    return df


def grouped_stats(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    g = df.groupby(["method", x_col])[y_col]
    stats = g.agg(["mean", "std", "count"]).reset_index()
    stats.columns = ["method", "x", "mean", "std", "count"]
    stats["se"] = stats["std"] / np.sqrt(stats["count"])
    stats["ci_lo"] = stats["mean"] - 1.96 * stats["se"]
    stats["ci_hi"] = stats["mean"] + 1.96 * stats["se"]
    return stats


def save_fig(fig: matplotlib.figure.Figure, base_name: str) -> None:
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"{base_name}.{ext}"
        fig.savefig(path)
    plt.close(fig)
    print(f"  -> {base_name}.pdf / .png")


def plot_method(
    ax: matplotlib.axes.Axes,
    x: ArrayLike,
    mean: ArrayLike,
    ci_lo: ArrayLike,
    ci_hi: ArrayLike,
    method: str,
    label: bool = True,
    ms: float = 5,
    lw: float = 1.2,
) -> None:
    lbl = METHOD_LABELS[method] if label else None
    ax.plot(
        x,
        mean,
        label=lbl,
        color=COLORS[method],
        marker=MARKERS[method],
        markersize=ms,
        linestyle=LINESTYLES[method],
        linewidth=lw,
    )
    ax.fill_between(x, ci_lo, ci_hi, color=COLORS[method], alpha=0.12)


def compute_gain_stats(stats: pd.DataFrame) -> pd.DataFrame:
    avg_stats = stats[stats["method"] == "Average"][["x", "mean", "se"]].rename(
        columns={"mean": "avg_mean", "se": "avg_se"}
    )
    gain_rows = []
    for method in METHOD_ORDER:
        if method == "Average":
            continue
        ms = stats[stats["method"] == method].copy()
        if ms.empty:
            continue
        merged = ms.merge(avg_stats, on="x")
        merged["gain_mean"] = merged["mean"] - merged["avg_mean"]
        merged["gain_se"] = np.sqrt(merged["se"] ** 2 + merged["avg_se"] ** 2)
        merged["gain_ci_lo"] = merged["gain_mean"] - 1.96 * merged["gain_se"]
        merged["gain_ci_hi"] = merged["gain_mean"] + 1.96 * merged["gain_se"]
        merged["method"] = method
        gain_rows.append(merged)
    return (
        pd.DataFrame(pd.concat(gain_rows, ignore_index=True))
        if gain_rows
        else pd.DataFrame()
    )


def compute_improvement_stats(stats: pd.DataFrame) -> pd.DataFrame:
    avg_stats = stats[stats["method"] == "Average"][["x", "mean", "se"]].rename(
        columns={"mean": "avg_mean", "se": "avg_se"}
    )
    imp_rows = []
    for method in METHOD_ORDER:
        if method == "Average":
            continue
        ms = stats[stats["method"] == method].copy()
        if ms.empty:
            continue
        merged = ms.merge(avg_stats, on="x")
        merged["imp_mean"] = (
            (merged["avg_mean"] - merged["mean"]) / merged["avg_mean"] * 100
        )
        merged["imp_se"] = (
            np.sqrt(
                (merged["se"] / merged["avg_mean"]) ** 2
                + (merged["mean"] * merged["avg_se"] / merged["avg_mean"] ** 2) ** 2
            )
            * 100
        )
        merged["imp_ci_lo"] = merged["imp_mean"] - 1.96 * merged["imp_se"]
        merged["imp_ci_hi"] = merged["imp_mean"] + 1.96 * merged["imp_se"]
        merged["method"] = method
        imp_rows.append(merged)
    return (
        pd.DataFrame(pd.concat(imp_rows, ignore_index=True))
        if imp_rows
        else pd.DataFrame()
    )


# ===================================================================
# Simulation Noise Sweep: Simulation Noise Sweep
# ===================================================================
def plot_sim_noise_sweep() -> None:
    print("Simulation Noise Sweep: Simulation Noise Sweep")
    df = load_csv("sim_final2_noise.csv")
    if df.empty:
        return

    snr_col = "SNR [dB]" if "SNR [dB]" in df.columns else "SNR"
    noise_col = "Noise"

    # Filter to no-overlap if column exists
    if "overlap" in df.columns:
        df = df[df["overlap"] == "No"].copy()

    stats = grouped_stats(df, noise_col, snr_col)
    gain_df = compute_gain_stats(stats)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # (a) Absolute SNR
    for method in METHOD_ORDER:
        s = stats[stats["method"] == method].sort_values("x")
        if s.empty:
            continue
        plot_method(
            ax1,
            s["x"].values,
            s["mean"].values,
            s["ci_lo"].values,
            s["ci_hi"].values,
            method,
        )
    ax1.set_xscale("log")
    ax1.set_xlabel("Noise Std")
    ax1.set_ylabel("SNR [dB]")
    ax1.legend(ncol=2, loc="upper right", framealpha=0.9, fontsize=7)
    ax1.set_title("(a) Absolute SNR")

    # (b) Gain over Average
    for method in METHOD_ORDER:
        if method == "Average":
            continue
        g = gain_df[gain_df["method"] == method].sort_values("x")
        if g.empty:
            continue
        plot_method(
            ax2,
            g["x"].values,
            g["gain_mean"].values,
            g["gain_ci_lo"].values,
            g["gain_ci_hi"].values,
            method,
        )
    ax2.set_xscale("log")
    ax2.set_xlabel("Noise Std")
    ax2.set_ylabel("SNR Gain over Average [dB]")
    ax2.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=7)
    ax2.set_title("(b) Improvement over Average")

    fig.tight_layout()
    save_fig(fig, "fig3_noise_sweep")


# ===================================================================
# Simulation N-events Sweep: Simulation N-Events Sweep
# ===================================================================
def plot_sim_nevents_sweep() -> None:
    print("Simulation N-events Sweep: Simulation N-Events Sweep")
    df = load_csv("sim_final2_nevents.csv")
    if df.empty:
        return

    snr_col = "SNR [dB]" if "SNR [dB]" in df.columns else "SNR"
    n_col = "N_events" if "N_events" in df.columns else "N"

    stats = grouped_stats(df, n_col, snr_col)
    gain_df = compute_gain_stats(stats)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # (a) Absolute SNR
    for method in METHOD_ORDER:
        s = stats[stats["method"] == method].sort_values("x")
        if s.empty:
            continue
        plot_method(
            ax1,
            s["x"].values,
            s["mean"].values,
            s["ci_lo"].values,
            s["ci_hi"].values,
            method,
        )
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of Events per Type")
    ax1.set_ylabel("SNR [dB]")
    ax1.legend(ncol=2, loc="lower right", framealpha=0.9, fontsize=7)
    ax1.set_title("(a) Absolute SNR")

    # (b) Gain over Average
    for method in METHOD_ORDER:
        if method == "Average":
            continue
        g = gain_df[gain_df["method"] == method].sort_values("x")
        if g.empty:
            continue
        plot_method(
            ax2,
            g["x"].values,
            g["gain_mean"].values,
            g["gain_ci_lo"].values,
            g["gain_ci_hi"].values,
            method,
        )
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of Events per Type")
    ax2.set_ylabel("SNR Gain over Average [dB]")
    ax2.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax2.legend(loc="center left", framealpha=0.9, fontsize=7)
    ax2.set_title("(b) Improvement over Average")

    fig.tight_layout()
    save_fig(fig, "fig4_nevents_sweep")


# ===================================================================
# EEG Template Distance: EEG Template Distance
# ===================================================================
def plot_eeg_template_distance() -> None:
    print("EEG Template Distance: EEG Template Distance")
    df = load_csv("eeg_final2.csv")
    if df.empty:
        return

    stats = grouped_stats(df, "K", "distance")
    imp_df = compute_improvement_stats(stats)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # (a) Absolute distance (log scale)
    for method in METHOD_ORDER:
        s = stats[stats["method"] == method].sort_values("x")
        if s.empty:
            continue
        plot_method(
            ax1,
            s["x"].values,
            s["mean"].values,
            s["ci_lo"].values,
            s["ci_hi"].values,
            method,
        )
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of Events ($K$)")
    ax1.set_ylabel("Template Distance (Frobenius)")
    ax1.legend(ncol=2, loc="upper right", framealpha=0.9, fontsize=7)
    ax1.set_title("(a) Absolute distance")

    # (b) Improvement over Average
    for method in METHOD_ORDER:
        if method == "Average":
            continue
        g = imp_df[imp_df["method"] == method].sort_values("x")
        if g.empty:
            continue
        plot_method(
            ax2,
            g["x"].values,
            g["imp_mean"].values,
            g["imp_ci_lo"].values,
            g["imp_ci_hi"].values,
            method,
        )
    ax2.set_xlabel("Number of Events ($K$)")
    ax2.set_ylabel("Improvement over Average [%]")
    ax2.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax2.legend(loc="best", framealpha=0.9, fontsize=7)
    ax2.set_title("(b) Relative improvement")

    fig.tight_layout()
    save_fig(fig, "fig_eeg")


# ===================================================================
# P300 Template Distance: P300 Template Distance
# ===================================================================
def plot_p300_template_distance() -> None:
    print("P300 Template Distance: P300 Template Distance")
    df = load_csv("p300_final2.csv")
    if df.empty:
        return

    stats = grouped_stats(df, "K", "distance")
    imp_df = compute_improvement_stats(stats)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # (a) Absolute distance (log scale)
    for method in METHOD_ORDER:
        s = stats[stats["method"] == method].sort_values("x")
        if s.empty:
            continue
        plot_method(
            ax1,
            s["x"].values,
            s["mean"].values,
            s["ci_lo"].values,
            s["ci_hi"].values,
            method,
        )
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of Events ($K$)")
    ax1.set_ylabel("Template Distance (Frobenius)")
    ax1.legend(ncol=2, loc="upper right", framealpha=0.9, fontsize=7)
    ax1.set_title("(a) Absolute distance")

    # (b) Improvement over Average
    for method in METHOD_ORDER:
        if method == "Average":
            continue
        g = imp_df[imp_df["method"] == method].sort_values("x")
        if g.empty:
            continue
        plot_method(
            ax2,
            g["x"].values,
            g["imp_mean"].values,
            g["imp_ci_lo"].values,
            g["imp_ci_hi"].values,
            method,
        )
    ax2.set_xlabel("Number of Events ($K$)")
    ax2.set_ylabel("Improvement over Average [%]")
    ax2.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax2.legend(loc="best", framealpha=0.9, fontsize=7)
    ax2.set_title("(b) Relative improvement")

    fig.tight_layout()
    save_fig(fig, "fig6_p300")


# ===================================================================
# ECG Template Distance: ECG Template Distance
# ===================================================================
def plot_ecg_template_distance() -> None:
    print("ECG Template Distance: ECG Template Distance")
    df = load_csv("ecg_final2.csv")
    if df.empty:
        return

    k_col = "K_per_class"
    stats = grouped_stats(df, k_col, "distance")
    imp_df = compute_improvement_stats(stats)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # (a) Absolute distance (log scale)
    for method in METHOD_ORDER:
        s = stats[stats["method"] == method].sort_values("x")
        if s.empty:
            continue
        plot_method(
            ax1,
            s["x"].values,
            s["mean"].values,
            s["ci_lo"].values,
            s["ci_hi"].values,
            method,
        )
    ax1.set_yscale("log")
    ax1.set_xlabel(r"Events per Class ($K$)")
    ax1.set_ylabel("Template Distance (Frobenius)")
    ax1.legend(ncol=2, loc="upper right", framealpha=0.9, fontsize=7)
    ax1.set_title("(a) Absolute distance")

    # (b) Relative change vs Average
    for method in METHOD_ORDER:
        if method == "Average":
            continue
        g = imp_df[imp_df["method"] == method].sort_values("x")
        if g.empty:
            continue
        plot_method(
            ax2,
            g["x"].values,
            g["imp_mean"].values,
            g["imp_ci_lo"].values,
            g["imp_ci_hi"].values,
            method,
        )
    ax2.set_xlabel(r"Events per Class ($K$)")
    ax2.set_ylabel("Improvement over Average [%]")
    ax2.axhline(0, color="k", lw=0.8, alpha=0.5, zorder=5)
    ax2.legend(loc="best", framealpha=0.9, fontsize=7)
    ax2.set_title("(b) Relative change")

    # Shade the "safe zone" near 0
    ylim = ax2.get_ylim()
    ax2.axhspan(-10, 10, color="green", alpha=0.04, zorder=0)
    ax2.set_ylim(ylim)

    fig.tight_layout()
    save_fig(fig, "fig7_ecg")


# ===================================================================
# Cross-Modality Summary: Cross-Modality Summary Bar Chart
# ===================================================================
def _compute_sim_improvement(
    methods_bar: list[str],
) -> tuple[dict[str, tuple[float, float]] | None, int]:
    """Compute SNR gain over Average at lowest N for simulation data."""
    df_sim = load_csv("sim_final2_nevents.csv")
    if df_sim.empty:
        return None, 0

    snr_col = "SNR [dB]" if "SNR [dB]" in df_sim.columns else "SNR"
    n_col = "N_events" if "N_events" in df_sim.columns else "N"
    n_min = df_sim[n_col].min()
    df_k = df_sim[df_sim[n_col] == n_min].copy()
    avg_snr = df_k[df_k["method"] == "Average"][snr_col]
    avg_mean_snr = avg_snr.mean()
    avg_se_snr = avg_snr.std() / np.sqrt(len(avg_snr))

    sim_entry: dict[str, tuple[float, float]] = {}
    for m in methods_bar:
        msub = df_k[df_k["method"] == m][snr_col]
        if msub.empty:
            continue
        gain = msub.mean() - avg_mean_snr
        gain_se = np.sqrt((msub.std() / np.sqrt(len(msub))) ** 2 + avg_se_snr**2)
        sim_entry[m] = (gain, 1.96 * gain_se)

    return sim_entry, int(n_min)


def _compute_real_improvements(
    methods_bar: list[str],
) -> tuple[
    dict[str, dict[str, tuple[float, float]]],
    dict[str, int],
]:
    """Compute percent improvement over Average at lowest K for real datasets."""
    real_datasets: list[tuple[str, pd.DataFrame, str, str]] = []

    df_eeg = load_csv("eeg_final2.csv")
    if not df_eeg.empty:
        real_datasets.append(("EEG", df_eeg, "K", "distance"))

    df_p300 = load_csv("p300_final2.csv")
    if not df_p300.empty:
        real_datasets.append(("P300", df_p300, "K", "distance"))

    df_ecg = load_csv("ecg_final2.csv")
    if not df_ecg.empty:
        real_datasets.append(("ECG", df_ecg, "K_per_class", "distance"))

    real_summary: dict[str, dict[str, tuple[float, float]]] = {}
    real_k_vals: dict[str, int] = {}

    for dname, df_full, k_col, y_col in real_datasets:
        k_min = df_full[k_col].min()
        real_k_vals[dname] = int(k_min)
        df_k = df_full[df_full[k_col] == k_min].copy()

        avg_mean = df_k[df_k["method"] == "Average"][y_col].mean()
        avg_se = df_k[df_k["method"] == "Average"][y_col].std() / np.sqrt(
            len(df_k[df_k["method"] == "Average"])
        )

        real_summary[dname] = {}
        for m in methods_bar:
            msub = df_k[df_k["method"] == m][y_col]
            if msub.empty:
                continue
            m_mean = msub.mean()
            m_se = msub.std() / np.sqrt(len(msub))
            imp = (avg_mean - m_mean) / avg_mean * 100
            imp_se = (
                np.sqrt((m_se / avg_mean) ** 2 + (m_mean * avg_se / avg_mean**2) ** 2)
                * 100
            )
            real_summary[dname][m] = (imp, 1.96 * imp_se)

    return real_summary, real_k_vals


def _collect_bar_values(
    method: str,
    has_sim: bool,
    sim_entry: dict[str, tuple[float, float]] | None,
    real_summary: dict[str, dict[str, tuple[float, float]]],
) -> tuple[list[float], list[float]]:
    """Collect bar chart values and errors for a single method across datasets."""
    vals: list[float] = []
    errs: list[float] = []

    if has_sim and sim_entry is not None:
        if method in sim_entry:
            vals.append(sim_entry[method][0])
            errs.append(sim_entry[method][1])
        else:
            vals.append(0)
            errs.append(0)

    for dname in real_summary:
        if method in real_summary[dname]:
            vals.append(real_summary[dname][method][0])
            errs.append(real_summary[dname][method][1])
        else:
            vals.append(0)
            errs.append(0)

    return vals, errs


def plot_cross_modality_summary() -> None:
    print("Cross-Modality Summary: Cross-Modality Summary")
    methods_bar = ["GCV", "L-curve", "EmpBayes", "SURE(lambda)", "SURE(kernel)"]

    sim_entry, n_min = _compute_sim_improvement(methods_bar)
    real_summary, real_k_vals = _compute_real_improvements(methods_bar)

    # ---- Build the figure ----
    has_sim = sim_entry is not None and len(sim_entry) > 0
    all_names: list[str] = []
    if has_sim:
        all_names.append(f"Simulation\n($N$={n_min})")
    for dname in real_summary:
        k_min = real_k_vals[dname]
        all_names.append(f"{dname}\n($K$={k_min})")

    n_datasets = len(all_names)
    n_methods = len(methods_bar)
    bar_width = 0.8 / n_methods
    x = np.arange(n_datasets)

    fig, ax = plt.subplots(figsize=(7.2, 3.5))

    for i, method in enumerate(methods_bar):
        vals, errs = _collect_bar_values(method, has_sim, sim_entry, real_summary)

        offset = (i - n_methods / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            vals,
            bar_width * 0.9,
            yerr=errs,
            capsize=3,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.4,
        )

        # Value labels on bars
        for j, (v, e) in enumerate(zip(vals, errs)):
            if abs(v) > 1:
                ypos = v + e + 1.5 if v >= 0 else v - e - 1.5
                va = "bottom" if v >= 0 else "top"
                ax.text(
                    x[j] + offset,
                    ypos,
                    f"{v:.0f}",
                    ha="center",
                    va=va,
                    fontsize=5,
                    color=COLORS[method],
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(all_names, fontsize=9)

    if has_sim:
        ax.set_ylabel("Improvement over Average\n[% or dB for Simulation]")
    else:
        ax.set_ylabel("Improvement over Average [%]")

    ax.axhline(0, color="k", lw=0.8, alpha=0.5, zorder=5)
    ax.legend(
        ncol=n_methods,
        loc="upper center",
        framealpha=0.9,
        fontsize=7,
        bbox_to_anchor=(0.5, 1.12),
    )

    ax.axhspan(-5, 5, color="green", alpha=0.04, zorder=0)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "cross_modality_summary")

    # Print summary table
    print("\n  Cross-modality summary:")
    if has_sim and sim_entry is not None:
        print(f"  Simulation (N={n_min}):")
        for m in methods_bar:
            if m in sim_entry:
                print(f"    {m}: {sim_entry[m][0]:.1f} dB gain")
    for dname in real_summary:
        print(f"  {dname} (K={real_k_vals[dname]}):")
        for m in methods_bar:
            if m in real_summary[dname]:
                print(f"    {m}: {real_summary[dname][m][0]:.1f}% improvement")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Generating definitive figures from *_final2.csv files")
    print("=" * 60, flush=True)

    plot_sim_noise_sweep()
    print()
    plot_sim_nevents_sweep()
    print()
    plot_eeg_template_distance()
    print()
    plot_p300_template_distance()
    print()
    plot_ecg_template_distance()
    print()
    plot_cross_modality_summary()

    print(f"\n{'=' * 60}")
    print(f"All figures saved to {FIG_DIR}/")
    print(f"{'=' * 60}")
