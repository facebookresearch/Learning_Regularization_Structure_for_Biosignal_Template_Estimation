# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

"""
Generate all PDF figures for the paper.
Reads from results/*_final.csv and produces paper/figures/*.pdf
"""

from pathlib import Path

import matplotlib as mpl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.use("Agg")

from core import generate_data, get_events_templates, ricker
from data_loaders import load_eeg_erp

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Consistent style
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
    }
)

METHOD_ORDER = [
    "Average",
    "OLS",
    "Reg. LS",
    "GCV",
    "L-curve",
    "EmpBayes",
    "SURE(lambda)",
    "SURE(kernel)",
]
METHOD_LABELS = {
    "Average": "Averaging",
    "OLS": "OLS",
    "Reg. LS": r"Reg. LS ($\lambda$=500)",
    "GCV": "GCV",
    "L-curve": "L-curve",
    "EmpBayes": "Emp. Bayes",
    "SURE(lambda)": r"SURE ($\lambda$)",
    "SURE(kernel)": r"SURE (kernel)",
}

COLORS = {
    "Average": "#999999",
    "OLS": "#666666",
    "Reg. LS": "#2ca02c",
    "GCV": "#1f77b4",
    "L-curve": "#9467bd",
    "EmpBayes": "#ff7f0e",
    "SURE(lambda)": "#d62728",
    "SURE(kernel)": "#000000",
}

MARKERS = {
    "Average": "v",
    "OLS": "^",
    "Reg. LS": "s",
    "GCV": "D",
    "L-curve": "p",
    "EmpBayes": "h",
    "SURE(lambda)": "o",
    "SURE(kernel)": "*",
}

LINESTYLES = {
    "Average": ":",
    "OLS": ":",
    "Reg. LS": "--",
    "GCV": "-.",
    "L-curve": "-.",
    "EmpBayes": "--",
    "SURE(lambda)": "-",
    "SURE(kernel)": "-",
}


def fig_eeg_setting1() -> None:
    """Fig 6: EEG template distance vs K — absolute (log) + relative improvement."""
    df = pd.read_csv(RESULTS_DIR / "eeg_setting1_final.csv")

    pivot = df.pivot_table(
        index="method", columns="K", values="distance", aggfunc="mean"
    )
    avg_dist = pivot.loc["Average"]
    show_methods = [
        "Reg. LS",
        "GCV",
        "L-curve",
        "EmpBayes",
        "SURE(lambda)",
        "SURE(kernel)",
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # Left: absolute distance (log scale)
    for method in METHOD_ORDER:
        if method not in pivot.index:
            continue
        vals = pivot.loc[method]
        ax1.plot(
            vals.index,
            vals.values,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=5,
            linestyle=LINESTYLES[method],
            linewidth=1.2,
        )

    ax1.set_yscale("log")
    ax1.set_xlabel("Number of Events (K)")
    ax1.set_ylabel("Template Distance (Frobenius)")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(ncol=2, loc="upper right", framealpha=0.9, fontsize=7)
    ax1.set_title("(a) Absolute distance")

    # Right: % improvement over averaging
    for method in show_methods:
        if method not in pivot.index:
            continue
        improvement = (avg_dist - pivot.loc[method]) / avg_dist * 100
        ax2.plot(
            improvement.index,
            improvement.values,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=5,
            linestyle=LINESTYLES[method],
            linewidth=1.2,
        )

    ax2.set_xlabel("Number of Events (K)")
    ax2.set_ylabel("Improvement over Averaging [%]")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=7)
    ax2.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax2.set_title("(b) Relative improvement")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "eeg_setting1.pdf")
    plt.close(fig)
    print("  eeg_setting1.pdf")


def fig_eeg_setting2() -> None:
    """Fig 7: EEG reconstruction error — horizontal bar, sorted, zoomed."""
    df = pd.read_csv(RESULTS_DIR / "eeg_setting2_final.csv")

    stats = df.groupby("method")["recon_error"].agg(["mean", "std"])

    # Sort by mean (best at top)
    order = stats.sort_values("mean").index.tolist()

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.0))

    y = np.arange(len(order))
    vals = (stats.loc[order, "mean"] - 1) * 1e6
    errs = stats.loc[order, "std"] * 1e6

    ax.barh(
        y,
        vals,
        xerr=errs,
        color=[COLORS[m] for m in order],
        capsize=3,
        edgecolor="black",
        linewidth=0.5,
        height=0.6,
    )

    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABELS[m] for m in order])
    ax.set_xlabel(r"Normalized Reconstruction Error $-1$ ($\times 10^{-6}$)")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "eeg_setting2.pdf")
    plt.close(fig)
    print("  eeg_setting2.pdf")


def fig_kernels() -> None:
    """Fig 2: Learned regularization kernel frequency responses."""
    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    import scipy.signal as sig

    T = 100
    half_win = T // 2
    C = 8
    real_templates = {
        "Template A": np.outer(ricker(T, 15), np.roll(np.hanning(C), 3)),
        "Template B": np.outer(ricker(T, 10), np.hanning(C)),
    }

    rs = np.random.RandomState([42])
    data, ix, labels = generate_data(
        real_templates, I=20, noise=5e-2, mean_interval=40, rs=rs
    )

    configs = [
        ("Sym(3)", {"vlen": 3, "symm": True}),
        ("NonSym(3)", {"vlen": 3, "symm": False}),
        ("Sym(5)", {"vlen": 5, "symm": True}),
        ("NonSym(5)", {"vlen": 5, "symm": False}),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.2))

    kernels = {}
    for name, cfg in configs:
        _, inner = get_events_templates(
            data,
            ix,
            labels,
            half_win=half_win,
            method="ls",
            reg_temporal=500,
            opt_reg_temporal=True,
            vec_mode=True,
            noise_est="hpf",
            **cfg,
        )
        assert inner is not None
        kernel = inner["opt_val"]
        kernels[name] = kernel
        w, h = sig.freqz(kernel, fs=1)
        ls = "-" if "Sym" in name else "--"
        ax.semilogx(w[1:], 20 * np.log10(np.abs(h[1:])), label=name, ls=ls, lw=1.5)

    # Add fixed references
    for lam, kernel, name, color in [
        (500, [1, -2, 1], r"$\lambda$=500, [1,-2,1]", "green"),
        (500, [1, -4, 6, -4, 1], r"$\lambda$=500, 4th-order", "magenta"),
    ]:
        kernel_scaled = lam * np.array(kernel)
        w, h = sig.freqz(kernel_scaled, fs=1)
        ax.semilogx(
            w[1:], 20 * np.log10(np.abs(h[1:])), label=name, ls=":", lw=1, color=color
        )

    ax.set_xlabel("Normalized Frequency")
    ax.set_ylabel("Magnitude [dB]")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3, which="both")

    fig.savefig(FIG_DIR / "kernel_frequency_response.pdf")
    plt.close(fig)
    print("  kernel_frequency_response.pdf")


def fig_synthetic_templates() -> None:
    """Fig 1b: The synthetic templates used in simulations."""

    T = 100
    C = 8
    real_templates = {
        "Template A": np.outer(ricker(T, 15), np.roll(np.hanning(C), 3)),
        "Template B": np.outer(ricker(T, 10), np.hanning(C)),
    }

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5), sharey=True)
    ts = np.arange(-T // 2, T // 2)

    for ax, (name, template) in zip(axes, real_templates.items()):
        colors = mpl.colormaps["twilight"](np.linspace(0.1, 0.9, C))
        for ch in range(C):
            ax.plot(ts, template[:, ch], color=colors[ch], lw=1, alpha=0.8)
        ax.set_title(name)
        ax.set_xlabel("Time [samples]")
        ax.axvline(0, ls="--", color="k", lw=0.5, alpha=0.3)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Amplitude")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "synthetic_templates.pdf")
    plt.close(fig)
    print("  synthetic_templates.pdf")


def fig_eeg_template_comparison() -> None:
    """Fig 5: EEG templates — top row: templates from K=5 events; bottom row: error vs reference."""

    eeg_bandpass = (1.0, 40.0)
    eeg_fs = 600.615

    eeg_raw, eeg_ix, eeg_lb = load_eeg_erp(
        event_ids={"auditory_left": 1, "auditory_right": 2}, bandpass=None
    )
    eeg_filt, _, _ = load_eeg_erp(
        event_ids={"auditory_left": 1, "auditory_right": 2}, bandpass=eeg_bandpass
    )
    lb_str = np.array(["aud_L" if lbl == 1 else "aud_R" for lbl in eeg_lb])
    hw = int(0.2 * eeg_fs)
    ts = np.arange(-hw, hw) / eeg_fs * 1000  # ms

    # Reference (all events)
    ref, _ = get_events_templates(
        eeg_filt,
        eeg_ix,
        lb_str,
        half_win=hw,
        method="ls",
        reg_temporal=500,
        opt_reg_temporal=True,
        noise_est="outofband",
        data_raw=eeg_raw,
        bandpass=eeg_bandpass,
        fs=eeg_fs,
        noise_band=(48, 200),
    )

    # Subsample K=5 (10 total events) — the regime where methods differ most
    rs = np.random.RandomState([42])
    idx = rs.choice(len(eeg_ix), size=10, replace=False)

    methods_to_show = [
        ("Averaging", {"method": "average"}),
        ("Reg. LS (fixed)", {"method": "ls", "reg_temporal": 500}),
        ("Emp. Bayes", {"method": "ls", "reg_method": "empbayes"}),
        (
            "SURE (kernel)",
            {
                "method": "ls",
                "reg_temporal": 500,
                "opt_reg_temporal": True,
                "vec_mode": True,
                "symm": True,
                "vlen": 3,
                "noise_est": "outofband",
                "data_raw": eeg_raw,
                "bandpass": eeg_bandpass,
                "fs": eeg_fs,
                "noise_band": (48, 200),
            },
        ),
    ]

    # Pick 5 channels with highest GFP
    gfp = np.mean([np.sum(ref[k] ** 2, axis=0) for k in ref], axis=0)
    top_ch = np.argsort(gfp)[-5:]

    label = "aud_L"
    n_methods = len(methods_to_show)

    fig, axes = plt.subplots(
        2,
        n_methods,
        figsize=(3.2 * n_methods, 5),
        gridspec_kw={"height_ratios": [1.2, 1]},
    )

    # Estimate templates for each method
    estimated = {}
    for mname, mkwargs in methods_to_show:
        t, _ = get_events_templates(
            eeg_filt,
            eeg_ix[idx],
            lb_str[idx],
            half_win=hw,
            **mkwargs,
        )
        estimated[mname] = t

    # Top row: templates (with reference overlaid as dashed black)
    for j, (mname, _) in enumerate(methods_to_show):
        ax = axes[0, j]
        for ch in top_ch:
            ax.plot(ts, ref[label][:, ch] * 1e6, color="k", lw=0.5, ls="--", alpha=0.3)
            ax.plot(ts, estimated[mname][label][:, ch] * 1e6, lw=0.9, alpha=0.8)
        ax.set_title(mname, fontsize=9)
        ax.axvline(0, ls=":", color="k", lw=0.4, alpha=0.3)
        ax.grid(True, alpha=0.15)
        if j == 0:
            ax.set_ylabel(r"Template [$\mu$V]")
        ax.tick_params(labelbottom=False)

    # Bottom row: estimation error (template - reference)
    for j, (mname, _) in enumerate(methods_to_show):
        ax = axes[1, j]
        err = estimated[mname][label] - ref[label]
        for ch in top_ch:
            ax.plot(ts, err[:, ch] * 1e6, lw=0.8, alpha=0.7)
        ax.axhline(0, color="k", lw=0.4, alpha=0.3)
        ax.axvline(0, ls=":", color="k", lw=0.4, alpha=0.3)
        ax.grid(True, alpha=0.15)
        ax.set_xlabel("Time [ms]")
        if j == 0:
            ax.set_ylabel(r"Error [$\mu$V]")
        # RMSE annotation
        rmse = np.sqrt(np.mean(err[:, top_ch] ** 2)) * 1e6
        ax.text(
            0.95,
            0.92,
            f"RMSE={rmse:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.8},
        )

    # Share y-axes within each row
    for row in range(2):
        ylims = [axes[row, j].get_ylim() for j in range(n_methods)]
        ymin = min(y[0] for y in ylims)
        ymax = max(y[1] for y in ylims)
        for j in range(n_methods):
            axes[row, j].set_ylim(ymin, ymax)

    fig.suptitle(
        "Auditory-Left ERP templates (K=5 events, dashed=reference)",
        fontsize=10,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eeg_templates.pdf")
    plt.close(fig)
    print("  eeg_templates.pdf")


def fig_eeg_relative_improvement() -> None:
    """Fig 7 replacement: Relative improvement over averaging vs K on real EEG."""
    df = pd.read_csv(RESULTS_DIR / "eeg_setting1_final.csv")

    # Compute mean distance per method per K
    pivot = df.pivot_table(
        index="method", columns="K", values="distance", aggfunc="mean"
    )

    # Relative improvement over averaging: (avg - method) / avg * 100
    avg_dist = pivot.loc["Average"]

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.2))

    show_methods = [
        "Reg. LS",
        "GCV",
        "L-curve",
        "EmpBayes",
        "SURE(lambda)",
        "SURE(kernel)",
    ]
    for method in show_methods:
        improvement = (avg_dist - pivot.loc[method]) / avg_dist * 100
        ax.plot(
            improvement.index,
            improvement.values,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=6,
            linestyle=LINESTYLES[method],
            linewidth=1.3,
        )

    ax.set_xlabel("Number of Events (K)")
    ax.set_ylabel("Improvement over Averaging [%]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)

    fig.savefig(FIG_DIR / "eeg_improvement.pdf")
    plt.close(fig)
    print("  eeg_improvement.pdf")


def fig_eeg_kernels() -> None:
    """Fig 8: Learned regularization kernel frequency response on real EEG data."""
    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    import scipy.signal as sig

    eeg_bandpass = (1.0, 40.0)
    eeg_fs = 600.615

    eeg_raw, eeg_ix, eeg_lb = load_eeg_erp(
        event_ids={"auditory_left": 1, "auditory_right": 2}, bandpass=None
    )
    eeg_filt, _, _ = load_eeg_erp(
        event_ids={"auditory_left": 1, "auditory_right": 2}, bandpass=eeg_bandpass
    )
    lb_str = np.array(["aud_L" if lbl == 1 else "aud_R" for lbl in eeg_lb])
    hw = int(0.2 * eeg_fs)

    configs = [
        ("Sym(3)", {"vlen": 3, "symm": True}),
        ("NonSym(3)", {"vlen": 3, "symm": False}),
        ("Sym(5)", {"vlen": 5, "symm": True}),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.2))

    for name, cfg in configs:
        _, inner = get_events_templates(
            eeg_filt,
            eeg_ix,
            lb_str,
            half_win=hw,
            method="ls",
            reg_temporal=500,
            opt_reg_temporal=True,
            vec_mode=True,
            noise_est="outofband",
            data_raw=eeg_raw,
            bandpass=eeg_bandpass,
            fs=eeg_fs,
            noise_band=(48, 200),
            **cfg,
        )
        assert inner is not None
        kernel = inner["opt_val"]
        w, h = sig.freqz(kernel, fs=1)
        ls = "-" if "Sym" in name else "--"
        ax.semilogx(
            w[1:], 20 * np.log10(np.abs(h[1:])), label=f"EEG {name}", ls=ls, lw=1.5
        )

    # Fixed reference
    kernel_fixed = 500 * np.array([1, -2, 1])
    w, h = sig.freqz(kernel_fixed, fs=1)
    ax.semilogx(
        w[1:],
        20 * np.log10(np.abs(h[1:])),
        label=r"Fixed $\lambda$=500, [1,-2,1]",
        ls=":",
        lw=1,
        color="green",
    )

    ax.set_xlabel("Normalized Frequency")
    ax.set_ylabel("Magnitude [dB]")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title("Learned kernels on EEG data")

    fig.savefig(FIG_DIR / "eeg_kernels.pdf")
    plt.close(fig)
    print("  eeg_kernels.pdf")


def fig_eeg_gfp_error() -> None:
    """Fig 5 replacement: GFP of estimation error for each method at K=5."""

    eeg_bandpass = (1.0, 40.0)
    eeg_fs = 600.615

    eeg_raw, eeg_ix, eeg_lb = load_eeg_erp(
        event_ids={"auditory_left": 1, "auditory_right": 2}, bandpass=None
    )
    eeg_filt, _, _ = load_eeg_erp(
        event_ids={"auditory_left": 1, "auditory_right": 2}, bandpass=eeg_bandpass
    )
    lb_str = np.array(["aud_L" if lbl == 1 else "aud_R" for lbl in eeg_lb])
    hw = int(0.2 * eeg_fs)
    ts = np.arange(-hw, hw) / eeg_fs * 1000

    # Reference
    ref, _ = get_events_templates(
        eeg_filt,
        eeg_ix,
        lb_str,
        half_win=hw,
        method="ls",
        reg_temporal=500,
        opt_reg_temporal=True,
        noise_est="outofband",
        data_raw=eeg_raw,
        bandpass=eeg_bandpass,
        fs=eeg_fs,
        noise_band=(48, 200),
    )

    methods_to_show = {
        "Average": {"method": "average"},
        "Reg. LS": {"method": "ls", "reg_temporal": 500},
        "GCV": {"method": "ls", "reg_method": "gcv"},
        "EmpBayes": {"method": "ls", "reg_method": "empbayes"},
        "SURE(lambda)": {
            "method": "ls",
            "reg_temporal": 500,
            "opt_reg_temporal": True,
            "noise_est": "outofband",
            "data_raw": eeg_raw,
            "bandpass": eeg_bandpass,
            "fs": eeg_fs,
            "noise_band": (48, 200),
        },
        "SURE(kernel)": {
            "method": "ls",
            "reg_temporal": 500,
            "opt_reg_temporal": True,
            "vec_mode": True,
            "symm": True,
            "vlen": 3,
            "noise_est": "outofband",
            "data_raw": eeg_raw,
            "bandpass": eeg_bandpass,
            "fs": eeg_fs,
            "noise_band": (48, 200),
        },
    }

    # Average GFP error over multiple random subsamples
    n_rep = 15
    rs = np.random.RandomState([42])

    gfp_errors = {m: np.zeros(2 * hw) for m in methods_to_show}
    label = "aud_L"

    for _rep in range(n_rep):
        idx = rs.choice(len(eeg_ix), size=10, replace=False)
        for mname, mkwargs in methods_to_show.items():
            t, _ = get_events_templates(
                eeg_filt,
                eeg_ix[idx],
                lb_str[idx],
                half_win=hw,
                **mkwargs,
            )
            err = t[label] - ref[label]
            gfp = np.sqrt(np.mean(err**2, axis=1))
            gfp_errors[mname] += gfp

    # Average over reps
    for m in gfp_errors:
        gfp_errors[m] /= n_rep

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.2))

    for mname in [
        "Average",
        "Reg. LS",
        "GCV",
        "EmpBayes",
        "SURE(lambda)",
        "SURE(kernel)",
    ]:
        ax.plot(
            ts,
            gfp_errors[mname] * 1e6,
            label=METHOD_LABELS[mname],
            color=COLORS[mname],
            linestyle=LINESTYLES[mname],
            linewidth=1.3,
        )

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel(r"GFP of Error [$\mu$V]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax.axvline(0, ls=":", color="k", lw=0.4, alpha=0.3)
    ax.set_title("K=5 events, averaged over 15 subsamples")

    fig.savefig(FIG_DIR / "eeg_gfp_error.pdf")
    plt.close(fig)
    print("  eeg_gfp_error.pdf")


if __name__ == "__main__":
    print("Generating figures...", flush=True)

    # Some panels read sweep/setting CSVs that are not produced by
    # run_benchmarks.py. Render each figure independently so a missing input
    # only skips that panel instead of aborting the whole script.
    figure_fns = [
        ("synthetic templates", fig_synthetic_templates),
        ("kernel frequency response", fig_kernels),
        ("EEG setting 1", fig_eeg_setting1),
        ("EEG GFP error", fig_eeg_gfp_error),
        ("EEG relative improvement", fig_eeg_relative_improvement),
        ("EEG kernels", fig_eeg_kernels),
    ]
    for name, fn in figure_fns:
        try:
            fn()
        except FileNotFoundError as e:
            print(f"  SKIP {name}: missing input ({e.filename})")
        except Exception as e:
            print(f"  SKIP {name}: {type(e).__name__}: {e}")

    print(f"\nAll available figures saved to {FIG_DIR}/")
