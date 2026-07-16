# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Composite double-column template figure for the paper.

Top row:    P300 target on Fz, Cz, Pz (BNCI2014-009 subject 1, K=5)
Bottom row: ECG Normal beat (MIT-BIH record 106, QRS crop, K=5)

Each panel shows mean +/- 1 std bands across 20 random subsamples for
Average, Empirical Bayes, and SURE kernel, against a reference template
estimated from all available events.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

import matplotlib as mpl
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.axes import Axes

mpl.use("Agg")
from core import get_events_templates
from data_loaders_ecg import download_ecg_mitbih, load_ecg_mitbih
from data_loaders_p300 import load_p300_bnci2014_009

# Note: MOABB / MNE may mutate rcParams during data loading, so we re-apply
# this style immediately before figure creation rather than relying on a
# single module-level update.
PAPER_STYLE = {
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

METHODS_P300: dict[str, dict[str, Any]] = {
    "Average": {"method": "average"},
    "EmpBayes": {"method": "ls", "reg_method": "empbayes"},
    "SUREkernel": {
        "method": "ls",
        "reg_temporal": 500,
        "opt_reg_temporal": True,
        "vec_mode": True,
        "symm": True,
        "vlen": 3,
        "noise_est": "ar1_colored",
    },
}

METHODS_ECG: dict[str, dict[str, Any]] = {
    "Average": {"method": "average"},
    "EmpBayes": {"method": "ls", "reg_method": "empbayes"},
    "SUREkernel": {
        "method": "ls",
        "reg_temporal": 500,
        "opt_reg_temporal": True,
        "vec_mode": True,
        "symm": True,
        "vlen": 3,
        "noise_est": "pairwise",
    },
}

COLORS = {
    "Reference": "#aaaaaa",
    "Average": "#888888",
    "EmpBayes": "#ff7f0e",
    "SUREkernel": "#000000",
}
LINESTYLES = {
    "Reference": "-",
    "Average": ":",
    "EmpBayes": "--",
    "SUREkernel": "-",
}
METHOD_ORDER = ["Average", "EmpBayes", "SUREkernel"]
LEGEND_LABELS = {
    "Reference": "Reference (all events)",
    "Average": "Average",
    "EmpBayes": "Emp. Bayes",
    "SUREkernel": "SURE kernel",
}

K = 5
N_SEEDS = 20


def subsample_events(
    ix: NDArray, lb: NDArray, k: int, seed: int
) -> tuple[NDArray, NDArray]:
    rs = np.random.RandomState(seed)
    unique_labels = np.unique(lb)
    idx_sel = []
    for lbl in unique_labels:
        label_idx = np.where(lb == lbl)[0]
        idx_sel.extend(rs.choice(label_idx, size=min(k, len(label_idx)), replace=False))
    idx = np.array(sorted(idx_sel))
    return ix[idx], lb[idx]


def collect_templates(
    data: NDArray,
    ix: NDArray,
    lb: NDArray,
    half_win: int,
    methods: dict[str, dict[str, Any]],
    target_label: str,
    n_seeds: int = N_SEEDS,
) -> dict[str, NDArray]:
    T = 2 * half_win
    C = data.shape[1]
    result = {m: np.full((n_seeds, T, C), np.nan) for m in methods}

    for seed in range(n_seeds):
        ix_sub, lb_sub = subsample_events(ix, lb, K, seed)
        for method_name, method_kwargs in methods.items():
            try:
                t, _ = get_events_templates(
                    data, ix_sub, lb_sub, half_win=half_win, **method_kwargs
                )
                if target_label in t:
                    result[method_name][seed] = t[target_label]
            except Exception as e:
                print(f"    Warning: seed={seed} {method_name}: {e}")
        if (seed + 1) % 5 == 0:
            print(f"    Seeds {seed + 1}/{n_seeds} done", flush=True)
    return result


def plot_panel(
    ax: Axes,
    time_crop: NDArray,
    ref_wf: NDArray,
    templates_by_method: dict[str, NDArray],
    crop_mask: NDArray,
    ch_idx: int,
    scale: float = 1.0,
    title: str = "",
    xlabel: str = "Time (ms)",
    ylabel: str = "",
) -> None:
    ax.plot(
        time_crop,
        ref_wf * scale,
        color=COLORS["Reference"],
        linewidth=1.6,
        linestyle=LINESTYLES["Reference"],
        alpha=0.7,
        zorder=1,
    )
    for method in METHOD_ORDER:
        all_wf = templates_by_method[method][:, crop_mask, ch_idx] * scale
        mean_wf = np.nanmean(all_wf, axis=0)
        std_wf = np.nanstd(all_wf, axis=0)
        ax.fill_between(
            time_crop,
            mean_wf - std_wf,
            mean_wf + std_wf,
            color=COLORS[method],
            alpha=0.18,
            zorder=2,
        )
        ax.plot(
            time_crop,
            mean_wf,
            color=COLORS[method],
            linewidth=1.1,
            linestyle=LINESTYLES[method],
            zorder=4 if method == "SUREkernel" else 3,
        )
    if title:
        ax.set_title(title, pad=2)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=2)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=2)
    ax.tick_params(axis="both", pad=2)


def main() -> None:
    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    import matplotlib.pyplot as plt

    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    from matplotlib import gridspec

    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    from matplotlib.lines import Line2D

    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(exist_ok=True)

    # ---------------------- P300 ----------------------
    print("P300: Loading and collecting templates across 20 seeds")
    p300_data, p300_ix, p300_lb, sfreq_p300, p300_info = load_p300_bnci2014_009(
        subjects=[1], bandpass=(1.0, 20.0), resample=128.0
    )
    p300_lb_str = np.array([str(lbl) for lbl in p300_lb])
    half_win_p300 = int(0.4 * sfreq_p300)
    ch_names = p300_info["ch_names"]
    valid = (p300_ix - half_win_p300 >= 0) & (p300_ix + half_win_p300 <= len(p300_data))
    p300_ix = p300_ix[valid]
    p300_lb_str = p300_lb_str[valid]

    print("  Computing P300 reference (Emp. Bayes, all events)...")
    p300_ref, _ = get_events_templates(
        p300_data,
        p300_ix,
        p300_lb_str,
        half_win=half_win_p300,
        method="ls",
        reg_method="empbayes",
    )

    t0 = time.time()
    p300_templates = collect_templates(
        p300_data, p300_ix, p300_lb_str, half_win_p300, METHODS_P300, "1", N_SEEDS
    )
    print(f"  Done ({time.time() - t0:.0f}s)")

    # ---------------------- ECG ----------------------
    print("ECG: Loading record 106 and collecting templates")
    sfreq_ecg = 360.0
    half_win_ecg = int(0.35 * sfreq_ecg)
    download_ecg_mitbih(records=["106"])
    ecg_data, ecg_ix, ecg_lb, _ = load_ecg_mitbih(
        records=["106"], lead=0, bandpass=(0.5, 40.0), beat_types=["N", "V"]
    )
    label_names = {0: "Normal", 1: "PVC"}
    ecg_lb_str = np.array([label_names.get(lbl, f"type_{lbl}") for lbl in ecg_lb])
    valid = (ecg_ix - half_win_ecg >= 0) & (ecg_ix + half_win_ecg <= len(ecg_data))
    ecg_ix = ecg_ix[valid]
    ecg_lb_str = ecg_lb_str[valid]
    n_normal = int(np.sum(ecg_lb_str == "Normal"))
    print(f"  Record 106: {n_normal} Normal beats")

    print("  Computing ECG reference (Emp. Bayes, all beats)...")
    ecg_ref, _ = get_events_templates(
        ecg_data,
        ecg_ix,
        ecg_lb_str,
        half_win=half_win_ecg,
        method="ls",
        reg_method="empbayes",
    )

    t0 = time.time()
    ecg_templates = collect_templates(
        ecg_data, ecg_ix, ecg_lb_str, half_win_ecg, METHODS_ECG, "Normal", N_SEEDS
    )
    print(f"  Done ({time.time() - t0:.0f}s)")

    # ---------------------- Figure ----------------------
    print("Building composite figure...")
    plt.rcParams.update(PAPER_STYLE)  # MOABB / MNE may have mutated rcParams
    time_p300 = np.linspace(
        -half_win_p300 / sfreq_p300 * 1000,
        half_win_p300 / sfreq_p300 * 1000,
        2 * half_win_p300,
        endpoint=False,
    )
    p300_mask = (time_p300 >= -100) & (time_p300 <= 500)
    time_p300_crop = time_p300[p300_mask]

    ch_pick_names = ["Fz", "Cz", "Pz"]
    ch_pick_idx = [ch_names.index(c) for c in ch_pick_names]
    p300_ref_by_ch = {
        c: p300_ref["1"][p300_mask, ch_pick_idx[i]] for i, c in enumerate(ch_pick_names)
    }

    time_ecg = np.linspace(
        -half_win_ecg / sfreq_ecg * 1000,
        half_win_ecg / sfreq_ecg * 1000,
        2 * half_win_ecg,
        endpoint=False,
    )
    ecg_mask = (time_ecg >= -100) & (time_ecg <= 100)
    time_ecg_crop = time_ecg[ecg_mask]
    ecg_ref_wf = ecg_ref["Normal"][ecg_mask, 0]

    # Double-column: IEEE textwidth ~ 7.16 in. Match other paper figures (7.2 wide).
    fig = plt.figure(figsize=(7.2, 3.8))
    gs = gridspec.GridSpec(
        2,
        6,
        figure=fig,
        height_ratios=[1.0, 1.0],
        hspace=0.95,
        wspace=0.5,
        top=0.93,
        bottom=0.22,
        left=0.07,
        right=0.98,
    )
    ax_fz = fig.add_subplot(gs[0, 0:2])
    ax_cz = fig.add_subplot(gs[0, 2:4])
    ax_pz = fig.add_subplot(gs[0, 4:6])
    ax_ecg = fig.add_subplot(gs[1, 1:5])

    # P300 row: shared y to keep amplitude comparable
    p300_axes = [ax_fz, ax_cz, ax_pz]
    for i, (ax, ch) in enumerate(zip(p300_axes, ch_pick_names)):
        plot_panel(
            ax,
            time_p300_crop,
            p300_ref_by_ch[ch],
            p300_templates,
            p300_mask,
            ch_pick_idx[i],
            scale=1e6,
            title=f"P300 -- {ch}",
            xlabel="Time (ms)",
            ylabel=r"Amplitude ($\mu$V)" if i == 0 else "",
        )
        if i > 0:
            ax.tick_params(labelleft=False)
    # share y-axis across P300 panels
    y_lo = min(a.get_ylim()[0] for a in p300_axes)
    y_hi = max(a.get_ylim()[1] for a in p300_axes)
    for a in p300_axes:
        a.set_ylim(y_lo, y_hi)

    # ECG row
    plot_panel(
        ax_ecg,
        time_ecg_crop,
        ecg_ref_wf,
        ecg_templates,
        ecg_mask,
        0,
        scale=1.0,
        title=f"ECG Normal beat -- record 106 ({n_normal} beats total)",
        xlabel="Time relative to R-peak (ms)",
        ylabel="Amplitude (mV)",
    )
    ax_ecg.axvline(0.0, color="k", linewidth=0.5, alpha=0.4)

    # Shared legend
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["Reference"],
            linewidth=1.6,
            alpha=0.7,
            label=LEGEND_LABELS["Reference"],
        )
    ]
    for m in METHOD_ORDER:
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS[m],
                linewidth=1.2,
                linestyle=LINESTYLES[m],
                label=LEGEND_LABELS[m],
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.005),
        frameon=False,
    )

    for ext in ("pdf", "png"):
        out = out_dir / f"fig_template_composite.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
