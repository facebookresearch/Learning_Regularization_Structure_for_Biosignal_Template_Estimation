# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

"""
EEG auditory ERP template figures for the supplementary.

Produces fig_template_eeg_aud_L.{pdf,png} and fig_template_eeg_aud_R.{pdf,png}.
Each shows templates on 3 channels (EEG 001, EEG 007, EEG 005) at K=5 events
per class, with mean and +/-1 std bands across N_SEEDS=20 random subsamples,
comparing Average, Empirical Bayes, and SURE kernel against a reference
estimated from all events. Style matches the other paper template figures.

Run with: OPENBLAS_NUM_THREADS=1 python plot_template_eeg.py
"""

import time
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.lines import Line2D
from mne.datasets import sample
from mne.io import read_raw_fif
from numpy.typing import NDArray

mpl.use("Agg")
import matplotlib.pyplot as plt
from core import get_events_templates
from data_loaders import load_eeg_erp

# IEEE paper style. Re-applied after MNE / MOABB data loading, which silently
# mutates rcParams (font.size 9 -> 12, titlesize 10 -> 18 etc).
PAPER_STYLE: dict[str, Any] = {
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

METHODS: dict[str, dict[str, Any]] = {
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
METHOD_ORDER: list[str] = ["Average", "EmpBayes", "SUREkernel"]
COLORS: dict[str, str] = {
    "Reference": "#aaaaaa",
    "Average": "#888888",
    "EmpBayes": "#ff7f0e",
    "SUREkernel": "#000000",
}
LINESTYLES: dict[str, str] = {
    "Reference": "-",
    "Average": ":",
    "EmpBayes": "--",
    "SUREkernel": "-",
}
LEGEND_LABELS: dict[str, str] = {
    "Reference": "Reference (all events)",
    "Average": "Average",
    "EmpBayes": "Emp. Bayes",
    "SUREkernel": "SURE kernel",
}

CHANNELS: list[str] = ["EEG 001", "EEG 007", "EEG 005"]
K: int = 5
N_SEEDS: int = 20

FIG_DIR: Path = Path(__file__).resolve().parent / "figures"


def subsample_events(
    ix: NDArray, lb: NDArray, k: int, seed: int
) -> tuple[NDArray, NDArray]:
    rs = np.random.RandomState(seed)
    unique_labels = np.unique(lb)
    idx_sel: list[int] = []
    for label in unique_labels:
        label_idx = np.where(lb == label)[0]
        idx_sel.extend(rs.choice(label_idx, size=min(k, len(label_idx)), replace=False))
    order = np.array(sorted(idx_sel))
    return ix[order], lb[order]


def collect_templates(
    data: NDArray,
    ix: NDArray,
    lb: NDArray,
    half_win: int,
    label: str,
    n_seeds: int = N_SEEDS,
) -> dict[str, NDArray]:
    T = 2 * half_win
    C = data.shape[1]
    result: dict[str, NDArray] = {m: np.zeros((n_seeds, T, C)) for m in METHODS}
    for seed in range(n_seeds):
        ix_s, lb_s = subsample_events(ix, lb, K, seed)
        for name, kw in METHODS.items():
            try:
                t, _ = get_events_templates(data, ix_s, lb_s, half_win=half_win, **kw)
            except Exception as e:
                print(f"    seed={seed} method={name}: {e}")
                continue
            if label in t:
                result[name][seed] = t[label]
        if (seed + 1) % 5 == 0:
            print(f"    Templates seed {seed + 1}/{n_seeds} [{label}]", flush=True)
    return result


def get_eeg_ch_names() -> list[str]:
    raw = read_raw_fif(
        sample.data_path() / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif",
        preload=False,
        verbose="ERROR",
    )
    return [c for c in raw.ch_names if c.startswith("EEG")]


def plot_one_label(
    label_short: str,
    templates: dict[str, NDArray],
    ref: dict[str, NDArray],
    time_ms: NDArray,
    ch_idx: list[int],
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(CHANNELS),
        figsize=(7.2, 2.8),
        sharey=False,
        gridspec_kw={
            "wspace": 0.35,
            "left": 0.07,
            "right": 0.99,
            "top": 0.90,
            "bottom": 0.30,
        },
    )

    for col, (ch_name, cidx) in enumerate(zip(CHANNELS, ch_idx)):
        ax = axes[col]
        ax.axvline(0, color="#cccccc", linewidth=0.5, zorder=0)

        ref_wf = ref[label_short][:, cidx] * 1e6
        ax.plot(
            time_ms,
            ref_wf,
            color=COLORS["Reference"],
            linewidth=1.6,
            linestyle=LINESTYLES["Reference"],
            alpha=0.7,
            zorder=1,
        )

        for method in METHOD_ORDER:
            all_wf = templates[method][:, :, cidx] * 1e6  # (n_seeds, T)
            mean_wf = np.mean(all_wf, axis=0)
            std_wf = np.std(all_wf, axis=0)
            ax.fill_between(
                time_ms,
                mean_wf - std_wf,
                mean_wf + std_wf,
                color=COLORS[method],
                alpha=0.18,
                zorder=2,
            )
            ax.plot(
                time_ms,
                mean_wf,
                color=COLORS[method],
                linewidth=1.1,
                linestyle=LINESTYLES[method],
                zorder=4 if method == "SUREkernel" else 3,
            )

        ax.set_title(ch_name, pad=2)
        ax.set_xlabel("Time (ms)", labelpad=2)
        if col == 0:
            ax.set_ylabel(r"Amplitude ($\mu$V)", labelpad=2)

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["Reference"],
            linewidth=1.6,
            linestyle=LINESTYLES["Reference"],
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
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
    )

    suffix = label_short[-1]  # 'L' or 'R'
    for ext in ("pdf", "png"):
        out = out_dir / f"fig_template_eeg_aud_{suffix}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)

    print("Loading EEG (MNE sample, auditory L/R)...")
    eeg_data, eeg_ix, eeg_lb = load_eeg_erp(
        event_ids={"auditory_left": 1, "auditory_right": 2},
        bandpass=(1.0, 40.0),
    )
    label_map = {1: "aud_L", 2: "aud_R"}
    eeg_lb_str = np.array([label_map[int(lbl)] for lbl in eeg_lb])
    sfreq = 600.615
    half_win = int(0.3 * sfreq)
    print(f"  Data: {eeg_data.shape}, {len(eeg_ix)} events, half_win={half_win}")

    ch_names_eeg = get_eeg_ch_names()
    ch_idx = [ch_names_eeg.index(c) for c in CHANNELS]
    print(f"  Channels: {dict(zip(CHANNELS, ch_idx))}")

    print("Computing reference (EmpBayes, all events)...")
    t0 = time.time()
    ref, _ = get_events_templates(
        eeg_data,
        eeg_ix,
        eeg_lb_str,
        half_win=half_win,
        method="ls",
        reg_method="empbayes",
    )
    print(f"  Reference done ({time.time() - t0:.0f}s)")

    print(f"Collecting {N_SEEDS} subsamples per class (K={K})...")
    templates_l = collect_templates(eeg_data, eeg_ix, eeg_lb_str, half_win, "aud_L")
    templates_r = collect_templates(eeg_data, eeg_ix, eeg_lb_str, half_win, "aud_R")

    # Re-apply paper style AFTER MNE has mutated rcParams
    plt.rcParams.update(PAPER_STYLE)

    T = 2 * half_win
    time_ms = (np.arange(T) - half_win) / sfreq * 1000.0

    print("Plotting aud_L...")
    plot_one_label("aud_L", templates_l, ref, time_ms, ch_idx, FIG_DIR)

    print("Plotting aud_R...")
    plot_one_label("aud_R", templates_r, ref, time_ms, ch_idx, FIG_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
