# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
from typing import Any

"""
Example: ECG beat template estimation on the MIT-BIH Arrhythmia Database.

Estimates Normal and PVC beat morphology from record 119 using only K=5 beats,
comparing:

    Average        - epoch averaging (baseline)
    Emp. Bayes     - evidence-maximization lambda
    SURE kernel    - the proposed jointly-optimized convolution kernel

For each method we draw 20 random subsamples of K=5 beats per class and plot
the mean template +/- 1 std band against the full reference (estimated from
all available beats via Emp. Bayes).

Run directly from this folder:
    python example_ecg_template.py
"""

from pathlib import Path

import matplotlib as mpl

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

mpl.use("Agg")

from core import get_events_templates
from data_loaders_ecg import download_ecg_mitbih, load_ecg_mitbih

# ---------------------------------------------------------------------------
# Style (IEEE, serif, 9 pt) -- matches example_meg_grad_auditory_right.py
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
        "axes.linewidth": 1.2,
    }
)

RECORD = "106"
BEAT_TYPES = ["N", "V"]
LABEL_NAMES = {0: "Normal", 1: "PVC"}
SFREQ = 360.0
HALF_WIN = int(0.35 * SFREQ)  # +/- 350 ms around R-peak
K = 5
N_SEEDS = 20

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
    "Reference": "Reference (all beats)",
    "Average": "Average",
    "EmpBayes": "Emp. Bayes",
    "SUREkernel": "SURE kernel",
}


def subsample_events(
    ix: NDArray,
    lb: NDArray,
    k: int,
    seed: int,
) -> tuple[NDArray, NDArray]:
    """Pick k events per class using a fixed seed."""
    rs = np.random.RandomState(seed)
    unique_labels = np.unique(lb)
    idx_sel = []
    for label in unique_labels:
        label_idx = np.where(lb == label)[0]
        idx_sel.extend(rs.choice(label_idx, size=min(k, len(label_idx)), replace=False))
    idx = np.array(sorted(idx_sel))
    return ix[idx], lb[idx]


def collect_templates(
    data: NDArray,
    ix: NDArray,
    lb: NDArray,
    target_label: str,
    n_seeds: int = N_SEEDS,
) -> dict[str, NDArray]:
    """Run n_seeds subsamples and collect templates for one beat type.

    Returns method -> array of shape (n_seeds, 2 * HALF_WIN, 1).
    """
    result = {m: np.zeros((n_seeds, 2 * HALF_WIN, 1)) for m in METHODS}

    for seed in range(n_seeds):
        ix_sub, lb_sub = subsample_events(ix, lb, K, seed)
        for method_name, method_kwargs in METHODS.items():
            try:
                templates, _ = get_events_templates(
                    data, ix_sub, lb_sub, half_win=HALF_WIN, **method_kwargs
                )
                if target_label in templates:
                    result[method_name][seed] = templates[target_label]
            except Exception as e:
                print(f"    Warning: seed={seed} {method_name}: {e}")
        if (seed + 1) % 5 == 0:
            print(f"    Seeds {seed + 1}/{n_seeds} done", flush=True)

    return result


def main(beat_type: str = "Normal") -> None:
    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    from matplotlib.lines import Line2D

    fig_dir = Path(__file__).resolve().parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print(f"ECG beat template: {beat_type} (record {RECORD}, K={K})")
    print("=" * 60)

    # Download if needed
    download_ecg_mitbih(records=[RECORD])

    # Load data
    data, ix, lb, sfreq = load_ecg_mitbih(
        records=[RECORD],
        lead=0,
        bandpass=(0.5, 40.0),
        beat_types=BEAT_TYPES,
    )

    lb_str = np.array([LABEL_NAMES.get(l, f"type_{l}") for l in lb])

    # Boundary check
    valid = (ix - HALF_WIN >= 0) & (ix + HALF_WIN <= len(data))
    ix = ix[valid]
    lb_str = lb_str[valid]

    for label in np.unique(lb_str):
        print(f"  {label}: {np.sum(lb_str == label)} beats")

    # Reference: Emp. Bayes from all events
    print("  Computing reference (Emp. Bayes from all beats)...")
    ref, _ = get_events_templates(
        data,
        ix,
        lb_str,
        half_win=HALF_WIN,
        method="ls",
        reg_method="empbayes",
    )

    # Collect subsampled templates
    print(f"  Collecting {beat_type} templates across {N_SEEDS} seeds...")
    templates = collect_templates(data, ix, lb_str, beat_type, N_SEEDS)

    # -------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------
    time_ms = np.linspace(
        -HALF_WIN / sfreq * 1000,
        HALF_WIN / sfreq * 1000,
        2 * HALF_WIN,
        endpoint=False,
    )

    # Crop to QRS region for visible uncertainty bands
    crop = (time_ms >= -100) & (time_ms <= 100)
    time_crop = time_ms[crop]
    ref_wf = ref[beat_type][crop, 0]

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.0))

    # Reference
    ax.plot(
        time_crop,
        ref_wf,
        color=COLORS["Reference"],
        linewidth=2.0,
        alpha=0.6,
        zorder=1,
    )

    # Methods: mean +/- std
    for method in METHOD_ORDER:
        all_wf = templates[method][:, crop, 0]  # (n_seeds, T_crop)
        mean_wf = np.mean(all_wf, axis=0)
        std_wf = np.std(all_wf, axis=0)
        ax.fill_between(
            time_crop,
            mean_wf - std_wf,
            mean_wf + std_wf,
            color=COLORS[method],
            alpha=0.15,
            zorder=2,
        )
        ax.plot(
            time_crop,
            mean_wf,
            color=COLORS[method],
            linewidth=1.3,
            linestyle=LINESTYLES[method],
            zorder=4 if method == "SUREkernel" else 3,
        )

    ax.axvline(0.0, color="k", linewidth=0.6, alpha=0.4)

    n_beats = int(np.sum(lb_str == beat_type))
    ax.set_title(f"ECG {beat_type} (record {RECORD}, K={K} of {n_beats} beats)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (mV)")

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["Reference"],
            linewidth=2.0,
            alpha=0.6,
            label=LEGEND_LABELS["Reference"],
        )
    ]
    handles += [
        Line2D(
            [0],
            [0],
            color=COLORS[m],
            linewidth=1.5,
            linestyle=LINESTYLES[m],
            label=LEGEND_LABELS[m],
        )
        for m in METHOD_ORDER
    ]
    ax.legend(handles=handles, loc="best", frameon=False)
    fig.tight_layout()

    bt_slug = beat_type.lower()
    out_png = fig_dir / f"fig_template_ecg_{bt_slug}.png"
    out_pdf = fig_dir / f"fig_template_ecg_{bt_slug}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  Saved: {out_png}")
    print(f"  Saved: {out_pdf}")
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG beat template estimation example")
    parser.add_argument(
        "--beat-type",
        default="Normal",
        choices=["Normal", "PVC"],
        help="which beat type to plot (default: Normal)",
    )
    cli_args = parser.parse_args()
    main(beat_type=cli_args.beat_type)
