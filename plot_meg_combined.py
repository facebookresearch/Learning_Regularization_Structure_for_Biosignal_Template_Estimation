# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Combined MEG figure for the supplementary -- both broadband and filtered regimes.

For each regime, a single double-column figure is produced with:
  Left:  estimated template at K=5 (mean +/- 1 std bands across 20 subsamples)
  Right: relative L2 distance to the full evoked vs K

Run with --filtered for the 1-30 Hz bandpass case, otherwise the 1 Hz
highpass-only (broadband) case is generated.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib as mpl
import mne
import numpy as np
from numpy.typing import NDArray

mpl.use("Agg")
from core import get_events_templates
from data_loaders import EEG_DIR

# IEEE paper style (must be re-applied after MNE/MOABB data loading, which
# silently mutates rcParams).
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
    "axes.linewidth": 1.0,
}

# Data + analysis constants (match example_meg_grad_auditory_right.py)
AUDITORY_RIGHT_ID = 2
LABEL = str(AUDITORY_RIGHT_ID)
SFREQ = 600.614990234375
HIGHPASS = 1.0
HALF_WIN = int(0.15 * SFREQ)
TMIN, TMAX = -(HALF_WIN + 2) / SFREQ, (HALF_WIN + 2) / SFREQ
RESPONSE_WINDOW = (0.05, 0.30)
REJECT = dict(grad=4000e-13)
K = 5
N_SEEDS = 20
KS = [2, 3, 5, 8, 12, 20]

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
    "Reference": "Full evoked (all epochs)",
    "Average": "Average",
    "EmpBayes": "Emp. Bayes",
    "SUREkernel": "SURE kernel",
}


def load_good_grad_epochs(lowpass: float | None) -> mne.Epochs:
    fif_path = EEG_DIR / "MNE-sample-data" / "MEG" / "sample" / "sample_audvis_raw.fif"
    if not fif_path.exists():
        sample_dir = Path(mne.datasets.sample.data_path(path=str(EEG_DIR)))
        fif_path = sample_dir / "MEG" / "sample" / "sample_audvis_raw.fif"
    if not fif_path.exists():
        raise FileNotFoundError(
            f"MNE sample MEG recording not found at {fif_path} after "
            f"attempting download into {EEG_DIR}"
        )
    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
    events = mne.find_events(raw, stim_channel="STI 014", verbose=False)
    raw.pick(picks="grad", verbose=False)
    raw.filter(HIGHPASS, lowpass, verbose=False)
    epochs = mne.Epochs(
        raw,
        events,
        event_id={"auditory_right": AUDITORY_RIGHT_ID},
        tmin=TMIN,
        tmax=TMAX,
        baseline=(None, 0),
        reject=REJECT,
        preload=True,
        verbose=False,
    )
    print(f"  Epochs: {len(epochs)} kept")
    return epochs


def select_best_channel(epochs: mne.Epochs) -> tuple[int, str, float]:
    evoked = epochs.average()
    times = evoked.times
    post = (times >= RESPONSE_WINDOW[0]) & (times <= RESPONSE_WINDOW[1])
    base = times < 0.0
    peak = np.max(np.abs(evoked.data[:, post]), axis=1)
    noise = np.std(evoked.data[:, base], axis=1)
    snr = peak / np.maximum(noise, 1e-30)
    best = int(np.argmax(snr))
    return best, evoked.ch_names[best], float(snr[best])


def channel_epoch_matrix(epochs: mne.Epochs, ch_idx: int) -> NDArray:
    x = epochs.get_data(copy=True)[:, ch_idx, :]
    t0 = int(epochs.time_as_index(0.0)[0])
    return x[:, t0 - HALF_WIN : t0 + HALF_WIN]


def make_continuous(
    epoch_mat: NDArray, rows: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    seg = epoch_mat[rows]
    block = 2 * HALF_WIN
    data = np.concatenate([seg.reshape(-1, 1), np.zeros((HALF_WIN, 1))])
    ix = np.arange(len(rows)) * block + HALF_WIN
    lb = np.array([LABEL] * len(rows))
    return data, ix, lb


def subsample_rows(n_epochs: int, k: int, seed: int) -> NDArray:
    rs = np.random.RandomState(seed)
    return np.sort(rs.choice(n_epochs, size=min(k, n_epochs), replace=False))


def collect_templates(epoch_mat: NDArray) -> dict[str, NDArray]:
    n_epochs = epoch_mat.shape[0]
    result = {m: np.full((N_SEEDS, 2 * HALF_WIN), np.nan) for m in METHODS}
    for seed in range(N_SEEDS):
        rows = subsample_rows(n_epochs, K, seed)
        data, ix, lb = make_continuous(epoch_mat, rows)
        for name, kw in METHODS.items():
            try:
                t, _ = get_events_templates(data, ix, lb, half_win=HALF_WIN, **kw)
                if LABEL in t:
                    result[name][seed] = t[LABEL][:, 0]
            except Exception as e:
                print(f"    seed={seed} {name}: {e}")
        if (seed + 1) % 5 == 0:
            print(f"    Templates seed {seed + 1}/{N_SEEDS}", flush=True)
    return result


def distance_vs_k(epoch_mat: NDArray) -> dict[str, list[float]]:
    gt = epoch_mat.mean(axis=0)
    n_epochs = epoch_mat.shape[0]
    result: dict[str, list[float]] = {m: [] for m in METHODS}
    for k in KS:
        for name, kw in METHODS.items():
            dists: list[float] = []
            for seed in range(N_SEEDS):
                rows = subsample_rows(n_epochs, k, seed)
                data, ix, lb = make_continuous(epoch_mat, rows)
                try:
                    t, _ = get_events_templates(data, ix, lb, half_win=HALF_WIN, **kw)
                    if LABEL in t:
                        est = t[LABEL][:, 0]
                        dists.append(
                            float(np.linalg.norm(est - gt) / np.linalg.norm(gt))
                        )
                except Exception as e:
                    print(f"    K={k} {name} seed={seed}: {e}")
            result[name].append(float(np.mean(dists)) if dists else float("nan"))
        print(f"    K={k:2d} done", flush=True)
    return result


def main(lowpass: float | None) -> None:
    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    import matplotlib.pyplot as plt

    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    from matplotlib.lines import Line2D

    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(exist_ok=True)
    suffix = "filtered" if lowpass is not None else "nofilter"

    print(f"Loading MEG epochs ({suffix}, lowpass={lowpass})...")
    epochs = load_good_grad_epochs(lowpass)
    best_idx, best_name, best_snr = select_best_channel(epochs)
    print(f"  Best gradiometer: {best_name} (SNR {best_snr:.1f})")
    epoch_mat = channel_epoch_matrix(epochs, best_idx)
    n_epochs = epoch_mat.shape[0]

    print("Computing templates...")
    templates = collect_templates(epoch_mat)

    print("Computing distance vs K...")
    kcurves = distance_vs_k(epoch_mat)

    # Re-apply paper style AFTER MNE has mutated rcParams
    plt.rcParams.update(PAPER_STYLE)

    time_ms = np.linspace(
        -HALF_WIN / SFREQ * 1000,
        HALF_WIN / SFREQ * 1000,
        2 * HALF_WIN,
        endpoint=False,
    )
    mask = (time_ms >= -100) & (time_ms <= 150)
    time_crop = time_ms[mask]
    scale = 1e13  # T/m -> fT/cm
    ref_wf = epoch_mat.mean(axis=0)[mask] * scale

    fig, (ax_t, ax_k) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.2),
        gridspec_kw={
            "wspace": 0.32,
            "left": 0.08,
            "right": 0.99,
            "top": 0.92,
            "bottom": 0.30,
        },
    )

    # Left: template
    ax_t.plot(
        time_crop, ref_wf, color=COLORS["Reference"], linewidth=1.6, alpha=0.7, zorder=1
    )
    for m in METHOD_ORDER:
        all_wf = templates[m][:, mask] * scale
        mean_wf = np.nanmean(all_wf, axis=0)
        std_wf = np.nanstd(all_wf, axis=0)
        ax_t.fill_between(
            time_crop,
            mean_wf - std_wf,
            mean_wf + std_wf,
            color=COLORS[m],
            alpha=0.18,
            zorder=2,
        )
        ax_t.plot(
            time_crop,
            mean_wf,
            color=COLORS[m],
            linewidth=1.1,
            linestyle=LINESTYLES[m],
            zorder=4 if m == "SUREkernel" else 3,
        )
    ax_t.axvline(0.0, color="k", linewidth=0.5, alpha=0.4)
    ax_t.set_title(f"(a) Template -- {best_name} (K=5 of {n_epochs} epochs)")
    ax_t.set_xlabel("Time (ms)")
    ax_t.set_ylabel("Amplitude (fT/cm)")

    # Right: distance vs K
    for m in METHOD_ORDER:
        ax_k.plot(
            KS,
            kcurves[m],
            color=COLORS[m],
            linestyle=LINESTYLES[m],
            marker="o",
            markersize=4,
            linewidth=1.2,
        )
    ax_k.set_title("(b) Distance vs K")
    ax_k.set_xlabel("Number of epochs K")
    ax_k.set_ylabel(r"Relative $L_2$ distance")
    ax_k.set_xticks(KS)

    # Shared legend below
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
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
    )

    for ext in ("pdf", "png"):
        out = out_dir / f"fig_meg_combined_{suffix}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filtered",
        action="store_true",
        help="use 1-30 Hz bandpass (default: 1 Hz highpass only)",
    )
    args = parser.parse_args()
    main(lowpass=30.0 if args.filtered else None)
