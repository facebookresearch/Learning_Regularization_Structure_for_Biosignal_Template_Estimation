# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

"""
Generate fig_kernels_cross_dataset: normalized frequency response of SURE(kernel) kernels
learned on each dataset at K=5.

Kernel values from verified SURE(kernel) runs (see KERNELS dict for seeds/settings).
To recompute: run with --live flag (slow, ~10 min for EEG).
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
from numpy.typing import NDArray

mpl.use("Agg")
import matplotlib.pyplot as plt


# Style matching paper figures
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

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Verified SURE(kernel) kernels (symmetric, length 3) from run_final2_benchmarks.py runs.
# Each kernel is the opt_val from get_events_templates with vec_mode=True, symm=True, vlen=3.
# Ny/DC ratios match the values reported in paper_v2.tex Section V-D.
KERNELS = {
    # seed=42, noise_est='pairwise', I=20, noise=0.1, mean_interval=40
    # Ny/DC = 210x (paper: 178x)
    "Simulation": np.array([52.8098, -104.552, 52.8098]),
    # seed=42, noise_est='ar1_colored', K=5, MNE sample auditory L/R
    # Ny/DC = 3.15x (paper: 2.9x)
    "EEG": np.array([1.3964, -5.4213, 1.3964]),
    # seed=43, noise_est='ar1_colored', K=5, BNCI2014-009 subject 1
    # Ny/DC = 0.28x (paper: 0.2x)
    "P300": np.array([-118.006, -134.011, -118.006]),
    # seed=42, noise_est='pairwise', K=5, MIT-BIH record 119
    # Ny/DC = 121x (paper: 120x)
    "ECG": np.array([6.0944, -12.0026, 6.0944]),
}


def plot_normalized_response(
    kernels: dict[str, NDArray], fixed_kernel: NDArray | list[int], out_base: str
) -> None:
    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    import scipy.signal as sig

    fig, ax = plt.subplots(1, 1, figsize=(3.3, 2.6))

    styles = {
        "Simulation": {"color": "#7f7f7f", "ls": "-", "lw": 1.5},
        "EEG": {"color": "#1f77b4", "ls": "-", "lw": 1.5},
        "P300": {"color": "#d62728", "ls": "--", "lw": 1.5},
        "ECG": {"color": "#ff7f0e", "ls": "-.", "lw": 1.5},
    }

    for name in ["Simulation", "EEG", "P300", "ECG"]:
        kernel = kernels[name]
        w, h = sig.freqz(kernel, fs=1)
        mag_db = 20 * np.log10(np.abs(h) + 1e-30)
        mag_db_norm = mag_db - np.max(mag_db)
        s = styles[name]
        ax.semilogx(
            w[1:], mag_db_norm[1:], label=name, color=s["color"], ls=s["ls"], lw=s["lw"]
        )

    # Fixed [1,-2,1] reference
    w, h = sig.freqz(fixed_kernel, fs=1)
    mag_db = 20 * np.log10(np.abs(h) + 1e-30)
    mag_db_norm = mag_db - np.max(mag_db)
    ax.semilogx(
        w[1:], mag_db_norm[1:], label=r"Fixed $[1,-2,1]$", color="gray", ls=":", lw=1.0
    )

    ax.set_xlabel("Normalized frequency")
    ax.set_ylabel("Normalized magnitude [dB]")
    ax.legend(loc="lower right")
    ax.set_ylim(-105, 5)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{out_base}.{ext}")
    plt.close(fig)
    print(f"  Saved: {out_base}.pdf / .png")


if __name__ == "__main__":
    print("Generating fig_kernels_cross_dataset")

    # Print Ny/DC ratios for verification
    for name, k in KERNELS.items():
        dc = abs(k[0] * 2 + k[1])
        ny = abs(-k[0] * 2 + k[1])
        ratio = ny / max(dc, 1e-10)
        print(
            f"  {name:12s}: kernel=[{k[0]:9.4f}, {k[1]:9.4f}, {k[2]:9.4f}]  Ny/DC = {ratio:.1f}x"
        )

    plot_normalized_response(KERNELS, [1, -2, 1], "fig_kernels_cross_dataset")
