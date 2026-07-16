# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

"""Regenerate the phi-sensitivity figure (fig:phi_sensitivity) for the paper.

Reads ``supplementary/phi_sensitivity.csv`` and writes the SURE(kernel)
phi-mismatch sensitivity plot to ``figures/phi_sensitivity.pdf`` and
``figures/phi_sensitivity.png``.
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR: Path = Path(__file__).resolve().parent
CSV_PATH: Path = SCRIPT_DIR / "phi_sensitivity.csv"
FIG_DIR: Path = SCRIPT_DIR.parent / "figures"

# IEEE-style plot settings (paper style).
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


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    # --- Extract data ---
    colored = df[df["method"] == "SURE(kernel) colored"].copy()
    empbayes = df[df["method"] == "EmpBayes"]
    white = df[df["method"] == "SURE(kernel) white"]

    # True phi value (from phi_used at factor=1.0).
    true_phi = colored[colored["phi_factor"] == 1.0]["phi_used"].iloc[0]

    # Once phi_used saturates at the 0.99 stability bound, larger perturbation
    # factors are redundant (identical clipped phi -> identical distance). Keep
    # only up to the first clipped factor so a single clipped point is shown,
    # matching the published figure.
    all_factors = np.sort(colored["phi_factor"].unique())
    phi_used_all = np.clip(true_phi * all_factors, 0, 0.99)
    clipped_all = phi_used_all >= 0.99
    if bool(np.any(clipped_all)):
        first_clip = int(np.argmax(clipped_all))
        keep_factors = all_factors[: first_clip + 1]
        colored = colored[colored["phi_factor"].isin(keep_factors)].copy()

    # Scale distances to micro-units for readability.
    scale = 1e6
    unit_label = r"distance ($\times 10^{-6}$)"

    # Colored SURE stats by factor.
    colored_stats = colored.groupby("phi_factor")["distance"].agg(
        ["mean", "std", "count"]
    )
    colored_stats["se"] = colored_stats["std"] / np.sqrt(colored_stats["count"])

    # Baseline stats.
    empbayes_mean = empbayes["distance"].mean() * scale
    empbayes_se = empbayes["distance"].std() / np.sqrt(len(empbayes)) * scale
    white_mean = white["distance"].mean() * scale
    white_se = white["distance"].std() / np.sqrt(len(white)) * scale

    # Factors whose phi_used hits the 0.99 stability bound are marked as clipped.
    phi_factors = colored_stats.index.to_numpy()
    phi_used = np.array([np.clip(true_phi * f, 0, 0.99) for f in phi_factors])
    clipped = phi_used >= 0.99

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    means = colored_stats["mean"].to_numpy() * scale
    ses = colored_stats["se"].to_numpy() * scale

    # Colored SURE line.
    ax.errorbar(
        phi_factors,
        means,
        yerr=1.96 * ses,
        fmt="o-",
        color="black",
        linewidth=1.5,
        markersize=5,
        capsize=3,
        label="SURE(kernel) colored",
        zorder=3,
    )

    # Mark clipped points.
    if np.any(clipped):
        ax.scatter(
            phi_factors[clipped],
            means[clipped],
            marker="D",
            color="black",
            s=30,
            zorder=4,
            edgecolors="red",
            linewidth=1.2,
        )

    # EmpBayes horizontal line.
    ax.axhline(
        empbayes_mean,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.2,
        label=f"Emp. Bayes ({empbayes_mean:.1f})",
    )
    ax.fill_between(
        [phi_factors[0] - 0.05, phi_factors[-1] + 0.05],
        empbayes_mean - 1.96 * empbayes_se,
        empbayes_mean + 1.96 * empbayes_se,
        color="#ff7f0e",
        alpha=0.12,
    )

    # White SURE horizontal line.
    ax.axhline(
        white_mean,
        color="#1f77b4",
        linestyle=":",
        linewidth=1.2,
        label=f"SURE(kernel) white ({white_mean:.1f})",
    )
    ax.fill_between(
        [phi_factors[0] - 0.05, phi_factors[-1] + 0.05],
        white_mean - 1.96 * white_se,
        white_mean + 1.96 * white_se,
        color="#1f77b4",
        alpha=0.12,
    )

    # True phi vertical line.
    ax.axvline(1.0, color="gray", linestyle="-.", linewidth=0.8, alpha=0.5)

    ax.set_xlabel(r"$\varphi$ perturbation factor")
    ax.set_ylabel(f"Template distance ({unit_label})")
    ax.set_title(r"SURE(kernel) sensitivity to $\varphi$ mismatch (P300, K=5)")
    ax.legend(loc="upper left", framealpha=0.9)

    # Annotate phi_used on the top x-axis.
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(phi_factors)
    ax2.set_xticklabels([f"{p:.2f}" for p in phi_used], fontsize=7)
    ax2.set_xlabel(r"$\varphi_{\mathrm{used}}$", fontsize=8)

    # Note about clipping.
    if np.any(clipped):
        clipped_factors = phi_factors[clipped]
        ax.annotate(
            r"clipped to $\varphi=0.99$",
            xy=(clipped_factors[0], means[np.argmax(clipped)]),
            xytext=(clipped_factors[0] - 0.227, means[np.argmax(clipped)] + 6.0),
            fontsize=7,
            color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8),
        )

    plt.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIG_DIR / "phi_sensitivity.pdf"
    png_path = FIG_DIR / "phi_sensitivity.png"
    plt.savefig(pdf_path)
    plt.savefig(png_path)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
