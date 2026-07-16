# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

"""Standalone plotting for the ``fig:convergence_kernels`` submission figure.

Reads ``supplementary/convergence_analysis.csv`` (produced by the heavy
``run_convergence_analysis.py`` pipeline) and renders the three-panel
convergence figure to ``figures/convergence_kernels.{pdf,png}``.

Only the plotting is reproduced here; the expensive optimiser analysis is not
re-run.  The three panels are:

1. Converged kernels over 50 seeds (blue = success, red = fail).
2. Local-minima test: converged coefficients over 10 realisations x 5 inits,
   coloured by initialisation and sign-canonicalised.
3. P300 kernel variability over 20 repetitions with mean and 1-std ellipse.
"""

import os

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

HERE: str = os.path.dirname(os.path.abspath(__file__))
CSV_PATH: str = os.path.join(HERE, "convergence_analysis.csv")
FIG_DIR: str = os.path.abspath(os.path.join(HERE, "..", "figures"))

# The published ``convergence_kernels.pdf`` is the original figsize=(15, 5)
# ``run_convergence_analysis.py`` plot reproduced at half scale: its PDF page
# is 536 x 174 pt, exactly half of the original's 1072 x 350 pt tight bounding
# box.  We therefore draw at half figsize and scale every font, marker and line
# dimension by SCALE (marker *areas* by SCALE**2) so the emitted vector
# geometry matches the published PDF.  FIGSIZE is tuned so the 300-dpi raster is
# 2234 x 728 px, matching the published figure exactly.
SCALE: float = 0.5

FIGSIZE: tuple[float, float] = (7.4935, 2.483)


def set_rcparams() -> None:
    """Apply the (half-scale) matplotlib style of the published figure.

    The figure uses matplotlib's default DejaVu Sans family at half the
    default point sizes (title 12 -> 6, labels/ticks 10 -> 5) with all line
    weights halved, matching the 50%-scaled published PDF.
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10.0 * SCALE,
            "axes.titlesize": 12.0 * SCALE,
            "axes.labelsize": 10.0 * SCALE,
            "xtick.labelsize": 10.0 * SCALE,
            "ytick.labelsize": 10.0 * SCALE,
            "legend.fontsize": 8.0 * SCALE,
            "axes.linewidth": 0.8 * SCALE,
            "grid.linewidth": 0.8 * SCALE,
            "grid.alpha": 0.3,
            "xtick.major.width": 0.8 * SCALE,
            "ytick.major.width": 0.8 * SCALE,
            "xtick.major.size": 3.5 * SCALE,
            "ytick.major.size": 3.5 * SCALE,
            "xtick.major.pad": 3.5 * SCALE,
            "ytick.major.pad": 3.5 * SCALE,
            "axes.labelpad": 4.0 * SCALE,
            "axes.titlepad": 6.0 * SCALE,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.grid": True,
        }
    )


def make_figure(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame) -> Figure:
    """Build the three-panel convergence figure from the analysis frames."""
    has_p300 = len(df3) > 0
    n_panels = 2 + (1 if has_p300 else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=FIGSIZE)

    # Panel 1: convergence rate -- scatter of converged kernel coefficients.
    ax = axes[0]
    colors = ["tab:blue" if s else "tab:red" for s in df1["success"]]
    ax.scatter(
        df1["coeff_0"],
        df1["coeff_1"],
        c=colors,
        alpha=0.6,
        s=30 * SCALE**2,
        edgecolors="k",
        linewidths=0.5 * SCALE,
    )
    ax.scatter(
        [1.0],
        [-3.0],
        marker="x",
        c="red",
        s=100 * SCALE**2,
        linewidths=2 * SCALE,
        zorder=5,
        label="Default init",
    )
    ax.set_xlabel("Kernel coeff 0")
    ax.set_ylabel("Kernel coeff 1")
    ax.set_title("Analysis 1: Converged Kernels\n(50 seeds, blue=success, red=fail)")
    ax.legend(fontsize=8 * SCALE)
    ax.grid(True, alpha=0.3)

    # Panel 2: local minima -- converged coefficients coloured by init.
    # SURE cost is invariant under h -> -h, so flip negative-coeff_0 solutions
    # to the positive half for comparison.
    ax = axes[1]
    init_names = df2["init_name"].unique()
    cmap = plt.get_cmap("Set1")
    for i, init_name in enumerate(init_names):
        mask = df2["init_name"] == init_name
        c0 = df2.loc[mask, "coeff_0"].to_numpy().copy()
        c1 = df2.loc[mask, "coeff_1"].to_numpy().copy()
        flip = c0 < 0
        c0[flip] *= -1
        c1[flip] *= -1
        ax.scatter(
            c0,
            c1,
            c=[cmap(i)],
            alpha=0.7,
            s=30 * SCALE**2,
            label=init_name,
            edgecolors="k",
            linewidths=0.5 * SCALE,
        )
    ax.set_xlabel("Kernel coeff 0 (sign-canonicalized)")
    ax.set_ylabel("Kernel coeff 1 (sign-canonicalized)")
    ax.set_title("Analysis 2: Local Minima Test\n(10 realizations x 5 inits)")
    ax.legend(fontsize=7 * SCALE, loc="best")
    ax.grid(True, alpha=0.3)

    # Panel 3: P300 kernel variability (mean + 1-std ellipse).
    if has_p300:
        ax = axes[2]
        ax.scatter(
            df3["coeff_0"],
            df3["coeff_1"],
            c="tab:green",
            alpha=0.6,
            s=30 * SCALE**2,
            edgecolors="k",
            linewidths=0.5 * SCALE,
        )
        mean_c0 = float(df3["coeff_0"].mean())
        mean_c1 = float(df3["coeff_1"].mean())
        ax.scatter(
            [mean_c0],
            [mean_c1],
            marker="D",
            c="red",
            s=80 * SCALE**2,
            zorder=5,
            label="Mean",
        )
        std_c0 = float(df3["coeff_0"].std())
        std_c1 = float(df3["coeff_1"].std())
        ell = Ellipse(
            (mean_c0, mean_c1),
            width=2 * std_c0,
            height=2 * std_c1,
            fill=False,
            edgecolor="red",
            linestyle="--",
            linewidth=1.5 * SCALE,
        )
        ax.add_patch(ell)
        ax.set_xlabel("Kernel coeff 0")
        ax.set_ylabel("Kernel coeff 1")
        ax.set_title(f"Analysis 3: P300 Kernel Variability\n(K=5, {len(df3)} reps)")
        ax.legend(fontsize=8 * SCALE)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the combined analysis CSV back into its three sub-frames."""
    df = pd.read_csv(CSV_PATH)
    df1 = df[df["analysis"] == "convergence_rate"].reset_index(drop=True)
    df2 = df[df["analysis"] == "local_minima"].reset_index(drop=True)
    df3 = df[df["analysis"] == "p300_variability"].reset_index(drop=True)
    return df1, df2, df3


def main() -> None:
    """Render the convergence figure to PDF and PNG."""
    set_rcparams()
    df1, df2, df3 = load_frames()
    fig = make_figure(df1, df2, df3)
    os.makedirs(FIG_DIR, exist_ok=True)
    pdf_path = os.path.join(FIG_DIR, "convergence_kernels.pdf")
    png_path = os.path.join(FIG_DIR, "convergence_kernels.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
