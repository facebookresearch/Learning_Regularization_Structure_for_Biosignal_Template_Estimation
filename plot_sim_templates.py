# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Stacked-trace plot of the simulation templates (paper Figure 1).

Replaces the previous overlaid 8-channel plot, which hid the outer channels
when their Hanning-window amplitude was small. Per-channel vertical offset,
twilight palette, hidden y-ticks, t=0 reference line.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
from numpy.typing import NDArray

mpl.use("Agg")
from core import ricker

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


def make_real_templates(T: int = 100, C: int = 8) -> dict[str, NDArray]:
    """Two Ricker-wavelet templates with Hanning spatial profiles across C channels."""
    return {
        "Template A": np.outer(ricker(T, 15), np.roll(np.hanning(C), 3)),
        "Template B": np.outer(ricker(T, 10), np.hanning(C)),
    }


def plot_stacked(
    templates: dict[str, NDArray],
    scale: float = 0.75,
    out_dir: Path | None = None,
) -> None:
    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    import matplotlib.pyplot as plt

    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "figures"
    plt.rcParams.update(PAPER_STYLE)

    n_templates = len(templates)
    # Single-column. Pick figsize so the bbox-tight PDF width matches the other
    # single-column figures -- otherwise LaTeX scales this PDF up to
    # \myfigwidth and the text renders larger relative to the rest of the paper.
    fig, axes = plt.subplots(
        1,
        n_templates,
        figsize=(4.2, 3.4),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.05},
    )
    if n_templates == 1:
        axes = [axes]

    # Common gap derived from peak amplitude across all templates
    peak = max(np.max(np.abs(t)) for t in templates.values())
    gap = scale * peak

    # Time axis centered at 0 in samples
    T_full = next(iter(templates.values())).shape[0]
    t_axis = np.arange(T_full) - (T_full - 1) / 2

    for ax, (name, tpl) in zip(axes, templates.items()):
        C = tpl.shape[1]
        colors = plt.get_cmap("twilight")(np.linspace(0.15, 0.85, C))
        ax.axvline(0, color="k", linestyle="--", linewidth=0.7, alpha=0.3, zorder=0)
        for ch in range(C):
            ax.plot(
                t_axis,
                tpl[:, ch] + gap * ch,
                color=colors[ch],
                linewidth=1.2,
                alpha=0.9,
            )
        ax.set_title(name, pad=2)
        ax.set_xlabel("Time (samples)", labelpad=2)
        # Hide y-ticks: offsets are not physically meaningful, but keep the full
        # frame for visual consistency with the other paper figures.
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = out_dir / f"fig1b_templates.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close(fig)


def main() -> None:
    templates = make_real_templates(T=100, C=8)
    print("Templates:")
    for name, tpl in templates.items():
        print(f"  {name}: shape={tpl.shape}, peak={np.max(np.abs(tpl)):.4f}")
    plot_stacked(templates)


if __name__ == "__main__":
    main()
