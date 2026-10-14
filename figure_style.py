# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Figure style of the paper: the rcParams, method colours, markers, line styles
and labels of the submitted figures, the plot primitives, and the two-panel
layout of the submitted simulation sweep figures (panel (a) mean SNR with a 95%
band, mean +/- 1.96 SE; panel (b) gain over Average, SE sqrt(SE_m^2 + SE_avg^2)).

Importing the module changes no global state: ``paper_style()`` scopes the
rcParams, and ``save_figure`` must run inside it because the ``savefig.*`` keys
are read at save time. Figures use the object-oriented API (``new_figure``).
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, NullLocator
from numpy.typing import ArrayLike

PAPER_RC: dict[str, Any] = {
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
LEGEND_KW: dict[str, Any] = {"framealpha": 0.9, "fontsize": 7}
CI_ALPHA: float = 0.12
ZERO_LINE_KW: dict[str, Any] = {"color": "k", "lw": 0.5, "alpha": 0.3}
LINE_WIDTH: float = 1.2
MARKER_SIZE: float = 5.0

# Inches before bbox-tight cropping; these sizes keep the printed font size of the
# submission (double-column figures are included at 0.9\textwidth).
DOUBLE_COLUMN: tuple[float, float] = (7.2, 3.2)
DOUBLE_COLUMN_TALL: tuple[float, float] = (7.2, 3.8)
SINGLE_COLUMN: tuple[float, float] = (3.5, 2.8)


def new_figure(figsize: tuple[float, float]) -> Figure:
    """An Agg-backed figure, without pyplot."""
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    return fig


@dataclass(frozen=True)
class MethodStyle:
    color: str
    marker: str
    linestyle: str | tuple[int, tuple[int, ...]]
    label: str
    markerfacecolor: str | None = None  # None = filled with `color`
    marker_scale: float = 1.0


# The first seven entries are the submission's. Molina filtered DTW takes the next
# unused tab10 hue (green is avoided: it was "Reg. LS" in the submission); Population
# is a control, so it gets a hue no estimator uses.
METHOD_STYLES: dict[str, MethodStyle] = {
    "Average": MethodStyle("#888888", "s", ":", "Average"),
    "OLS": MethodStyle("#444444", "v", ":", "OLS"),
    "GCV": MethodStyle("#1f77b4", "^", "-.", "GCV"),
    "L-curve": MethodStyle("#9467bd", "<", "-.", "L-curve"),
    "EmpBayes": MethodStyle("#ff7f0e", ">", "--", "Emp. Bayes"),
    "SURE(lambda)": MethodStyle("#d62728", "o", "-", r"SURE $\lambda$"),
    "SURE(kernel)": MethodStyle("#000000", "*", "-", "SURE kernel"),
    "Molina filtered DTW": MethodStyle(
        "#8c564b", "p", (0, (5, 1)), "Molina filtered DTW"
    ),
    "Population": MethodStyle("#17becf", "P", "--", "Population"),
}
CORE_METHODS: list[str] = list(METHOD_STYLES)[:7]
METHOD_LABELS: dict[str, str] = {m: s.label for m, s in METHOD_STYLES.items()}

# Colored-noise factorial: the estimator's colour and marker (fixed [1,-2,1] kernel =
# SURE(lambda), learned = SURE(kernel)); white noise dash-dot and open, colored solid.
# Effects use tab10 hues no cell uses.
FACTORIAL_STYLES: dict[str, MethodStyle] = {
    "fixed_white": MethodStyle("#d62728", "o", "-.", "Fixed kernel, white", "none"),
    "fixed_colored": MethodStyle("#d62728", "o", "-", "Fixed kernel, colored"),
    "learned_white": MethodStyle(
        "#000000", "*", "-.", "Learned kernel, white", "none", 1.2
    ),
    "learned_colored": MethodStyle("#000000", "*", "-", "Learned kernel, colored"),
    "Oracle": MethodStyle("#888888", "s", ":", "Oracle"),
    "kernel_effect_white": MethodStyle(
        "#2ca02c", "D", "-.", "Kernel effect (white)", "none"
    ),
    "kernel_effect_colored": MethodStyle(
        "#2ca02c", "D", "-", "Kernel effect (colored)"
    ),
    "noise_effect_fixed": MethodStyle(
        "#1f77b4", "^", "-.", "Noise effect (fixed)", "none"
    ),
    "noise_effect_learned": MethodStyle("#1f77b4", "^", "-", "Noise effect (learned)"),
    "interaction": MethodStyle("#9467bd", "h", "-", "Interaction"),
}
REFERENCE_COLOR: str = "#aaaaaa"
DATASET_STYLES: dict[str, dict[str, Any]] = {
    "Simulation": {"color": "#7f7f7f", "ls": "-", "lw": 1.5},
    "EEG": {"color": "#1f77b4", "ls": "-", "lw": 1.5},
    "P300": {"color": "#d62728", "ls": "--", "lw": 1.5},
    "ECG": {"color": "#ff7f0e", "ls": "-.", "lw": 1.5},
}


def method_style(method: str) -> MethodStyle:
    """Style of a method or series; an unknown name raises rather than get a default."""
    style = METHOD_STYLES.get(method) or FACTORIAL_STYLES.get(method)
    if style is None:
        raise KeyError(f"no paper style for {method!r}")
    return style


@contextlib.contextmanager
def paper_style() -> Iterator[None]:
    """The paper rcParams on top of matplotlib's defaults (independent of any local
    matplotlibrc), restored on exit."""
    with mpl.rc_context():
        mpl.rcdefaults()
        mpl.rcParams.update(PAPER_RC)
        yield


def plot_method(
    ax: Axes,
    x: ArrayLike,
    mean: ArrayLike,
    ci_lo: ArrayLike,
    ci_hi: ArrayLike,
    method: str,
    label: bool = True,
    ms: float = MARKER_SIZE,
    lw: float = LINE_WIDTH,
) -> None:
    """Mean curve with a shaded 95% band."""
    s = method_style(method)
    ax.plot(
        x,
        mean,
        label=s.label if label else None,
        color=s.color,
        marker=s.marker,
        markersize=ms * s.marker_scale,
        markerfacecolor=s.markerfacecolor or s.color,
        linestyle=s.linestyle,
        linewidth=lw,
    )
    ax.fill_between(x, ci_lo, ci_hi, color=s.color, alpha=CI_ALPHA)


def zero_line(ax: Axes) -> None:
    ax.axhline(0, **ZERO_LINE_KW)


def legend_handles(methods: Sequence[str]) -> list[Line2D]:
    """Proxy legend entries in paper style."""
    out = []
    for m in methods:
        s = method_style(m)
        out.append(
            Line2D(
                [],
                [],
                color=s.color,
                marker=s.marker,
                markersize=MARKER_SIZE * s.marker_scale,
                markerfacecolor=s.markerfacecolor or s.color,
                linestyle=s.linestyle,
                linewidth=LINE_WIDTH,
                label=s.label,
            )
        )
    return out


def figure_legend(fig: Figure, handles: Sequence[Artist], ncol: int) -> None:
    """One legend centred under the axes, clear of the data."""
    fig.legend(
        handles=list(handles),
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, 0.0),
        frameon=True,
        **LEGEND_KW,
    )


def shared_axis_labels(
    fig: Figure, axes: Sequence[Axes], xlabel: str, ylabel: str, pad: float = 0.01
) -> None:
    """One x and one y label for a row of axes, just outside the tick labels.

    Call after the layout is final: the labels are placed from the drawn extents.
    """
    renderer = cast(FigureCanvasAgg, fig.canvas).get_renderer()
    boxes = [b for ax in axes if (b := ax.get_tightbbox(renderer)) is not None]
    if not boxes:
        raise ValueError(
            f"shared_axis_labels: none of the {len(axes)} axes has a drawn extent"
        )
    x0 = min(b.x0 for b in boxes) / fig.bbox.width
    y0 = min(b.y0 for b in boxes) / fig.bbox.height
    pos = [ax.get_position() for ax in axes]
    xc = (min(p.x0 for p in pos) + max(p.x1 for p in pos)) / 2
    yc = (min(p.y0 for p in pos) + max(p.y1 for p in pos)) / 2
    size = mpl.rcParams["axes.labelsize"]
    fig.text(xc, y0 - pad, xlabel, ha="center", va="top", fontsize=size)
    fig.text(x0 - pad, yc, ylabel, ha="right", va="center", rotation=90, fontsize=size)


def log_k_axis(
    ax: Axes, ticks: Sequence[float], labels: Sequence[str] | None = None
) -> None:
    """Log-scaled x axis with exactly the given ticks."""
    ax.set_xscale("log")
    ax.set_xticks(list(ticks))
    ax.set_xticklabels(
        list(labels) if labels is not None else [f"{t:g}" for t in ticks]
    )
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def save_figure(fig: Figure, out_dir: Path, name: str) -> None:
    """Vector PDF and a 300-dpi PNG preview (run inside ``paper_style``)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")


# stats[method][x] = (mean, se)
Stats = Mapping[str, Mapping[float, tuple[float, float]]]


def gain(
    m: tuple[float, float], avg: tuple[float, float]
) -> tuple[float, float, float]:
    """(gain over Average, lo, hi) in dB."""
    g = m[0] - avg[0]
    se = math.sqrt(m[1] ** 2 + avg[1] ** 2)
    return g, g - 1.96 * se, g + 1.96 * se


def sweep_figure(
    stats: Stats, xlabel: str, methods: Sequence[str] = CORE_METHODS
) -> Figure:
    """The submitted two-panel sweep figure, with one legend below the panels."""
    fig = new_figure(DOUBLE_COLUMN)
    ax1, ax2 = fig.subplots(1, 2)
    shown = [m for m in methods if m in stats]
    for m in shown:
        xs = sorted(stats[m])
        mean = np.array([stats[m][x][0] for x in xs])
        se = np.array([stats[m][x][1] for x in xs])
        plot_method(ax1, xs, mean, mean - 1.96 * se, mean + 1.96 * se, m)
    avg = stats.get("Average", {})
    for m in shown:
        xs = [x for x in sorted(stats[m]) if x in avg]
        if m == "Average" or not xs:
            continue
        g = np.array([gain(stats[m][x], avg[x]) for x in xs])
        plot_method(ax2, xs, g[:, 0], g[:, 1], g[:, 2], m)
    for ax, ylabel, title in (
        (ax1, "SNR [dB]", "(a) Absolute SNR"),
        (ax2, "SNR Gain over Average [dB]", "(b) Improvement over Average"),
    ):
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    zero_line(ax2)
    if shown:
        figure_legend(fig, legend_handles(shown), ncol=len(shown))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig
