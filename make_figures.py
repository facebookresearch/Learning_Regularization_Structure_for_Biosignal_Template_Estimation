# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Paper figures from the analysis outputs.

Reads ``<runs>/<analysis>/summary.json`` and ``per_fit.csv`` written by
``run_simulations.py`` (``white_sims``, ``colored_factorial``),
``run_real_data.py`` (``real_data``) and ``run_p300.py`` (``p300``), and writes
``<out>/<name>.pdf`` and ``.png`` for each figure of ``FIGURES``.
``fig_kernels_magnitude`` also writes ``<out>/fig_kernels_magnitude.json``: the
kernel counts and Nyquist/DC ratios the paper cites. Values that a failed or
sliced run left undefined (None or absent) are not drawn.

Usage: python3 -B make_figures.py [--runs DIR] [--out DIR] [NAME ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import analysis_tools
import figure_style as fs
import numpy as np
import protocol
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numpy.typing import NDArray


def run_file(runs: Path, analysis: str, name: str) -> Path:
    path = runs / analysis / name
    if not path.is_file():
        raise FileNotFoundError(
            f"make_figures: {path} not found; the figure needs the {analysis!r} run"
        )
    return path


def summary(runs: Path, analysis: str) -> dict[str, Any]:
    return json.loads(run_file(runs, analysis, "summary.json").read_text())


def per_fit(runs: Path, analysis: str) -> list[dict[str, str]]:
    with run_file(runs, analysis, "per_fit.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def value(tree: object, *keys: str) -> float:
    """The number at ``tree[k1][k2]...``; NaN where a key is absent or the value is
    not a number (None for an undefined statistic), so that it is not drawn."""
    for k in keys:
        if not isinstance(tree, Mapping) or k not in tree:
            return math.nan
        tree = tree[k]
    return float(tree) if isinstance(tree, (int, float)) else math.nan


def branch(tree: object, *keys: str) -> Mapping[str, Any]:
    """The mapping at ``tree[k1][k2]...``; empty where a key is absent."""
    for k in keys:
        tree = tree.get(k) if isinstance(tree, Mapping) else None
    return tree if isinstance(tree, Mapping) else {}


def axhline_defined(ax: Axes, y: float, **kw: Any) -> None:  # noqa: ANN401
    if math.isfinite(y):
        ax.axhline(y, **kw)


# ---------------------------------------------------------------------------
# Main paper: white-noise event-count sweep
# ---------------------------------------------------------------------------
def sweep_stats(S: Mapping[str, Any], sweep: str) -> fs.Stats:
    """{method: {x: (mean, se)}}; a cell without a mean is left out, a missing SE
    (one repetition) is NaN."""
    out: dict[str, dict[float, tuple[float, float]]] = {}
    for x, per_method in branch(S, "sweeps", sweep).items():
        for m in per_method:
            mean = value(per_method, m, "mean")
            if math.isfinite(mean):
                se = value(per_method, m, "se")
                out.setdefault(m, {})[float(x)] = (mean, se)
    return out


def sim_nevents_sweep(runs: Path) -> Figure:
    stats = sweep_stats(summary(runs, "white_sims"), "nevents")
    return fs.sweep_figure(stats, "Number of Events per Type")


# ---------------------------------------------------------------------------
# Colored-noise factorial: the dense case (main paper) and all cells (supplement)
# ---------------------------------------------------------------------------
FACTORIAL_CELLS: list[str] = [
    "Oracle",
    "fixed_white",
    "fixed_colored",
    "learned_white",
    "learned_colored",
]
MAIN_EFFECTS: list[str] = [
    "kernel_effect_white",
    "kernel_effect_colored",
    "noise_effect_fixed",
    "noise_effect_learned",
]
MEAN_INTERVALS: list[str] = ["40", "200"]


def _factorial_series(
    ax: Axes, phis: Sequence[float], cells: Sequence[Mapping[str, Any]], key: str
) -> None:
    mean, lo, hi = (
        np.array([value(c, k) for c in cells]) for k in ("mean", "lo", "hi")
    )
    fs.plot_method(ax, phis, mean, lo, hi, key, ms=4, lw=1.0)


def _factorial_row(axes_row: Sequence[Axes], S: Mapping[str, Any], mi: str) -> None:
    sim = branch(S, "sim")
    grid = sorted(
        (float(k.split("_phi")[1]), k) for k in sim if k.startswith(f"mi{mi}_phi")
    )
    phis = [p for p, _ in grid]
    a0, a1, a2 = axes_row
    for c in FACTORIAL_CELLS:
        _factorial_series(a0, phis, [branch(sim, k, "cells", c) for _, k in grid], c)
    for e in [*MAIN_EFFECTS, "interaction"]:
        ax = a2 if e == "interaction" else a1
        _factorial_series(ax, phis, [branch(sim, k, e) for _, k in grid], e)
    fs.zero_line(a1)
    fs.zero_line(a2)
    word = "dense" if mi == "40" else "sparse"
    a0.text(
        0.97,
        0.95,
        f"mean_interval {mi}\n({word})",
        transform=a0.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    a0.set_ylabel("SNR [dB]")
    a1.set_ylabel("Effect on SNR [dB]")
    a2.set_ylabel("Interaction [dB]")


def _phi_axis(ax: Axes) -> None:
    ax.set_xlabel(r"AR(1) coefficient $\varphi$")
    ax.set_xticks([0, 0.3, 0.5, 0.7, 0.98])
    ax.set_xticklabels(["0", "0.3", "0.5", "0.7", "0.98"])
    ax.set_xticks([0.15, 0.85, 0.95], minor=True)


# Main paper: the 2x2 cells of the dense case. Colour and marker of the estimator
# (fixed [1,-2,1] kernel = SURE lambda, learned = SURE kernel); dashed, open = white
# noise procedure, solid, filled = colored. The oracle curve is left out: in the dense
# case it lies within 0.1 dB of the fixed-kernel colored curve at every phi.
SIMPLE_CELLS: list[tuple[str, str, str]] = [
    ("fixed_white", "SURE(lambda)", r"Fixed $[1,-2,1]$, SURE $\lambda$, white"),
    ("fixed_colored", "SURE(lambda)", r"Fixed $[1,-2,1]$, SURE $\lambda$, colored"),
    ("learned_white", "SURE(kernel)", "Learned kernel (SURE kernel), white"),
    ("learned_colored", "SURE(kernel)", "Learned kernel (SURE kernel), colored"),
]


def _simple_line_kw(cell: str, method: str) -> dict[str, Any]:
    st = fs.method_style(method)
    white = cell.endswith("_white")
    return {
        "color": st.color,
        "ls": (0, (4, 2)) if white else "-",
        "lw": fs.LINE_WIDTH,
        "marker": st.marker,
        "markersize": 4 * st.marker_scale,
        "markerfacecolor": "white" if white else st.color,
    }


def sim_colored_simple(runs: Path) -> Figure:
    sim = branch(summary(runs, "colored_factorial"), "sim")
    mi = MEAN_INTERVALS[0]
    grid = sorted(
        (float(k.split("_phi")[1]), k) for k in sim if k.startswith(f"mi{mi}_phi")
    )
    phis = [p for p, _ in grid]
    fig = fs.new_figure((fs.SINGLE_COLUMN[0], 3.2))
    ax = fig.subplots()
    handles = []
    for cell, method, label in SIMPLE_CELLS:
        kw = _simple_line_kw(cell, method)
        cells = [branch(sim, k, "cells", cell) for _, k in grid]
        mean, lo, hi = (
            np.array([value(c, k) for c in cells]) for k in ("mean", "lo", "hi")
        )
        ax.fill_between(
            phis, lo, hi, color=kw["color"], alpha=fs.CI_ALPHA, lw=0, zorder=2
        )
        ax.plot(phis, mean, zorder=3, **kw)
        handles.append(Line2D([], [], label=label, **kw))
    ax.set_ylabel("Reconstruction SNR (dB)")
    _phi_axis(ax)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=1,
        frameon=False,
        fontsize=7,
        handlelength=2.6,
    )
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    return fig


def sim_colored_factorial(runs: Path) -> Figure:
    S = summary(runs, "colored_factorial")
    fig = fs.new_figure((fs.DOUBLE_COLUMN[0], 5.0))
    axes = fig.subplots(2, 3, sharex=True)
    for r, mi in enumerate(MEAN_INTERVALS):
        _factorial_row(list(axes[r]), S, mi)
    titles = ["(a) Cell SNR", "(b) Main effects", "(c) Interaction"]
    for c in range(3):
        axes[0, c].set_title(titles[c])
        _phi_axis(axes[1, c])
    series = FACTORIAL_CELLS + MAIN_EFFECTS + ["interaction"]
    fs.figure_legend(fig, fs.legend_handles(series), ncol=4)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    return fig


# ---------------------------------------------------------------------------
# EEG/ECG and P300 figures: trivial baselines (zero template, lambda = inf) against K
# ---------------------------------------------------------------------------
ZERO_KW: dict[str, Any] = {
    "color": "#000000",
    "ls": (0, (6, 2)),
    "lw": 1.4,
    "label": "zero template",
}
INF_KW: dict[str, Any] = {
    "color": "#7f7f7f",
    "marker": "d",
    "ls": (0, (3, 1, 1, 1)),
    "lw": 1.2,
    "ms": 5,
    "label": r"$\lambda=\infty$ (linear trend)",
}
P300_KS: list[int] = [5, 10, 20, 50]


def _trivial_line(ax: Axes, x: Sequence[float], y: Sequence[float], name: str) -> None:
    st = fs.method_style(name)
    ax.plot(
        x,
        y,
        color=st.color,
        marker=st.marker,
        ls=st.linestyle,
        lw=fs.LINE_WIDTH,
        ms=fs.MARKER_SIZE,
        mfc=st.markerfacecolor or st.color,
        label=st.label,
    )


def _real_data_panel(ax: Axes, levels: Mapping[str, Any], title: str) -> None:
    """Estimators, lambda = inf and zero; `levels[K][series]["OLS"]`."""
    ks = sorted(int(k) for k in levels)
    for n in fs.CORE_METHODS:
        _trivial_line(ax, ks, [value(levels, str(k), n, "OLS") for k in ks], n)
    ax.plot(ks, [value(levels, str(k), "lambda=inf", "OLS") for k in ks], **INF_KW)
    zero = value(levels, str(ks[0]), "zero", "OLS") if ks else math.nan
    axhline_defined(ax, zero, **ZERO_KW)
    if math.isfinite(zero):
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, max(hi, 1.06 * zero))
    fs.log_k_axis(ax, ks)
    ax.set_xlabel("events per class K")
    ax.set_ylabel("distance to OLS ref.")
    ax.set_title(title)


def eeg_ecg_baselines(runs: Path) -> Figure:
    S = summary(runs, "real_data")
    fig = fs.new_figure(fs.DOUBLE_COLUMN)
    axes = fig.subplots(1, 2)
    for ax, ds, title in (
        (axes[0], "EEG", r"(a) auditory EEG [$\mu$V]"),
        (axes[1], "ECG", "(b) ECG record 119 [mV]"),
    ):
        _real_data_panel(ax, branch(S, ds, "baselines", "K"), title)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, **fs.LEGEND_KW)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    return fig


def _population(ax: Axes, y: float, label: bool) -> None:
    """The K-independent Population level (undefined with one subject)."""
    pop = fs.method_style(protocol.POPULATION)
    kw = {"label": pop.label} if label else {}
    axhline_defined(ax, y, color=pop.color, ls=pop.linestyle, lw=1.0, **kw)


def _distance_panel(ax: Axes, levels: Mapping[str, Any], key: str, title: str) -> None:
    """Estimators, lambda = inf, zero and Population; `levels[K][series][key]`."""
    ks = P300_KS
    for n in [*fs.CORE_METHODS, "Molina filtered DTW"]:
        _trivial_line(ax, ks, [value(levels, str(k), n, key) for k in ks], n)
    ax.plot(ks, [value(levels, str(k), "lambda=inf", key) for k in ks], **INF_KW)
    axhline_defined(ax, value(levels, "5", "zero", key), **ZERO_KW)
    _population(ax, value(levels, "5", protocol.POPULATION, key), label=True)
    fs.log_k_axis(ax, ks)
    ax.set_xlabel("events per class K")
    ax.set_ylabel(r"$\mu$V")
    ax.set_title(title)


def _auc_panel(ax: Axes, auc: Mapping[str, Any]) -> None:
    kk = ["5", "10", "20", "50", "all"]
    xs = [5, 10, 20, 50, 100]
    for n in [*fs.CORE_METHODS, "Molina filtered DTW"]:
        _trivial_line(ax, xs, [value(auc, k, n, "mean") for k in kk], n)
    ax.plot(xs, [value(auc, k, "lambda=inf", "mean") for k in kk], **INF_KW)
    ax.axhline(0.5, **ZERO_KW)
    _population(ax, value(auc, "5", protocol.POPULATION, "mean"), label=False)
    fs.log_k_axis(ax, xs, kk)
    ax.set_xlabel("training events per class")
    ax.set_ylabel("held-out AUC")
    ax.set_title("(b) Held-out AUC")


def p300_kcurve_baselines(runs: Path) -> Figure:
    """(a) Distance to the OLS reference and (b) held-out AUC from run_p300.py."""
    base = branch(summary(runs, "p300"), "baselines")
    fig = fs.new_figure(fs.DOUBLE_COLUMN_TALL)
    axes = fig.subplots(1, 2)
    _distance_panel(axes[0], branch(base, "K"), "OLS", "(a) Distance to OLS reference")
    _auc_panel(axes[1], branch(base, "auc"))
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, **fs.LEGEND_KW)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return fig


# ---------------------------------------------------------------------------
# Learned-kernel magnitude responses (main paper). |H(f)| on n_freqs frequencies in
# [0, Nyquist] divided by its maximum. P300 (K = 5 central fits): median response
# per subject, then median and 25-75% band across subjects; EEG and ECG (K = 5) and
# the simulation (noise sweep at SD 0.1): across repetitions. Failed fits and a
# zero response are excluded and counted; Nyquist/DC only where |H(0)| > threshold
# x max |H| (decision OQ-B1).
# ---------------------------------------------------------------------------
KERNEL_K = "5"
KERNEL_SIM_NOISE = 0.1
KERNEL_PANELS: list[tuple[str, str]] = [
    ("simulation", "Simulation"),
    ("EEG", "EEG"),
    ("P300", "P300"),
    ("ECG", "ECG"),
]
Items = Sequence[tuple[str, Mapping[str, str]]]


def magnitude(kernel: Sequence[float], n: int) -> NDArray | None:
    f = np.linspace(0.0, 0.5, n)
    h = np.abs(
        np.exp(-2j * np.pi * np.outer(f, np.arange(len(kernel))))
        @ np.asarray(kernel, dtype=float)
    )
    return None if not np.isfinite(h).all() or h.max() == 0 else h


def kernel_fits(runs: Path) -> dict[str, list[tuple[str, dict[str, str]]]]:
    """(group, per-fit row) of every SURE(kernel) fit per panel; group = subject
    for P300, "-" otherwise."""
    panels: dict[str, list[tuple[str, dict[str, str]]]] = {
        p: [] for p, _ in KERNEL_PANELS
    }
    for r in per_fit(runs, "p300"):
        if (r["analysis"], r["K"], r["estimator"]) == ("A", KERNEL_K, "SURE(kernel)"):
            panels["P300"].append((r["subject"], r))
    for r in per_fit(runs, "real_data"):
        if (r["K"], r["estimator"]) == (KERNEL_K, "SURE(kernel)") and r[
            "dataset"
        ] in panels:
            panels[r["dataset"]].append(("-", r))
    for r in per_fit(runs, "white_sims"):
        noise = float(r["grid_value"]) if r["sweep"] == "noise" else math.nan
        if r["label"] == "SURE(kernel)" and math.isclose(
            noise, KERNEL_SIM_NOISE, rel_tol=0.0, abs_tol=0.0
        ):
            panels["simulation"].append(("-", r))
    return panels


def _quartiles(values: Sequence[float]) -> dict[str, Any]:
    return {
        "n_units": len(values),
        "median": float(np.median(values)) if values else None,
        "q25": float(np.percentile(values, 25)) if values else None,
        "q75": float(np.percentile(values, 75)) if values else None,
    }


def _responses(
    items: Items, n: int, thr: float
) -> tuple[dict[str, int], dict[str, list[NDArray]], dict[str, list[float]]]:
    """Counts, normalized responses and Nyquist/DC ratios per group."""
    counts = dict.fromkeys(
        ("failed_excluded", "zero_response_excluded", "flagged_kept", "dc_near_zero"),
        0,
    ) | {"n": len(items)}
    curves: dict[str, list[NDArray]] = {}
    ratios: dict[str, list[float]] = {}
    for group, rec in items:
        if rec["status"] != "success" or not rec["kernel"]:
            counts["failed_excluded"] += 1
            continue
        h = magnitude(json.loads(rec["kernel"]), n)
        if h is None:
            counts["zero_response_excluded"] += 1
            continue
        counts["flagged_kept"] += rec["flags"] not in {"", "[]"}
        if h[0] > thr * h.max():
            ratios.setdefault(group, []).append(float(h[-1] / h[0]))
        else:
            counts["dc_near_zero"] += 1
        curves.setdefault(group, []).append(h / h.max())
    return counts, curves, ratios


def aggregate(items: Items, unit: str) -> dict[str, Any]:
    thr = protocol.KERNEL_DC_REL_THRESHOLD
    counts, curves, ratios = _responses(items, protocol.KERNEL_N_FREQS, thr)
    out: dict[str, Any] = {"counts": counts, "unit": unit, "median": None}
    if not curves:
        return out | {"n_units": 0, "nyquist_over_dc": _quartiles([])}
    per_kernel = [x for v in ratios.values() for x in v]
    out["n_kernels_nyquist_below_dc"] = sum(x < 1 for x in per_kernel)
    if unit == "subject":
        per_subject = {g: np.median(np.array(v), axis=0) for g, v in curves.items()}
        stack = np.array([per_subject[g] for g in sorted(per_subject)])
        units = [float(np.median(v)) for _, v in sorted(ratios.items())]
        out["per_subject_median"] = per_subject
        out["subject_median_response_nyquist_over_dc"] = sorted(
            float(m[-1] / m[0]) for m in per_subject.values() if m[0] > thr * m.max()
        )
    else:
        stack = np.array([c for v in curves.values() for c in v])
        units = per_kernel
    return out | {
        "n_units": len(stack),
        "median": np.median(stack, axis=0),
        "q25": np.percentile(stack, 25, axis=0),
        "q75": np.percentile(stack, 75, axis=0),
        "nyquist_over_dc": _quartiles(units),
    }


def kernel_aggregates(runs: Path) -> dict[str, dict[str, Any]]:
    fits = kernel_fits(runs)
    return {
        p: aggregate(fits[p], "subject" if p == "P300" else "repetition")
        for p, _ in KERNEL_PANELS
    }


def kernel_numbers(runs: Path) -> dict[str, Any]:
    arrays = ("median", "q25", "q75", "per_subject_median")
    return {
        p: {k: v for k, v in a.items() if k not in arrays}
        for p, a in kernel_aggregates(runs).items()
    }


def _kernel_panel(ax: Axes, a: Mapping[str, Any], name: str, letter: str) -> None:
    unit = {"subject": "subjects", "repetition": "repetitions"}[a["unit"]]
    ax.set_title(f"({letter}) {name}\n$n$ = {a['n_units']} {unit}", fontsize=9)
    ax.set_xlim(0, 0.5)
    ax.set_xticks([0, 0.25, 0.5])
    ax.set_xticklabels(["0", "0.25", "0.5"])
    if a["median"] is None:
        return
    f = np.linspace(0.0, 0.5, len(a["median"]))
    st = fs.DATASET_STYLES[name]
    for s in sorted(a.get("per_subject_median", {}), key=int):
        ax.plot(f, a["per_subject_median"][s], color=st["color"], lw=0.5, alpha=0.35)
    ax.fill_between(f, a["q25"], a["q75"], color=st["color"], alpha=fs.CI_ALPHA)
    ax.plot(f, a["median"], color=st["color"], ls=st["ls"], lw=st["lw"])
    ax.plot(f, np.sin(np.pi * f) ** 2, color="gray", ls=":", lw=1.0)
    r = a["nyquist_over_dc"]
    if r["median"] is not None:
        ax.text(
            0.03,
            0.97,
            f"Nyquist/DC {r['median']:.3g}\n[{r['q25']:.3g}, {r['q75']:.3g}]",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
        )


def kernels_magnitude(runs: Path) -> Figure:
    """The dotted line is the fixed [1,-2,1] kernel: |H(f)| / max |H| = sin^2(pi f)."""
    agg = kernel_aggregates(runs)
    fig = fs.new_figure((fs.DOUBLE_COLUMN[0], 2.6))
    axes = fig.subplots(1, 4, sharey=True)
    for ax, (key, name), letter in zip(axes, KERNEL_PANELS, "abcd"):
        _kernel_panel(ax, agg[key], name, letter)
    axes[0].set_ylim(0, 1.05)
    handles = [
        Line2D([], [], color="k", lw=1.5, label="Median"),
        Patch(facecolor="k", alpha=fs.CI_ALPHA, label="25–75% across units"),  # noqa: RUF001
        Line2D(
            [],
            [],
            color=fs.DATASET_STYLES["P300"]["color"],
            lw=0.5,
            alpha=0.6,
            label="P300 subject medians",
        ),
        Line2D([], [], color="gray", ls=":", lw=1.0, label=r"Fixed $[1,-2,1]$"),
    ]
    fs.figure_legend(fig, handles, ncol=4)
    fig.tight_layout(rect=(0.04, 0.17, 1, 1))
    fs.shared_axis_labels(
        fig, list(axes), "Frequency [cycles/sample]", r"Normalized magnitude $|H(f)|$"
    )
    return fig


# ---------------------------------------------------------------------------
# Supplement: white-noise noise-level sweep
# ---------------------------------------------------------------------------
def sim_noise_sweep(runs: Path) -> Figure:
    stats = sweep_stats(summary(runs, "white_sims"), "noise")
    return fs.sweep_figure(stats, "Noise Std")


FIGURES: dict[str, Callable[[Path], Figure]] = {
    "fig_sim_nevents_sweep": sim_nevents_sweep,
    "fig_sim_colored_simple": sim_colored_simple,
    "fig_sim_colored_factorial": sim_colored_factorial,
    "fig_eeg_ecg_baselines": eeg_ecg_baselines,
    "fig_p300_kcurve_2panel": p300_kcurve_baselines,
    "fig_kernels_magnitude": kernels_magnitude,
    "figS_sim_noise_sweep": sim_noise_sweep,
}
NUMBERS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "fig_kernels_magnitude": kernel_numbers,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--runs", default="./runs", help="folder holding the analysis outputs"
    )
    parser.add_argument("--out", default="./figures")
    parser.add_argument("names", nargs="*", help=f"default: all of {list(FIGURES)}")
    args = parser.parse_args()
    unknown = sorted(set(args.names) - set(FIGURES))
    if unknown:
        parser.error(f"unknown figures {unknown}; expected names from {list(FIGURES)}")
    runs, out = Path(args.runs), Path(args.out)
    with fs.paper_style():
        for name in args.names or FIGURES:
            fs.save_figure(FIGURES[name](runs), out, name)
            if name in NUMBERS:
                analysis_tools.write_json(out / f"{name}.json", NUMBERS[name](runs))


if __name__ == "__main__":
    main()
