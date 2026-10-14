# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Make figure_style.py in the parent folder importable when this script is run
# directly from supplementary/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import figure_style as fs  # noqa: E402

DESCRIPTION = """Conditioning of D^T Sigma_W D for AR(1) noise (primary EEG).

Panels: condition number vs phi with the white-noise D^T D level; lambda_max and
lambda_min vs phi; condition number vs 1 - phi (log-log) with a (1 - phi)^-2 guide.
Points with negative eigenvalues (< -1e-10 lambda_max) are crosses; the exact builder
gives none. Fonts, sizes and save settings are those of the other paper figures
(figure_style.py).

    python3 -B supplementary/plot_stability_condition_numbers.py \\
        --results runs/numerical_stability/results/EEG.json \\
        --out runs/numerical_stability
"""

LINE: dict[str, float] = {"ms": fs.MARKER_SIZE * 0.8, "lw": fs.LINE_WIDTH}
CROSS: dict[str, float] = {"ms": fs.MARKER_SIZE * 1.2, "mew": fs.LINE_WIDTH}
GUIDE: dict[str, float] = {"lw": 0.8, "alpha": 0.5}


Points = list[tuple[float, float]]


def condition_panel(ax: Axes, psd: Points, neg: Points, cond_dtd: float) -> None:
    # a sweep with no PSD point (or no lambda_min >= 0) only draws what it has
    if psd:
        x, y = zip(*psd)
        ax.semilogy(x, y, "bo-", **LINE, label=r"$D^T \Sigma_W D$", gid="stab.cond|psd")
    if neg:
        x, y = zip(*neg)
        ax.semilogy(x, y, "rx", **CROSS, label="neg. eigenvalue", gid="stab.cond|neg")
    ax.axhline(
        cond_dtd,
        color="r",
        ls="--",
        **GUIDE,
        label=r"$D^T D$ (white)",
        gid="stab.cond|dtd",
    )


def eigenvalue_panel(
    ax: Axes, phi: list[float], eigmin: list[float], eigmax: list[float]
) -> None:
    ax.semilogy(
        phi, eigmax, "rs-", **LINE, label=r"$\lambda_{\max}$", gid="stab.eig|max"
    )
    pos = [(p, e) for p, e in zip(phi, eigmin) if e >= 0]
    negm = [(p, abs(e)) for p, e in zip(phi, eigmin) if e < 0]
    if pos:
        x, y = zip(*pos)
        y = [e if e != 0 else 1e-300 for e in y]
        ax.semilogy(x, y, "b^-", **LINE, label=r"$\lambda_{\min}$", gid="stab.eig|min")
    if negm:
        x, y = zip(*negm)
        ax.semilogy(
            x,
            y,
            "bx",
            **CROSS,
            label=r"$|\lambda_{\min}|$ (negative)",
            gid="stab.eig|min_negative",
        )


def scaling_panel(ax: Axes, psd: Points, neg: Points) -> None:
    omp = [1 - p for p, _ in psd]
    if psd:
        y = [c for _, c in psd]
        ax.loglog(omp, y, "go-", **LINE, label="Measured", gid="stab.scale|psd")
    if neg:
        x, y = [1 - p for p, _ in neg], [c for _, c in neg]
        ax.loglog(x, y, "rx", **CROSS, label="neg. eigenvalue", gid="stab.scale|neg")
    if psd:
        x_ref = np.array([1e-3, 1.0])
        y_ref = x_ref ** (-2) * psd[0][1] * omp[0] ** 2  # through the first PSD point
        ax.loglog(
            x_ref,
            y_ref,
            "k--",
            **GUIDE,
            label=r"$\propto (1-\varphi)^{-2}$",
            gid="stab.scale|guide",
        )


def figure(res: dict[str, Any]) -> Figure:
    """The three panels from one results/<config>.json of run_numerical_stability.py
    (build inside ``figure_style.paper_style``)."""
    sweep = sorted(res["phi_sweep"].items(), key=lambda kv: float(kv[0]))
    phi = [float(p) for p, _ in sweep]
    # lambda_max / |lambda_min|, the submitted figure's quantity
    cond = [r["cond"] for _, r in sweep]
    bad = [r["neg_eigenvalues"] > 0 for _, r in sweep]
    psd = [(p, c) for p, c, b in zip(phi, cond, bad) if not b]
    neg = [(p, c) for p, c, b in zip(phi, cond, bad) if b]
    fig = fs.new_figure(fs.DOUBLE_COLUMN)
    axes = fig.subplots(1, 3)
    condition_panel(axes[0], psd, neg, res["DtD"]["cond"])
    eigmin = [r["eigmin"] for _, r in sweep]
    eigenvalue_panel(axes[1], phi, eigmin, [r["eigmax"] for _, r in sweep])
    scaling_panel(axes[2], psd, neg)
    titles = ("Condition Number", "Eigenvalue Range", r"Scaling with $1-\varphi$")
    xlabels = (r"$\varphi$", r"$\varphi$", r"$1 - \varphi$")
    ylabels = ("Condition number", "Eigenvalue magnitude", "Condition number")
    # the condition number falls as 1 - phi grows: the scaling panel's lower left is
    # empty, while "best" can put its legend on a negative-eigenvalue cross
    locs = ("best", "best", "lower left")
    for ax, title, xl, yl, loc in zip(axes, titles, xlabels, ylabels, locs):
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
        ax.legend(loc=loc, **fs.LEGEND_KW)
    fig.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(
        description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--results",
        type=Path,
        default=Path("runs/numerical_stability/results/EEG.json"),
    )
    ap.add_argument("--out", type=Path, default=Path("runs/numerical_stability"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "figS_stability_condition_numbers.pdf"
    with fs.paper_style():
        fig = figure(json.loads(args.results.read_text()))
        fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
