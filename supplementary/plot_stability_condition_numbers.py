# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

"""Regenerate the numerical-stability condition-number figure (stability_condition_numbers).

Computes the condition number and eigenvalue range of ``D^T Sigma_W D`` for AR(1)
noise across a grid of phi values on the EEG/ERP (MNE sample) dataset, then renders
the three-panel figure used in the paper's supplement:

  1. Condition number vs phi (with the white-noise ``D^T D`` reference line).
  2. Eigenvalue range (lambda_max / lambda_min) vs phi.
  3. Condition number vs ``1 - phi`` on log-log axes, with the ``(1-phi)^-2`` guide.

This is the Task-1 computation of supplementary/run_numerical_stability.py
ported to a standalone, figure-only script (no CSV: the values are computed). The
computed condition numbers reproduce supplementary/numerical_stability.md exactly.

Styling matches the published PDF (submission/figures/stability_condition_numbers.pdf):
the original figure uses matplotlib's default (sans-serif) font family at roughly
half the original per-panel sizes, and axes are positioned to the published layout so
a 300-dpi render reproduces it. NOTE: this intentionally differs from the serif
paper-standard rcParams used by the main-text figure scripts (e.g.
plot_colored_noise_sim.py); reproducing this already-published supplement figure
pixel-for-pixel required matching its actual (sans-serif) styling.

Writes figures/stability_condition_numbers.{pdf,png}.

Run with: OPENBLAS_NUM_THREADS=1 python supplementary/plot_stability_condition_numbers.py
(the MNE sample dataset must be reachable via the MNE_DATA env var or ~/mne_data).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

# Make the repo root (parent of supplementary/) importable when this script is
# run directly from supplementary/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib as mpl
import matplotlib.axes
import numpy as np
import scipy.sparse
from core import build_DtSigmaD_ar1, est_noise_var_baseline, get_design_matrix
from data_loaders import load_eeg_erp
from numpy.typing import NDArray

mpl.use("Agg")
import matplotlib.pyplot as plt

HERE: Path = Path(__file__).resolve().parent
FIG_DIR: Path = HERE.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Sampling / subsampling parameters (identical to run_numerical_stability.py).
SFREQ: float = 600.615
K_EVENTS: int = 5
SEED: int = 42
PHI_VALUES: list[float] = [0.0, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]

# Published-figure geometry: the target PDF renders at 300 dpi to 2231 x 725 px, with
# the three panel axes spanning x = [98, 717], [849, 1467], [1598, 2216] and y = [48, 635].
# We fix the figure to that exact pixel size and position the axes to those fractions so a
# 300-dpi render reproduces the published layout (values below are px / [2231 or 725]).
RENDER_DPI: int = 300
FIG_W_PX: int = 2231
FIG_H_PX: int = 725
AXES_LEFT: float = 98.0 / FIG_W_PX
AXES_RIGHT: float = 2216.0 / FIG_W_PX
AXES_TOP: float = 1.0 - 48.0 / FIG_H_PX
AXES_BOTTOM: float = 1.0 - 635.0 / FIG_H_PX
AXES_WSPACE: float = 0.2125  # inter-panel gap (~131 px) / panel width (~619 px)

# Font / line / marker sizes matching the published (sans-serif, default family) figure.
TITLE_SIZE: float = 6.5
LABEL_SIZE: float = 6.0
LEGEND_SIZE: float = 5.5
LINE_WIDTH: float = 1.0
DASH_WIDTH: float = 0.7
MS_CIRCLE: float = 3.0
MS_SQUARE_TRI: float = 2.5
MS_X_RED: float = 4.5
MS_X_BLUE: float = 4.0
MARKER_EDGE_WIDTH: float = 0.8

plt.rcParams.update(
    {
        "font.size": 6,
        "xtick.labelsize": 5.3,
        "ytick.labelsize": 5.3,
        "figure.dpi": 150,
        "savefig.dpi": RENDER_DPI,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


@dataclass
class PhiStability:
    """Condition number and eigenvalue range of ``D^T Sigma_W D`` at one phi."""

    phi: float
    cond: float
    eigmin: float
    eigmax: float
    n_negative: int


def compute_stability() -> tuple[list[PhiStability], float]:
    """Compute cond(D^T Sigma_W D) over the phi grid and cond(D^T D) reference.

    Loads the EEG/ERP data, subsamples ``K_EVENTS`` events per class with a fixed
    seed, builds the design matrix, estimates the baseline noise variance, and for
    each phi assembles ``D^T Sigma_W D`` and takes its symmetric eigenvalues.

    Returns
    -------
    tuple[list[PhiStability], float]
        Per-phi stability records and the condition number of ``D^T D`` (white).
    """
    loaded = load_eeg_erp()
    assert isinstance(loaded, tuple), (
        "load_eeg_erp must return a (data, indices, labels) tuple when return_epochs=False"
    )
    data, event_indices, event_labels = loaded

    half_win = int(0.3 * SFREQ)
    events = np.unique(event_labels)

    # Subsample K_EVENTS events per class (fixed seed), then sort by onset.
    rng = np.random.RandomState(SEED)
    ix_parts: list[NDArray] = []
    lb_parts: list[NDArray] = []
    for ev in events:
        mask = event_labels == ev
        candidates = event_indices[mask]
        valid = (candidates - half_win >= 0) & (candidates + half_win <= len(data))
        candidates = candidates[valid]
        sel = rng.choice(len(candidates), min(K_EVENTS, len(candidates)), replace=False)
        ix_parts.append(candidates[sel])
        lb_parts.append(np.full(len(sel), ev))
    ix = np.concatenate(ix_parts)
    lb = np.concatenate(lb_parts)
    order = np.argsort(ix)
    ix = ix[order]
    lb = lb[order]

    # Design matrix and white-noise reference cond(D^T D).
    design = get_design_matrix(data, ix, lb, half_win=half_win, events=events)
    if scipy.sparse.issparse(design):
        d_dense = np.asarray(design.todense())
    else:
        d_dense = np.asarray(design)
    dtd = d_dense.T @ d_dense
    eig_dtd = np.linalg.eigvalsh(dtd)
    cond_dtd = float(
        np.max(np.abs(eig_dtd)) / max(float(np.min(np.abs(eig_dtd))), 1e-300)
    )

    sigsq = est_noise_var_baseline(data, ix, half_win)

    stab: list[PhiStability] = []
    for phi in PHI_VALUES:
        dt_sigma_d = build_DtSigmaD_ar1(
            ix, lb, events, half_win, phi, sigsq, data_length=len(data)
        )
        eigvals = np.linalg.eigvalsh(dt_sigma_d)
        eigmin = float(np.min(eigvals))
        eigmax = float(np.max(eigvals))
        cond = eigmax / max(abs(eigmin), 1e-300)
        n_negative = int(np.sum(eigvals < -1e-10 * eigmax))
        stab.append(
            PhiStability(
                phi=phi,
                cond=cond,
                eigmin=eigmin,
                eigmax=eigmax,
                n_negative=n_negative,
            )
        )
    return stab, cond_dtd


def _split_psd_bad(
    stab: list[PhiStability], phis: list[float], conds: list[float]
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Split (phi, cond) points by whether ``D^T Sigma_W D`` stayed PSD at that phi.

    Returns ``(psd_phis, psd_conds, bad_phis, bad_conds)`` where the ``bad_*`` lists
    hold the phi values at which a negative eigenvalue appeared.
    """
    neg_mask = [s.n_negative > 0 for s in stab]
    psd_phis = [p for p, neg in zip(phis, neg_mask) if not neg]
    psd_conds = [c for c, neg in zip(conds, neg_mask) if not neg]
    bad_phis = [p for p, neg in zip(phis, neg_mask) if neg]
    bad_conds = [c for c, neg in zip(conds, neg_mask) if neg]
    return psd_phis, psd_conds, bad_phis, bad_conds


def _plot_condition_number(
    ax: matplotlib.axes.Axes,
    psd_phis: list[float],
    psd_conds: list[float],
    bad_phis: list[float],
    bad_conds: list[float],
    cond_dtd: float,
) -> None:
    """Panel 1: condition number vs phi with the white-noise ``D^T D`` reference."""
    ax.semilogy(
        psd_phis,
        psd_conds,
        "bo-",
        ms=MS_CIRCLE,
        lw=LINE_WIDTH,
        label=r"$D^T \Sigma_W D$",
    )
    if bad_phis:
        ax.semilogy(
            bad_phis,
            bad_conds,
            "rx",
            ms=MS_X_RED,
            mew=MARKER_EDGE_WIDTH,
            label=r"neg. eigenvalue",
        )
    ax.axhline(
        cond_dtd, color="r", ls="--", lw=DASH_WIDTH, alpha=0.5, label=r"$D^T D$ (white)"
    )
    ax.set_xlabel(r"$\varphi$", fontsize=LABEL_SIZE)
    ax.set_ylabel("Condition number", fontsize=LABEL_SIZE)
    ax.set_title("Condition Number", fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(True, alpha=0.3)


def _plot_eigenvalue_range(
    ax: matplotlib.axes.Axes,
    phis: list[float],
    eigmins: list[float],
    eigmaxs: list[float],
) -> None:
    """Panel 2: eigenvalue range (``lambda_max`` and ``|lambda_min|``) vs phi."""
    ax.semilogy(
        phis, eigmaxs, "rs-", ms=MS_SQUARE_TRI, lw=LINE_WIDTH, label=r"$\lambda_{\max}$"
    )
    abs_mins = [abs(e) if e != 0 else 1e-300 for e in eigmins]
    neg_e = [e < 0 for e in eigmins]
    pos_phis = [p for p, neg in zip(phis, neg_e) if not neg]
    pos_vals = [v for v, neg in zip(abs_mins, neg_e) if not neg]
    neg_phis = [p for p, neg in zip(phis, neg_e) if neg]
    neg_vals = [v for v, neg in zip(abs_mins, neg_e) if neg]
    ax.semilogy(
        pos_phis,
        pos_vals,
        "b^-",
        ms=MS_SQUARE_TRI,
        lw=LINE_WIDTH,
        label=r"$\lambda_{\min}$",
    )
    if neg_phis:
        ax.semilogy(
            neg_phis,
            neg_vals,
            "bx",
            ms=MS_X_BLUE,
            mew=MARKER_EDGE_WIDTH,
            label=r"$|\lambda_{\min}|$ (negative)",
        )
    ax.set_xlabel(r"$\varphi$", fontsize=LABEL_SIZE)
    ax.set_ylabel("Eigenvalue magnitude", fontsize=LABEL_SIZE)
    ax.set_title("Eigenvalue Range", fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(True, alpha=0.3)


def _plot_scaling(
    ax: matplotlib.axes.Axes,
    psd_phis: list[float],
    psd_conds: list[float],
    bad_phis: list[float],
    bad_conds: list[float],
) -> None:
    """Panel 3: condition number vs ``1 - phi`` (log-log) with the guide line."""
    psd_omp = [1 - p for p in psd_phis]
    ax.loglog(psd_omp, psd_conds, "go-", ms=MS_CIRCLE, lw=LINE_WIDTH, label="Measured")
    if bad_phis:
        bad_omp = [1 - p for p in bad_phis]
        ax.loglog(
            bad_omp,
            bad_conds,
            "rx",
            ms=MS_X_RED,
            mew=MARKER_EDGE_WIDTH,
            label="neg. eigenvalue",
        )
    x_ref = np.array([1e-3, 1.0])
    ax.loglog(
        x_ref,
        x_ref ** (-2) * psd_conds[0] * psd_omp[0] ** 2,
        "k--",
        alpha=0.4,
        lw=DASH_WIDTH,
        label=r"$\propto (1-\varphi)^{-2}$",
    )
    ax.set_xlabel(r"$1 - \varphi$", fontsize=LABEL_SIZE)
    ax.set_ylabel("Condition number", fontsize=LABEL_SIZE)
    ax.set_title(r"Scaling with $1-\varphi$", fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE)
    ax.grid(True, alpha=0.3)


def make_plot(
    stab: list[PhiStability], cond_dtd: float, pdf_path: Path, png_path: Path
) -> None:
    """Render the three-panel stability figure to PDF and PNG.

    The figure is fixed to the published pixel size and the axes are positioned to the
    published layout (see the AXES_* constants) so a 300-dpi render reproduces it. The
    figure is saved without a tight bounding box so the output size stays exactly
    ``FIG_W_PX x FIG_H_PX`` at ``RENDER_DPI``.
    """
    phis = [s.phi for s in stab]
    conds = [s.cond for s in stab]
    eigmins = [s.eigmin for s in stab]
    eigmaxs = [s.eigmax for s in stab]
    psd_phis, psd_conds, bad_phis, bad_conds = _split_psd_bad(stab, phis, conds)

    fig, axes = plt.subplots(
        1, 3, figsize=(FIG_W_PX / RENDER_DPI, FIG_H_PX / RENDER_DPI)
    )
    fig.subplots_adjust(
        left=AXES_LEFT,
        right=AXES_RIGHT,
        bottom=AXES_BOTTOM,
        top=AXES_TOP,
        wspace=AXES_WSPACE,
    )

    _plot_condition_number(axes[0], psd_phis, psd_conds, bad_phis, bad_conds, cond_dtd)
    _plot_eigenvalue_range(axes[1], phis, eigmins, eigmaxs)
    _plot_scaling(axes[2], psd_phis, psd_conds, bad_phis, bad_conds)

    # Save without a tight bbox so the output keeps the exact published pixel size.
    fig.savefig(pdf_path, bbox_inches=None)
    fig.savefig(png_path, bbox_inches=None)
    plt.close(fig)


def main() -> None:
    """Compute the stability metrics and write the figure."""
    stab, cond_dtd = compute_stability()
    make_plot(
        stab,
        cond_dtd,
        FIG_DIR / "stability_condition_numbers.pdf",
        FIG_DIR / "stability_condition_numbers.png",
    )


if __name__ == "__main__":
    main()
