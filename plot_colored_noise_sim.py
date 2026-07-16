# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

"""
Colored-noise simulation figure (paper Fig 5).

Produces fig5_colored_noise_sim.{pdf,png}: template-estimation SNR vs the AR(1)
coefficient phi at sigma=0.1 on the dense (overlapping, ISI=40) event layout,
comparing Oracle / White SURE / Colored SURE. Colored SURE tracks the oracle at
all phi while white SURE loses up to ~0.7 dB at intermediate phi.

Data: reads the committed results/colored_noise_sim_noise01.csv (the paper's
simulation output, aggregated as snr_mean / snr_se over repetitions). If that
file is absent, run_colored_noise_simulation() regenerates an equivalent
realization (the original raw-repetition harness was not preserved, so a fresh
run reproduces the methodology and is numerically equivalent, not bit-identical).

Run with: OPENBLAS_NUM_THREADS=1 python plot_colored_noise_sim.py
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from core import calculate_snr, generate_data, get_events_templates, ricker
from numpy.typing import NDArray

mpl.use("Agg")
import matplotlib.pyplot as plt

HERE: Path = Path(__file__).resolve().parent
RESULTS_DIR: Path = HERE / "results"
FIG_DIR: Path = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

CSV_NAME: str = "colored_noise_sim_noise01.csv"

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

# Method -> (color, linestyle, marker). Oracle grey, White SURE blue, Colored red.
CN_STYLE: dict[str, tuple[str, str, str]] = {
    "Oracle": ("#888888", ":", "s"),
    "White SURE": ("#1f77b4", "-.", "^"),
    "Colored SURE": ("#d62728", "-", "o"),
}
CN_METHODS: list[str] = ["Oracle", "White SURE", "Colored SURE"]

# Figure is single-column: the "overlapping events" panel from the caption.
DENSE_LABEL: str = "Dense (ISI=40)"

PHI_VALUES: list[float] = [0.0, 0.15, 0.3, 0.5, 0.7, 0.85]
N_REP: int = 20
NOISE_STD: float = 0.1
I_EVENTS: int = 15
CONFIGS: list[tuple[str, int]] = [("Dense (ISI=40)", 40), ("Sparse (ISI=200)", 200)]


def run_colored_noise_simulation() -> pd.DataFrame:
    """Regenerate an equivalent colored-noise sweep (sigma=0.1), aggregated.

    Returns a frame with columns isi, phi, method, snr_mean, snr_se. This
    reproduces the methodology behind the committed colored_noise_sim_noise01.csv
    (the original raw-repetition harness was not preserved).
    """
    T = 100
    half_win = T // 2
    C = 8
    real_templates: dict[str, NDArray] = {
        "Template A": np.outer(ricker(T, 15), np.roll(np.hanning(C), 3)),
        "Template B": np.outer(ricker(T, 10), np.hanning(C)),
    }

    raw: list[dict[str, float | str]] = []
    for config_name, mean_interval in CONFIGS:
        for phi in PHI_VALUES:
            print(f"    {config_name}, phi={phi}...", flush=True)
            for rep in range(N_REP):
                rs = np.random.RandomState(1000 * rep + int(phi * 100))
                data_white, ix, labels = generate_data(
                    real_templates,
                    I=I_EVENTS,
                    noise=NOISE_STD,
                    mean_interval=mean_interval,
                    rs=rs,
                )
                if phi > 0:
                    ar_noise = np.zeros_like(data_white)
                    w = NOISE_STD * np.sqrt(1 - phi**2) * rs.randn(*data_white.shape)
                    ar_noise[0] = w[0]
                    for t_idx in range(1, len(data_white)):
                        ar_noise[t_idx] = phi * ar_noise[t_idx - 1] + w[t_idx]
                    signal = np.zeros_like(data_white)
                    for ii, lbl in zip(ix, labels):
                        ii = int(ii)
                        sl = slice(ii - half_win, ii + half_win)
                        if sl.start >= 0 and sl.stop <= len(signal):
                            signal[sl] += real_templates[lbl]
                    data = signal + ar_noise
                else:
                    data = data_white

                best_oracle = -np.inf
                for lam in np.logspace(-1, 6, 100):
                    t_est, _ = get_events_templates(
                        data,
                        ix,
                        labels,
                        half_win=half_win,
                        method="ls",
                        reg_temporal=lam,
                    )
                    best_oracle = max(best_oracle, calculate_snr(t_est, real_templates))

                t_white, _ = get_events_templates(
                    data,
                    ix,
                    labels,
                    half_win=half_win,
                    method="ls",
                    reg_temporal=500,
                    opt_reg_temporal=True,
                    noise_est="pairwise",
                )
                t_colored, _ = get_events_templates(
                    data,
                    ix,
                    labels,
                    half_win=half_win,
                    method="ls",
                    reg_temporal=500,
                    opt_reg_temporal=True,
                    noise_est="ar1_colored",
                )
                for method, snr in [
                    ("Oracle", best_oracle),
                    ("White SURE", calculate_snr(t_white, real_templates)),
                    ("Colored SURE", calculate_snr(t_colored, real_templates)),
                ]:
                    raw.append(
                        {"isi": config_name, "phi": phi, "method": method, "snr": snr}
                    )

    raw_df = pd.DataFrame(raw)
    agg = (
        raw_df.groupby(["isi", "phi", "method"])["snr"]
        .agg(snr_mean="mean", snr_se=lambda x: x.std(ddof=1) / np.sqrt(len(x)))
        .reset_index()
    )
    csv_path = RESULTS_DIR / CSV_NAME
    agg.to_csv(csv_path, index=False)
    print(f"  Saved colored-noise sweep -> {csv_path}")
    return agg


def plot_fig5(df: pd.DataFrame) -> None:
    d = df[df["isi"] == DENSE_LABEL].sort_values("phi")

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for method in CN_METHODS:
        dm = d[d["method"] == method]
        color, ls, marker = CN_STYLE[method]
        ax.errorbar(
            dm["phi"],
            dm["snr_mean"],
            yerr=1.96 * dm["snr_se"],
            label=method,
            color=color,
            marker=marker,
            markersize=5,
            linestyle=ls,
            linewidth=1.3,
            capsize=2,
        )
    ax.set_xlabel(r"AR(1) coefficient $\varphi$")
    ax.set_ylabel("Template estimation SNR [dB]")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig5_colored_noise_sim.{ext}")
    plt.close(fig)
    print("  -> fig5_colored_noise_sim.pdf / .png")


def main() -> None:
    csv_path = RESULTS_DIR / CSV_NAME
    if csv_path.exists():
        print(f"  Using committed {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("  Regenerating colored-noise sweep...")
        df = run_colored_noise_simulation()
    plot_fig5(df)


if __name__ == "__main__":
    main()
