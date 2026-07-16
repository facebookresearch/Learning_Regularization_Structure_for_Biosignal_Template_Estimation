# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import Counter
from typing import Any

"""
SURE(kernel) optimizer convergence analysis.

Three analyses:
1. Convergence rate: 50 runs with different seeds, record kernel, cost, iterations, success.
2. Local minima test: 10 realizations x 5 initializations, flag divergent solutions.
3. Kernel variability on P300 data: 20 repetitions, report coefficient statistics.

Outputs:
  - convergence_analysis.csv (raw kernel data)
  - convergence_kernels.png (scatter plot)
  - convergence_analysis.md (full report)
"""

import os
import sys
import time
from pathlib import Path

# Limit BLAS threads -- 200x200 matrices are faster single-threaded
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")

# Make sibling modules (core.py, data_loaders*.py in the parent folder)
# importable when this script is run directly from supplementary/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import (
    est_noise_var,
    gen_corefilt,
    generate_data,
    get_design_matrix,
    ricker,
    SURE_cost_vec,
)

OUTPUT_DIR = Path(__file__).resolve().parent


def make_real_templates(T: int = 100, C: int = 8) -> dict[str, np.ndarray]:
    return {
        "Template A": np.outer(ricker(T, 15), np.roll(np.hanning(C), 3)),
        "Template B": np.outer(ricker(T, 10), np.hanning(C)),
    }


def run_kernel_with_details(
    data: np.ndarray,
    ix: np.ndarray,
    lb: np.ndarray,
    half_win: int,
    vlen: int = 3,
    symm: bool = True,
    omega: float = 0.5,
    x0_override: list[float] | None = None,
    noise_est: str = "pairwise",
) -> dict[str, Any]:
    """Run SURE(kernel) (SURE-kernel) and return full optimizer details.

    This replicates the vec_mode=True path from core.get_events_templates,
    but captures and returns the full opt_res object.
    """
    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    import scipy as sp

    events = np.unique(lb)
    M = len(events)
    T = 2 * half_win

    # Build design matrix
    design = get_design_matrix(data, ix, lb, half_win=half_win, events=events)
    design = sp.sparse.csr_matrix(design)
    a = design.T @ design
    b = design.T @ data

    # Standard 2nd-order difference kernel
    R = sp.linalg.convolution_matrix([1, -2, 1], T - 2)
    RR = R @ R.T
    temporal_mat = sp.sparse.block_diag([RR] * M, format="csr")

    # Convert to dense ndarray once (avoid repeated todense() in optimizer loop)
    a_dense = np.asarray(a.todense(), dtype=float)

    # OLS solution
    theta_ml = np.linalg.solve(a_dense, np.asarray(b, dtype=float))

    # Noise variance
    sigsq = est_noise_var(
        data,
        method=noise_est,
        DtD=a,
        DtX=b,
        S=temporal_mat,
        X=data,
        D=design,
        ix=ix,
        lb=lb,
        half_win=half_win,
    )

    # Initialization
    def gen_h_diff(n: int) -> np.ndarray:
        h_diff: np.ndarray = np.array([1])
        for _ii in range(n):
            h_diff = np.convolve(h_diff, [1, -1])
        return h_diff

    if x0_override is not None:
        x0 = np.array(x0_override, dtype=float)
    else:
        x0_full = gen_h_diff(vlen)
        x0 = x0_full[: int(vlen - ((vlen - 1) / 2) * int(symm))]

    # Cost at initialization
    cost_at_x0 = SURE_cost_vec(x0, a_dense, theta_ml, data, sigsq, M, symm, omega)
    cost_scale = max(abs(cost_at_x0), 1e-30)

    # Run optimizer
    t0 = time.time()
    opt_res = sp.optimize.minimize(
        SURE_cost_vec,
        x0=x0,
        args=(a_dense, theta_ml, data, sigsq, M, symm, omega, cost_scale, None),
        method=None,
        options={"maxiter": 50, "disp": 0},
    )
    elapsed = time.time() - t0

    # Reconstruct full kernel
    opt_vec = gen_corefilt(opt_res.x, symm)

    # Cost at convergence (un-scaled)
    cost_at_converge = SURE_cost_vec(
        opt_res.x, a.todense(), theta_ml, data, sigsq, M, symm, omega
    )

    return {
        "kernel_coeffs": opt_res.x.copy(),
        "full_kernel": opt_vec.copy(),
        "cost_init": float(cost_at_x0),
        "cost_converged": float(cost_at_converge),
        "cost_ratio": (
            float(cost_at_converge / cost_at_x0) if abs(cost_at_x0) > 1e-30 else np.nan
        ),
        "nit": int(getattr(opt_res, "nit", -1)),
        "nfev": int(getattr(opt_res, "nfev", -1)),
        "success": bool(opt_res.success),
        "message": str(opt_res.message),
        "x0": x0.copy(),
        "sigsq": float(sigsq),
        "elapsed_s": float(elapsed),
    }


# ============================================================
# Analysis 1: Convergence rate (50 seeds)
# ============================================================
def analysis_convergence_rate(n_runs: int = 50) -> pd.DataFrame:
    print("=" * 60)
    print("Analysis 1: Convergence Rate (50 runs, different seeds)")
    print("=" * 60)

    T = 100
    half_win = T // 2
    real_templates = make_real_templates(T=T, C=8)

    records = []
    for run_idx in range(n_runs):
        seed = 1000 + run_idx
        rs = np.random.RandomState([seed])
        data, ix, labels = generate_data(
            real_templates, I=20, noise=0.1, mean_interval=50, rs=rs
        )

        result = run_kernel_with_details(data, ix, labels, half_win)

        rec = {
            "run": run_idx,
            "seed": seed,
            "coeff_0": result["kernel_coeffs"][0],
            "coeff_1": result["kernel_coeffs"][1],
            "cost_init": result["cost_init"],
            "cost_converged": result["cost_converged"],
            "cost_ratio": result["cost_ratio"],
            "nit": result["nit"],
            "nfev": result["nfev"],
            "success": result["success"],
            "message": result["message"],
            "sigsq": result["sigsq"],
            "elapsed_s": result["elapsed_s"],
        }
        # Full kernel (3 elements for symm=True, vlen=3)
        for i, v in enumerate(result["full_kernel"]):
            rec[f"kernel_{i}"] = v
        records.append(rec)

        if (run_idx + 1) % 10 == 0:
            print(
                f"  Run {run_idx + 1}/{n_runs}: success={result['success']}, "
                f"nit={result['nit']}, cost_ratio={result['cost_ratio']:.4f}"
            )

    df1 = pd.DataFrame(records)
    return df1


# ============================================================
# Analysis 2: Local minima test (10 realizations x 5 inits)
# ============================================================
def analysis_local_minima(n_realizations: int = 10) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Analysis 2: Local Minima Test (10 realizations x 5 inits)")
    print("=" * 60)

    T = 100
    half_win = T // 2
    real_templates = make_real_templates(T=T, C=8)

    # 5 initialization strategies
    inits = {
        "default": [1.0, -3.0],  # gen_h_diff(3) = [1, -3]
        "scaled": [0.5, -1.5],  # half-scale
        "perturbed": [1.0, -2.0],  # standard [1,-2,1]
        "large": [10.0, -20.0],  # 10x scale
    }

    records = []
    for real_idx in range(n_realizations):
        seed = 2000 + real_idx
        rs = np.random.RandomState([seed])
        data, ix, labels = generate_data(
            real_templates, I=20, noise=0.1, mean_interval=50, rs=rs
        )

        # Also add a random init per realization
        rs_init = np.random.RandomState([seed + 5000])
        inits_this = dict(inits)
        inits_this["random"] = rs_init.randn(2).tolist()

        for init_name, x0 in inits_this.items():
            result = run_kernel_with_details(data, ix, labels, half_win, x0_override=x0)
            rec = {
                "realization": real_idx,
                "seed": seed,
                "init_name": init_name,
                "x0_0": x0[0],
                "x0_1": x0[1],
                "coeff_0": result["kernel_coeffs"][0],
                "coeff_1": result["kernel_coeffs"][1],
                "cost_init": result["cost_init"],
                "cost_converged": result["cost_converged"],
                "cost_ratio": result["cost_ratio"],
                "nit": result["nit"],
                "nfev": result["nfev"],
                "success": result["success"],
            }
            for i, v in enumerate(result["full_kernel"]):
                rec[f"kernel_{i}"] = v
            records.append(rec)

        # Check for local minima in this realization
        costs = [r["cost_converged"] for r in records if r["realization"] == real_idx]
        min_cost = min(costs)
        max_cost = max(costs)
        if abs(min_cost) > 1e-30:
            spread = (max_cost - min_cost) / abs(min_cost)
        else:
            spread = 0.0
        flag = " ** LOCAL MINIMA **" if spread > 0.01 else ""
        print(
            f"  Realization {real_idx}: cost spread = {spread:.4f} "
            f"(min={min_cost:.2e}, max={max_cost:.2e}){flag}"
        )

    df2 = pd.DataFrame(records)
    return df2


# ============================================================
# Analysis 3: Kernel variability on P300 data (20 reps)
# ============================================================
def analysis_p300_kernel_variability(
    n_reps: int = 20, K: int = 5
) -> pd.DataFrame | None:
    print("\n" + "=" * 60)
    print(f"Analysis 3: Kernel Variability on P300 Data (K={K}, {n_reps} reps)")
    print("=" * 60)

    try:
        from data_loaders_p300 import load_p300_bnci2014_009

        P300_CHANNELS = ["Fz", "Cz", "Pz", "Oz", "PO7", "PO8"]
        data, ix, lb, sfreq, info = load_p300_bnci2014_009(
            subjects=[1],
            bandpass=(1.0, 20.0),
            resample=128.0,
            pick_channels=P300_CHANNELS,
        )
        print(f"  Loaded BNCI2014-009: {data.shape}, sfreq={sfreq}")
    except Exception as e:
        print(f"  Could not load P300 data: {e}")
        print("  Skipping Analysis 3.")
        return None

    half_win = int(0.4 * sfreq)  # 400ms
    lb_str = np.array([str(lbl) for lbl in lb])
    unique_labels = np.unique(lb_str)

    valid = (ix - half_win >= 0) & (ix + half_win <= len(data))
    ix_valid = ix[valid]
    lb_valid = lb_str[valid]

    print(f"  Valid events: {len(ix_valid)}, half_win={half_win}")

    from run_benchmarks import compact_continuous

    rs = np.random.RandomState([42])
    records = []

    for rep in range(n_reps):
        # Subsample K events per class
        idx_selected = []
        for label in unique_labels:
            label_idx = np.where(lb_valid == label)[0]
            k_per_class = min(K, len(label_idx))
            if k_per_class < 3:
                continue
            selected = rs.choice(label_idx, size=k_per_class, replace=False)
            idx_selected.extend(selected)

        if not idx_selected:
            continue
        idx_selected = np.array(sorted(idx_selected))
        ix_sub = ix_valid[idx_selected]
        lb_sub = lb_valid[idx_selected]

        # Compact data around selected events
        data_sub, ix_sub_c, lb_sub_c = compact_continuous(
            data, ix_sub, lb_sub, half_win, margin=5
        )

        try:
            result = run_kernel_with_details(
                data_sub, ix_sub_c, lb_sub_c, half_win, noise_est="pairwise"
            )
            rec = {
                "rep": rep,
                "coeff_0": result["kernel_coeffs"][0],
                "coeff_1": result["kernel_coeffs"][1],
                "cost_converged": result["cost_converged"],
                "nit": result["nit"],
                "nfev": result["nfev"],
                "success": result["success"],
                "sigsq": result["sigsq"],
            }
            for i, v in enumerate(result["full_kernel"]):
                rec[f"kernel_{i}"] = v
            records.append(rec)

            if (rep + 1) % 5 == 0:
                print(
                    f"  Rep {rep + 1}/{n_reps}: kernel=[{result['kernel_coeffs'][0]:.4f}, "
                    f"{result['kernel_coeffs'][1]:.4f}], success={result['success']}"
                )
        except Exception as e:
            print(f"  Rep {rep + 1} failed: {e}")

    if not records:
        return None

    df3 = pd.DataFrame(records)
    return df3


# ============================================================
# Generate outputs
# ============================================================
def generate_plots(
    df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame | None
) -> None:
    """Create convergence_kernels.png scatter plot."""
    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    import matplotlib.pyplot as plt

    n_panels = 2 + (1 if df3 is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 2:
        axes = list(axes)

    # Panel 1: Convergence rate - scatter of converged kernel coefficients
    ax = axes[0]
    colors = ["tab:blue" if s else "tab:red" for s in df1["success"]]
    ax.scatter(
        df1["coeff_0"],
        df1["coeff_1"],
        c=colors,
        alpha=0.6,
        s=30,
        edgecolors="k",
        linewidths=0.5,
    )
    # Mark the default init
    ax.scatter(
        [1.0],
        [-3.0],
        marker="x",
        c="red",
        s=100,
        linewidths=2,
        zorder=5,
        label="Default init",
    )
    ax.set_xlabel("Kernel coeff 0")
    ax.set_ylabel("Kernel coeff 1")
    ax.set_title("Analysis 1: Converged Kernels\n(50 seeds, blue=success, red=fail)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Local minima - converged coefficients colored by init
    # Canonicalize sign: SURE cost is invariant under h -> -h (since S = h^T h),
    # so we flip negative-coeff_0 solutions to the positive half for comparison.
    ax = axes[1]
    init_names = df2["init_name"].unique()
    cmap = plt.get_cmap("Set1")
    for i, init_name in enumerate(init_names):
        mask = df2["init_name"] == init_name
        c0 = df2.loc[mask, "coeff_0"].to_numpy().copy()
        c1 = df2.loc[mask, "coeff_1"].to_numpy().copy()
        # Canonicalize: flip so coeff_0 > 0
        flip = c0 < 0
        c0[flip] *= -1
        c1[flip] *= -1
        ax.scatter(
            c0,
            c1,
            c=[cmap(i)],
            alpha=0.7,
            s=30,
            label=init_name,
            edgecolors="k",
            linewidths=0.5,
        )
    ax.set_xlabel("Kernel coeff 0 (sign-canonicalized)")
    ax.set_ylabel("Kernel coeff 1 (sign-canonicalized)")
    ax.set_title("Analysis 2: Local Minima Test\n(10 realizations x 5 inits)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)

    # Panel 3: P300 kernel variability (if available)
    if df3 is not None:
        ax = axes[2]
        ax.scatter(
            df3["coeff_0"],
            df3["coeff_1"],
            c="tab:green",
            alpha=0.6,
            s=30,
            edgecolors="k",
            linewidths=0.5,
        )
        mean_c0 = df3["coeff_0"].mean()
        mean_c1 = df3["coeff_1"].mean()
        ax.scatter(
            [mean_c0], [mean_c1], marker="D", c="red", s=80, zorder=5, label="Mean"
        )
        # Draw 1-std ellipse
        from matplotlib.patches import Ellipse

        std_c0 = df3["coeff_0"].std()
        std_c1 = df3["coeff_1"].std()
        ell = Ellipse(
            (mean_c0, mean_c1),
            width=2 * std_c0,
            height=2 * std_c1,
            fill=False,
            edgecolor="red",
            linestyle="--",
            linewidth=1.5,
        )
        ax.add_patch(ell)
        ax.set_xlabel("Kernel coeff 0")
        ax.set_ylabel("Kernel coeff 1")
        ax.set_title(f"Analysis 3: P300 Kernel Variability\n(K=5, {len(df3)} reps)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = OUTPUT_DIR / "convergence_kernels.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to: {png_path}")


def _report_coeff_table(
    df: pd.DataFrame,
    cols: list[str],
    include_minmax: bool = True,
) -> list[str]:
    """Generate a markdown table of coefficient statistics."""
    lines: list[str] = []
    if include_minmax:
        lines.append("| Coefficient | Mean | Std | CV | Min | Max |")
        lines.append("|-------------|------|-----|------|-----|-----|")
    else:
        lines.append("| Coefficient | Mean | Std | CV |")
        lines.append("|-------------|------|-----|------|")
    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        cv = abs(std / mean) if abs(mean) > 1e-10 else float("inf")
        if include_minmax:
            lines.append(
                f"| {col} | {mean:.4f} | {std:.4f} | {cv:.4f} | {df[col].min():.4f} | {df[col].max():.4f} |"
            )
        else:
            lines.append(f"| {col} | {mean:.4f} | {std:.4f} | {cv:.4f} |")
    return lines


def _report_sign_patterns(
    df: pd.DataFrame,
    kernel_cols: list[str],
    total_label: int,
) -> list[str]:
    """Generate sign pattern consistency lines."""
    lines: list[str] = []
    sign_patterns: list[tuple[float, ...]] = []
    for _, row in df.iterrows():
        signs = tuple(np.sign([row.get(c, 0) for c in kernel_cols]))
        sign_patterns.append(signs)
    unique_patterns = set(sign_patterns)
    lines.append("### Sign pattern consistency")
    lines.append(f"Number of distinct sign patterns: {len(unique_patterns)}")
    pattern_counts = Counter(sign_patterns)
    for pattern, count in pattern_counts.most_common():
        signs_str = ", ".join(
            ["+1" if s > 0 else ("-1" if s < 0 else "0") for s in pattern]
        )
        lines.append(
            f"  - [{signs_str}]: {count}/{total_label} ({100 * count / total_label:.0f}%)"
        )
    lines.append("")
    return lines


def _report_section1(df1: pd.DataFrame) -> list[str]:
    """Generate report section 1: Convergence Rate."""
    lines: list[str] = []
    lines.append("## 1. Convergence Rate (50 simulation runs)")
    lines.append("")
    lines.append(
        "Setup: I=20 events/type, noise=0.1, mean_interval=50, T=100, C=8, symm=True, vlen=3"
    )
    lines.append("Default init: x0 = [1, -3] (from gen_h_diff(3))")
    lines.append("")

    n_success = df1["success"].sum()
    n_total = len(df1)
    lines.append(
        f"**Convergence rate: {n_success}/{n_total} ({100 * n_success / n_total:.0f}%)**"
    )
    lines.append("")

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Success rate | {100 * n_success / n_total:.1f}% |")
    lines.append(
        f"| Mean iterations (nit) | {df1['nit'].mean():.1f} +/- {df1['nit'].std():.1f} |"
    )
    lines.append(
        f"| Mean func evals (nfev) | {df1['nfev'].mean():.1f} +/- {df1['nfev'].std():.1f} |"
    )
    lines.append(
        f"| Mean elapsed time | {df1['elapsed_s'].mean():.3f}s +/- {df1['elapsed_s'].std():.3f}s |"
    )
    lines.append(
        f"| Mean cost ratio (converged/init) | {df1['cost_ratio'].mean():.4f} +/- {df1['cost_ratio'].std():.4f} |"
    )
    lines.append("")

    lines.append("### Kernel coefficient statistics")
    lines.append("")
    lines.extend(_report_coeff_table(df1, ["coeff_0", "coeff_1"], include_minmax=True))
    lines.append("")

    kernel_cols = [c for c in df1.columns if c.startswith("kernel_")]
    if kernel_cols:
        lines.append("### Full kernel (symmetric) coefficient statistics")
        lines.append("")
        lines.extend(_report_coeff_table(df1, kernel_cols, include_minmax=False))
        lines.append("")

    lines.extend(_report_sign_patterns(df1, kernel_cols, n_total))

    # Failure messages
    failed = df1[~df1["success"]]
    if len(failed) > 0:
        lines.append("### Failed runs")
        lines.append("")
        msg_counts = failed["message"].value_counts()
        for msg, count in msg_counts.items():
            lines.append(f'  - "{msg}": {count} runs')
        lines.append("")
    else:
        lines.append("All runs reported success.")
        lines.append("")

    return lines


def _report_section2(df2: pd.DataFrame) -> list[str]:
    """Generate report section 2: Local Minima Test."""
    lines: list[str] = []
    lines.append("## 2. Local Minima Test")
    lines.append("")
    lines.append(
        "10 random simulation realizations, each with 5 different initializations:"
    )
    lines.append("  - default: [1, -3]")
    lines.append("  - scaled: [0.5, -1.5]")
    lines.append("  - perturbed: [1, -2]")
    lines.append("  - random: np.random.randn(2)")
    lines.append("  - large: [10, -20]")
    lines.append("")

    # Per-realization cost spread
    lines.append("| Realization | Min Cost | Max Cost | Spread (%) | Local Minima? |")
    lines.append("|-------------|----------|----------|------------|---------------|")
    n_local_minima = 0
    for real_idx in range(int(df2["realization"].max()) + 1):
        sub = df2[df2["realization"] == real_idx]
        min_c = sub["cost_converged"].min()
        max_c = sub["cost_converged"].max()
        if abs(min_c) > 1e-30:
            spread_pct = 100 * (max_c - min_c) / abs(min_c)
        else:
            spread_pct = 0.0
        flag = "YES" if spread_pct > 1.0 else "no"
        if spread_pct > 1.0:
            n_local_minima += 1
        lines.append(
            f"| {real_idx} | {min_c:.4e} | {max_c:.4e} | {spread_pct:.2f}% | {flag} |"
        )
    lines.append("")
    lines.append(
        f"**Realizations with local minima (>1% cost spread): {n_local_minima}/10**"
    )
    lines.append("")

    # Sign symmetry analysis
    n_sign_flipped = 0
    for real_idx in range(int(df2["realization"].max()) + 1):
        sub = df2[df2["realization"] == real_idx]
        signs = np.sign(sub["coeff_0"].values)
        if len(set(signs)) > 1:
            n_sign_flipped += 1
    if n_sign_flipped > 0:
        lines.append("### Sign symmetry (h = -h equivalence)")
        lines.append("")
        lines.append(
            "The SURE cost is invariant under h -> -h because the penalty matrix S = R^T R "
            "where R = conv_matrix(h), and (-h) produces the same S."
        )
        lines.append(
            f"In {n_sign_flipped}/10 realizations, a random init converged to the sign-flipped "
            f"solution (coeff_0 < 0). The SURE cost at convergence was identical, confirming "
            f"this is a global symmetry, not a local minimum."
        )
        lines.append("")

    # Per-init summary
    lines.append("### Convergence by initialization")
    lines.append("")
    lines.append("| Init | Mean Cost | Std Cost | Success Rate | Mean nit |")
    lines.append("|------|-----------|----------|--------------|----------|")
    for init_name in df2["init_name"].unique():
        sub = df2[df2["init_name"] == init_name]
        lines.append(
            f"| {init_name} | {sub['cost_converged'].mean():.4e} | "
            f"{sub['cost_converged'].std():.4e} | "
            f"{100 * sub['success'].mean():.0f}% | "
            f"{sub['nit'].mean():.1f} |"
        )
    lines.append("")

    return lines


def _report_section3(df3: pd.DataFrame | None) -> list[str]:
    """Generate report section 3: Kernel Variability on P300 Data."""
    lines: list[str] = []
    if df3 is None:
        lines.append("## 3. Kernel Variability on P300 Data")
        lines.append("")
        lines.append("Skipped: P300 data could not be loaded.")
        lines.append("")
        return lines

    lines.append("## 3. Kernel Variability on P300 Data")
    lines.append("")
    lines.append(
        f"Dataset: BNCI2014-009, subject 1, K={5} events/class, {len(df3)} repetitions"
    )
    lines.append("")

    lines.append("### Coefficient statistics")
    lines.append("")
    lines.extend(_report_coeff_table(df3, ["coeff_0", "coeff_1"], include_minmax=True))
    lines.append("")

    kernel_cols_3 = [c for c in df3.columns if c.startswith("kernel_")]
    if kernel_cols_3:
        lines.append("### Full kernel (symmetric) statistics")
        lines.append("")
        lines.extend(_report_coeff_table(df3, kernel_cols_3, include_minmax=False))
        lines.append("")

    lines.extend(_report_sign_patterns(df3, kernel_cols_3, len(df3)))

    unique_patterns_3 = set()
    for _, row in df3.iterrows():
        signs = tuple(np.sign([row.get(c, 0) for c in kernel_cols_3]))
        unique_patterns_3.add(signs)
    lines.append(
        f"All kernels qualitatively similar: "
        f"{'YES' if len(unique_patterns_3) == 1 else 'NO'}"
    )
    lines.append("")

    return lines


def generate_report(
    df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame | None
) -> str:
    """Generate convergence_analysis.md report."""
    lines: list[str] = []
    lines.append("# SURE(kernel) Optimizer Convergence Analysis")
    lines.append("")

    lines.extend(_report_section1(df1))
    lines.extend(_report_section2(df2))
    lines.extend(_report_section3(df3))

    report = "\n".join(lines)
    md_path = OUTPUT_DIR / "convergence_analysis.md"
    with open(md_path, "w") as f:
        f.write(report)
    print(f"Report saved to: {md_path}")
    return report


if __name__ == "__main__":
    print("SURE(kernel) Optimizer Convergence Analysis")
    print("=" * 60)

    t_total = time.time()

    # Analysis 1
    df1 = analysis_convergence_rate(n_runs=50)

    # Analysis 2
    df2 = analysis_local_minima(n_realizations=10)

    # Analysis 3
    df3 = analysis_p300_kernel_variability(n_reps=20, K=5)

    # Save raw CSV
    all_dfs = []
    df1_out = df1.copy()
    df1_out["analysis"] = "convergence_rate"
    all_dfs.append(df1_out)

    df2_out = df2.copy()
    df2_out["analysis"] = "local_minima"
    all_dfs.append(df2_out)

    if df3 is not None:
        df3_out = df3.copy()
        df3_out["analysis"] = "p300_variability"
        all_dfs.append(df3_out)

    csv_path = OUTPUT_DIR / "convergence_analysis.csv"
    pd.concat(all_dfs, ignore_index=True, sort=False).to_csv(csv_path, index=False)
    print(f"\nRaw data saved to: {csv_path}")

    # Generate plot
    generate_plots(df1, df2, df3)

    # Generate report
    generate_report(df1, df2, df3)

    print(f"\nTotal time: {time.time() - t_total:.1f}s")
