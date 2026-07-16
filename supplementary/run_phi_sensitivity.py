# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

"""
Phi sensitivity analysis for SURE(kernel) on P300 data.

Tests how SURE(kernel) performance degrades when the AR(1) coefficient phi
is misspecified. The kernel is optimized once (white SURE), then the colored
SURE lambda optimization is re-run at each perturbed phi value. This isolates
the effect of phi on the noise-model-dependent lambda selection.

Uses BNCI2014-009 subject 1, K=5 events per class, 10 reps.
Matches the paper setup (Setting 1: template distance to full-data reference).

Methods:
  - SURE(kernel) with colored SURE at various phi perturbation factors
  - EmpBayes and white SURE(kernel) as phi-independent baselines

Outputs:
  supplementary/phi_sensitivity.csv
  supplementary/phi_sensitivity.png
  supplementary/phi_sensitivity.md
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make sibling modules (core.py, run_benchmarks.py, data_loaders*.py in the
# parent folder) importable when this script is run directly from supplementary/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core
from core import (
    build_DtSigmaD_ar1,
    est_noise_var_baseline,
    estimate_phi,
    gen_corefilt,
    get_design_matrix,
    get_events_templates,
)

# calc_template_distance and compact_continuous live in run_benchmarks.
from run_benchmarks import calc_template_distance, compact_continuous


OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
K = 5
N_REP = 10
SEED = 42
PHI_FACTORS = [0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]

# P300 channels (same as paper)
P300_CHANNELS = ["Fz", "Cz", "Pz", "Oz", "PO7", "PO8"]


# ---------------------------------------------------------------------------
# Fast colored SURE cost using precomputed eigendecomposition
# ---------------------------------------------------------------------------
def SURE_cost_colored_fast(
    regval: float,
    mu: np.ndarray,
    DtX_rot: np.ndarray,
    V: np.ndarray,
    theta_ml: np.ndarray,
    Q: np.ndarray,
    A_sym: np.ndarray,
    n_obs: int,
    n_cols: int,
    omega: float = 0.5,
) -> float:
    """Fast colored SURE cost avoiding per-evaluation np.linalg.solve.

    Uses precomputed:
      Q = V^T V  (V is B-orthonormal, NOT orthonormal)
      A_sym = V^T DtSigmaD V

    The trace term tr(M @ DtSigmaD) where M = DtD^{-1}(DtD+lam*S)^{-1}
    is computed as: sum_{ij} (1/mu_i) * Q_{ij} * g_j * A_sym_{ji}
    where g_j = 1/(mu_j + lam*(1-mu_j)).

    This is O(p^2) per evaluation instead of O(p^3) from np.linalg.solve.
    """
    g = 1.0 / (mu + regval * (1 - mu) + 1e-30)

    # Phi_hat = V @ diag(g) @ DtX_rot
    Phi_hat = V @ (g[:, None] * DtX_rot)
    Phi_hat_norm_sq = np.sum(Phi_hat**2)

    # tr(M @ DtSigmaD) = sum_{ij} (1/mu_i) Q_{ij} g_j A_{ji}
    # = sum_i (1/mu_i) sum_j Q_{ij} g_j A_{ji}
    # = sum_i (1/mu_i) [Q[i,:] * g * A_sym[:,i]]
    # Vectorized: (1/mu) @ ((Q * A_sym.T) @ g)
    inv_mu = 1.0 / (mu + 1e-30)
    QA = Q * A_sym.T  # element-wise multiply, shape (p, p)
    trace_term = 2 * np.dot(inv_mu, QA @ g)

    # Cross term
    cross_term = np.sum(theta_ml * Phi_hat)

    # G_correction for omega != 0.5
    G_correction = 0
    if abs(omega - 0.5) > 1e-5:
        V_col_norms_sq = np.sum(V**2, axis=0)
        GG_trace = np.sum(g**2 * V_col_norms_sq)
        sigsq_approx = np.trace(Q @ np.diag(mu)) / len(mu)  # rough estimate
        G_correction = (2 * omega - 1) * sigsq_approx * GG_trace

    norm_vec_sum = (
        omega * (Phi_hat_norm_sq + trace_term * n_cols) - G_correction * n_cols
    )
    cost = norm_vec_sum - omega * 2 * cross_term

    return cost


def sure_kernel_with_phi(
    data: np.ndarray,
    ix: np.ndarray,
    lb: np.ndarray,
    half_win: int,
    phi: float,
    sigsq: float,
    vlen: int = 3,
    symm: bool = True,
    omega: float = 0.5,
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    """Run SURE(kernel) with a specific phi for the colored noise model.

    Two-stage approach:
      1. Optimize kernel via white SURE (fast, noise-model independent)
      2. Re-optimize lambda via colored SURE with the specified phi

    Stage 2 uses the fast colored SURE that avoids per-evaluation linalg.solve.
    """
    import scipy as sp

    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    from scipy.optimize import minimize_scalar

    events = np.unique(lb)
    M = len(events)
    T = 2 * half_win
    C = data.shape[1]

    # Build design matrix and normal equations
    design = get_design_matrix(data, ix, lb, half_win=half_win, events=events)
    design = sp.sparse.csr_matrix(design)
    DtD = design.T @ design
    DtX = design.T @ data

    DtD_dense = np.asarray(DtD.todense(), dtype=float)
    DtX_dense = np.asarray(DtX, dtype=float)
    theta_ml = np.linalg.solve(DtD_dense, DtX_dense)

    # --- Stage 1: Optimize kernel via white SURE (fast) ---
    def gen_h_diff(n: int) -> np.ndarray:
        h_diff: np.ndarray = np.array([1])
        for _ii in range(n):
            h_diff = np.convolve(h_diff, [1, -1])
        return h_diff

    x0 = gen_h_diff(vlen)
    x0 = x0[: int(vlen - ((vlen - 1) / 2) * int(symm))]

    cost_at_x0 = core.SURE_cost_vec(
        x0, DtD.todense(), theta_ml, data, sigsq, M, symm, omega
    )
    cost_scale = max(abs(cost_at_x0), 1e-30)

    opt_res = sp.optimize.minimize(
        core.SURE_cost_vec,
        x0=x0,
        args=(DtD.todense(), theta_ml, data, sigsq, M, symm, omega, cost_scale, None),
        method=None,
        options={"maxiter": 50, "disp": 0},
    )

    opt_vec = gen_corefilt(opt_res.x, symm)

    # Build S from optimized kernel
    R = sp.linalg.convolution_matrix(opt_vec, T - len(opt_vec) + 1)
    RR = R @ R.T
    S_dense = sp.linalg.block_diag(*([RR] * M)).astype(float)

    # --- Stage 2: Optimize lambda via colored SURE with specified phi ---
    DtSigmaD = build_DtSigmaD_ar1(
        ix, lb, events, half_win, phi, sigsq, data_length=len(data)
    )

    mu, V = sp.linalg.eigh(DtD_dense, DtD_dense + S_dense)
    mu = np.clip(mu, 0, 1)
    DtX_rot = V.T @ DtX_dense

    # Precompute Q and A_sym for fast colored SURE
    Q = V.T @ V  # p x p
    A_sym = V.T @ DtSigmaD @ V  # p x p

    n_obs = data.shape[0]
    n_cols = C

    opt_lam = minimize_scalar(
        SURE_cost_colored_fast,
        bounds=(0, 1e5),
        method="bounded",
        args=(mu, DtX_rot, V, theta_ml, Q, A_sym, n_obs, n_cols, omega),
        options={"xatol": 1e-02, "maxiter": 25, "disp": 0},
    )

    reg_temporal = opt_lam.x
    temporal_mat = sp.sparse.csr_matrix(S_dense)

    # Build final templates
    templates = sp.sparse.linalg.spsolve(DtD + reg_temporal * temporal_mat, DtX).T
    templates = templates.reshape(C, M, T).transpose(1, 2, 0)

    return (
        {str(ev): templates[ii] for ii, ev in enumerate(events)},
        opt_vec,
        reg_temporal,
    )


def load_p300_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Load P300 dataset, compact, and return."""
    from data_loaders_p300 import load_p300_bnci2014_009

    data, ix, lb, sfreq, info = load_p300_bnci2014_009(
        subjects=[1],
        bandpass=(1.0, 20.0),
        resample=128.0,
        pick_channels=P300_CHANNELS,
    )
    lb_str = np.array([str(lbl) for lbl in lb])
    half_win = int(0.4 * sfreq)

    data, ix, lb_str = compact_continuous(data, ix, lb_str, half_win, margin=10)

    valid = (ix - half_win >= 0) & (ix + half_win <= len(data))
    ix = ix[valid]
    lb_str = lb_str[valid]

    print(
        f"P300 data: {data.shape}, {len(ix)} events, sfreq={sfreq}, half_win={half_win}"
    )
    return data, ix, lb_str, sfreq, half_win


def compute_reference(
    data: np.ndarray, ix: np.ndarray, lb: np.ndarray, half_win: int
) -> dict[str, np.ndarray]:
    """Full-data reference template (SURE(lambda) with all events)."""
    ref, _ = get_events_templates(
        data,
        ix,
        lb,
        half_win=half_win,
        method="ls",
        reg_temporal=500,
        opt_reg_temporal=True,
        omega=0.5,
    )
    return {str(k): v for k, v in ref.items()}


def main() -> tuple[pd.DataFrame, float]:
    t_start = time.time()

    # --- Load data ---
    data, ix, lb, sfreq, half_win = load_p300_data()
    unique_labels = np.unique(lb)

    # --- Estimate true phi from baselines ---
    true_phi = estimate_phi(data, ix, half_win)
    print(f"True phi (from baselines): {true_phi:.4f}")

    # --- Compute full-data reference ---
    print("Computing full-data reference template...")
    ref = compute_reference(data, ix, lb, half_win)
    print("  Reference computed.")

    # --- Subsample and evaluate ---
    rs = np.random.RandomState([SEED])
    results = []

    # Baseline method configs
    empbayes_kwargs = {
        "method": "ls",
        "reg_method": "empbayes",
    }
    kernel_white_kwargs = {
        "method": "ls",
        "reg_temporal": 500,
        "opt_reg_temporal": True,
        "vec_mode": True,
        "symm": True,
        "vlen": 3,
        "omega": 0.5,
        "noise_est": "pairwise",
    }

    for rep in range(N_REP):
        t_rep = time.time()

        # Subsample K events per class
        idx_selected = []
        for label in unique_labels:
            label_idx = np.where(lb == label)[0]
            k_per_class = min(K, len(label_idx))
            if k_per_class < 3:
                continue
            selected = rs.choice(label_idx, size=k_per_class, replace=False)
            idx_selected.extend(selected)

        if not idx_selected:
            continue
        idx_selected = np.array(sorted(idx_selected))
        ix_sub = ix[idx_selected]
        lb_sub = lb[idx_selected]

        # Compact
        data_sub, ix_sub_c, lb_sub_c = compact_continuous(
            data,
            ix_sub,
            lb_sub,
            half_win,
            margin=5,
        )

        # --- EmpBayes baseline (phi-independent) ---
        try:
            t, _ = get_events_templates(
                data_sub,
                ix_sub_c,
                lb_sub_c,
                half_win=half_win,
                **empbayes_kwargs,
            )
            dist = calc_template_distance(t, ref)
            results.append(
                {
                    "phi_factor": np.nan,
                    "phi_used": np.nan,
                    "method": "EmpBayes",
                    "distance": dist,
                    "rep": rep,
                }
            )
        except Exception as e:
            print(f"  EmpBayes rep {rep} failed: {e}")

        # --- White SURE(kernel) baseline ---
        try:
            t, _ = get_events_templates(
                data_sub,
                ix_sub_c,
                lb_sub_c,
                half_win=half_win,
                **kernel_white_kwargs,
            )
            dist = calc_template_distance(t, ref)
            results.append(
                {
                    "phi_factor": np.nan,
                    "phi_used": np.nan,
                    "method": "SURE(kernel) white",
                    "distance": dist,
                    "rep": rep,
                }
            )
        except Exception as e:
            print(f"  SURE(kernel) white rep {rep} failed: {e}")

        # Base sigma^2 for colored runs
        sigsq_base = est_noise_var_baseline(data_sub, ix_sub_c, half_win)

        # --- SURE(kernel) colored at various phi values ---
        for factor in PHI_FACTORS:
            phi_used = np.clip(true_phi * factor, 0.0, 0.99)

            try:
                templates, opt_vec, opt_lam = sure_kernel_with_phi(
                    data_sub,
                    ix_sub_c,
                    lb_sub_c,
                    half_win,
                    phi_used,
                    sigsq_base,
                    vlen=3,
                    symm=True,
                    omega=0.5,
                )
                dist = calc_template_distance(templates, ref)
                results.append(
                    {
                        "phi_factor": factor,
                        "phi_used": phi_used,
                        "method": "SURE(kernel) colored",
                        "distance": dist,
                        "rep": rep,
                    }
                )
            except Exception as e:
                print(f"  Colored factor={factor} rep {rep} failed: {e}")

        print(f"  Rep {rep + 1}/{N_REP} done ({time.time() - t_rep:.0f}s)", flush=True)

    # --- Save CSV ---
    df = pd.DataFrame(results)
    csv_path = OUT_DIR / "phi_sensitivity.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(f"Total rows: {len(df)}")

    # --- Summary ---
    print(f"\nTrue phi: {true_phi:.4f}")
    colored = df[df["method"] == "SURE(kernel) colored"]
    if len(colored) > 0:
        summary = colored.groupby("phi_factor")["distance"].agg(["mean", "std"])
        print("\nSURE(kernel) colored distance by phi factor:")
        print(summary.to_string(float_format=lambda x: f"{x:.6f}"))

    baselines = df[df["method"].isin(["EmpBayes", "SURE(kernel) white"])]
    if len(baselines) > 0:
        bl_summary = baselines.groupby("method")["distance"].agg(["mean", "std"])
        print("\nBaseline distances:")
        print(bl_summary.to_string(float_format=lambda x: f"{x:.6f}"))

    print(f"\nTotal time: {time.time() - t_start:.0f}s")
    return df, true_phi


if __name__ == "__main__":
    df, true_phi = main()
