# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

"""
Final2 benchmarks: definitive runs with the fixed-lambda Reg. LS baseline dropped.

Methods:
- Simulation: METHODS_PAIRWISE (white noise)
- P300, EEG: METHODS_AR1COL (AR1 colored noise)
- ECG: METHODS_PAIRWISE (baselines contaminated)

Results saved with _final2 suffix.
"""

import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from core import calculate_snr, generate_data, get_events_templates, ricker
from data_loaders import load_eeg_erp
from data_loaders_ecg import download_ecg_mitbih, load_ecg_mitbih
from data_loaders_p300 import load_p300_bnci2014_009
from numpy.typing import NDArray

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ===========================================================================
# METHOD DEFINITIONS
# ===========================================================================

METHODS_PAIRWISE: dict[str, dict[str, Any]] = {
    "Average": {"method": "average"},
    "OLS": {"method": "ls"},
    "GCV": {"method": "ls", "reg_method": "gcv"},
    "L-curve": {"method": "ls", "reg_method": "lcurve"},
    "EmpBayes": {"method": "ls", "reg_method": "empbayes"},
    "SURE(lambda)": {
        "method": "ls",
        "reg_temporal": 500,
        "opt_reg_temporal": True,
        "noise_est": "pairwise",
    },
    "SURE(kernel)": {
        "method": "ls",
        "reg_temporal": 500,
        "opt_reg_temporal": True,
        "vec_mode": True,
        "symm": True,
        "vlen": 3,
        "noise_est": "pairwise",
    },
}

METHODS_AR1COL: dict[str, dict[str, Any]] = {
    "Average": {"method": "average"},
    "OLS": {"method": "ls"},
    "GCV": {"method": "ls", "reg_method": "gcv"},
    "L-curve": {"method": "ls", "reg_method": "lcurve"},
    "EmpBayes": {"method": "ls", "reg_method": "empbayes"},
    "SURE(lambda)": {
        "method": "ls",
        "reg_temporal": 500,
        "opt_reg_temporal": True,
        "noise_est": "ar1_colored",
    },
    "SURE(kernel)": {
        "method": "ls",
        "reg_temporal": 500,
        "opt_reg_temporal": True,
        "vec_mode": True,
        "symm": True,
        "vlen": 3,
        "noise_est": "ar1_colored",
    },
}


# ===========================================================================
# HELPERS
# ===========================================================================


def make_real_templates(T: int = 100, C: int = 8) -> dict[str, NDArray]:
    return {
        "Template A": np.outer(ricker(T, 15), np.roll(np.hanning(C), 3)),
        "Template B": np.outer(ricker(T, 10), np.hanning(C)),
    }


def calc_template_distance(t1: dict[str, NDArray], t2: dict[str, NDArray]) -> float:
    d = 0
    n = 0
    for k in t1:
        if k in t2:
            d += np.linalg.norm(t1[k] - t2[k])
            n += 1
    return d / max(n, 1)


def compact_continuous(
    data: NDArray,
    ix: NDArray,
    lb: NDArray,
    half_win: int,
    margin: int = 10,
) -> tuple[NDArray, NDArray, NDArray]:
    """Crop a long continuous recording to just the segments around events.

    For each event we keep the window ``[idx - half_win - margin,
    idx + half_win + margin)`` (clipped to the signal bounds) and concatenate
    the kept windows into a shorter continuous signal. Event indices are
    remapped to point at the same samples in the compacted signal; labels are
    returned unchanged. This is used to speed up estimation after subsampling a
    handful of events from a large recording, without changing the
    ``2 * half_win`` template window itself.

    Parameters
    ----------
    data : NDArray
        Continuous data of shape ``(T, C)``.
    ix : NDArray
        1-D integer array of event-onset indices.
    lb : NDArray
        1-D array of event labels (returned unchanged).
    half_win : int
        Half-width of the template window.
    margin : int
        Extra samples kept on each side of the template window.

    Returns
    -------
    data_c : NDArray
        Compacted continuous data.
    ix_c : NDArray
        Event indices into ``data_c``.
    lb : NDArray
        Labels (unchanged).
    """
    seg = half_win + margin
    chunks: list[NDArray] = []
    new_ix: list[int] = []
    offset = 0
    for idx in ix:
        start = max(0, int(idx) - seg)
        stop = min(len(data), int(idx) + seg)
        chunk = data[start:stop]
        chunks.append(chunk)
        new_ix.append(offset + (int(idx) - start))
        offset += chunk.shape[0]

    data_c = np.vstack(chunks) if chunks else data[:0]
    return data_c, np.array(new_ix, dtype=int), np.asarray(lb)


def run_method_safe(
    data: NDArray,
    ix: NDArray,
    lb: NDArray,
    half_win: int,
    method_name: str,
    method_kwargs: dict[str, Any],
) -> tuple[dict[str, NDArray] | None, dict[str, object] | None, str, float]:
    try:
        t0 = time.time()
        t, inner = get_events_templates(
            data, ix, lb, half_win=half_win, **method_kwargs
        )
        elapsed = time.time() - t0
        lam = ""
        if inner and "opt_val" in inner:
            v = inner["opt_val"]
            lam = (
                float(v)
                if not isinstance(v, np.ndarray)
                else str(np.round(v, 2).tolist())
            )
    except Exception as e:
        print(f"    Warning: {method_name}: {e}")
        traceback.print_exc()
        return None, None, "", 0
    else:
        return t, inner, lam, elapsed


# ===========================================================================
# 1. SIMULATION (METHODS_PAIRWISE)
# ===========================================================================


def run_sim_noise_sweep(
    real_templates: dict[str, NDArray],
    half_win: int,
    N_rep: int = 20,
    I: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    noises = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    results = []
    rs = np.random.RandomState(seed)

    for ni, noise in enumerate(noises):
        for kk in range(N_rep):
            data, ix, labels = generate_data(
                real_templates, I=I, noise=noise, mean_interval=40, rs=rs
            )

            for method_name, method_kwargs in METHODS_PAIRWISE.items():
                try:
                    templates, inner = get_events_templates(
                        data, ix, labels, half_win=half_win, **method_kwargs
                    )
                    snr = calculate_snr(templates, real_templates)
                    noise_var = ""
                    if inner and "noise_var" in inner:
                        noise_var = inner["noise_var"]
                    results.append(
                        {
                            "Noise": noise,
                            "SNR [dB]": snr,
                            "method": method_name,
                            "repetition": kk,
                            "noise_var_est": noise_var,
                        }
                    )
                except Exception as e:
                    print(
                        f"  Warning: noise_sweep {method_name} noise={noise:.3f} rep={kk}: {e}"
                    )
        print(f"  Noise level {ni + 1}/{len(noises)}: {noise:.4f} done", flush=True)

    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "sim_final2_noise.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Noise sweep: {len(df)} rows -> {csv_path}")
    return df


def run_sim_nevents_sweep(
    real_templates: dict[str, NDArray], half_win: int, N_rep: int = 20, seed: int = 42
) -> pd.DataFrame:
    Ns = [4, 8, 15, 30, 50, 100]
    results = []
    rs = np.random.RandomState(seed)

    for ii, I in enumerate(Ns):
        for kk in range(N_rep):
            data, ix, labels = generate_data(
                real_templates, I=I, noise=0.1, mean_interval=50, rs=rs
            )

            for method_name, method_kwargs in METHODS_PAIRWISE.items():
                try:
                    templates, inner = get_events_templates(
                        data, ix, labels, half_win=half_win, **method_kwargs
                    )
                    snr = calculate_snr(templates, real_templates)
                    results.append(
                        {
                            "N_events": int(I),
                            "SNR [dB]": snr,
                            "method": method_name,
                            "repetition": kk,
                        }
                    )
                except Exception as e:
                    print(f"  Warning: nevents {method_name} I={I} rep={kk}: {e}")
        print(f"  N_events {ii + 1}/{len(Ns)}: I={I} done", flush=True)

    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "sim_final2_nevents.csv"
    df.to_csv(csv_path, index=False)
    print(f"  N-events sweep: {len(df)} rows -> {csv_path}")
    return df


# ===========================================================================
# 2. P300 BCI (METHODS_AR1COL)
# ===========================================================================


def run_p300_benchmark() -> pd.DataFrame:
    print("\n--- Loading P300 data (BNCI2014-009, subject 1) ---", flush=True)
    data, ix, lb, sfreq, info = load_p300_bnci2014_009(
        subjects=[1], bandpass=(1.0, 20.0), resample=128.0
    )

    lb_str = np.array([str(lbl) for lbl in lb])
    half_win = int(0.4 * sfreq)
    valid = (ix - half_win >= 0) & (ix + half_win <= len(data))
    ix_v = ix[valid]
    lb_v = lb_str[valid]
    unique_labels = np.unique(lb_v)

    print(f"  Data: {data.shape}, sfreq={sfreq}, half_win={half_win}")
    print(
        f"  Events: {len(ix_v)} (target={info['n_target']}, non-target={info['n_nontarget']})"
    )
    print(f"  Channels: {len(info['ch_names'])}")

    # Reference: EmpBayes from ALL events
    print("  Computing reference (EmpBayes from all events)...", flush=True)
    ref, _ = get_events_templates(
        data, ix_v, lb_v, half_win=half_win, method="ls", reg_method="empbayes"
    )

    Ks = [5, 10, 20, 50]
    N_rep = 20
    results = []

    for K in Ks:
        t0 = time.time()
        rs = np.random.RandomState(42)
        for rep in range(N_rep):
            idx_sel = []
            for label in unique_labels:
                label_idx = np.where(lb_v == label)[0]
                idx_sel.extend(
                    rs.choice(label_idx, size=min(K, len(label_idx)), replace=False)
                )
            idx_sel = np.array(sorted(idx_sel))

            for name, kw in METHODS_AR1COL.items():
                t, inner, lam, elapsed = run_method_safe(
                    data, ix_v[idx_sel], lb_v[idx_sel], half_win, name, kw
                )
                if t is not None:
                    dist = calc_template_distance(t, ref)
                    results.append(
                        dict(K=K, distance=dist, method=name, rep=rep, lam=lam)
                    )
        print(f"  K={K} done ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "p300_final2.csv"
    df.to_csv(csv_path, index=False)

    if len(df) > 0:
        pivot = df.pivot_table(
            index="method", columns="K", values="distance", aggfunc="mean"
        )
        print("\n  P300 - Template distance (lower is better):")
        print(pivot.to_string(float_format=lambda x: f"{x:.2e}"))

    return df


# ===========================================================================
# 3. EEG (METHODS_AR1COL)
# ===========================================================================


def run_eeg_benchmark() -> pd.DataFrame:
    print("\n--- Loading EEG data (MNE sample, auditory L/R) ---", flush=True)
    eeg_data, eeg_ix, eeg_lb = load_eeg_erp(
        event_ids={"auditory_left": 1, "auditory_right": 2}, bandpass=(1.0, 40.0)
    )

    label_map = {1: "aud_L", 2: "aud_R"}
    eeg_lb_str = np.array([label_map[lbl] for lbl in eeg_lb])
    sfreq = 600.615

    half_win = int(0.3 * sfreq)
    label_names = np.unique(eeg_lb_str)

    print(
        f"  EEG: {eeg_data.shape}, {len(eeg_ix)} events, sfreq={sfreq:.1f}, half_win={half_win}"
    )

    # Reference: EmpBayes from all events
    print("  Computing reference (EmpBayes from all events)...", flush=True)
    ref, _ = get_events_templates(
        eeg_data,
        eeg_ix,
        eeg_lb_str,
        half_win=half_win,
        method="ls",
        reg_method="empbayes",
    )

    Ks = [5, 10, 20, 50]
    N_rep_default = 20
    N_rep_kernel = 10  # SURE(kernel) with ar1_colored takes ~60s/call on 60ch EEG
    results = []

    # Split methods
    METHODS_FAST = {k: v for k, v in METHODS_AR1COL.items() if k != "SURE(kernel)"}
    METHODS_SLOW = {"SURE(kernel)": METHODS_AR1COL["SURE(kernel)"]}

    for K in Ks:
        # Fast methods
        t0 = time.time()
        rs = np.random.RandomState(42)
        for rep in range(N_rep_default):
            idx_sel = []
            for lbl in label_names:
                label_idx = np.where(eeg_lb_str == lbl)[0]
                idx_sel.extend(
                    rs.choice(label_idx, size=min(K, len(label_idx)), replace=False)
                )
            idx_sel = np.array(sorted(idx_sel))

            for name, kw in METHODS_FAST.items():
                t, inner, lam, elapsed = run_method_safe(
                    eeg_data, eeg_ix[idx_sel], eeg_lb_str[idx_sel], half_win, name, kw
                )
                if t is not None:
                    dist = calc_template_distance(t, ref)
                    results.append(
                        dict(K=K, distance=dist, method=name, rep=rep, lam=lam)
                    )
        print(f"  K={K} fast methods done ({time.time() - t0:.0f}s)", flush=True)

        # SURE(kernel): fewer reps
        t0 = time.time()
        rs_slow = np.random.RandomState(42)
        for rep in range(N_rep_kernel):
            idx_sel = []
            for lbl in label_names:
                label_idx = np.where(eeg_lb_str == lbl)[0]
                idx_sel.extend(
                    rs_slow.choice(
                        label_idx, size=min(K, len(label_idx)), replace=False
                    )
                )
            idx_sel = np.array(sorted(idx_sel))

            for name, kw in METHODS_SLOW.items():
                t, inner, lam, elapsed = run_method_safe(
                    eeg_data, eeg_ix[idx_sel], eeg_lb_str[idx_sel], half_win, name, kw
                )
                if t is not None:
                    dist = calc_template_distance(t, ref)
                    results.append(
                        dict(K=K, distance=dist, method=name, rep=rep, lam=lam)
                    )
                    print(
                        f"    SURE(kernel) K={K} rep={rep}: dist={dist:.4f} ({elapsed:.0f}s)",
                        flush=True,
                    )
        print(f"  K={K} SURE(kernel) done ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "eeg_final2.csv"
    df.to_csv(csv_path, index=False)

    if len(df) > 0:
        pivot = df.pivot_table(
            index="method", columns="K", values="distance", aggfunc="mean"
        )
        method_order = [
            m
            for m in [
                "Average",
                "OLS",
                "GCV",
                "L-curve",
                "EmpBayes",
                "SURE(lambda)",
                "SURE(kernel)",
            ]
            if m in pivot.index
        ]
        pivot = pivot.reindex(method_order)
        print("\n  EEG - Template distance (lower is better):")
        print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    return df


# ===========================================================================
# 4. ECG (METHODS_PAIRWISE)
# ===========================================================================


def run_ecg_benchmark() -> pd.DataFrame:
    records = ["119"]
    print("\n--- Downloading MIT-BIH ---")
    download_ecg_mitbih(records=records)

    print("\n--- Loading ECG (record 119, Normal + PVC) ---")
    data, ix, lb, sfreq = load_ecg_mitbih(
        records=records,
        lead=0,
        bandpass=(0.5, 40.0),
        beat_types=["N", "V"],
    )

    label_map = {0: "Normal", 1: "PVC"}
    lb_str = np.array([label_map.get(lbl, f"type_{lbl}") for lbl in lb])

    half_win = int(0.35 * sfreq)  # ~126 samples at 360 Hz
    valid = (ix - half_win >= 0) & (ix + half_win <= len(data))
    ix = ix[valid]
    lb_str = lb_str[valid]

    print(f"  Data: {data.shape}, sfreq={sfreq}, half_win={half_win}")
    for lbl in np.unique(lb_str):
        print(f"    {lbl}: {np.sum(lb_str == lbl)} beats")

    unique_labels = np.unique(lb_str)
    counts = {lbl: np.sum(lb_str == lbl) for lbl in unique_labels}

    # Reference: SURE(lambda) (pairwise) from all events
    print(
        f"  Computing reference templates (SURE(lambda) from all {len(ix)} events)...",
        flush=True,
    )
    t0 = time.time()
    ref, _ = get_events_templates(
        data,
        ix,
        lb_str,
        half_win=half_win,
        method="ls",
        reg_temporal=500,
        opt_reg_temporal=True,
        noise_est="pairwise",
    )
    print(f"  Reference computed ({time.time() - t0:.0f}s)", flush=True)

    METHODS_FAST = {k: v for k, v in METHODS_PAIRWISE.items() if k != "SURE(kernel)"}
    METHODS_SLOW = {"SURE(kernel)": METHODS_PAIRWISE["SURE(kernel)"]}

    Ks = [5, 10, 25]
    N_rep_fast = 20
    N_rep_slow = 5
    seed = 42
    results = []

    for K in Ks:
        min_available = min(counts.values())
        if K > min_available:
            print(f"    K={K} > min class count ({min_available}), skipping")
            continue

        # Fast methods
        t0 = time.time()
        rs = np.random.RandomState(seed)
        for rep in range(N_rep_fast):
            idx_selected = []
            for label in unique_labels:
                label_idx = np.where(lb_str == label)[0]
                selected = rs.choice(label_idx, size=K, replace=False)
                idx_selected.extend(selected)
            idx_selected = np.array(sorted(idx_selected))
            ix_sub = ix[idx_selected]
            lb_sub = lb_str[idx_selected]

            for method_name, method_kwargs in METHODS_FAST.items():
                t, inner, lam, elapsed = run_method_safe(
                    data, ix_sub, lb_sub, half_win, method_name, method_kwargs
                )
                if t is not None:
                    dist = calc_template_distance(t, ref)
                    results.append(
                        {
                            "K_per_class": K,
                            "distance": dist,
                            "method": method_name,
                            "rep": rep,
                            "lambda": lam,
                        }
                    )
        print(f"    K={K} fast methods: done ({time.time() - t0:.0f}s)", flush=True)

        # SURE(kernel): fewer reps
        t0 = time.time()
        rs_slow = np.random.RandomState(seed)
        for rep in range(N_rep_slow):
            idx_selected = []
            for label in unique_labels:
                label_idx = np.where(lb_str == label)[0]
                selected = rs_slow.choice(label_idx, size=K, replace=False)
                idx_selected.extend(selected)
            idx_selected = np.array(sorted(idx_selected))
            ix_sub = ix[idx_selected]
            lb_sub = lb_str[idx_selected]

            for method_name, method_kwargs in METHODS_SLOW.items():
                t, inner, lam, elapsed = run_method_safe(
                    data, ix_sub, lb_sub, half_win, method_name, method_kwargs
                )
                if t is not None:
                    dist = calc_template_distance(t, ref)
                    results.append(
                        {
                            "K_per_class": K,
                            "distance": dist,
                            "method": method_name,
                            "rep": rep,
                            "lambda": lam,
                        }
                    )
                    print(
                        f"      SURE(kernel) K={K} rep={rep}: dist={dist:.6f} ({elapsed:.0f}s)",
                        flush=True,
                    )
        print(f"    K={K} SURE(kernel): done ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "ecg_final2.csv"
    df.to_csv(csv_path, index=False)

    if len(df) > 0:
        pivot = df.pivot_table(
            index="method", columns="K_per_class", values="distance", aggfunc="mean"
        )
        print("\n  ECG - Template distance (lower is better):")
        print(pivot.to_string(float_format=lambda x: f"{x:.6f}"))

    return df


# ===========================================================================
# SUMMARY
# ===========================================================================


def _print_dataset_block(
    df: pd.DataFrame,
    k_col: str,
    metric_col: str,
    header: str,
    fmt: str,
    method_order: list[str],
) -> None:
    """Print one dataset summary block (pivot table at lowest K)."""
    k_val = df[k_col].min()
    sub = df[df[k_col] == k_val]
    print(f"\n  {header}")
    pivot = sub.groupby("method")[metric_col].agg(["mean", "std"])
    for m in method_order:
        if m in pivot.index:
            r = pivot.loc[m]
            print(f"    {m:12s}: {r['mean']:{fmt}} +/- {r['std']:{fmt}}")


def _print_kernel_vs_empbayes(
    name: str,
    df: pd.DataFrame | None,
    k_col: str,
    metric_col: str,
    direction: str,
) -> None:
    """Print SURE(kernel) vs EmpBayes verification for one dataset."""
    if df is None or len(df) == 0:
        return
    k_val = df[k_col].min()
    sub = df[df[k_col] == k_val]
    means = sub.groupby("method")[metric_col].mean()
    kernel_val = means.get("SURE(kernel)", None)
    eb_val = means.get("EmpBayes", None)
    if kernel_val is None or eb_val is None:
        return
    if direction == "higher":
        improvement = kernel_val - eb_val
        wins = "SURE(kernel)" if kernel_val > eb_val else "EmpBayes"
        detail = f"SURE(kernel)={kernel_val:.4f}, EB={eb_val:.4f}, delta={improvement:+.2f} dB"
    else:
        improvement = (eb_val - kernel_val) / eb_val * 100
        wins = "SURE(kernel)" if kernel_val < eb_val else "EmpBayes"
        detail = f"SURE(kernel)={kernel_val:.6f}, EB={eb_val:.6f}, improvement={improvement:+.1f}%"
    status = "OK" if wins == "SURE(kernel)" else "FAILED"
    print(f"  {name:25s}: {detail} -> {wins} [{status}]")


def print_summary(
    df_sim_noise: pd.DataFrame | None,
    df_sim_nevents: pd.DataFrame | None,
    df_p300: pd.DataFrame | None,
    df_ecg: pd.DataFrame | None,
    df_eeg: pd.DataFrame | None,
) -> None:
    method_order = [
        "Average",
        "OLS",
        "GCV",
        "L-curve",
        "EmpBayes",
        "SURE(lambda)",
        "SURE(kernel)",
    ]

    print("\n" + "=" * 90)
    print("COMPREHENSIVE SUMMARY - ALL METHODS x ALL DATASETS at lowest K")
    print("=" * 90)

    if df_sim_noise is not None and len(df_sim_noise) > 0:
        lowest_noise = df_sim_noise["Noise"].min()
        _print_dataset_block(
            df_sim_noise[df_sim_noise["Noise"] == lowest_noise].copy(),
            "Noise",
            "SNR [dB]",
            f"SIMULATION noise sweep (noise={lowest_noise}, SNR dB higher=better)",
            "8.2f",
            method_order,
        )

    if df_sim_nevents is not None and len(df_sim_nevents) > 0:
        lowest_n = df_sim_nevents["N_events"].min()
        _print_dataset_block(
            df_sim_nevents[df_sim_nevents["N_events"] == lowest_n].copy(),
            "N_events",
            "SNR [dB]",
            f"SIMULATION N-events sweep (N={lowest_n}, SNR dB higher=better)",
            "8.2f",
            method_order,
        )

    if df_p300 is not None and len(df_p300) > 0:
        _print_dataset_block(
            df_p300,
            "K",
            "distance",
            f"P300 BCI (K={df_p300['K'].min()}, distance lower=better)",
            ".6f",
            method_order,
        )

    if df_eeg is not None and len(df_eeg) > 0:
        _print_dataset_block(
            df_eeg,
            "K",
            "distance",
            f"EEG (K={df_eeg['K'].min()}, distance lower=better)",
            ".6f",
            method_order,
        )

    if df_ecg is not None and len(df_ecg) > 0:
        _print_dataset_block(
            df_ecg,
            "K_per_class",
            "distance",
            f"ECG (K_per_class={df_ecg['K_per_class'].min()}, distance lower=better)",
            ".6f",
            method_order,
        )

    # --- SURE(kernel) vs EmpBayes verification ---
    print("\n" + "-" * 60)
    print("VERIFICATION: SURE(kernel) vs EmpBayes at lowest K")
    print("-" * 60)

    for name, df, k_col, metric_col, direction in [
        ("Simulation noise", df_sim_noise, "Noise", "SNR [dB]", "higher"),
        ("Simulation N-events", df_sim_nevents, "N_events", "SNR [dB]", "higher"),
        ("P300", df_p300, "K", "distance", "lower"),
        ("EEG", df_eeg, "K", "distance", "lower"),
        ("ECG", df_ecg, "K_per_class", "distance", "lower"),
    ]:
        _print_kernel_vs_empbayes(name, df, k_col, metric_col, direction)

    print("\n" + "=" * 90)


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    overall_start = time.time()
    print("=" * 90)
    print("FINAL2 BENCHMARK SUITE")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 90)

    df_sim_noise = None
    df_sim_nevents = None
    df_p300 = None
    df_ecg = None
    df_eeg = None

    # ===== 1. SIMULATION (pairwise, white noise) =====
    print("\n" + "=" * 90)
    print("PART 1: SIMULATION (METHODS_PAIRWISE, white noise)")
    print("=" * 90)

    T = 100
    half_win = T // 2
    real_templates = make_real_templates(T=T, C=8)

    print("\n--- Noise sweep (mean_interval=40) ---")
    t0 = time.time()
    df_sim_noise = run_sim_noise_sweep(real_templates, half_win, N_rep=20, I=20)
    print(f"  Elapsed: {time.time() - t0:.0f}s")

    print("\n--- N-events sweep (noise=0.02, mean_interval=50) ---")
    t0 = time.time()
    df_sim_nevents = run_sim_nevents_sweep(real_templates, half_win, N_rep=20)
    print(f"  Elapsed: {time.time() - t0:.0f}s")

    # ===== 2. P300 BCI (ar1_colored) =====
    print("\n" + "=" * 90)
    print("PART 2: P300 BCI (METHODS_AR1COL)")
    print("=" * 90)

    t0 = time.time()
    try:
        df_p300 = run_p300_benchmark()
    except Exception as e:
        print(f"  P300 FAILED: {e}")
        traceback.print_exc()
    print(f"  Elapsed: {time.time() - t0:.0f}s")

    # ===== 3. EEG (ar1_colored) =====
    print("\n" + "=" * 90)
    print("PART 3: EEG (METHODS_AR1COL)")
    print("=" * 90)

    t0 = time.time()
    try:
        df_eeg = run_eeg_benchmark()
    except Exception as e:
        print(f"  EEG FAILED: {e}")
        traceback.print_exc()
    print(f"  Elapsed: {time.time() - t0:.0f}s")

    # ===== 4. ECG (pairwise) =====
    print("\n" + "=" * 90)
    print("PART 4: ECG (METHODS_PAIRWISE)")
    print("=" * 90)

    t0 = time.time()
    try:
        df_ecg = run_ecg_benchmark()
    except Exception as e:
        print(f"  ECG FAILED: {e}")
        traceback.print_exc()
    print(f"  Elapsed: {time.time() - t0:.0f}s")

    # ===== SUMMARY =====
    print_summary(df_sim_noise, df_sim_nevents, df_p300, df_ecg, df_eeg)

    total_min = (time.time() - overall_start) / 60
    print(f"\nALL BENCHMARKS COMPLETE. Total time: {total_min:.1f} min")
    print(f"Finished: {datetime.now().isoformat()}")
    print("=" * 90)
