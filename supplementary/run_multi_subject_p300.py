# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

"""
Multi-subject P300 validation across BNCI2014-009 (MOABB) -- reproduces the
paper's tab:p300_subjects (template distance at K=5 aggregated across all 10
subjects).

For each subject (1-10): load P300 (1-20 Hz, resampled to 64 Hz, midline
channels), estimate templates at K=5 over N_REP subsamples with methods
Average / EmpBayes / GCV / SURE(lambda) / SURE(kernel), and measure the
Frobenius distance to a per-subject EmpBayes-from-all-events reference.

Writes supplementary/multi_subject_p300.csv and _summary.md. The committed CSV
is the paper's data; a fresh run reproduces it (scalar seed 42).

Run with: OPENBLAS_NUM_THREADS=1 python supplementary/run_multi_subject_p300.py
"""

import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Make the repo root (parent of supplementary/) importable when this script is
# run directly from supplementary/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from core import get_events_templates
from data_loaders_p300 import load_p300_bnci2014_009
from run_benchmarks import calc_template_distance, compact_continuous
from scipy.stats import wilcoxon

OUTPUT_DIR: Path = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(exist_ok=True)

# P300 midline channels (strongest P300), 64 Hz (Nyquist 32 Hz >> 20 Hz BW).
P300_CHANNELS: list[str] = ["Fz", "Cz", "Pz", "Oz", "PO7", "PO8"]
SFREQ: float = 64.0

METHODS: dict[str, dict[str, Any]] = {
    "Average": {"method": "average"},
    "EmpBayes": {"method": "ls", "reg_method": "empbayes"},
    "GCV": {"method": "ls", "reg_method": "gcv"},
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

K: int = 5
N_REP: int = 10
SEED: int = 42
METHOD_ORDER: list[str] = ["Average", "EmpBayes", "GCV", "SURE(lambda)", "SURE(kernel)"]


def run_subject(subj_id: int) -> list[dict[str, Any]]:
    """Run the K=5 template-distance benchmark for one subject."""
    print(f"\n{'=' * 60}\n  Subject {subj_id}\n{'=' * 60}", flush=True)
    data, event_indices, event_labels, sfreq, info = load_p300_bnci2014_009(
        subjects=[subj_id],
        bandpass=(1.0, 20.0),
        resample=SFREQ,
        pick_channels=P300_CHANNELS,
    )
    print(
        f"  Loaded: {data.shape}, sfreq={sfreq}, "
        f"target={info['n_target']}, nontarget={info['n_nontarget']}",
        flush=True,
    )

    half_win = int(0.25 * sfreq)
    lb_str = np.array([str(lbl) for lbl in event_labels])
    unique_labels = np.unique(lb_str)

    valid = (event_indices - half_win >= 0) & (event_indices + half_win <= len(data))
    ix_valid = event_indices[valid]
    lb_valid = lb_str[valid]

    # Reference: EmpBayes from ALL events (compacted for speed).
    data_c, ix_c, lb_c = compact_continuous(
        data, ix_valid, lb_valid, half_win, margin=10
    )
    ref, _ = get_events_templates(
        data_c, ix_c, lb_c, half_win=half_win, method="ls", reg_method="empbayes"
    )

    rs = np.random.RandomState(SEED)
    results: list[dict[str, Any]] = []
    t0 = time.time()
    for rep in range(N_REP):
        idx_selected: list[int] = []
        for label in unique_labels:
            label_idx = np.where(lb_valid == label)[0]
            k_per_class = min(K, len(label_idx))
            if k_per_class < 3:
                continue
            idx_selected.extend(rs.choice(label_idx, size=k_per_class, replace=False))
        if not idx_selected:
            continue
        sel = np.array(sorted(idx_selected))
        data_sub, ix_sub_c, lb_sub_c = compact_continuous(
            data, ix_valid[sel], lb_valid[sel], half_win, margin=5
        )
        for method_name, method_kwargs in METHODS.items():
            try:
                t, _ = get_events_templates(
                    data_sub, ix_sub_c, lb_sub_c, half_win=half_win, **method_kwargs
                )
                dist = calc_template_distance(t, ref)
                results.append(
                    {
                        "subject": subj_id,
                        "method": method_name,
                        "K": K,
                        "distance": dist,
                        "rep": rep,
                    }
                )
            except Exception as e:
                print(f"    WARNING: {method_name} rep={rep} failed: {e}", flush=True)
    print(
        f"  Subject {subj_id}: {len(results)} results ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return results


def make_summary(df: pd.DataFrame) -> str:
    """Aggregate across subjects into the tab:p300_subjects markdown table."""
    lines: list[str] = [
        "# Multi-Subject P300 Validation (BNCI2014-009)",
        "",
        f"- K={K}, N_rep={N_REP}, channels={', '.join(P300_CHANNELS)}, {SFREQ} Hz",
        "- Metric: template distance to per-subject EmpBayes reference (x1e-5)",
        "",
        "| Method | Mean +/- Std | Median |",
        "|--------|-------------:|-------:|",
    ]
    for m in METHOD_ORDER:
        per = df[df["method"] == m].groupby("subject")["distance"].mean() * 1e5
        if len(per):
            lines.append(
                f"| {m} | {per.mean():.2f} +/- {per.std():.2f} | {per.median():.2f} |"
            )

    # Overall paired Wilcoxon: SURE(kernel) vs EmpBayes
    eb = df[df["method"] == "EmpBayes"][["subject", "rep", "distance"]].rename(
        columns={"distance": "eb"}
    )
    sk = df[df["method"] == "SURE(kernel)"][["subject", "rep", "distance"]].rename(
        columns={"distance": "sk"}
    )
    paired = eb.merge(sk, on=["subject", "rep"])
    if len(paired):
        _, pval = wilcoxon(paired["eb"], paired["sk"], alternative="two-sided")
        lines += [
            "",
            f"- Overall paired Wilcoxon (SURE(kernel) vs EmpBayes): p = {pval:.2e}",
        ]
    return "\n".join(lines)


def main() -> None:
    all_results: list[dict[str, Any]] = []
    for subj_id in range(1, 11):
        try:
            all_results.extend(run_subject(subj_id))
        except Exception as e:
            print(f"  Subject {subj_id} FAILED: {e}", flush=True)
            traceback.print_exc()

    if not all_results:
        raise RuntimeError("No results collected across any P300 subject (1-10).")

    df = pd.DataFrame(all_results)
    csv_path = OUTPUT_DIR / "multi_subject_p300.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved: {csv_path} ({len(df)} rows)", flush=True)

    summary = make_summary(df)
    md_path = OUTPUT_DIR / "multi_subject_p300_summary.md"
    md_path.write_text(summary + "\n")
    print(f"  Summary saved: {md_path}\n\n{summary}", flush=True)


if __name__ == "__main__":
    main()
