# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Noise-model diagnostics and noise inputs (supplement), and the residual/input
noise ratios of the response letter (R2.2).

P300 subjects 1-3 (1-20 Hz, three blocks joined with block starts), the auditory EEG
and ECG record 119, every eligible event:

1. Per channel, AR(0)-AR(3) diagnostics of the noise estimator's baseline segments
   with a bootstrap-calibrated residual test, and the contamination readout
   (baseline over full-data OLS residual variance; share of baseline samples inside
   another event's window). The 250 ms P300 SOA leaves no response-free baseline.
2. Per channel, the same models on the full-data OLS residual series (descriptive),
   and its noise variance and AR(1) coefficient over the K = 5 SURE(kernel) values.
3. The EEG/ECG trigger (OQ-B17) from the diagnostics of 1 alone.
4. K = 5 noise-model performance: SURE(lambda) and SURE(kernel) with the pairwise,
   AR(1) and AR(3) inputs, distances to the full-data references; on EEG/ECG only
   where the trigger fires (SURE(kernel) with the dataset's own input always runs:
   it gives the estimator values of 2).

    python3 -B run_noise_diagnostics.py --selections RUNS/p300/selections.json

``--selections``: ``run_p300.py``'s ``selections.json`` (K = 5 whole-recording
selections). EEG and ECG are loaded, and their selections drawn, by
``run_real_data.py``'s functions. Writes ``<out>/noise_diagnostics/``:
``per_fit.csv``, ``summary.json``, ``run_info.json`` and ``inputs/`` (prepared arrays
the workers read).
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import os
import statistics
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import analysis_tools
import ar_tools
import core
import data_loaders_p300
import heldout_p300
import numpy as np
import protocol
import run_real_data
from numpy.typing import NDArray

OQ9 = protocol.AR_DIAGNOSTICS
TRIGGER = protocol.TRIGGER


@functools.cache
def get_settings() -> ar_tools.Bootstrap:
    return ar_tools.Bootstrap(
        OQ9["drop_first"], OQ9["resid_lags"], OQ9["bootstrap"], OQ9["burn_in"]
    )


@functools.cache
def get_correction() -> ar_tools.Correction:
    return ar_tools.Correction(**protocol.BIAS_CORRECTION)


K: int = heldout_p300.K
ORDERS: tuple[int, ...] = protocol.AR_ORDERS
REFS: tuple[str, ...] = protocol.REFERENCES
REALDATA: tuple[str, ...] = run_real_data.DATASETS
HALF_WIN: dict[str, int] = {"P300": heldout_p300.HALF_WIN, **run_real_data.HALF_WIN}
# The noise input of each dataset's paper fits (SURE(kernel)'s row of its table).
OWN_NOISE: dict[str, str] = {
    "P300": protocol.METHODS_AR1COL["SURE(kernel)"]["noise_est"],
    "EEG": protocol.METHODS_AR1COL["SURE(kernel)"]["noise_est"],
    "ECG": protocol.METHODS_PAIRWISE["SURE(kernel)"]["noise_est"],
}
INNER: tuple[str, ...] = ("phi_estimated", "phi_used", "noise_var_estimated")
COLUMNS: tuple[str, ...] = (
    ("dataset", "repetition", "label", "status", "failure_type", "flags", "seconds")
    + ("lambda", "noise_var")
    + INNER
    + tuple(f"distance_{r}" for r in REFS)
    + ("scoring_failure", "kernel")
)
RUN_SETTINGS: dict[str, object] = {
    "ar_diagnostics": OQ9,
    "bias_correction": protocol.BIAS_CORRECTION,
    "trigger": TRIGGER,
    "residual_bootstrap_channels": protocol.RESIDUAL_BOOTSTRAP_CHANNELS,
}
# One K = 5 selection as indices into a dataset's prepared events (P300 selections
# are read as [block, event] pairs and mapped onto the joined recording's events).
Selection = list[int]
ChannelTask = tuple[str, str, int, str | None]
PerformanceTask = tuple[str, str, int, Selection, tuple[str, ...]]


def modality(ds: str) -> str:
    return "P300" if ds.startswith("P300") else ds


def half_win(ds: str) -> int:
    return HALF_WIN[modality(ds)]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def prepare_p300(subject: int) -> dict[str, NDArray]:
    """The three blocks joined (with block starts) and every eligible event."""
    ds = f"P300_s{subject:02d}"
    blocks = data_loaders_p300.load_bnci2014_009_blocks(
        subject, bandpass=protocol.BANDPASS, resample=protocol.RESAMPLE
    )
    rows = []
    for b in blocks:
        ok = b.eligible(heldout_p300.HALF_WIN, heldout_p300.GUARD)
        counts = {c: int(np.sum(b.labels[ok] == c)) for c in (0, 1)}
        if not np.all(ok):
            raise RuntimeError(
                f"{ds} block {b.block}: {int(np.sum(~ok))} of {len(ok)} events are "
                "not eligible (window or guard outside the block)"
            )
        if counts != protocol.BLOCK_CLASS_COUNTS:
            raise RuntimeError(
                f"{ds} block {b.block}: eligible class counts {counts}, "
                f"expected {protocol.BLOCK_CLASS_COUNTS}"
            )
        rows += [(b.block, e) for e in range(len(ok))]
    starts = np.cumsum([0] + [len(b.data) for b in blocks[:-1]])
    return {
        "data": np.vstack([b.data for b in blocks]),
        "ix": np.array([starts[b] + blocks[b].onsets[e] for b, e in rows]),
        "lb": np.array([blocks[b].labels[e] for b, e in rows]),
        "block_starts": starts,
        "rows": np.array(rows),
        "channels": np.array(blocks[0].channels),
    }


def p300_selections(
    a: Mapping[str, NDArray], reps: Sequence[list[list[int]]]
) -> tuple[list[Selection], list[str]]:
    """[block, event] selections as event indices, and why a repetition is not
    exactly K distinct eligible events per class."""
    where = {(int(b), int(e)): i for i, (b, e) in enumerate(a["rows"])}
    want = {str(c): K for c in np.unique(a["lb"])}
    out: list[Selection] = []
    errors = []
    for i, sel in enumerate(reps):
        idx = [where.get((b, e)) for b, e in sel]
        missing = [pair for pair, j in zip(sel, idx) if j is None]
        found = [j for j in idx if j is not None]
        counts = dict(Counter(str(a["lb"][j]) for j in found))
        if missing:
            errors.append(f"repetition {i} names non-eligible events {missing[:5]}")
        elif len(set(found)) != len(found):
            errors.append(f"repetition {i} repeats events ({len(found)} rows)")
        elif counts != want:
            errors.append(f"repetition {i} has class counts {counts}, expected {want}")
        out.append(found)
    return out, errors


@functools.cache
def inputs(folder: str, ds: str, kind: str) -> dict[str, NDArray]:
    """One prepared array file (data, refs or residual), read once per process."""
    with np.load(Path(folder) / f"{ds}_{kind}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def fit(
    a: Mapping[str, NDArray], sel: NDArray | slice, ds: str, kwargs: Mapping[str, Any]
) -> tuple[dict[str, Any], NDArray | None]:
    """``core.get_events_templates`` on selected events under the failure policy;
    templates stacked in np.unique label order."""
    lb = a["lb"][sel]
    record, array = analysis_tools.fit_call(
        lambda: core.get_events_templates(
            a["data"],
            a["ix"][sel],
            lb,
            half_win=half_win(ds),
            block_starts=a.get("block_starts"),
            **kwargs,
        ),
        [str(e) for e in np.unique(lb)],
        bool(protocol.DECISIONS["OQ6_exceptions_are_failures"]),
    )
    inner = record.pop("inner") or {}
    record.update({k: inner[k] for k in INNER if k in inner})
    if (record.get("phi_estimated") or 0.0) >= protocol.PHI_CAP:
        record["flags"] = record["flags"] + ["phi_hat_at_cap"]
    return record, array


def reference_task(args: tuple[str, str, str]) -> tuple[dict[str, Any], NDArray | None]:
    folder, ds, ref = args
    return fit(
        inputs(folder, ds, "data"), slice(None), ds, protocol.METHODS_AR1COL[ref]
    )


def residual_series(folder: Path, ds: str, refs: Mapping[str, NDArray]) -> str | None:
    """Save data - D theta_hat(OLS reference); the reason when it is unavailable."""
    if "OLS" not in refs:
        return "the OLS reference failed"
    a = inputs(str(folder), ds, "data")
    try:
        r, ortho, _ = ar_tools.ols_residual(
            a["data"], a["ix"], a["lb"], half_win(ds), refs["OLS"]
        )
        var = np.var(r, axis=0)
    except analysis_tools.INFRASTRUCTURE_ERRORS:
        raise
    except Exception as error:  # failure policy: a scientific failure is counted
        return f"OLS residual failed: {type(error).__name__}: {error}"
    tol = protocol.RESIDUAL_ORTHOGONALITY_TOLERANCE
    if not math.isfinite(ortho):
        return f"OLS residual orthogonality is not finite: {ortho}"
    if ortho >= tol:
        return f"OLS residual not orthogonal: ||D^T r||/||D^T X|| = {ortho} >= {tol}"
    np.savez(folder / f"{ds}_residual.npz", residual=r, var=var)
    return None


# ---------------------------------------------------------------------------
# Per channel: baseline-segment and residual-series diagnostics
# ---------------------------------------------------------------------------


def baseline(
    x: NDArray, a: Mapping[str, NDArray], ds: str, resid_var: float | None
) -> dict[str, Any]:
    hw, n = half_win(ds), len(x)
    n_b = ar_tools.baseline_length(hw, OQ9["baseline_frac"], OQ9["min_baseline"])
    # The contamination readout measures the noise estimator's own window
    # (core.est_noise_var_baseline); equal to n_b here (30, 72, 75 samples).
    n_est = ar_tools.baseline_length(hw, OQ9["baseline_frac"], 1)
    if n_est > n_b:
        raise ValueError(f"{ds}: estimator window {n_est} exceeds the segment {n_b}")
    bs = a.get("block_starts")
    bounds = None if bs is None else [int(s) for s in bs]
    starts = ar_tools.segment_starts(a["ix"], hw, n_b, n, bounds)
    diag: dict[str, Any] = dict(
        ar_tools.channel_diagnostics(
            x,
            ar_tools.Placement(starts, n_b, n, bounds),
            ORDERS,
            get_settings(),
            OQ9["acf_lags"],
            np.random.RandomState(OQ9["seed"]),
            get_correction(),
            np.random.RandomState([OQ9["seed"], 1]),
        )
    )
    raw = x[starts[:, None] + np.arange(n_est)[None, :]]
    var = float(np.mean(np.var(raw, axis=1)))
    valid = resid_var is not None and math.isfinite(resid_var) and resid_var > 0
    diag["contamination"] = {
        "baseline_var": var,
        "ols_residual_var": resid_var if valid else None,
        "ratio": var / resid_var if valid and resid_var else None,
        "overlap_fraction": ar_tools.overlap_fraction(a["ix"], hw, starts, n_est, n),
    }
    if not valid:
        diag["contamination"]["unavailable"] = f"OLS residual variance {resid_var}"
    return diag


def residual(r: NDArray, a: Mapping[str, NDArray], ds: str, c: int) -> dict[str, Any]:
    bounds = [int(s) for s in a.get("block_starts", [0])] + [len(r)]
    segs = [r[lo:hi] for lo, hi in zip(bounds[:-1], bounds[1:])]
    drop, lags = OQ9["drop_first"], OQ9["resid_lags"]
    diag: dict[str, Any] = dict(
        ar_tools.series_diagnostics(segs, ORDERS, drop, OQ9["acf_lags"], lags)
    )
    if c in protocol.RESIDUAL_BOOTSTRAP_CHANNELS.get(ds, ()):
        rs = np.random.RandomState(OQ9["seed"])  # one stream, orders in sequence
        diag["bootstrap"] = {
            str(p): ar_tools.bootstrap_residual_test(
                e["coef"],
                e["sigma2"],
                [len(s) for s in segs],
                p,
                get_settings(),
                rs,
                e["q"],
            )
            for p, e in zip(ORDERS, diag["orders"].values())
        }
    hw = half_win(ds)
    n_est = ar_tools.baseline_length(hw, OQ9["baseline_frac"], 1)
    diag["window_matched"] = ar_tools.window_matched_variances(
        r, bounds, a["ix"], hw, n_est
    )
    return diag


def channel_task(args: ChannelTask) -> dict[str, Any]:
    """Both diagnostics of one channel; an exception or a nonfinite value is a failed
    part. `unavailable`: why the dataset has no OLS residual series (None if it has)."""
    folder, ds, c, unavailable = args
    a = inputs(folder, ds, "data")
    res = None if unavailable else inputs(folder, ds, "residual")
    record: dict[str, Any] = {"dataset": ds, "channel": c}
    record["channel_name"] = str(a["channels"][c])
    record["residual"] = {"status": "unavailable", "reason": unavailable}
    var = None if res is None else float(res["var"][c])
    parts: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("baseline", lambda: baseline(a["data"][:, c], a, ds, var))
    ]
    if res is not None:
        series = res["residual"][:, c]
        parts.append(("residual", lambda: residual(series, a, ds, c)))
    for part, call in parts:
        try:
            out = {"status": "success", **call()}
        except analysis_tools.INFRASTRUCTURE_ERRORS:
            raise
        except Exception as error:  # failure policy: a scientific failure is counted
            out = {"status": "failure", "failure_type": type(error).__name__}
            out["failure_message"] = str(error)
        else:
            bad = nonfinite(out)
            if bad:
                out = {"status": "failure", "failure_type": "FloatingPointError"}
                out["failure_message"] = f"nonfinite {part} diagnostics at {bad[:5]}"
        record[part] = out
    return record


def nonfinite(obj: object, path: str = "") -> list[str]:
    """Locations of NaN or infinite numbers in a nested structure."""
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in nonfinite(v, f"{path}.{k}")]
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [p for i, v in enumerate(obj) for p in nonfinite(v, f"{path}[{i}]")]
    if isinstance(obj, (float, np.floating)) and not math.isfinite(float(obj)):
        return [path or "."]
    return []


# ---------------------------------------------------------------------------
# Noise-model performance at K = 5
# ---------------------------------------------------------------------------


def cells(ds: str, full: bool) -> tuple[str, ...]:
    if not full:
        return (f"SURE(kernel)|{OWN_NOISE[modality(ds)]}",)
    methods = protocol.NOISE_MODEL_METHODS
    return tuple(f"{m}|{ne}" for m in methods for ne in protocol.NOISE_MODELS)


def score(
    array: NDArray, refs: Mapping[str, NDArray]
) -> tuple[dict[str, float | None], str | None]:
    """Distances to the available references; None, with the reason, where the shapes
    differ, a class is missing, or the distance is not finite."""
    distances: dict[str, float | None] = dict.fromkeys(REFS)
    problems = []
    for r in (r for r in REFS if r in refs):  # a failed reference is absent
        if array.shape != refs[r].shape or array.shape[0] != 2:
            problems.append(
                f"template shape {array.shape} vs reference {r} shape "
                f"{refs[r].shape}; the distance needs both classes"
            )
            continue
        try:
            d = analysis_tools.distance(array, refs[r])
        except ArithmeticError as error:  # overflow when NumPy raises on it
            problems.append(f"distance to {r}: {type(error).__name__}: {error}")
            continue
        if not math.isfinite(d):
            problems.append(f"distance to {r} is {d}")
            continue
        distances[r] = d
    return distances, "; ".join(problems) or None


def performance_task(args: PerformanceTask) -> list[dict[str, Any]]:
    """Every (method, noise input) cell of one selection, scored on the references."""
    folder, ds, rep, selection, labels = args
    a, refs = inputs(folder, ds, "data"), inputs(folder, ds, "refs")
    sel = np.array(selection, dtype=np.int64)
    out = []
    for label in labels:
        method, noise = label.split("|")
        kwargs = {**protocol.METHODS_AR1COL[method], "noise_est": noise}
        record, array = fit(a, sel, ds, kwargs)
        if record["status"] == "success" and array is not None:
            record["distances"], problem = score(array, refs)
            if problem:
                record["scoring_failure"] = problem
        out.append({**record, "dataset": ds, "repetition": rep, "label": label})
    return out


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def med_range(values: Sequence[float | None]) -> dict[str, Any]:
    v = [float(x) for x in values if x is not None and math.isfinite(x)]
    if not v:
        return {"median": None, "min": None, "max": None, "n": 0}
    return {"median": float(np.median(v)), "min": min(v), "max": max(v), "n": len(v)}


def group(r: Mapping[str, Any]) -> str:
    return (
        str(int(r["dataset"][6:])) if r["dataset"].startswith("P300") else r["dataset"]
    )


def performance_summary(perf: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean log distance per reference, cell and group (P300 subject or dataset)."""
    out: dict[str, Any] = {}
    for ref in REFS:
        for label in sorted({r["label"] for r in perf}):
            for g in sorted({group(r) for r in perf}):
                rs = [r for r in perf if r["label"] == label and group(r) == g]
                d = [(r.get("distances") or {}).get(ref) for r in rs]
                v = [  # successful fits with a finite positive distance
                    math.log(x)
                    for x, r in zip(d, rs)
                    if r["status"] == "success" and x and math.isfinite(x) and x > 0
                ]
                out.setdefault(ref, {}).setdefault(label, {})[g] = {
                    "mean": float(np.mean(v)) if v else None,
                    "n_valid": len(v),
                    "n": len(rs),
                }
    return out


def fit_counts(perf: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Failures and flags; a group|cell is flagged above 10% failures."""
    groups: dict[str, list[bool]] = {}
    for r in perf:
        groups.setdefault(f"{group(r)}|{r['label']}", []).append(
            r["status"] != "success"
        )
    return {
        "n": len(perf),
        "failures": sum(r["status"] != "success" for r in perf),
        "flagged": sum(bool(r["flags"]) for r in perf),
        "scoring_failures": sum("scoring_failure" in r for r in perf),
        "groups_over_10pct_failures": [
            g for g, f in sorted(groups.items()) if sum(f) > 0.1 * len(f)
        ],
    }


def baseline_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per dataset (failed channels counted): the figure's per-order fields, the
    lag-2 PACF, the contamination readout and the AR(1) coefficients."""
    out = {}
    for ds in sorted({r["dataset"] for r in records}):
        ch = [r["baseline"] for r in records if r["dataset"] == ds]
        ok = [b for b in ch if b["status"] == "success"]
        orders = {}
        for p in map(str, ORDERS):
            o = [b["orders"][p] for b in ok]
            pv = [e["residual_p_calibrated"] for e in o]
            orders[p] = {
                "delta_bic": med_range(
                    [e["bic"] - b["orders"]["0"]["bic"] for e, b in zip(o, ok)]
                ),
                "rejections": sum(x is not None and x < 0.05 for x in pv),
                "calibration_unavailable_channels": sum(
                    not e["calibration_valid"] for e in o
                ),
                "coef": [med_range([e["coef"][j] for e in o]) for j in range(int(p))],
                "bic_gain": med_range([e["bic_gain_best_higher"] for e in o]),
                "bic_gain_null_q95": med_range([e.get("bic_gain_null_q95") for e in o]),
            }
        out[ds] = {
            "n_channels": len(ch),
            "failed_channels": len(ch) - len(ok),
            "orders": orders,
            "pacf": (
                [med_range([b["pacf"][k] for b in ok]) for k in range(OQ9["acf_lags"])]
                if ok
                else []
            ),
            "contamination_ratio": med_range([b["contamination"]["ratio"] for b in ok]),
            "overlap_fraction": (
                ok[0]["contamination"]["overlap_fraction"] if ok else None
            ),
        }
    return out


def estimator_values(perf: Sequence[Mapping[str, Any]], ds: str) -> dict[str, Any]:
    """Noise variance and phi of the K = 5 SURE(kernel) fits with the own input."""
    own = [r for r in perf if r["dataset"] == ds and r["label"] == cells(ds, False)[0]]
    ok = [r for r in own if r["status"] == "success"]
    nv = [r["noise_var"] for r in ok if r.get("noise_var") is not None]
    phi = [r["phi_estimated"] for r in ok if r.get("phi_estimated") is not None]
    return {
        "noise_var": med_range(nv) if nv else None,
        "phi": med_range(phi) if phi else None,
        "noise_est": OWN_NOISE[modality(ds)],
        "n": len(own),
    }


def ratio_rows(
    records: Sequence[Mapping[str, Any]], est: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Per channel: residual quantities over the estimator's median values."""
    rows = []
    for r in records:
        res, e = r["residual"], est[r["dataset"]]
        row = {k: r[k] for k in ("dataset", "channel", "channel_name")}
        row["status"] = res["status"]
        if res["status"] == "success":
            nv = (e["noise_var"] or {}).get("median")
            ph = (e["phi"] or {}).get("median")
            wm = res["window_matched"]
            phi = float(res["orders"]["1"]["coef"][0])
            row.update(
                residual_variance=res["marginal_variance"],
                residual_ar1_phi=phi,
                residual_window_matched_variance=wm,
                variance_ratio=res["marginal_variance"] / nv if nv else None,
                tiled_variance_ratio=wm["tiled"] / nv if nv else None,
                event_baseline_variance_ratio=(
                    wm["event_baselines"] / nv if nv else None
                ),
                phi_ratio=phi / ph if ph else None,
                bootstrap=res.get("bootstrap"),
                orders={
                    p: {k: o[k] for k in ("coef", "bic", "q", "p_asymptotic")}
                    for p, o in res["orders"].items()
                },
            )
        rows.append(row)
    return rows


def residual_summary(
    rows: Sequence[Mapping[str, Any]], est: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Per dataset: BIC-best orders and the residual/estimator comparison."""
    out = {}
    for ds in est:
        ch = [r for r in rows if r["dataset"] == ds]
        ok = [r for r in ch if r["status"] == "success"]
        best = [min(map(str, ORDERS), key=lambda p: r["orders"][p]["bic"]) for r in ok]
        nv = (est[ds]["noise_var"] or {}).get("median")
        var = [r["residual_variance"] for r in ok]
        out[ds] = {
            "n_channels": len(ch),
            "n_failed": len(ch) - len(ok),
            "estimator": est[ds],
            "orders": {
                p: {"bic_best_channels": best.count(p)} for p in map(str, ORDERS)
            },
            "estimator_comparison": {
                "residual_ar1_phi": med_range([r["residual_ar1_phi"] for r in ok]),
                "variance_ratio_residual_over_estimator": med_range(
                    [r["variance_ratio"] for r in ok]
                ),
                "channel_mean_variance_ratio": (
                    float(np.mean(var)) / nv if nv and ok else None
                ),
                "window_matched": {
                    "tiled_ratio_residual_over_estimator": med_range(
                        [r["tiled_variance_ratio"] for r in ok]
                    ),
                    "event_baseline_ratio_residual_over_estimator": med_range(
                        [r["event_baseline_variance_ratio"] for r in ok]
                    ),
                },
                "phi_ratio_residual_over_estimator": med_range(
                    [r["phi_ratio"] for r in ok]
                ),
            },
        }
    return out


def best_noise_input(perf: Mapping[str, Any]) -> dict[str, Any]:
    """Per reference, method and group: the noise input with the smallest mean log
    distance, None unless every input has a finite mean."""
    out: dict[str, Any] = {}
    for ref, table in perf.items():
        for method in protocol.NOISE_MODEL_METHODS:
            means = {
                ne: table.get(f"{method}|{ne}", {}) for ne in protocol.NOISE_MODELS
            }
            for g in means["ar1_colored"]:
                m = {ne: means[ne].get(g, {}).get("mean") for ne in means}
                ok = all(v is not None and math.isfinite(v) for v in m.values())
                best = min(m, key=lambda ne: m[ne]) if ok else None
                out.setdefault(ref, {}).setdefault(method, {})[g] = best
    return out


def noise_table(perf: Mapping[str, Any]) -> dict[str, dict[str, float | None]]:
    """Mean log distance to OLS per cell: EEG, ECG and P300 (the mean of the subject
    means, None unless every subject has one; the blocks report's statistics.mean)."""
    ols = perf.get("OLS", {})
    p300 = [g for g in ols.get("SURE(kernel)|ar1_colored", {}) if g.isdigit()]
    table: dict[str, dict[str, float | None]] = {}
    for label, v in ols.items():
        means = [v[g]["mean"] for g in p300 if g in v]
        p = None if None in means or not means else statistics.mean(means)
        table[label] = {"P300": p, **{g: v[g]["mean"] for g in REALDATA if g in v}}
    return table


def min_gain_over_null(diag: Mapping[str, Any]) -> float | None:
    """Smallest observed BIC gain over the largest bootstrap-null q95, per dataset and
    order 0-2 with an available positive null (the blocks report's arithmetic)."""
    gains = [
        o["bic_gain"]["min"] / o["bic_gain_null_q95"]["max"]
        for d in diag.values()
        for o in (d["orders"][p] for p in ("0", "1", "2"))
        if o["bic_gain_null_q95"]["n"] and o["bic_gain_null_q95"]["median"] > 0
    ]
    return min(gains, default=None)


def manuscript_numbers(S: Mapping[str, Any]) -> dict[str, Any]:
    """The derived numbers that the supplement's noise-model diagnostics and the R2.2
    response-letter material cite."""
    diag, res = S["baseline"]["diagnostics"], S["residual"]
    rows = [b for b in S["baseline"]["per_channel"] if b["status"] == "success"]
    cal = [b["orders"]["1"] for b in rows if b["orders"]["1"]["calibration_valid"]]
    boot = [b for r in res["per_channel"] for b in (r.get("bootstrap") or {}).values()]
    return {
        "baseline_channels": sum(d["n_channels"] for d in diag.values()),
        "rejected_every_channel": {  # None when no channel was diagnosed
            p: (
                all(
                    d["orders"][p]["rejections"] == d["n_channels"]
                    for d in diag.values()
                )
                if any(d["n_channels"] for d in diag.values())
                else None
            )
            for p in map(str, ORDERS)
        },
        "ar1_calibration_unavailable": sum(
            d["orders"]["1"]["calibration_unavailable_channels"] for d in diag.values()
        ),
        "ar1_calibrated_channels": len(cal),
        "ar1_calibrated_rejected": sum(e["residual_p_calibrated"] < 0.05 for e in cal),
        "ar1_calibrated_max_p": max(
            (e["residual_p_calibrated"] for e in cal), default=None
        ),
        "smallest_attainable_p": 1 / (OQ9["bootstrap"] + 1),
        "min_bic_gain_over_null_q95": min_gain_over_null(diag),
        "noise_model_mean_log_distance_OLS": noise_table(S["performance"]),
        "noise_model_best_input": best_noise_input(S["performance"]),
        "residual_bootstrap_channels": sorted(
            f"{r['dataset']} {r['channel_name']}"
            for r in res["per_channel"]
            if r.get("bootstrap")
        ),
        "residual_bootstrap_max_p": None if None in boot else max(boot, default=None),
    }


def summarize(
    records: Sequence[Mapping[str, Any]],
    perf: Sequence[Mapping[str, Any]],
    trigger: Mapping[str, ar_tools.TriggerDecision],
    references: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    datasets = list(references)
    est = {ds: estimator_values(perf, ds) for ds in datasets}
    shown = [
        r
        for r in perf
        if r["dataset"] not in trigger or trigger[r["dataset"]]["run_performance"]
    ]
    rows = ratio_rows(records, est)
    S: dict[str, Any] = {
        "baseline": {
            "diagnostics": baseline_summary(records),
            "segment_length": {
                ds: ar_tools.baseline_length(
                    half_win(ds), OQ9["baseline_frac"], OQ9["min_baseline"]
                )
                for ds in datasets
            },
            "per_channel": [baseline_row(r) for r in records],
        },
        "trigger": trigger,
        "performance": performance_summary(shown),
        "fits": fit_counts(shown),
        "residual": {"datasets": residual_summary(rows, est), "per_channel": rows},
        "references": references,
        "failed_channels": [
            [r["dataset"], r["channel"], part, r[part]["failure_type"]]
            for r in records
            for part in ("baseline", "residual")
            if r[part]["status"] == "failure"
        ],
        "residual_unavailable": {
            r["dataset"]: r["residual"]["reason"]
            for r in records
            if r["residual"]["status"] == "unavailable"
        },
    }
    S["manuscript"] = manuscript_numbers(S)
    return S


def baseline_row(r: Mapping[str, Any]) -> dict[str, Any]:
    b = r["baseline"]
    row = {k: r[k] for k in ("dataset", "channel", "channel_name")}
    row["status"] = b["status"]
    if b["status"] == "success":
        row.update({k: b[k] for k in ("pacf", "contamination")})
        row["orders"] = {
            p: {
                k: v
                for k, v in e.items()
                if k not in {"residual_acf", "bias_correction"}
            }
            for p, e in b["orders"].items()
        }
    return row


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def check_selections(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> dict[str, list[list[list[int]]]]:
    """The K = 5 whole-recording selections of the requested subjects; stops with a
    message naming the file, subject and repetition before any input is prepared."""
    k, name = str(K), args.selections
    sel = json.loads(Path(name).read_text())
    if not isinstance(sel, dict):
        parser.error(f"{name} is not a JSON object (got {type(sel).__name__})")
    if not isinstance(sel.get("A"), dict):
        parser.error(f"{name} has no whole-recording selections (key 'A')")
    if not isinstance(sel["A"].get(k), dict):
        parser.error(f"{name} has no K = {k} selections (key 'A' -> '{k}')")
    available = sel["A"][k]
    for s in args.subjects:
        reps = available.get(str(s))
        if not isinstance(reps, list) or len(reps) < args.reps:
            parser.error(
                f"{name}: subject {s} has {len(reps) if isinstance(reps, list) else 0} "
                f"K = {k} selections, {args.reps} requested (subjects in the file: "
                f"{sorted(available)})"
            )
        for i, rows in enumerate(reps[: args.reps]):
            pairs = isinstance(rows, list) and all(
                isinstance(x, list)
                and len(x) == 2
                and all(isinstance(v, int) and not isinstance(v, bool) for v in x)
                for x in rows
            )
            if not rows or not pairs:
                parser.error(
                    f"{name}: subject {s} repetition {i} is not a nonempty list of "
                    f"integer [block, event] pairs: {str(rows)[:80]}"
                )
    return available


def prepare_inputs(
    parser: argparse.ArgumentParser, args: argparse.Namespace, folder: Path
) -> tuple[dict[str, list[Selection]], dict[str, int], dict[str, str]]:
    """Save every dataset's arrays; its K = 5 selections, channel count, and the reason
    when EEG/ECG classes cannot fill K = 5 selections."""
    p300_sel = check_selections(parser, args)
    selections: dict[str, list[Selection]] = {}
    n_channels: dict[str, int] = {}
    no_fits: dict[str, str] = {}
    for s in args.subjects:
        ds = f"P300_s{s:02d}"
        a = prepare_p300(s)
        selections[ds], errors = p300_selections(a, p300_sel[str(s)][: args.reps])
        if errors:
            parser.error(f"{args.selections}, {ds}: {'; '.join(errors[:3])}")
        np.savez(folder / f"{ds}_data.npz", **a)
        n_channels[ds] = len(a["channels"])
    for ds in REALDATA:
        a, _ = run_real_data.load_input(ds, Path(args.data_dir))
        np.savez(folder / f"{ds}_data.npz", **a)
        n_channels[ds] = len(a["channels"])
        sel, unavailable = run_real_data.make_selections(a["lb"], ds, args.reps)
        selections[ds] = list(sel.get(str(K), []))
        if str(K) in unavailable:
            no_fits[ds] = unavailable[str(K)]
    n_channels = {ds: min(n, args.channels or n) for ds, n in n_channels.items()}
    return selections, n_channels, no_fits


def references_and_residuals(
    workers: int, folder: Path, datasets: Sequence[str]
) -> tuple[dict[str, dict[str, str]], dict[str, str | None]]:
    """Full-data references (saved when successful) and residual series; the
    reference statuses and why a residual series is unavailable."""
    tasks = [(str(folder), ds, ref) for ds in datasets for ref in REFS]
    fits = analysis_tools.run_pool(workers, reference_task, tasks)
    references: dict[str, dict[str, str]] = {ds: {} for ds in datasets}
    unavailable: dict[str, str | None] = {}
    for ds in datasets:
        refs = {}
        for (_, d, ref), (record, array) in zip(tasks, fits):
            if d == ds:
                references[ds][ref] = record["status"]
                if record["status"] == "success" and array is not None:
                    refs[ref] = array
        np.savez(folder / f"{ds}_refs.npz", **refs)
        unavailable[ds] = residual_series(folder, ds, refs)
    return references, unavailable


def decide_triggers(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, ar_tools.TriggerDecision]:
    """The EEG/ECG trigger from the baseline diagnostics (a failed channel is None)."""
    return {
        ds: ar_tools.trigger_decision(
            [
                r["baseline"] if r["baseline"]["status"] == "success" else None
                for r in records
                if r["dataset"] == ds
            ],
            ORDERS,
            TRIGGER["baseline_order"][ds],
            TRIGGER["bic_margin"],
            TRIGGER["alpha"],
            TRIGGER["channel_fraction"],
        )
        for ds in REALDATA
    }


def write_outputs(
    out: Path, S: Mapping[str, Any], perf: Sequence[Mapping[str, Any]]
) -> None:
    analysis_tools.write_json(out / "summary.json", S)
    with (out / "per_fit.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(COLUMNS)
        for r in perf:
            row = {
                **r,
                **{f"distance_{k}": v for k, v in (r.get("distances") or {}).items()},
            }
            row["flags"] = ";".join(r["flags"])
            if r.get("kernel") is not None:
                row["kernel"] = json.dumps(analysis_tools.plain(r["kernel"]))
            writer.writerow(["" if row.get(c) is None else row[c] for c in COLUMNS])


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="Noise-model diagnostics and noise inputs."
    )
    parser.add_argument("--selections", required=True)
    parser.add_argument(
        "--data-dir", default=os.environ.get("MNE_DATA", str(Path.home() / "mne_data"))
    )
    parser.add_argument("--out", default="runs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--subjects", type=int, nargs="*", default=protocol.DIAG_SUBJECTS
    )
    parser.add_argument("--reps", type=int, default=heldout_p300.REPETITIONS)
    parser.add_argument("--channels", type=int, default=None, help="first N only")
    args = parser.parse_args()
    if args.reps < 1:
        parser.error(f"--reps must be at least 1, got {args.reps}")
    if args.channels is not None and args.channels < 1:
        parser.error(f"--channels must be at least 1, got {args.channels}")
    return parser, args


def main() -> None:
    parser, args = parse_args()
    started = datetime.now(UTC)
    os.environ["MNE_DATA"] = os.environ["MNE_DATASETS_BNCI_PATH"] = args.data_dir
    out = Path(args.out) / "noise_diagnostics"
    folder = out / "inputs"
    folder.mkdir(parents=True, exist_ok=True)
    datasets = [f"P300_s{s:02d}" for s in args.subjects] + list(REALDATA)
    selections, n_channels, no_fits = prepare_inputs(parser, args, folder)
    references, unavailable = references_and_residuals(args.workers, folder, datasets)
    chan: list[ChannelTask] = [
        (str(folder), ds, c, unavailable[ds])
        for ds in datasets
        for c in range(n_channels[ds])
    ]
    records = analysis_tools.run_pool(args.workers, channel_task, chan)
    trigger = decide_triggers(records)
    jobs: list[PerformanceTask] = [
        (
            str(folder),
            ds,
            i,
            s,
            cells(ds, ds not in trigger or trigger[ds]["run_performance"]),
        )
        for ds in datasets
        for i, s in enumerate(selections[ds])
    ]
    perf = [
        x
        for rows in analysis_tools.run_pool(args.workers, performance_task, jobs)
        for x in rows
    ]
    S = summarize(records, perf, trigger, references)
    S["performance_unavailable"] = no_fits
    write_outputs(out, S, perf)
    analysis_tools.write_run_info(out, {**RUN_SETTINGS, "args": vars(args)}, started)


if __name__ == "__main__":
    main()
