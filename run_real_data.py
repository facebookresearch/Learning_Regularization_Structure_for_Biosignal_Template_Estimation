# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Auditory EEG and ECG analyses of the paper (descriptive, one recording each).

MNE sample auditory EEG (59 channels, left/right tones, 1-40 Hz, half window
120, METHODS_AR1COL) and MIT-BIH record 119 (lead 0, 0.5-40 Hz, Normal/PVC
beats, half window 126, METHODS_PAIRWISE). Every estimator, the zero template
and the lambda = inf template are scored against three full-data references
(OLS, EmpBayes, Average) on 20 random selections of K events per class; the
L-curve verticality is measured on every selection.

Writes ``<out>/real_data/``: ``cache/`` (inputs), ``selections.json``,
``references/``, ``fits/`` (templates as stored: EEG in V, ECG in mV),
``per_fit.csv``, ``summary.json``, ``run_info.json``.

Usage: python3 -B run_real_data.py [--data-dir DIR] [--out DIR] [--workers N]
           [--datasets EEG ECG] [--reps N]
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import os
import statistics
import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import analysis_tools
import core
import data_loaders
import data_loaders_ecg
import numpy as np
import protocol
from numpy.typing import NDArray

DATASETS: tuple[str, ...] = ("EEG", "ECG")
REPETITIONS = 20
SELECTION_SEED = 42  # decision OQ-B14: a fresh RandomState(42) per dataset and K
FAILURES = bool(protocol.DECISIONS["OQ6_exceptions_are_failures"])
REFS: tuple[str, ...] = protocol.REFERENCES
BASELINE_FRACTION = 0.3  # core's baseline segment: the first 30% of the window

# Template order: class names sorted (as the references and selections use them).
CLASSES: dict[str, tuple[str, ...]] = {
    "EEG": tuple(sorted(protocol.EEG_LABELS.values())),
    "ECG": tuple(sorted(protocol.ECG_LABELS.values())),
}
BANDPASS: dict[str, tuple[float, float]] = {
    "EEG": protocol.EEG_BANDPASS,
    "ECG": protocol.ECG_BANDPASS,
}
HALF_WIN: dict[str, int] = {"EEG": protocol.EEG_HALF_WIN, "ECG": protocol.ECG_HALF_WIN}
KS: dict[str, tuple[int, ...]] = {"EEG": (5, 10, 20, 50), "ECG": (5, 10, 25)}
METHODS: dict[str, dict[str, dict[str, Any]]] = {
    "EEG": protocol.METHODS_AR1COL,
    "ECG": protocol.METHODS_PAIRWISE,
}
SCALE: dict[str, float] = {"EEG": 1e6, "ECG": 1.0}
UNIT: dict[str, str] = {"EEG": "uV", "ECG": "mV"}
PHI_FIELDS: tuple[str, ...] = ("phi_estimated", "phi_used", "noise_var_estimated")
COLUMNS: tuple[str, ...] = (
    *[
        "dataset",
        "K",
        "repetition",
        "estimator",
        "status",
        "failure_type",
        "failure_message",
    ],
    *["flags", "lambda", "noise_var"],
    *PHI_FIELDS,
    *(f"distance_{r}" for r in REFS),
    *[
        "lcurve_deviation_deg",
        "lcurve_deviation_missing",
        "kernel",
        "template",
        "seconds",
    ],
)


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


# ---------------------------------------------------------------------------
# Inputs, selections, references
# ---------------------------------------------------------------------------


def load_eeg(data_dir: Path) -> tuple[NDArray, NDArray, NDArray, list[str], float]:
    """Auditory EEG: data, event onsets, class names, channel names, sampling rate."""
    if not (data_dir / "MNE-sample-data" / "MEG" / "sample").exists():
        data_loaders.download_eeg_mne(data_dir)
    data, ix, lb, info = data_loaders.load_eeg_erp(
        data_dir=data_dir,
        event_ids=dict(protocol.EEG_EVENT_IDS),
        bandpass=BANDPASS["EEG"],
        exclude_bads=True,
        return_info=True,
    )
    names = [str(n) for n in info["ch_names"]]
    if len(names) != protocol.EEG_N_CHANNELS or protocol.EEG_BAD in names:
        raise RuntimeError(
            f"EEG input has {len(names)} channels, expected {protocol.EEG_N_CHANNELS} "
            f"without {protocol.EEG_BAD}: {names}"
        )
    labels = np.array([protocol.EEG_LABELS[int(v)] for v in lb])
    return data, ix, labels, names, float(info["sfreq"])


def load_ecg(data_dir: Path) -> tuple[NDArray, NDArray, NDArray, list[str], float]:
    """MIT-BIH record 119: data, beat onsets, class names, channel names, rate."""
    data_loaders_ecg.download_ecg_mitbih(
        data_dir / "mitdb", records=[protocol.ECG_RECORD]
    )
    data, ix, lb, sfreq = data_loaders_ecg.load_ecg_mitbih(
        data_dir=data_dir / "mitdb",
        records=[protocol.ECG_RECORD],
        lead=protocol.ECG_LEAD,
        bandpass=BANDPASS["ECG"],
        beat_types=list(protocol.ECG_BEATS),
    )
    labels = np.array([protocol.ECG_LABELS.get(int(v), f"type_{v}") for v in lb])
    return data, ix, labels, [f"lead{protocol.ECG_LEAD}"], float(sfreq)


def load_input(ds: str, data_dir: Path) -> tuple[dict[str, NDArray], dict[str, Any]]:
    """The recording, its complete event windows with class names, and input facts.

    A missing recording is downloaded into *data_dir* first.
    """
    data, ix, labels, names, sfreq = (load_eeg if ds == "EEG" else load_ecg)(data_dir)
    if data.shape[1] != len(names):
        raise RuntimeError(f"{ds}: {data.shape[1]} data columns for channels {names}")
    checks: dict[str, Any] = (
        {"n_channels": len(names), "bad_excluded": protocol.EEG_BAD not in names}
        if ds == "EEG"
        else {}
    )
    hw = HALF_WIN[ds]
    valid = (ix - hw >= 0) & (ix + hw <= len(data))
    data = np.asarray(data)
    checks |= {
        "sfreq": sfreq,
        "n_samples": len(data),
        "duration_s": float(len(data) / sfreq),
        "window_samples": 2 * hw,
        "window_s": 2 * hw / sfreq,
        "n_events": len(ix),
        "n_incomplete_dropped": int(np.sum(~valid)),
        "eligible_counts": {
            str(c): int(np.sum(labels[valid] == c)) for c in np.unique(labels)
        },
        "channels": names,
    }
    arrays = {
        "data": data,
        "ix": np.asarray(ix[valid], dtype=np.int64),
        "lb": labels[valid],
        "channels": np.array(names),
        "sfreq": np.array(sfreq),
    }
    return arrays, checks


def make_selections(
    lb: NDArray, ds: str, reps: int
) -> tuple[dict[str, list[list[int]]], dict[str, str]]:
    """Selections of every K both classes can fill, and the reason for each K they
    cannot (a fresh RandomState(42) per K)."""
    counts = {c: int(np.sum(lb == c)) for c in CLASSES[ds]}
    extra = sorted(set(lb.tolist()) - set(CLASSES[ds]))
    out: dict[str, list[list[int]]] = {}
    unavailable: dict[str, str] = {}
    for k in KS[ds]:
        short = {c: n for c, n in counts.items() if n < k}
        if extra or short:
            unavailable[str(k)] = (
                f"unexpected classes {extra}" if extra else f"eligible events {short}"
            )
            continue
        rs = np.random.RandomState(SELECTION_SEED)
        out[str(k)] = [
            analysis_tools.rb_selection(rs, lb, k).tolist() for _ in range(reps)
        ]
    return out, unavailable


@functools.cache
def inputs(out: str, ds: str) -> dict[str, NDArray]:
    with np.load(Path(out) / "cache" / f"{ds.lower()}.npz", allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def fit(
    a: Mapping[str, NDArray], rows: NDArray | None, ds: str, kwargs: Mapping[str, Any]
) -> tuple[dict[str, Any], NDArray | None]:
    """One estimator on the selected (all when *rows* is None) events, failure policy
    applied; templates stacked in sorted class-name order."""
    ix, lb = (a["ix"], a["lb"]) if rows is None else (a["ix"][rows], a["lb"][rows])
    call = functools.partial(
        core.get_events_templates, a["data"], ix, lb, half_win=HALF_WIN[ds], **kwargs
    )
    record, array = analysis_tools.fit_call(call, CLASSES[ds], FAILURES)
    for key in PHI_FIELDS:
        if key in (record["inner"] or {}):
            record[key] = record["inner"][key]
    if (record.get("phi_estimated") or 0.0) >= protocol.PHI_CAP:
        record["flags"] = record["flags"] + ["phi_hat_at_cap"]
    return record, array


def fit_references(task: tuple[str, str]) -> dict[str, str]:
    """OLS, EmpBayes and Average on all eligible events; references/<ds>.npz.

    Returns the failed references with their reasons (each a counted failure).
    """
    out, ds = task
    a = inputs(out, ds)
    arrays, failed = {}, {}
    for ref in REFS:
        record, array = fit(a, None, ds, METHODS[ds][ref])
        if array is not None and record["status"] == "success":
            arrays[slug(ref)] = array
        else:
            failed[ref] = f"{record['failure_type']}: {record['failure_message']}"
    np.savez(Path(out) / "references" / f"{ds.lower()}.npz", **arrays)
    return failed


def references_of(out: str, ds: str) -> dict[str, NDArray | None]:
    with np.load(Path(out) / "references" / f"{ds.lower()}.npz") as z:
        return {r: (z[slug(r)] if slug(r) in z.files else None) for r in REFS}


# ---------------------------------------------------------------------------
# Fit units
# ---------------------------------------------------------------------------


Failure = tuple[str, str]  # (failure_type, failure_message)


def forced_lambda_inf(
    a: Mapping[str, NDArray], rows: NDArray, ds: str
) -> tuple[NDArray | None, list[analysis_tools.SelectorCall], Failure | None]:
    """The lambda = inf template of one selection, the selector calls it made, and
    the failure when the fit failed (a counted failure, as for every estimator)."""
    try:
        with (
            analysis_tools.forced_infinite_lambda() as calls,
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            t, inner = core.get_events_templates(
                a["data"],
                a["ix"][rows],
                a["lb"][rows],
                half_win=HALF_WIN[ds],
                **METHODS[ds]["L-curve"],
            )
    except analysis_tools.INFRASTRUCTURE_ERRORS:
        raise
    except Exception as error:  # a counted failure (decision OQ6)
        return None, [], (type(error).__name__, str(error))
    selected = None if inner is None else inner.get("opt_val")
    if selected != math.inf:
        message = f"forced lambda = inf fit selected lambda = {selected}"
        return None, calls, ("ForcedFitInvalid", message)
    if not calls:
        message = "forced lambda = inf fit made no L-curve selection call"
        return None, calls, ("ForcedFitInvalid", message)
    missing = [c for c in CLASSES[ds] if c not in t]
    if missing:
        message = f"forced lambda = inf fit returned no template for classes {missing}"
        return None, calls, ("ForcedFitInvalid", message)
    inf = np.stack([np.asarray(t[c], dtype=float) for c in CLASSES[ds]])
    if not np.all(np.isfinite(inf)):
        n_bad = int(np.sum(~np.isfinite(inf)))
        message = f"forced lambda = inf template has {n_bad} nonfinite values"
        return None, calls, ("NonfiniteTemplate", message)
    return inf, calls, None


def lambda_inf_fit(
    a: Mapping[str, NDArray], rows: NDArray, ds: str
) -> tuple[dict[str, Any], NDArray | None]:
    """The lambda = inf record (with the L-curve verticality) and its template."""
    array, calls, failure = forced_lambda_inf(a, rows, ds)
    if failure is not None:
        failure_type, message = failure
        record: dict[str, Any] = {"status": "failure", "flags": []}
        return record | {"failure_type": failure_type, "failure_message": message}, None
    # The selection's L-curve, captured from the forced fit's selector call.
    deviation = analysis_tools.lcurve_deviation(calls)
    return {"status": "success", "flags": []} | deviation, array


def fit_unit(task: tuple[str, str, int, int]) -> list[dict[str, Any]]:
    """Every estimator and the lambda = inf template on one selection."""
    out, ds, k, rep = task
    a = inputs(out, ds)
    refs = references_of(out, ds)
    sel = json.loads((Path(out) / "selections.json").read_text())[ds][str(k)][rep]
    rows = np.array(sel, dtype=int)
    unit = f"{ds}_K{k:02d}_r{rep:02d}"
    (Path(out) / "fits" / unit).mkdir(parents=True, exist_ok=True)
    records = []
    for name in [*METHODS[ds], "lambda=inf"]:
        if name == "lambda=inf":
            record, array = lambda_inf_fit(a, rows, ds)
        else:
            record, array = fit(a, rows, ds, METHODS[ds][name])
        record.update(dataset=ds, K=k, repetition=rep, estimator=name)
        if array is not None:
            record["template"] = f"fits/{unit}/{slug(name)}.npz"
            np.savez(Path(out) / record["template"], templates=array)
            if record["status"] == "success":
                record["distances"] = {
                    r: analysis_tools.distance(array, ref) if ref is not None else None
                    for r, ref in refs.items()
                }
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def failure_policy(
    records: Sequence[Mapping[str, Any]],
    failed_references: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Failed and flagged estimator fits, failed lambda = inf fits and failed
    full-data references; a (dataset, K, estimator) group with more than 10% failed
    repetitions is flagged (the failure policy)."""
    groups: dict[str, list[bool]] = {}
    for r in records:
        if r["estimator"] != "lambda=inf":
            group = f"{r['dataset']}|{r['K']}|{r['estimator']}"
            groups.setdefault(group, []).append(r["status"] == "failure")
    return {
        "n_fits": sum(len(keys) for keys in groups.values()),
        "n_failures": int(sum(sum(v) for v in groups.values())),
        "n_flagged_fits": sum(bool(r.get("flags")) for r in records),
        "n_lambda_inf_failures": sum(
            r["status"] == "failure" for r in records if r["estimator"] == "lambda=inf"
        ),
        "n_reference_failures": sum(len(v) for v in failed_references.values()),
        "groups_over_10pct": sorted(
            g for g, v in groups.items() if sum(v) > 0.1 * len(v)
        ),
    }


def baselines(
    records: Sequence[Mapping[str, Any]], ds: str, zero: Mapping[str, float | None]
) -> dict[str, Any]:
    """Per K: mean distance of each estimator and of the lambda = inf template over
    its successful repetitions, the zero template's distance (EEG in uV, ECG in mV),
    and the repetitions below zero; None where undefined (no fit or no reference)."""
    sc = SCALE[ds]
    out: dict[str, Any] = {"unit": UNIT[ds], "K": {}}
    for k in sorted({r["K"] for r in records}):
        d: dict[str, Any] = {
            "zero": {ref: None if z is None else z * sc for ref, z in zero.items()}
        }
        for name in ["lambda=inf", *sorted(METHODS[ds])]:
            rs = [
                r
                for r in records
                if r["K"] == k and r["estimator"] == name and r["status"] == "success"
            ]
            e: dict[str, Any] = {}
            for ref, z in zero.items():
                v = [r["distances"][ref] for r in rs if r["distances"][ref] is not None]
                e[ref] = float(np.mean(v)) * sc if v else None
                e[f"n_reps_below_zero_{ref}"] = (
                    None if z is None else int(sum(x < z for x in v))
                )
            if name == "lambda=inf":
                d["n_reps"] = len(rs)
            else:
                e["n_reps"] = len(rs)
            d[name] = e
        out["K"][str(k)] = d
    return out


def phi_summary(
    records: Sequence[Mapping[str, Any]], ds: str, sfreq: float
) -> dict[str, Any]:
    """The baseline segment and the AR(1) coefficients of the SURE(kernel) fits."""
    hw = HALF_WIN[ds]
    n_base = int(2 * hw * BASELINE_FRACTION)
    out: dict[str, Any] = {
        "baseline_segment": {
            "samples": n_base,
            "first_ms": -hw / sfreq * 1000,
            "last_ms": (n_base - 1 - hw) / sfreq * 1000,
        }
    }
    sk = [
        r
        for r in records
        if r["estimator"] == "SURE(kernel)"
        and r["status"] == "success"
        and r.get("phi_estimated") is not None
    ]
    if sk:  # pairwise (white-noise) fits estimate no AR(1) coefficient
        out["sure_kernel_phi_estimated"] = {
            "n": len(sk),
            "max": max(float(r["phi_estimated"]) for r in sk),
        }
    used = [float(r["phi_used"]) for r in sk if r["K"] == 5]
    if used:
        out["sure_kernel_phi_used_K05"] = {
            "n": len(used),
            "median": statistics.median(used),
            "min": min(used),
            "max": max(used),
        }
    return out


def csv_value(record: Mapping[str, Any], column: str) -> object:
    if column.startswith("distance_"):
        value = (record.get("distances") or {}).get(column[len("distance_") :])
    elif column == "kernel" and record.get("kernel") is not None:
        value = json.dumps(np.asarray(record["kernel"]).tolist())
    elif column == "flags":
        value = ";".join(record.get("flags") or [])
    else:
        value = record.get(column)
    return "" if value is None else value


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def prepare(
    out: Path, datasets: Sequence[str], data_dir: Path, reps: int
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, dict[str, str]]]:
    """Inputs (cache/), selections (selections.json): input facts, selections, and
    the K each dataset cannot fill with the reason."""
    checks: dict[str, dict[str, Any]] = {}
    selections: dict[str, Any] = {}
    unavailable: dict[str, dict[str, str]] = {}
    for ds in datasets:
        arrays, checks[ds] = load_input(ds, data_dir)
        np.savez(out / "cache" / f"{ds.lower()}.npz", **arrays)
        selections[ds], unavailable[ds] = make_selections(arrays["lb"], ds, reps)
    analysis_tools.write_json(out / "selections.json", selections)
    return checks, selections, unavailable


def fit_all(
    out: Path, selections: Mapping[str, Any], workers: int, reps: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """References and every fit unit of the datasets with at least one K; per-fit
    records in (dataset, K, repetition) order, and the failed references."""
    fitted = [ds for ds in selections if selections[ds]]
    failed = analysis_tools.run_pool(
        min(workers, max(len(fitted), 1)),
        fit_references,
        [(str(out), ds) for ds in fitted],
    )
    # SURE(kernel)-heavy EEG units and large K first, so the long ones do not trail.
    tasks = [
        (str(out), ds, k, rep)
        for ds in fitted
        for k in KS[ds][::-1]
        if str(k) in selections[ds]
        for rep in range(reps)
    ]
    units = analysis_tools.run_pool(workers, fit_unit, tasks)
    records = [
        r for _, unit in sorted(zip(tasks, units), key=lambda tu: tu[0]) for r in unit
    ]
    return records, dict(zip(fitted, failed))


def dataset_summary(
    out: Path, ds: str, records: Sequence[Mapping[str, Any]], sfreq: float
) -> dict[str, Any]:
    """Baselines, L-curve verticality and phi of one fitted dataset."""
    refs = references_of(str(out), ds)
    zero = {
        r: None if a is None else analysis_tools.distance(np.zeros_like(a), a)
        for r, a in refs.items()
    }
    inf = [r for r in records if r["estimator"] == "lambda=inf"]
    deviations = [r.get("lcurve_deviation_deg") for r in inf]
    dev = [v for v in deviations if v is not None]
    return {
        "status": "available",
        "baselines": baselines(records, ds, zero),
        "lcurve_verticality": {
            "n": len(dev),
            "max_deg": max(dev, default=None),
            "n_over_0p1_deg": sum(v > 0.1 for v in dev),
            "n_missing": len(inf) - len(dev),
        },
        "phi": phi_summary(records, ds, sfreq),
    }


def write_per_fit(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows([csv_value(r, c) for c in COLUMNS] for r in records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data-dir", default=os.environ.get("MNE_DATA", str(Path.home() / "mne_data"))
    )
    parser.add_argument("--out", default="./runs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--reps", type=int, default=REPETITIONS)
    args = parser.parse_args()
    started = datetime.now(UTC)
    inputs.cache_clear()  # a second run in this process must not read the last inputs
    out = Path(args.out) / "real_data"
    datasets = [d for d in DATASETS if d in args.datasets]
    for sub in ("cache", "references", "fits"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    checks, selections, unavailable = prepare(
        out, datasets, Path(args.data_dir), args.reps
    )
    records, failed_refs = fit_all(out, selections, args.workers, args.reps)
    write_per_fit(out / "per_fit.csv", records)
    summary: dict[str, Any] = {"failure_policy": failure_policy(records, failed_refs)}
    for ds in datasets:
        summary[ds] = {"input": checks[ds], "unavailable_K": unavailable[ds]}
        summary[ds]["failed_references"] = failed_refs.get(ds, {})
        if not selections[ds]:  # no K both classes can fill: nothing was fitted
            summary[ds]["status"] = "unavailable"
            continue
        mine = [r for r in records if r["dataset"] == ds]
        sfreq = float(checks[ds]["sfreq"])
        summary[ds] |= dataset_summary(out, ds, mine, sfreq)
    analysis_tools.write_json(out / "summary.json", summary)
    settings = {
        "datasets": datasets,
        "reps": args.reps,
        "ks": {ds: KS[ds] for ds in datasets},
        "half_win": HALF_WIN,
        "methods": {ds: METHODS[ds] for ds in datasets},
        "selection_seed": SELECTION_SEED,
        "workers": args.workers,
        "data_dir": args.data_dir,
    }
    analysis_tools.write_run_info(out, settings, started)


if __name__ == "__main__":
    main()
