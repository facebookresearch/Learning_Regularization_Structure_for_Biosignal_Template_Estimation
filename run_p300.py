# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""P300 analyses of the paper (BNCI2014-009, 10 subjects).

Every estimator is fitted on K = 5, 10, 20, 50 events per class: analysis A on
the whole recording, scored against three full-data references; analysis B on
two blocks, scored by AUC on the held-out block (also on all training events).
K = 5 is the central batch with its signed-rank tests. Also: the Population
control, the zero and lambda = inf baselines, the AUC difference to Average,
the learned kernels' null space and the L-curve's verticality.

Writes ``<out>/p300/``: ``cache/``, ``references/``, ``population/``,
``selections.json``, ``fits/`` (templates, volts), ``per_fit.csv``,
``summary.json``, ``run_info.json``.
Usage: python3 -B run_p300.py [--data-dir DIR] [--out DIR] [--workers N]
           [--subjects 1 2 ...] [--reps N]
"""

from __future__ import annotations

import argparse
import csv
import functools
import itertools
import json
import math
import os
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, cast

import analysis_tools
import data_loaders_p300
import heldout_p300
import numpy as np
import protocol
import scipy
from numpy.typing import NDArray

HALF, GUARD = heldout_p300.HALF_WIN, heldout_p300.GUARD
KS: tuple[int, ...] = (5, 10, 20, 50)
ALL = "all"
UV = 1e6
SELECTION_SEED = int(str(protocol.DECISIONS["OQ2_selection_seed"]))
RESAMPLES = int(str(protocol.DECISIONS["OQ3_bootstrap_resamples"]))
SEED = int(str(protocol.DECISIONS["OQ3_bootstrap_seed"]))
ALPHA = float(str(protocol.DECISIONS["OQ3_alpha"]))
FAILURES = bool(protocol.DECISIONS["OQ6_exceptions_are_failures"])
ESTIMATORS = protocol.ESTIMATORS
REFS = protocol.REFERENCES
ANCHOR = protocol.ANCHOR
ENDPOINTS = ("distance_OLS", "distance_EmpBayes", "distance_Average", "heldout_auc")
TRAINING_COUNTS = {0: 960, 1: 192}  # all eligible events of two blocks
PAIRED_ESTIMATORS = (*protocol.CORE_METHODS[1:], protocol.FILTERED)  # vs Average
COLUMNS = (
    *[
        "analysis",
        "K",
        "subject",
        "fold",
        "repetition",
        "estimator",
        "status",
        "failure_type",
    ],
    *["failure_message", "phi_estimate_phi"],
    *(f"distance_{r}" for r in REFS),
    *(f"distance_{r}_uV" for r in REFS),
    *[
        "auc",
        "n_heldout_events",
        "n_heldout_targets",
        "lambda",
        "kernel",
        "noise_var",
        "flags",
    ],
    *["warnings", "inner", "template", "lcurve_deviation_deg"],
    *["lcurve_deviation_missing", "seconds"],
)

Condition = int | str
Key = tuple[str, Condition, int, int | None, int, str]
Records = Mapping[Key, Mapping[str, Any]]
CellKey = int | tuple[int, int]  # a repetition, or a (fold, repetition) pair
Cells = dict[int, dict[CellKey, float | None]]


def kname(k: Condition) -> str:
    return "Kall" if k == ALL else f"K{int(k):02d}"


def unit_name(an: str, k: Condition, s: int, fold: int | None, rep: int) -> str:
    if an == "A":
        return f"A_{kname(k)}_s{s:02d}_r{rep:02d}"
    return f"B_{kname(k)}_s{s:02d}_f{fold}_r{rep:02d}"


def fit_key(
    an: str, k: Condition, s: int, fold: int | None, rep: int, name: str
) -> str:
    if an == "A":
        return f"A|{kname(k)}|s{s:02d}|r{rep:02d}|{name}"
    return f"B|{kname(k)}|s{s:02d}|f{fold}|r{rep:02d}|{name}"


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


# ---------------------------------------------------------------------------
# Inputs: preprocessing, selections, references, Population
# ---------------------------------------------------------------------------


def preprocess(
    task: tuple[str, int], bandpass: tuple[float, float] = protocol.BANDPASS
) -> None:
    out, s = Path(task[0]), task[1]
    blocks = data_loaders_p300.load_bnci2014_009_blocks(
        s, bandpass=bandpass, resample=protocol.RESAMPLE
    )
    ids = [block.block for block in blocks]
    # Held-out folds are numbered 0 .. N_BLOCKS - 1 throughout the analysis.
    if ids != list(range(protocol.N_BLOCKS)):
        raise RuntimeError(
            f"subject {s}: block ids {ids}, expected {list(range(protocol.N_BLOCKS))}"
        )
    if len(blocks[0].channels) != protocol.N_CHANNELS:
        raise RuntimeError(
            f"subject {s}: {len(blocks[0].channels)} channels "
            f"{blocks[0].channels}, expected {protocol.N_CHANNELS}"
        )
    for block in blocks:
        eligible = block.eligible(HALF, GUARD)
        if not np.all(eligible):
            raise RuntimeError(
                f"subject {s} block {block.block}: {int(np.sum(~eligible))} of "
                f"{len(eligible)} events lack a complete window {GUARD} samples "
                "from the block edges; expected all eligible"
            )
        counts = {c: int(np.sum(block.labels == c)) for c in (0, 1)}
        if counts != protocol.BLOCK_CLASS_COUNTS:
            raise RuntimeError(
                f"subject {s} block {block.block}: class counts {counts}, "
                f"expected {protocol.BLOCK_CLASS_COUNTS}"
            )
    analysis_tools.save_blocks(out / "cache" / f"subject_{s:02d}.npz", blocks)


@functools.cache
def blocks_of(out: str, s: int) -> list[heldout_p300.RecordingBlock]:
    return analysis_tools.load_blocks(Path(out) / "cache" / f"subject_{s:02d}.npz", s)


@functools.cache
def selections_of(out: str) -> dict[str, Any]:
    return json.loads((Path(out) / "selections.json").read_text())


@functools.cache
def references_of(out: str, s: int) -> dict[str, NDArray | None]:
    with np.load(Path(out) / "references" / f"subject_{s:02d}.npz") as z:
        return {r: (z[slug(r)] if slug(r) in z.files else None) for r in REFS}


def make_selections(out: Path, subjects: Sequence[int], reps: int) -> dict[str, Any]:
    """A: fresh RandomState(42) per subject and K; B: ``select_events``; B all events."""
    sel: dict[str, Any] = {"A": {}, "B": {}, "B_all": {}}
    for k in KS:
        sel["A"][str(k)], sel["B"][str(k)] = {}, {}
        for s in subjects:
            blocks = blocks_of(str(out), s)
            sel["A"][str(k)][str(s)] = analysis_tools.whole_recording_selections(
                blocks, np.random.RandomState(SELECTION_SEED), k, reps, HALF, GUARD
            )
            sel["B"][str(k)][str(s)] = {}
            for b in blocks:
                training = heldout_p300.training_blocks(blocks, b.block)
                drawn = heldout_p300.select_events(
                    training, k, reps, heldout_p300.SEED, HALF, GUARD
                )
                sel["B"][str(k)][str(s)][str(b.block)] = [x.tolist() for x in drawn]
    for s in subjects:
        blocks = blocks_of(str(out), s)
        by_id = {b.block: b for b in blocks}
        sel["B_all"][str(s)] = {}
        for block in blocks:
            training = heldout_p300.training_blocks(blocks, block.block)
            rows = [
                [int(b.block), int(e)]
                for b in training
                for e in np.flatnonzero(b.eligible(HALF, GUARD))
            ]
            counts = {c: sum(by_id[b].labels[e] == c for b, e in rows) for c in (0, 1)}
            if counts != TRAINING_COUNTS:
                raise RuntimeError(
                    f"subject {s} fold {block.block}: all-events class counts "
                    f"{counts}, expected {TRAINING_COUNTS}"
                )
            sel["B_all"][str(s)][str(block.block)] = rows
    analysis_tools.write_json(out / "selections.json", sel)
    return sel


def nesting(sel: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """Per pair of consecutive K, the selections that are subsets of the larger K's."""
    out = {}
    for small, large in zip(KS, KS[1:]):
        a, b = sel["A"][str(small)], sel["B"][str(small)]
        pairs = {
            "A": [
                (x, sel["A"][str(large)][s][r]) for s in a for r, x in enumerate(a[s])
            ],
            "B": [
                (x, sel["B"][str(large)][s][f][r])
                for s in b
                for f in b[s]
                for r, x in enumerate(b[s][f])
            ],
        }
        out[f"{small}<={large}"] = {}
        for an, p in pairs.items():
            out[f"{small}<={large}"][f"{an}_nested"] = sum(
                {tuple(x) for x in i} <= {tuple(x) for x in j} for i, j in p
            )
            out[f"{small}<={large}"][f"{an}_total"] = len(p)
    return out


def reference_subject(task: tuple[str, int]) -> dict[str, Any]:
    """Full-data references on every eligible event, joined with block starts."""
    out, s = task
    blocks = blocks_of(out, s)
    rows = np.array(
        [(b.block, e) for b in blocks for e in np.flatnonzero(b.eligible(HALF, GUARD))]
    )
    estimators = analysis_tools.default_estimators()
    records, arrays = {}, {}
    for ref in REFS:
        records[ref], array = analysis_tools.fit_estimator(
            estimators, ref, blocks, rows, HALF, FAILURES
        )
        if records[ref]["status"] == "success":
            arrays[slug(ref)] = array
    np.savez(Path(out) / "references" / f"subject_{s:02d}.npz", **arrays)
    return {"records": records, "n_events": len(rows)}


def make_population(out: Path, subjects: Sequence[int]) -> dict[int, NDArray | None]:
    ref = str(protocol.DECISIONS["OQ8_population_reference"])
    available = {s: references_of(str(out), s)[ref] for s in subjects}
    population = {s: analysis_tools.population_template(available, s) for s in subjects}
    for s, array in population.items():
        if array is not None:
            np.savez(out / "population" / f"subject_{s:02d}.npz", templates=array)
    return population


# ---------------------------------------------------------------------------
# Fit units
# ---------------------------------------------------------------------------


def unit_inputs(
    out: str, an: str, k: Condition, s: int, fold: int | None, rep: int
) -> tuple[
    NDArray, list[heldout_p300.RecordingBlock], heldout_p300.RecordingBlock | None
]:
    """The saved selection rows, the training blocks and (analysis B) the test block."""
    blocks = blocks_of(out, s)
    sel = selections_of(out)
    if an == "A":
        return np.array(sel["A"][str(k)][str(s)][rep], dtype=int), list(blocks), None
    assert fold is not None, (
        f"analysis B unit of subject {s}, K {k} has no held-out fold"
    )
    chosen = sel["B_all"] if k == ALL else sel["B"][str(k)]
    reps = chosen[str(s)][str(fold)]
    rows = np.array(reps if k == ALL else reps[rep], dtype=int)
    test = next(b for b in blocks if b.block == fold)
    return rows, heldout_p300.training_blocks(blocks, fold), test


def run_unit(
    task: tuple[str, str, Condition, int, int | None, int],
) -> list[dict[str, Any]]:
    """Every estimator, lambda = inf and (B) zero on one saved selection."""
    out, an, k, s, fold, rep = task
    rows, training, test = unit_inputs(out, an, k, s, fold, rep)
    unit = unit_name(an, k, s, fold, rep)
    (Path(out) / "fits" / unit).mkdir(parents=True, exist_ok=True)
    refs = references_of(out, s)
    base = {"analysis": an, "K": k, "subject": s, "fold": fold, "repetition": rep}
    phi = analysis_tools.training_phi(training, rows, HALF)
    records = []
    fits = analysis_tools.fit_unit(
        analysis_tools.default_estimators(),
        ESTIMATORS,
        training,
        rows,
        test,
        refs,
        HALF,
        GUARD,
        FAILURES,
    )
    for name, (record, array) in zip(ESTIMATORS, fits):
        record.update(base, estimator=name, phi_estimate_phi=phi)
        record["key"] = fit_key(an, k, s, fold, rep, name)
        if array is not None:
            record["template"] = f"fits/{unit}/{slug(name)}.npz"
            np.savez(Path(out) / record["template"], templates=array)
        records.append(record)
    inf, calls, failure = forced_lambda_inf(training, rows)
    if test is None:
        records.append(inf_distance_record(out, unit, base, refs, inf, failure))
        if k == 5:
            # The selection's L-curve, captured from the forced fit's selector call.
            records[-1] |= analysis_tools.lcurve_deviation(calls)
    else:
        records += baseline_auc_records(base, test, inf, failure)
    return records


def inf_distance_record(
    out: str,
    unit: str,
    base: Mapping[str, Any],
    refs: Mapping[str, NDArray | None],
    inf: NDArray | None,
    failure: str | None,
) -> dict[str, Any]:
    """Analysis A: the lambda = inf template (saved) and its distances in uV."""
    record: dict[str, Any] = {**base, "estimator": "lambda=inf"}
    record["key"] = baseline_key(base, "lambda=inf")
    if inf is None:
        record |= {"status": "failure", "failure_message": failure}
        return record | {"distances_uV": dict.fromkeys(REFS)}
    rel = f"fits/{unit}/lambda_inf.npz"
    np.savez(Path(out) / rel, templates=inf)
    raw = {r: uv_distance(inf * UV, refs[r]) for r in REFS}
    return record | {"status": "success", "template": rel, "distances_uV": raw}


def baseline_auc_records(
    base: Mapping[str, Any],
    test: heldout_p300.RecordingBlock,
    inf: NDArray | None,
    failure: str | None,
) -> list[dict[str, Any]]:
    """Analysis B: held-out AUC of the lambda = inf and zero templates."""
    zero = np.zeros((2, 2 * HALF, test.data.shape[1]))
    records = []
    for name, a in (("lambda=inf", inf), ("zero", zero)):
        record = {**base, "estimator": name, "key": baseline_key(base, name)}
        if a is None:
            records.append(record | {"status": "failure", "failure_message": failure})
            records[-1]["auc"] = None
        else:
            records.append(record | baseline_auc(test, a))
    return records


def baseline_key(base: Mapping[str, Any], name: str) -> str:
    return fit_key(
        base["analysis"],
        base["K"],
        base["subject"],
        base["fold"],
        base["repetition"],
        name,
    )


def baseline_auc(test: heldout_p300.RecordingBlock, a: NDArray) -> dict[str, Any]:
    try:
        scores, labels = heldout_p300.score_events(
            test, {"0": a[0], "1": a[1]}, HALF, GUARD
        )
        return {"status": "success", "auc": float(heldout_p300.auc(scores, labels))}
    except ValueError as error:  # nonfinite scores or a one-class block
        return {"status": "failure", "auc": None} | {
            "failure_message": f"held-out scoring: {error}"
        }


def all_units(
    out: Path, subjects: Sequence[int], reps: int
) -> list[tuple[str, str, Condition, int, int | None, int]]:
    """All-events units first (the slowest), then K from large to small."""
    units: list[tuple[str, str, Condition, int, int | None, int]] = [
        (str(out), "B", ALL, s, f, 0)
        for s in subjects
        for f in range(protocol.N_BLOCKS)
    ]
    for k in reversed(KS):
        for s in subjects:
            units += [(str(out), "A", k, s, None, r) for r in range(reps)]
            units += [
                (str(out), "B", k, s, f, r)
                for f in range(protocol.N_BLOCKS)
                for r in range(reps)
            ]
    return units


def uv_distance(array_uv: NDArray, reference: NDArray | None) -> float | None:
    """Distance in uV to a reference in volts; None when the reference fit failed."""
    if reference is None:
        return None
    return analysis_tools.distance(array_uv, reference * UV)


def forced_lambda_inf(
    training: Sequence[heldout_p300.RecordingBlock], rows: NDArray
) -> tuple[NDArray | None, list[analysis_tools.SelectorCall], str | None]:
    """The lambda = inf template of one selection, the selector calls it made, and
    the reason when the fit failed (a counted failure, as for every estimator)."""
    try:
        with (
            analysis_tools.forced_infinite_lambda() as calls,
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            t, inner = heldout_p300.fit_templates(
                training, rows, protocol.METHODS_AR1COL["L-curve"], HALF
            )
    except analysis_tools.INFRASTRUCTURE_ERRORS:
        raise
    except Exception as error:  # a counted failure (decision OQ6)
        return None, [], f"{type(error).__name__}: {error}"
    selected = None if inner is None else inner.get("opt_val")
    if selected != math.inf:
        return None, calls, f"forced lambda = inf fit selected lambda = {selected}"
    if not calls:
        return None, calls, "forced lambda = inf fit made no L-curve selection call"
    inf = np.stack([np.asarray(t[c], dtype=float) for c in ("0", "1")])
    if not np.all(np.isfinite(inf)):
        n_bad = int(np.sum(~np.isfinite(inf)))
        return None, calls, f"forced lambda = inf template has {n_bad} nonfinite values"
    return inf, calls, None


def score_population(
    out: Path, subjects: Sequence[int], population: Mapping[int, NDArray | None]
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Population distances and per-block AUC, and the zero template's distances."""
    pop: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for s in subjects:
        refs = references_of(str(out), s)
        zero = np.zeros((2, 2 * HALF, protocol.N_CHANNELS))
        rows.append(
            {"analysis": "A", "subject": s, "estimator": "zero", "status": "success"}
            | {"distances_uV": {r: uv_distance(zero, refs[r]) for r in REFS}}
        )
        array = population[s]
        if array is None:
            pop[s] = {"status": "unavailable", "auc_by_fold": {}, "distances": None}
            continue
        distances = {
            r: None if refs[r] is None else analysis_tools.distance(array, refs[r])
            for r in REFS
        }
        folds, fold_rows = population_folds(blocks_of(str(out), s), array)
        pop[s] = {
            "status": "success",
            "distances": distances,
            "auc_by_fold": folds,
            "auc": mean_or_none(list(folds.values())),
        }
        raw = {r: uv_distance(array * UV, refs[r]) for r in REFS}
        rows.append(
            {"analysis": "A", "subject": s, "estimator": protocol.POPULATION}
            | {"status": "success", "distances": distances, "distances_uV": raw}
        )
        rows += fold_rows
    return pop, rows


def population_folds(
    blocks: Sequence[heldout_p300.RecordingBlock], array: NDArray
) -> tuple[dict[int, float], list[dict[str, Any]]]:
    """Held-out AUC of the Population template on each block; a block whose
    scoring fails is a counted failure and the other blocks are kept."""
    templates: Mapping[object, NDArray] = {"0": array[0], "1": array[1]}
    folds: dict[int, float] = {}
    rows: list[dict[str, Any]] = []
    for block in blocks:
        row = {"analysis": "B", "subject": block.subject, "fold": block.block}
        row |= {"repetition": 0, "estimator": protocol.POPULATION}
        row["key"] = f"P|B|s{block.subject:02d}|f{block.block}"
        try:
            cells = heldout_p300.population_cells(
                [block], templates, protocol.POPULATION, HALF, GUARD
            )
        except ValueError as error:  # nonfinite scores or a one-class block
            message = f"held-out scoring: {error}"
            rows.append(
                row | {"status": "failure", "auc": None, "failure_message": message}
            )
            continue
        folds[block.block] = cells[0][4]
        rows.append(row | {"status": "success", "auc": cells[0][4]})
    return folds, rows


# ---------------------------------------------------------------------------
# Central batch (K = 5): tests, descriptive statistics, rankings
# ---------------------------------------------------------------------------


def endpoint_values(
    records: Records,
    subjects: Sequence[int],
    reps: int,
    endpoint: str,
    name: str,
    k: Condition = 5,
) -> Cells:
    """Per subject, cell -> value (None when the fit failed): repetitions for
    distances, (fold, repetition) for AUC."""
    out: Cells = {}
    for s in subjects:
        cells: dict[CellKey, float | None] = {}
        if endpoint.startswith("distance_"):
            ref = endpoint.removeprefix("distance_")
            for r in range(reps):
                rec = records["A", k, s, None, r, name]
                cells[r] = rec["distances"][ref] if rec["status"] == "success" else None
        else:
            for f in range(protocol.N_BLOCKS):
                for r in range(1 if k == ALL else reps):
                    rec = records["B", k, s, f, r, name]
                    cells[f, r] = rec["auc"] if rec["status"] == "success" else None
        out[s] = cells
    return out


def families() -> dict[str, list[tuple[str, str]]]:
    s4 = cast(Sequence[str], protocol.DECISIONS["OQ4_s4_endpoints"])
    return {
        "primary": [("distance_OLS", m) for m in protocol.PRIMARY_COMPARATORS],
        "S1": [("distance_EmpBayes", m) for m in protocol.PRIMARY_COMPARATORS],
        "S2": [("distance_Average", m) for m in protocol.PRIMARY_COMPARATORS],
        "S3": [("heldout_auc", m) for m in protocol.PRIMARY_COMPARATORS],
        "S4": [(str(e), protocol.FILTERED) for e in s4],
    }


def holm_families(comparisons: Mapping[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Holm within each family at its prespecified size; a test without a p-value
    enters with p = 1 and gets no adjusted value."""
    out = {}
    for fam, pairs in families().items():
        tests = [comparisons[f"{e}|{m}"]["test"] for e, m in pairs]
        adjusted = analysis_tools.holm(
            [t["p"] if t["p"] is not None else 1.0 for t in tests]
        )
        for t, adj in zip(tests, adjusted):
            t["p_holm"] = adj if t["p"] is not None else None
            t["holm_placeholder"] = t["p"] is None
            t["reject_at_alpha"] = (adj <= ALPHA) if t["p"] is not None else None
        out[fam] = [f"{e}|{m}" for e, m in pairs]
    return out


def central_statistics(
    records: Records,
    subjects: Sequence[int],
    reps: int,
    pop: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    inferential = {pair: fam for fam, pairs in families().items() for pair in pairs}
    comparisons = {}
    for endpoint in ENDPOINTS:
        anchor = endpoint_values(records, subjects, reps, endpoint, ANCHOR)
        for name in ESTIMATORS:
            if name == ANCHOR:
                continue
            effects = analysis_tools.paired_effects(
                anchor,
                endpoint_values(records, subjects, reps, endpoint, name),
                log_ratio=endpoint.startswith("distance_"),
            )
            summary = analysis_tools.summarize(
                effects, RESAMPLES, SEED, test=(endpoint, name) in inferential
            )
            summary["family"] = inferential.get((endpoint, name), "descriptive")
            if endpoint.startswith("distance_") and summary["mean"] is not None:
                summary["geometric_distance_ratio"] = math.exp(
                    float(str(summary["mean"]))
                )
            comparisons[f"{endpoint}|{name}"] = summary
    return {
        "comparisons": comparisons,
        "families": holm_families(comparisons),
        "descriptive": descriptive(records, subjects, reps, pop),
    }


def subject_entry(
    a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]]
) -> dict[str, float | None]:
    entry: dict[str, float | None] = {}
    for ref in REFS:
        vals = [
            x["distances"][ref]
            for x in a
            if x["status"] == "success" and x["distances"][ref] is not None
        ]
        entry[f"distance_{ref}_uV"] = float(np.mean(vals)) * 1e6 if vals else None
    aucs = [x["auc"] for x in b if x["status"] == "success"]
    entry["heldout_auc"] = float(np.mean(aucs)) if aucs else None
    return entry


def population_entry(p: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": p["status"],
        "distances_uV": {
            r: (v * 1e6 if v is not None else None)
            for r, v in (p["distances"] or {}).items()
        },
        "heldout_auc": p.get("auc"),
    }


def descriptive(
    records: Records,
    subjects: Sequence[int],
    reps: int,
    pop: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "subject_means": {},
        "failures": {},
        "flags": {},
        "subject_flags": [],
        "population": {},
    }
    for s in subjects:
        means, fails, flags = {}, {}, {}
        for name in ESTIMATORS:
            a = [records["A", 5, s, None, r, name] for r in range(reps)]
            b = [
                records["B", 5, s, f, r, name]
                for f in range(protocol.N_BLOCKS)
                for r in range(reps)
            ]
            means[name] = subject_entry(a, b)
            fails[name] = {
                "A": sum(x["status"] == "failure" for x in a),
                "B": sum(x["status"] == "failure" for x in b),
            }
            flags[name] = {
                "A": sum(bool(x.get("flags")) for x in a),
                "B": sum(bool(x.get("flags")) for x in b),
            }
            if fails[name]["A"] > 0.1 * len(a) or fails[name]["B"] > 0.1 * len(b):
                out["subject_flags"].append(
                    {"subject": s, "estimator": name, **fails[name]}
                )
        out["subject_means"][str(s)] = means
        out["failures"][str(s)] = fails
        out["flags"][str(s)] = flags
        out["population"][str(s)] = population_entry(pop[s])
    out["rankings"] = rankings(out["subject_means"])
    molina = [
        r for key, r in records.items() if key[1] == 5 and key[5] == protocol.FILTERED
    ]
    out["molina_inclusion_all_finite"] = all(r["status"] == "success" for r in molina)
    return out


def rankings(
    subject_means: Mapping[str, Mapping[str, Mapping[str, float | None]]],
) -> dict[str, Any]:
    """Estimators ordered by the mean over subjects of their subject means."""
    fields = {f"distance_{r}": f"distance_{r}_uV" for r in REFS} | {
        "heldout_auc": "heldout_auc"
    }
    out: dict[str, Any] = {"order": {}}
    means: dict[str, dict[str, float]] = {}
    for endpoint, field in fields.items():
        vals = {}
        for name in ESTIMATORS:
            per = [
                subject_means[s][name][field]
                for s in subject_means
                if subject_means[s][name][field] is not None
            ]
            if per:
                vals[name] = float(np.mean(per))
        means[endpoint] = vals
        out["order"][endpoint] = sorted(
            vals, key=lambda n: vals[n], reverse=(endpoint == "heldout_auc")
        )
    out["mean_over_subjects"] = means
    return out


# ---------------------------------------------------------------------------
# K curve (descriptive) and the paired AUC difference to Average
# ---------------------------------------------------------------------------


def level_summary(
    per_subject: Mapping[int, float | None], scale: float = 1.0
) -> dict[str, Any]:
    vals = {s: v * scale for s, v in per_subject.items() if v is not None}
    arr = list(vals.values())
    return {
        "per_subject": {str(s): v for s, v in vals.items()},
        "n_subjects": len(arr),
        "mean": float(np.mean(arr)) if arr else None,
        "median": float(np.median(arr)) if arr else None,
        "ci95": analysis_tools.bootstrap_mean_ci(arr, RESAMPLES, SEED),
    }


def mean_cells(cells: Cells) -> dict[int, float | None]:
    out: dict[int, float | None] = {}
    for s, c in cells.items():
        v = [x for x in c.values() if x is not None and math.isfinite(x)]
        out[s] = float(np.mean(v)) if v else None
    return out


def kcurve_statistics(
    records: Records,
    subjects: Sequence[int],
    reps: int,
    pop: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    endpoints = [f"distance_{r}" for r in REFS] + ["heldout_auc"]
    out: dict[str, Any] = {"levels": {}, "paired_vs_anchor": {}}
    for k in [*KS, ALL]:
        kk = kname(k)
        levels, paired = {}, {}
        for ep in ["heldout_auc"] if k == ALL else endpoints:
            scale = 1e6 if ep.startswith("distance_") else 1.0
            anchor = endpoint_values(records, subjects, reps, ep, ANCHOR, k)
            for name in ESTIMATORS:
                cells = endpoint_values(records, subjects, reps, ep, name, k)
                levels[f"{ep}|{name}"] = level_summary(mean_cells(cells), scale)
                if name != ANCHOR:
                    eff = analysis_tools.paired_effects(
                        anchor, cells, log_ratio=ep.startswith("distance_")
                    )
                    summ = analysis_tools.summarize(eff, RESAMPLES, SEED, test=False)
                    if ep.startswith("distance_") and summ["mean"] is not None:
                        summ["geometric_distance_ratio"] = math.exp(
                            cast(float, summ["mean"])
                        )
                    paired[f"{ep}|{name}"] = summ
        out["levels"][kk], out["paired_vs_anchor"][kk] = levels, paired
    out["population"] = {
        "auc": level_summary({s: pop[s].get("auc") for s in subjects}),
        **{
            f"distance_{r}": level_summary(
                {s: (pop[s]["distances"] or {}).get(r) for s in subjects}, 1e6
            )
            for r in REFS
        },
    }
    out["all_events_estimators"] = list(ESTIMATORS)
    out["all_events_skipped"] = []
    return out


def paired_vs_average(
    records: Records, subjects: Sequence[int], reps: int
) -> list[dict[str, Any]]:
    rows = []
    for k in [*KS, ALL]:
        avg = endpoint_values(records, subjects, reps, "heldout_auc", "Average", k)
        for est in PAIRED_ESTIMATORS:
            cells = endpoint_values(records, subjects, reps, "heldout_auc", est, k)
            rows.append(
                {"level": kname(k), "estimator": est}
                | analysis_tools.paired_difference(avg, cells)
            )
    return rows


# ---------------------------------------------------------------------------
# Baselines (raw distances and AUC; the noise-corrected part is run_reference_noise.py)
# ---------------------------------------------------------------------------


def baselines(
    rows: Sequence[Mapping[str, Any]],
    records: Records,
    subjects: Sequence[int],
    reps: int,
    pop: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Cohort mean of subject means per K and estimator, next to the zero, lambda = inf
    and Population templates (distances in uV), and the held-out AUC table."""
    failed: dict[str, list[str]] = defaultdict(list)
    res = distance_baselines(rows, records, subjects, reps, failed)
    res["auc"] = auc_baselines(rows, records, subjects, pop, failed)
    for row in rows:  # failed lambda = inf, zero and Population cells
        if row["status"] == "failure":
            failed[baseline_group(row)].append(row["key"])
    res["failed_fits"] = dict(failed)
    return res


def baseline_group(row: Mapping[str, Any]) -> str:
    """The ``failed_fits`` group of a baseline row, named like the estimators'."""
    k = row.get("K", "")
    if row["analysis"] == "A":
        return f"{k}|{row['estimator']}"
    return f"B{k}|{row['estimator']}"


def distance_baselines(
    rows: Sequence[Mapping[str, Any]],
    records: Records,
    subjects: Sequence[int],
    reps: int,
    failed: dict[str, list[str]],
) -> dict[str, Any]:
    cell: dict[tuple[int, str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for k, s, r, name in itertools.product(KS, subjects, range(reps), ESTIMATORS):
        rec = records["A", k, s, None, r, name]
        if rec["status"] != "success":
            failed[f"{k}|{name}"].append(rec["key"])
            continue
        for ref in REFS:
            if rec["distances"][ref] is not None:
                cell[k, name, s][ref].append(rec["distances"][ref] * UV)
    inf_a = [u for u in rows if u["estimator"] == "lambda=inf" and u["analysis"] == "A"]
    for u in sorted(inf_a, key=lambda u: (u["K"], u["subject"], u["repetition"])):
        for ref in REFS:
            if u["distances_uV"][ref] is not None:
                cell[u["K"], "lambda=inf", u["subject"]][ref].append(
                    u["distances_uV"][ref]
                )
    fixed = {
        (u["estimator"], u["subject"]): u
        for u in rows
        if u["analysis"] == "A" and "K" not in u
    }
    for s, k, name, ref in itertools.product(
        subjects, KS, ("zero", protocol.POPULATION), REFS
    ):
        # No Population row when its template is unavailable.
        value = fixed[name, s]["distances_uV"][ref] if (name, s) in fixed else None
        if value is not None:
            cell[k, name, s][ref].append(value)
    return cohort_distances(cell, subjects)


def mean_or_none(values: Sequence[float]) -> float | None:
    return statistic_or_none(np.mean, values)


def statistic_or_none(
    statistic: Callable[[Sequence[float] | NDArray], object],
    values: Sequence[float] | NDArray,
) -> float | None:
    """*statistic* of *values* as a float; None for an empty slice."""
    return float(cast(float, statistic(values))) if len(values) else None


def cohort_distances(
    cell: Mapping[tuple[int, str, int], Mapping[str, list[float]]],
    subjects: Sequence[int],
) -> dict[str, Any]:
    """Cohort mean of subject means; like the K-curve levels, a subject without a
    successful fit (or an available Population template or reference) is left out."""
    names = ["zero", "lambda=inf", *ESTIMATORS, protocol.POPULATION]
    res: dict[str, Any] = {"K": {}}
    for k in KS:
        res["K"][str(k)] = {}
        for n in names:
            subj = {
                s: {ref: mean_or_none(cell[k, n, s].get(ref, [])) for ref in REFS}
                for s in subjects
                if cell.get((k, n, s))
            }
            row: dict[str, Any] = {
                ref: mean_or_none([v[ref] for v in subj.values() if v[ref] is not None])
                for ref in REFS
            }
            row["subject"] = subj
            for ref in REFS:
                zero = {s: mean_or_none(cell[k, "zero", s].get(ref, [])) for s in subj}
                row[f"n_subjects_below_zero_{ref}"] = sum(
                    v[ref] is not None
                    and zero[s] is not None
                    and v[ref] < cast(float, zero[s])
                    for s, v in subj.items()
                )
            res["K"][str(k)][n] = row
    res["first_K_below_zero"] = {}
    for n in names[1:]:
        res["first_K_below_zero"][n] = {}
        for ref in REFS:
            cohort = [(k, res["K"][str(k)][n][ref]) for k in KS]
            zero = {k: res["K"][str(k)]["zero"][ref] for k in KS}
            below = [
                k
                for k, v in cohort
                if v is not None and zero[k] is not None and v < zero[k]
            ]
            res["first_K_below_zero"][n] |= {ref: below[0] if below else None}
            res["first_K_below_zero"][n] |= {f"{ref}_all_K_below": below}
    return res


def auc_baselines(
    rows: Sequence[Mapping[str, Any]],
    records: Records,
    subjects: Sequence[int],
    pop: Mapping[int, Mapping[str, Any]],
    failed: dict[str, list[str]],
) -> dict[str, Any]:
    """Cohort mean of subject-mean held-out AUC per K and template."""
    base_auc = {
        (u["K"], u["subject"], u["fold"], u["repetition"], u["estimator"]): u["auc"]
        for u in rows
        if u["analysis"] == "B" and "K" in u
    }
    auc = subject_aucs(base_auc, records, failed)
    # The zero template scores every event 0, so its AUC is exactly 1/2.
    zero_half = all(
        math.isclose(value, 0.5, rel_tol=0.0, abs_tol=0.0)
        for u, value in base_auc.items()
        if u[4] == "zero" and value is not None
    )
    res: dict[str, Any] = {"zero_all_exactly_half": zero_half}
    for k in [str(x) for x in [*KS, ALL]]:
        res[k] = {}
        for n in ["zero", "lambda=inf", *ESTIMATORS, protocol.POPULATION]:
            d = (
                {s: list(pop[s]["auc_by_fold"].values()) for s in subjects}
                if n == protocol.POPULATION
                else {s: auc[k, s][n] for s in subjects}
            )
            # A subject without a successful fit (or Population) is left out.
            subj_auc = {s: float(np.mean(v)) for s, v in sorted(d.items()) if v}
            res[k][n] = {
                "mean": float(np.mean(list(subj_auc.values()))) if subj_auc else None,
                "subject": subj_auc,
                "n_cells": int(sum(len(v) for v in d.values())),
                "n_subjects_above_half": int(sum(v > 0.5 for v in subj_auc.values())),
            }
    return res


def subject_aucs(
    base_auc: Mapping[tuple[Condition, int, int | None, int, str], float | None],
    records: Records,
    failed: dict[str, list[str]],
) -> dict[tuple[str, int], dict[str, list[float]]]:
    """(K, subject) -> template -> the successful held-out AUCs, in unit order;
    failed estimator fits are recorded in *failed*."""
    auc: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    b_units = sorted(
        {(key[1], key[2], key[3], key[4]) for key in records if key[0] == "B"},
        key=lambda u: (u[0] == ALL, 0 if u[0] == ALL else u[0], u[1], u[2], u[3]),
    )
    for k, s, f, r in b_units:
        for name in ("lambda=inf", "zero"):
            value = base_auc.get((k, s, f, r, name))
            if value is not None:
                auc[str(k), s][name].append(value)
        for n in ESTIMATORS:
            rec = records["B", k, s, f, r, n]
            if rec["status"] != "success":
                failed[f"B{k}|{n}"].append(rec["key"])
                continue
            auc[str(k), s][n].append(rec["auc"])
    return auc


# ---------------------------------------------------------------------------
# Learned kernels, facts and output
# ---------------------------------------------------------------------------


def kernel_nullspace(
    records: Records, subjects: Sequence[int], reps: int
) -> dict[str, Any]:
    """Null space of the SURE(kernel) penalty ||R^T theta||^2, R = conv(h, T - 2),
    h = [a, b, a], in the K = 5 whole-recording fits: real vs complex roots of h,
    the decay factor, and the share of the 2-D null space within 5 samples of the
    window edges."""
    fits = [
        (f"A_s{s:02d}_r{r:02d}", records["A", 5, s, None, r, ANCHOR])
        for s in subjects
        for r in range(reps)
        if records["A", 5, s, None, r, ANCHOR]["status"] == "success"
    ]
    T = 2 * HALF
    edge = np.r_[0:5, T - 5 : T]
    t = np.arange(T)
    line_basis = np.linalg.qr(np.stack([np.ones(T), t - t.mean()], 1))[0]
    real, edge_frac, line_frac, rho = [], [], [], []
    for _, rec in fits:
        h = np.asarray(rec["kernel"], dtype=float)
        real.append(bool(np.abs(h[1]) > 2 * np.abs(h[0])))
        R = scipy.linalg.convolution_matrix(h, T - 2)
        N = np.linalg.svd(R.T)[2][-2:].T
        P = N @ N.T
        edge_frac.append(np.trace(P[np.ix_(edge, edge)]) / 2)
        line_frac.append(np.linalg.norm(N.T @ line_basis) ** 2 / 2)
        roots = np.roots([h[0], h[1], h[2]])
        rho.append(float(np.min(np.abs(roots))) if roots.size else None)
    ef, rh = np.asarray(edge_frac), [v for v in rho if v is not None]
    return {
        "n_fits": len(fits),
        "n_edge5_ge_0.99": int(np.count_nonzero(ef >= 0.99)),
        "exceptions_edge5_lt_0.99": [
            {"fit": fits[i][0], "real_roots": real[i]}
            | {"decay_factor": rho[i], "edge5_energy_fraction": float(ef[i])}
            for i in np.flatnonzero(ef < 0.99)
        ],
        "n_real_roots": int(sum(real)),
        "decay_factor_median": statistic_or_none(np.median, rh),
        "decay_factor_max": statistic_or_none(np.max, rh),
        "edge5_energy_fraction_median": statistic_or_none(np.median, ef),
        "edge5_energy_fraction_min": statistic_or_none(np.min, ef),
        "overlap_with_line_space_median": statistic_or_none(np.median, line_frac),
        "overlap_with_line_space_max": statistic_or_none(np.max, line_frac),
    }


def facts(
    records: Records,
    rows: Sequence[Mapping[str, Any]],
    sel: Mapping[str, Any],
    subjects: Sequence[int],
    reps: int,
) -> dict[str, Any]:
    inf = Counter(
        (key[0], key[1])
        for key, rec in records.items()
        if key[5] == "SURE(lambda)" and rec.get("lambda") == math.inf
    )
    deviations = [
        r["lcurve_deviation_deg"] for r in rows if "lcurve_deviation_deg" in r
    ]
    vert = [v for v in deviations if v is not None]
    return phi_facts(records, subjects, reps) | {
        "sure_lambda_inf": {
            f"{kname(k)}_{an}": inf[an, k] for k in [*KS, ALL] for an in "AB"
        },
        "lcurve_verticality_K05_A": {
            "n_fits": len(vert),
            "max_deviation_deg": statistic_or_none(max, vert),
        }
        | {"n_over_0.1_deg": sum(v > 0.1 for v in vert)}
        | {"n_verticality_missing": len(deviations) - len(vert)},
        "all_events_training_counts": TRAINING_COUNTS,
        "nesting": nesting(sel),
    }


def phi_facts(records: Records, subjects: Sequence[int], reps: int) -> dict[str, Any]:
    """The AR(1) coefficient of the K = 5 whole-recording SURE(kernel) selections,
    and the largest phi SURE(kernel) estimated per K."""
    phi_all = [
        records["A", 5, s, None, r, ANCHOR]["phi_estimate_phi"]
        for s in subjects
        for r in range(reps)
    ]
    phi = [v for v in phi_all if v is not None]
    sk = [
        r for key, r in records.items() if key[5] == ANCHOR and r["status"] == "success"
    ]
    phi_hat = {
        kname(k): [r["inner"]["phi_estimated"] for r in sk if r["K"] == k]
        for k in [*KS, ALL]
    }
    return {
        "n_fits_K05": sum(1 for key in records if key[1] == 5),
        "phi_K05_A_sure_kernel": {
            "n": len(phi),
            "median": statistic_or_none(np.median, phi),
        }
        | {"min": statistic_or_none(min, phi), "max": statistic_or_none(max, phi)}
        | {"n_missing": len(phi_all) - len(phi)},
        "phi_estimated_sure_kernel": {
            k: {"n": len(v), "max": statistic_or_none(max, v)}
            for k, v in phi_hat.items()
        },
    }


def csv_row(rec: Mapping[str, Any]) -> dict[str, object]:
    row: dict[str, object] = {c: rec.get(c) for c in COLUMNS}
    for r in REFS:
        row[f"distance_{r}"] = (rec.get("distances") or {}).get(r)
        row[f"distance_{r}_uV"] = (rec.get("distances_uV") or {}).get(r)
    for c in ("kernel", "flags", "warnings", "inner"):
        row[c] = (
            json.dumps(analysis_tools.plain(rec[c]), allow_nan=False)
            if c in rec
            else None
        )
    return {c: ("" if v is None else analysis_tools.plain(v)) for c, v in row.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data-dir", default=os.environ.get("MNE_DATA", str(Path.home() / "mne_data"))
    )
    parser.add_argument("--out", default="./runs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--subjects", type=int, nargs="+", default=list(protocol.SUBJECTS)
    )
    parser.add_argument("--reps", type=int, default=heldout_p300.REPETITIONS)
    return parser.parse_args()


def write_per_fit(
    out: Path,
    refs: Sequence[Mapping[str, Any]],
    subjects: Sequence[int],
    fixed_rows: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
) -> None:
    """``per_fit.csv``: references, fixed templates, then every unit's rows."""
    ref_rows = [
        rec | {"analysis": "R", "subject": s, "estimator": r}
        for s, v in zip(subjects, refs)
        for r, rec in v["records"].items()
    ]
    order = ("analysis", "K", "subject", "fold", "repetition")
    table = [
        *ref_rows,
        *fixed_rows,
        *sorted(rows, key=lambda r: str([r[f] for f in order])),
    ]
    with (out / "per_fit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(csv_row(r) for r in table)


def main() -> None:
    args = parse_args()
    started = datetime.now(UTC)
    # MOABB reads its BNCI location from the MNE configuration; the environment wins.
    os.environ["MNE_DATA"] = os.environ["MNE_DATASETS_BNCI_PATH"] = args.data_dir
    out = Path(args.out).resolve() / "p300"
    for sub in ("cache", "references", "population", "fits"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    subjects, reps = tuple(args.subjects), args.reps
    per_subject = [(str(out), s) for s in subjects]
    analysis_tools.run_pool(min(args.workers, len(subjects)), preprocess, per_subject)
    sel = make_selections(out, subjects, reps)
    refs = analysis_tools.run_pool(
        min(args.workers, len(subjects)), reference_subject, per_subject
    )
    population = make_population(out, subjects)
    units = analysis_tools.run_pool(
        args.workers, run_unit, all_units(out, subjects, reps)
    )
    pop, fixed_rows = score_population(out, subjects, population)
    rows = [rec for unit in units for rec in unit]
    write_per_fit(out, refs, subjects, fixed_rows, rows)
    fields = ("analysis", "K", "subject", "fold", "repetition", "estimator")
    records: dict[Key, dict[str, Any]] = {
        cast(Key, tuple(r[f] for f in fields)): r
        for r in rows
        if r["estimator"] in ESTIMATORS
    }
    baseline_rows = [r for r in rows if r["estimator"] not in ESTIMATORS] + fixed_rows
    summary = {
        "settings": {"subjects": list(subjects), "repetitions": reps, "Ks": list(KS)},
        "central": central_statistics(records, subjects, reps, pop),
        "kcurve": kcurve_statistics(records, subjects, reps, pop),
        "paired_vs_average": paired_vs_average(records, subjects, reps),
        "baselines": baselines(baseline_rows, records, subjects, reps, pop),
        "kernel_nullspace": kernel_nullspace(records, subjects, reps),
        "facts": facts(records, rows, sel, subjects, reps),
    }
    analysis_tools.write_json(out / "summary.json", summary)
    analysis_tools.write_run_info(out, vars(args), started)


if __name__ == "__main__":
    main()
