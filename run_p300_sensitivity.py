# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""P300 sensitivity analyses of the paper (BNCI2014-009 subjects 1-3, K = 5).

Each analysis refits the central selections of a ``run_p300.py`` output
folder (its ``selections.json``, and its caches and references at 1-20 Hz):

- ``p300_colored_factorial``: fixed/learned kernel x white/colored noise
  procedure (the colored cells are the central SURE fits, refitted).
- ``p300_q_symmetry``: kernel length and symmetry; Q5 convergence at BFGS
  limits 500 (and 5000 for fits still capped).
- ``p300_phi_variance``: phi = phi_hat ** (1 / f) and the noise variance
  scaled, for SURE(kernel) and SURE(lambda), at 1-20 Hz and at 1-30 Hz
  (preprocessed and referenced here, in ``bw1-30/``).

Writes ``<out>/<analysis>/``: ``per_fit.csv`` (the estimator fits, and the
full-data reference fits as ``reference:<name>`` rows), ``templates.npz``
(keyed ``unit|label``), ``summary.json``, ``run_info.json``.

Usage: python3 -B run_p300_sensitivity.py [--central DIR] [--data-dir DIR]
           [--out DIR] [--workers N] [--subjects 1 2 ...] [--reps N]
           [--analyses p300_colored_factorial ...]
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import os
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, cast

import analysis_tools
import heldout_p300
import numpy as np
import protocol
import run_p300
import run_simulations
from numpy.typing import NDArray

ANALYSES: tuple[str, ...] = (
    "p300_colored_factorial",
    "p300_q_symmetry",
    "p300_phi_variance",
)
HALF, GUARD = heldout_p300.HALF_WIN, heldout_p300.GUARD
REFS = protocol.REFERENCES
FAILURES = bool(protocol.DECISIONS["OQ6_exceptions_are_failures"])
BANDS: dict[str, tuple[float, float]] = {
    "bw1-20": protocol.BANDPASS,
    "bw1-30": protocol.SECOND_BANDPASS,
}
CELLS, EFFECTS = run_simulations.CELLS, run_simulations.EFFECTS
Q5, Q_CONVERGENCE = run_simulations.Q5, run_simulations.Q_CONVERGENCE
LIMIT_DEFAULT, LIMIT, RERUN_LIMIT = protocol.BFGS_LIMITS
SURE = ("SURE(kernel)", "SURE(lambda)")
PHI_CELLS: tuple[str, ...] = (
    "baseline",
    *(f"phi_f{f:g}" for f in protocol.PHI_FACTORS),
    *(f"var_x{v:g}" for v in protocol.VARIANCE_SCALES),
)
INNER_FIELDS: tuple[str, ...] = (
    "phi_estimated",
    "phi_used",
    "noise_var_estimated",
    "noise_var_scale",
    "optimizer_status",
    "optimizer_iterations",
    "optimizer_gradient_norm",
)
FIELDS: tuple[str, ...] = ("unit", "label", "subject", "bandpass", "repetition")
COLUMNS: tuple[str, ...] = (
    *FIELDS,
    *("status", "failure_type", "failure_message", "seconds", "flags"),
    *("lambda", "noise_var"),
    *INNER_FIELDS,
    *(f"distance_{r}" for r in REFS),
    "kernel",
)
# exp overflows above ~709.78; the cap turns that into an infinite percentage.
EXP_CAP = 709.0
Record = dict[str, Any]
inner = run_simulations.inner
status = run_simulations.status
stats = run_simulations.stats


# ---------------------------------------------------------------------------
# Inputs and fits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    label: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    maxiter: int | None = None
    phi_factor: float | None = None  # ar1_phi = phi_hat ** (1 / phi_factor)


@dataclass(frozen=True)
class Unit:
    central: str  # the run_p300 output (selections)
    inputs: str  # cache/ and references/ of the unit's band
    band: str
    subject: int
    repetition: int
    specs: tuple[Spec, ...]

    @property
    def name(self) -> str:
        return f"p300_s{self.subject:02d}_{self.band}_r{self.repetition:02d}"


def check_same_events(
    s: int, a: heldout_p300.RecordingBlock, b: heldout_p300.RecordingBlock
) -> None:
    """The 1-30 Hz block *b* must have the events of the central block *a*:
    the fits reuse the central (block, event) selections."""
    where = f"subject {s} block {a.block}"
    if len(a.data) != len(b.data):
        raise RuntimeError(
            f"{where}: {len(b.data)} samples at 1-30 Hz, {len(a.data)} in the central cache"
        )
    for name, x, y in (("onsets", a.onsets, b.onsets), ("labels", a.labels, b.labels)):
        if x.shape != y.shape:
            raise RuntimeError(
                f"{where}: {name} shape {y.shape} at 1-30 Hz, {x.shape} in the central cache"
            )
        bad = np.flatnonzero(x != y)
        if bad.size:
            i = int(bad[0])
            raise RuntimeError(
                f"{where}: {bad.size} {name} differ; first at event {i}: "
                f"{y[i]} at 1-30 Hz, {x[i]} in the central cache"
            )
    if a.channels != b.channels:
        raise RuntimeError(
            f"{where}: channels {list(b.channels)} at 1-30 Hz, "
            f"{list(a.channels)} in the central cache"
        )


def reference_record(s: int, band: str, ref: str, fit: Mapping[str, Any]) -> Record:
    """One full-data reference fit as a per-fit row (label ``reference:<name>``)."""
    return {
        "unit": f"references_s{s:02d}_{band}",
        "subject": s,
        "bandpass": list(BANDS[band]),
        "repetition": None,
        "label": f"reference:{ref}",
        **{k: fit.get(k) for k in ("status", "failure_type", "failure_message")},
    }


def central_references(central: str, subjects: Sequence[int]) -> list[Record]:
    """The 1-20 Hz references of the run_p300 output: available or not (its
    references npz), with the failure reason its per_fit.csv recorded."""
    with (Path(central) / "per_fit.csv").open(newline="") as stream:
        rows = {
            (int(r["subject"]), r["estimator"]): r
            for r in csv.DictReader(stream)
            if r["analysis"] == "R"
        }
    records = []
    for s in subjects:
        refs = run_p300.references_of(central, s)
        for ref in REFS:
            row = rows.get((s, ref), {})
            fit = (
                {"status": "success"}
                if refs[ref] is not None
                else {
                    "status": "failure",
                    "failure_type": row.get("failure_type") or "MissingReference",
                    "failure_message": row.get("failure_message")
                    or f"{ref} is not in the references of {central}",
                }
            )
            records.append(reference_record(s, "bw1-20", ref, fit))
    return records


def second_band_inputs(
    workers: int, central: str, out: Path, subjects: Sequence[int]
) -> tuple[str, list[Record]]:
    """1-30 Hz caches and full-data references in ``out/bw1-30``, and the
    reference fits' records."""
    inputs = out / "bw1-30"
    (inputs / "references").mkdir(parents=True, exist_ok=True)
    tasks = [(str(inputs), s) for s in subjects]
    preprocess = functools.partial(
        run_p300.preprocess, bandpass=protocol.SECOND_BANDPASS
    )
    analysis_tools.run_pool(min(workers, len(tasks)), preprocess, tasks)
    for s in subjects:
        a, b = run_p300.blocks_of(central, s), run_p300.blocks_of(str(inputs), s)
        if len(a) != len(b):
            raise RuntimeError(
                f"subject {s}: {len(b)} blocks at 1-30 Hz, {len(a)} in the central cache"
            )
        for x, y in zip(a, b):
            check_same_events(s, x, y)
    fitted = analysis_tools.run_pool(
        min(workers, len(tasks)), run_p300.reference_subject, tasks
    )
    records = [
        reference_record(s, "bw1-30", ref, fit)
        for s, done in zip(subjects, fitted)
        for ref, fit in done["records"].items()
    ]
    return str(inputs), records


def split(records: Sequence[Record]) -> tuple[list[Record], list[Record]]:
    """(reference records, estimator fits)."""
    refs = [r for r in records if r["label"].startswith("reference:")]
    return refs, [r for r in records if not r["label"].startswith("reference:")]


def references_summary(refs: Sequence[Record]) -> dict[str, Any]:
    """Failed or missing full-data references: their fits have no distance to them."""
    failed = [
        {
            "subject": r["subject"],
            "bandpass": r["bandpass"],
            "reference": r["label"].removeprefix("reference:"),
            "failure_type": r["failure_type"],
            "failure_message": r["failure_message"],
        }
        for r in refs
        if r["status"] != "success"
    ]
    return {"n": len(refs), "failures": len(failed), "failed": failed}


def fit(
    blocks: Sequence[heldout_p300.RecordingBlock],
    rows: NDArray,
    kwargs: Callable[[], Mapping[str, Any]],
    maxiter: int | None,
) -> tuple[Record, NDArray | None]:
    """One fit under ``fit_call``'s failure policy; *kwargs* (which may estimate
    phi_hat) runs inside it, so its scientific errors are counted failures too."""

    def call() -> tuple[dict[object, NDArray], dict[str, object] | None]:
        return heldout_p300.fit_templates(blocks, rows, kwargs(), HALF)

    if maxiter is None:
        record, array = analysis_tools.fit_call(call, ("0", "1"), FAILURES)
    else:
        with analysis_tools.kernel_bfgs_maxiter(maxiter) as n:
            record, array = analysis_tools.fit_call(call, ("0", "1"), FAILURES)
        if n["unexpected"]:
            raise RuntimeError(
                f"BFGS limit override met unexpected minimize calls: {n}"
            )
        if record["status"] == "success" and n["patched"] != 1:
            raise RuntimeError(f"BFGS limit override applied {n['patched']} times: {n}")
    return record, array


def spec_kwargs(
    spec: Spec,
    phi_hat: dict[str, float],
    blocks: Sequence[heldout_p300.RecordingBlock],
    rows: NDArray,
) -> dict[str, Any]:
    """The fit settings of *spec*; the first covariance perturbation of a unit
    estimates phi_hat (kept in *phi_hat* for the unit's later specs)."""
    kw = dict(spec.kwargs)
    perturbs = spec.phi_factor is not None or "noise_var_scale" in kw
    if perturbs and "value" not in phi_hat:
        phi = analysis_tools.training_phi(blocks, rows, HALF)
        if phi is None:  # a counted failure of this fit (raised inside fit_call)
            raise FloatingPointError("phi_hat of the fit input is not finite")
        phi_hat["value"] = phi
    if spec.phi_factor is not None:
        kw["ar1_phi"] = phi_hat["value"] ** (1.0 / spec.phi_factor)
    return kw


def fit_unit(unit: Unit) -> tuple[list[Record], dict[str, NDArray]]:
    """Every spec of one unit, scored against the band's full-data references."""
    blocks = run_p300.blocks_of(unit.inputs, unit.subject)
    chosen = run_p300.selections_of(unit.central)["A"][str(heldout_p300.K)]
    rows = np.array(chosen[str(unit.subject)][unit.repetition], dtype=int)
    refs = run_p300.references_of(unit.inputs, unit.subject)
    # Once estimated, phi_hat also flags a later failed fit (phi cap).
    phi_hat: dict[str, float] = {}
    fields = {
        "subject": unit.subject,
        "bandpass": list(BANDS[unit.band]),
        "repetition": unit.repetition,
    }
    records: list[Record] = []
    arrays: dict[str, NDArray] = {}
    for spec in unit.specs:
        kwargs = functools.partial(spec_kwargs, spec, phi_hat, blocks, rows)
        record, array = fit(blocks, rows, kwargs, spec.maxiter)
        estimated = inner(record).get("phi_estimated") or 0.0
        if max(phi_hat.get("value", 0.0), estimated) >= run_simulations.PHI_CAP:
            record["flags"] = record["flags"] + ["phi_hat_at_cap"]
        try:
            analysis_tools.score_fit(record, array, refs, None, HALF, GUARD)
        except analysis_tools.INFRASTRUCTURE_ERRORS:
            raise
        except Exception as error:  # a counted failure that keeps its template
            if not FAILURES:
                raise
            record |= {
                "status": "failure",
                "failure_type": type(error).__name__,
                "failure_message": str(error),
                "distances": dict.fromkeys(refs),
            }
        records.append({"unit": unit.name, **fields, "label": spec.label, **record})
        if array is not None:
            arrays[f"{unit.name}|{spec.label}"] = array
    return records, arrays


def check_central(central: str, subjects: Sequence[int], reps: int) -> None:
    """The run_p300 output must hold its per_fit.csv, and a cache, references and
    *reps* K = 5 selections for every requested subject (checked before any fit)."""
    if not (Path(central) / "per_fit.csv").exists():
        raise FileNotFoundError(f"no per_fit.csv in {central}")
    selections = run_p300.selections_of(central)["A"].get(str(heldout_p300.K), {})
    for s in subjects:
        for kind in ("cache", "references"):
            path = Path(central) / kind / f"subject_{s:02d}.npz"
            if not path.exists():
                raise FileNotFoundError(f"subject {s}: no {kind} file {path}")
        have = len(selections.get(str(s), []))
        if have < reps:
            raise ValueError(
                f"subject {s}: {central} has {have} K = {heldout_p300.K} "
                f"selections, {reps} requested"
            )


def make_units(
    central: str,
    inputs: str,
    band: str,
    subjects: Sequence[int],
    reps: int,
    specs: Callable[[int, int], tuple[Spec, ...]],
) -> list[Unit]:
    """One unit per (subject, repetition) with any spec."""
    units = [
        Unit(central, inputs, band, s, r, specs(s, r))
        for s in subjects
        for r in range(reps)
    ]
    return [u for u in units if u.specs]


def run_units(
    workers: int, units: Sequence[Unit]
) -> tuple[list[Record], dict[str, NDArray]]:
    records: list[Record] = []
    arrays: dict[str, NDArray] = {}
    for recs, arrs in analysis_tools.run_pool(workers, fit_unit, units):
        records += recs
        arrays.update(arrs)
    return records, arrays


def present(values: Iterable[float | None]) -> list[float]:
    return [v for v in values if v is not None and math.isfinite(v)]


def positive(d: float | None) -> bool:
    return d is not None and math.isfinite(d) and d > 0


def largest(items: Iterable[tuple[float | None, Record]]) -> Record | None:
    """The largest |value| (the first on ties) as a log change, its percent
    change and where it occurred."""
    best: tuple[float, Record] | None = None
    for v, where in items:
        if v is not None and (best is None or abs(v) > best[0]):
            best = (abs(v), where)
    if best is None:
        return None
    return {"value": best[0], "percent": percent(best[0]), **best[1]}


def percent(x: float) -> float:
    """The percent change 100 (e^x - 1) of a log change *x*."""
    return 100 * (math.exp(min(x, EXP_CAP)) - 1)


def log_ratio(a: float, b: float) -> float:
    """log(a / b) of two finite positive values; by logs when the quotient
    underflows to 0 or overflows."""
    q = a / b
    return math.log(q) if 0 < q < math.inf else math.log(a) - math.log(b)


def log_distance(rec: Record | None, ref: str) -> float | None:
    """log of a successful fit's distance to *ref*; None when failed or not positive."""
    if rec is None or rec["status"] != "success":
        return None
    d = (rec.get("distances") or {}).get(ref)
    return math.log(d) if positive(d) else None


# ---------------------------------------------------------------------------
# Colored factorial
# ---------------------------------------------------------------------------


def run_factorial(
    workers: int, central: str, out: Path, subjects: Sequence[int], reps: int
) -> tuple[list[Record], dict[str, NDArray]]:
    specs = tuple(Spec(c, kw) for c, kw in protocol.FACTORIAL_CELLS.items())
    units = make_units(central, central, "bw1-20", subjects, reps, lambda s, r: specs)
    records, arrays = run_units(workers, units)
    return central_references(central, subjects) + records, arrays


def factorial_subject(
    groups: Mapping[tuple[int, int], Mapping[str, Record]], s: int, ref: str
) -> dict[str, Any]:
    """Mean factorial effects of one subject under one reference. Common valid
    cells: a selection counts only if all four cells succeeded."""
    vals = [
        {c: log_distance(groups[k].get(c), ref) for c in CELLS}
        for k in sorted(groups)
        if k[0] == s
    ]
    eff = [
        run_simulations.effects(cast(dict[str, float], v))
        for v in vals
        if all(x is not None for x in v.values())
    ]
    return {
        "n_valid": len(eff),
        "n_selections": len(vals),
        "excluded_selections": len(vals) - len(eff),
        **{e: (float(np.mean([x[e] for x in eff])) if eff else None) for e in EFFECTS},
    }


def factorial_facts(p300: Mapping[str, Any], subjects: Sequence[int]) -> dict[str, Any]:
    cells = [(ref, str(s)) for ref in REFS for s in subjects]
    values = {e: present(p300[ref][s][e] for ref, s in cells) for e in EFFECTS}
    # Undecided (None) unless every effect is available under every reference.
    complete = all(p300[ref][s][e] is not None for ref, s in cells for e in EFFECTS)
    same_signs = complete and all(
        (p300[ref][s][e] < 0) == (p300["OLS"][s][e] < 0)
        for ref, s in cells
        for e in EFFECTS
    )
    return {
        "ranges_over_references": {
            e: [min(v), max(v)] if v else None for e, v in values.items()
        },
        "interaction_negative_cells": sum(v < 0 for v in values["interaction"]),
        "n_cells": len(values["interaction"]),
        "same_signs_all_references": same_signs if complete and cells else None,
    }


def factorial_summary(records: Sequence[Record]) -> dict[str, Any]:
    refs, records = split(records)
    groups: dict[tuple[int, int], dict[str, Record]] = {}
    for r in records:
        groups.setdefault((r["subject"], r["repetition"]), {})[r["label"]] = r
    subjects = sorted({s for s, _ in groups})
    p300: dict[str, Any] = {}
    for ref in REFS:
        p300[ref] = {str(s): factorial_subject(groups, s, ref) for s in subjects}
        p300[ref]["n_subjects_contributing"] = sum(
            p300[ref][str(s)]["n_valid"] > 0 for s in subjects
        )
    return {
        "p300": p300,
        "failure_policy": {
            "p300": run_simulations.flagged_groups(records, ("subject",))
        },
        "facts": factorial_facts(p300, subjects),
        "references": references_summary(refs),
    }


# ---------------------------------------------------------------------------
# Kernel length and symmetry; Q5 convergence
# ---------------------------------------------------------------------------


def q_units(
    central: str,
    subjects: Sequence[int],
    reps: int,
    limits: Callable[[int, int, str], Sequence[int | None]],
) -> list[Unit]:
    """*limits*(subject, repetition, cell) gives the cell's BFGS limits (None: core's own)."""

    def specs(s: int, r: int) -> tuple[Spec, ...]:
        return tuple(
            Spec(
                cell if lim is None else f"{cell}@{lim}",
                {**protocol.SURE_KERNEL, **q, "noise_est": "ar1_colored"},
                maxiter=lim,
            )
            for cell, q in protocol.Q_CELLS.items()
            for lim in limits(s, r, cell)
        )

    return make_units(central, central, "bw1-20", subjects, reps, specs)


def run_q(
    workers: int, central: str, out: Path, subjects: Sequence[int], reps: int
) -> tuple[list[Record], dict[str, NDArray]]:
    """Every cell at core's limit (50) and the convergence cells at 500, then 5000
    where still capped. The 50 fits of the convergence follow-up are the plain fits."""
    records, arrays = run_units(
        workers,
        q_units(
            central,
            subjects,
            reps,
            lambda s, r, c: [None, LIMIT] if c in Q_CONVERGENCE else [None],
        ),
    )
    capped = {
        (r["subject"], r["repetition"], r["label"])
        for r in records
        if status(r) == "capped"
    }
    more, arrs = run_units(
        workers,
        q_units(
            central,
            subjects,
            reps,
            lambda s, r, c: [RERUN_LIMIT] if (s, r, f"{c}@{LIMIT}") in capped else [],
        ),
    )
    arrays.update(arrs)
    for r in (r for r in records if r["label"] == "Q3-sym"):
        r["equal_to_central"] = equal_to_central(
            central, r, arrays.get(r["unit"] + "|Q3-sym")
        )
    return central_references(central, subjects) + records + more, arrays


def equal_to_central(central: str, rec: Record, array: NDArray | None) -> bool:
    """Whether a Q3-sym template (the central SURE(kernel) configuration) is the
    run_p300 K = 5 fit of the same selection, bitwise."""
    unit = run_p300.unit_name("A", 5, rec["subject"], None, rec["repetition"])
    path = Path(central) / "fits" / unit / f"{run_p300.slug('SURE(kernel)')}.npz"
    if array is None or not path.exists():
        return False
    with np.load(path) as z:
        return z["templates"].shape == array.shape and (
            z["templates"].tobytes() == array.tobytes()
        )


def q_cell(rs: Sequence[Record], ref: str) -> dict[str, Any]:
    """Mean log distance per subject (successful fits), and over subjects."""
    per: dict[int, float | None] = {}
    counts: dict[int, dict[str, int]] = {}
    for s in sorted({r["subject"] for r in rs}):
        v = present(log_distance(r, ref) for r in rs if r["subject"] == s)
        per[s] = float(np.mean(v)) if v else None
        counts[s] = {"n_valid": len(v), "n": sum(r["subject"] == s for r in rs)}
    vals = present(per.values())
    return {
        "per_subject": per,
        "counts": counts,
        "n_subjects_contributing": len(vals),
        "mean": float(np.mean(vals)) if vals else None,
        "range": [min(vals), max(vals)] if vals else None,
    }


def asymmetry(h: Sequence[float]) -> float:
    """max|h_i - h_(Q-1-i)| / max|h| of a finite nonzero kernel; on overflow,
    from the kernel scaled by its largest coefficient."""
    top = max(map(abs, h))
    r = max(abs(h[i] - h[-1 - i]) for i in range(len(h))) / top
    if math.isfinite(r):
        return r
    g = [x / top for x in h]
    return max(abs(g[i] - g[-1 - i]) for i in range(len(g)))


def relative_asymmetry(base: Sequence[Record]) -> dict[str, float] | None:
    """Max and median of max|h_i - h_(Q-1-i)| / max|h| over the successful
    Q3-free kernels (nonfinite or all-zero kernels skipped)."""
    asym = [
        asymmetry(h)
        for h in (
            r["kernel"]
            for r in base
            if r["label"] == "Q3-free" and r["status"] == "success"
        )
        if np.all(np.isfinite(h)) and max(map(abs, h)) > 0
    ]
    if not asym:
        return None
    return {"max": float(max(asym)), "median": float(statistics.median(asym))}


def q_symmetry_part(records: Sequence[Record]) -> dict[str, Any]:
    base = [r for r in records if "@" not in r["label"]]
    p = {
        ref: {
            c: q_cell([r for r in base if r["label"] == c], ref)
            for c in protocol.Q_CELLS
        }
        for ref in REFS
    }
    diff = {
        ref: p[ref]["Q1"]["mean"] - p[ref]["Q3-sym"]["mean"]
        for ref in REFS
        if p[ref]["Q1"]["mean"] is not None and p[ref]["Q3-sym"]["mean"] is not None
    }
    lower = [100 * (1 - math.exp(min(x, EXP_CAP))) for x in diff.values()]
    q3sym = [r for r in base if r["label"] == "Q3-sym"]
    # Cells where all five kernels have a subject mean.
    full = [
        [p[ref][c]["per_subject"][s] for c in protocol.Q_CELLS]
        for ref in REFS
        for s in p[ref]["Q1"]["per_subject"]
        if all(p[ref][c]["per_subject"].get(s) is not None for c in protocol.Q_CELLS)
    ]
    return {
        "p300": {"bw1-20": p},
        "failure_policy": {
            "p300": run_simulations.flagged_groups(base, ("subject", "bandpass"))
        },
        "facts": {
            "q3free_relative_asymmetry": relative_asymmetry(base),
            "q1_minus_q3sym_mean_log_distance": diff,
            "q1_lower_distance_percent_range": (
                [min(lower), max(lower)] if lower else None
            ),
            "q1_lowest_cells": sum(all(v[0] < w for w in v[1:]) for v in full),
            "n_cells": len(full),
            "q3sym_bitwise_equal_central": {
                "equal": sum(r.get("equal_to_central", False) for r in q3sym),
                "n": len(q3sym),
            },
        },
    }


def limit_fits(fs: Mapping[str, Record], cell: str) -> dict[str, Record | None]:
    """A unit's fits of *cell* at the limit of 50, the raised limit, the rerun
    limit (if any) and the final one (the rerun's, else the raised limit's)."""
    lim, rlim = fs.get(f"{cell}@{LIMIT}"), fs.get(f"{cell}@{RERUN_LIMIT}")
    return {"50": fs.get(cell), "lim": lim, "final": rlim or lim, "rerun": rlim}


def convergence(f: Sequence[Mapping[str, Record | None]]) -> dict[str, Any]:
    """Optimizer status counts of one cell's fits over the units."""
    st50, final = [status(x["50"]) for x in f], [status(x["final"]) for x in f]
    st_lim = [status(x["lim"]) for x in f]
    st_rerun = [status(x["rerun"]) for x in f if x["rerun"] is not None]
    iters = [
        inner(cast(Record, x["final"]))["optimizer_iterations"]
        for x, s in zip(f, final)
        if s == "converged"
    ]
    return {
        "n": len(f),
        "converged_50": st50.count("converged"),
        "capped_50": st50.count("capped"),
        f"converged_{LIMIT}": st_lim.count("converged"),
        f"capped_{LIMIT}": st_lim.count("capped"),
        f"rerun_{RERUN_LIMIT}": len(st_rerun),
        f"converged_{RERUN_LIMIT}": st_rerun.count("converged"),
        f"still_capped_{RERUN_LIMIT}": st_rerun.count("capped"),
        "other_or_failed_final": sum(s not in {"converged", "capped"} for s in final),
        # BFGS status 2 is "Desired error not necessarily achieved due to precision loss".
        "final_status_counts": {s: final.count(s) for s in sorted(set(final))},
        "median_iterations_final_converged": (
            float(np.median(iters)) if iters else None
        ),
    }


def capped_change(f: Sequence[Mapping[str, Record | None]], ref: str) -> dict[str, Any]:
    """Final minus limit-50 log distance of the fits capped at 50."""
    change = [
        (log_distance(x["final"], ref), log_distance(x["50"], ref))
        for x in f
        if status(x["50"]) == "capped"
    ]
    d = [a - b for a, b in change if a is not None and b is not None]
    return {
        "capped_at_50": stats(d),
        "capped_improved": sum(x < 0 for x in d),
        "capped_at_50_values": d,
    }


def minus_q3sym(
    units: Mapping[str, Mapping[str, Record]],
    subject: Mapping[str, int],
    cell: str,
    ref: str,
) -> dict[str, Any]:
    """Per subject, the mean paired log distance of the converged final *cell*
    fit minus the final Q3-sym fit; then the mean over subjects."""
    per: dict[int, dict[str, Any]] = {}
    for s in sorted(set(subject.values())):
        d = []
        for u, fs in units.items():
            x, base = limit_fits(fs, cell)["final"], limit_fits(fs, "Q3-sym")["final"]
            if subject[u] != s or status(x) != "converged":
                continue
            a, b = log_distance(x, ref), log_distance(base, ref)
            if a is not None and b is not None:
                d.append(a - b)
        per[s] = {"mean": float(np.mean(d)) if d else None, "n": len(d)}
    vals = present(v["mean"] for v in per.values())
    return {
        "converged final only": {
            "per_subject": per,
            "mean_over_subjects": float(np.mean(vals)) if vals else None,
        }
    }


def q5_part(records: Sequence[Record]) -> dict[str, Any]:
    units = run_simulations.by_unit(records)
    subject = {u: next(iter(fs.values()))["subject"] for u, fs in units.items()}
    out: dict[str, Any] = {
        k: {} for k in ("convergence", "kernels", "log_distance_change", "q5_vs_q3sym")
    }
    not_converged = 0
    for cell in Q_CONVERGENCE:
        f = [limit_fits(fs, cell) for fs in units.values()]
        out["convergence"][cell] = convergence(f)
        converged = [
            cast(Record, x["final"]) for x in f if status(x["final"]) == "converged"
        ]
        if cell in Q5:
            not_converged += len(f) - len(converged)
        ratios = [run_simulations.nyquist_over_dc(r.get("kernel")) for r in converged]
        out["kernels"][cell] = {
            "final_converged": stats(ratios),
            "final_converged_dc_near_zero": sum(v is None for v in ratios),
        }
        for ref in REFS:
            out["log_distance_change"].setdefault(ref, {})[cell] = capped_change(f, ref)
    for ref in REFS:
        out["q5_vs_q3sym"][ref] = {
            cell: minus_q3sym(units, subject, cell, ref) for cell in (*Q5, "Q3-free")
        }
    grid = [
        r if "@" in r["label"] else {**r, "label": f"{r['label']}@{LIMIT_DEFAULT}"}
        for r in records
        if "@" in r["label"] or r["label"] in Q5
    ]
    out["failure_policy"] = run_simulations.flagged_groups(grid, ("subject",))
    out["facts"] = q5_facts(records, out["q5_vs_q3sym"], not_converged, subject)
    return out


def q5_facts(
    records: Sequence[Record],
    paired: Mapping[str, Mapping[str, Any]],
    not_converged: int,
    subject: Mapping[str, int],
) -> dict[str, Any]:
    raised = [r for r in records if "@" in r["label"] and r["status"] == "success"]
    return {
        "max_iterations_raised_limit": max(
            (inner(r)["optimizer_iterations"] for r in raised), default=None
        ),
        "q5_final_not_converged": not_converged,
        "positive_subject_reference_cells": {
            cell: sum(
                v["mean"] > 0
                for ref in REFS
                for v in paired[ref][cell]["converged final only"][
                    "per_subject"
                ].values()
                if v["mean"] is not None
            )
            for cell in (*Q5, "Q3-free")
        },
        "n_subject_reference_cells": len(REFS) * len(set(subject.values())),
    }


def q_summary(records: Sequence[Record]) -> dict[str, Any]:
    refs, records = split(records)
    return {
        "q_symmetry": q_symmetry_part(records),
        "q5_convergence": q5_part(records),
        "references": references_summary(refs),
    }


# ---------------------------------------------------------------------------
# phi and noise-variance sensitivity
# ---------------------------------------------------------------------------


def run_phi(
    workers: int, central: str, out: Path, subjects: Sequence[int], reps: int
) -> tuple[list[Record], dict[str, NDArray]]:
    specs: list[Spec] = []
    for m in SURE:
        kw = protocol.METHODS_AR1COL[m]
        specs.append(Spec(f"{m}|baseline", kw))
        specs += [
            Spec(f"{m}|phi_f{f:g}", kw, phi_factor=f) for f in protocol.PHI_FACTORS
        ]
        specs += [
            Spec(f"{m}|var_x{v:g}", {**kw, "noise_var_scale": v})
            for v in protocol.VARIANCE_SCALES
        ]
    second, refs = second_band_inputs(workers, central, out, subjects)
    inputs = {"bw1-20": central, "bw1-30": second}
    units = [
        u
        for band, path in inputs.items()
        for u in make_units(
            central, path, band, subjects, reps, lambda s, r: tuple(specs)
        )
    ]
    records, arrays = run_units(workers, units)
    return central_references(central, subjects) + refs + records, arrays


def phi_cell(rs: Sequence[Record], base: Sequence[Record], ref: str) -> dict[str, Any]:
    """Per subject, the mean log(d_cell / d_baseline) over the selections where
    both fits succeeded; then the mean over subjects."""
    per: dict[int, float | None] = {}
    counts: dict[int, dict[str, int]] = {}
    for s in sorted({r["subject"] for r in rs}):
        b = {r["repetition"]: r for r in base if r["subject"] == s}
        this = {r["repetition"]: r for r in rs if r["subject"] == s}
        both_ok = [
            k
            for k in this
            if k in b and this[k]["status"] == "success" and b[k]["status"] == "success"
        ]
        v = [
            log_ratio(this[k]["distances"][ref], b[k]["distances"][ref])
            for k in both_ok
            if positive(this[k]["distances"][ref]) and positive(b[k]["distances"][ref])
        ]
        per[s] = float(np.mean(v)) if v else None
        counts[s] = {
            "n_valid_pairs": len(v),
            "n_pairs": len(this),
            "excluded_failed": len(this) - len(both_ok),
            "excluded_nonpositive_or_missing_distance": len(both_ok) - len(v),
        }
    vals = present(per.values())
    phis = present(inner(r).get("phi_used") for r in rs)
    estimated = present(inner(r).get("phi_estimated") for r in rs)
    return {
        "per_subject": per,
        "counts": counts,
        "n_subjects_contributing": len(vals),
        "mean": float(np.mean(vals)) if vals else None,
        "phi_used_mean": float(np.mean(phis)) if phis else None,
        "phi_estimated_mean": float(np.mean(estimated)) if estimated else None,
        "flagged": sum(bool(r.get("flags")) for r in rs),
        "phi_hat_at_cap": sum("phi_hat_at_cap" in r["flags"] for r in rs),
        "failures": sum(r["status"] == "failure" for r in rs),
    }


def phi_tables(records: Sequence[Record]) -> dict[str, Any]:
    """band -> estimator -> reference -> cell -> ``phi_cell``."""
    p: dict[str, Any] = {}
    for band, bandpass in BANDS.items():
        mine = [r for r in records if tuple(r["bandpass"]) == bandpass]
        if not mine:
            continue
        p[band] = {
            m: {
                ref: {
                    c: phi_cell(
                        [r for r in mine if r["label"] == f"{m}|{c}"],
                        [r for r in mine if r["label"] == f"{m}|baseline"],
                        ref,
                    )
                    for c in PHI_CELLS
                }
                for ref in REFS
            }
            for m in SURE
        }
    return p


def changes(
    p: Mapping[str, Any], m: str, prefix: str
) -> Iterable[tuple[float | None, Record]]:
    """Mean changes of estimator *m* in the cells named *prefix*, with where."""
    for band in p:
        for ref in REFS:
            for c in (c for c in PHI_CELLS if c.startswith(prefix)):
                yield p[band][m][ref][c]["mean"], {"band": band, "ref": ref, "cell": c}


def subject_changes(p: Mapping[str, Any]) -> Iterable[tuple[float | None, Record]]:
    """Per-subject SURE(kernel) changes under the phi perturbations, with where."""
    for _, where in changes(p, "SURE(kernel)", "phi"):
        k = p[where["band"]]["SURE(kernel)"][where["ref"]][where["cell"]]
        for s, x in k["per_subject"].items():
            yield x, {**where, "subject": s}


def signs(p: Mapping[str, Any], cell: str, sign: float) -> list[bool]:
    """Per (band, reference, subject): does SURE(kernel)'s change in *cell* have *sign*?"""
    return [
        v * sign > 0
        for band in p
        for ref in REFS
        for v in p[band]["SURE(kernel)"][ref][cell]["per_subject"].values()
        if v is not None
    ]


def phi_summary(records: Sequence[Record]) -> dict[str, Any]:
    refs, records = split(records)
    p = phi_tables(records)
    flagged = [r for r in records if r["flags"]]
    return {
        "p300": p,
        "failure_policy": {
            "p300": run_simulations.flagged_groups(records, ("subject", "bandpass"))
        },
        "facts": {
            "sure_kernel_largest_phi_change": largest(
                changes(p, "SURE(kernel)", "phi")
            ),
            "sure_kernel_largest_variance_change": largest(
                changes(p, "SURE(kernel)", "var")
            ),
            "sure_lambda_largest_change": largest(changes(p, "SURE(lambda)", "")),
            "sure_kernel_largest_single_subject_phi_change": largest(
                subject_changes(p)
            ),
            "sure_kernel_var_x1.25_lower_cells": sum(signs(p, "var_x1.25", -1.0)),
            "sure_kernel_var_x1.25_cells": len(signs(p, "var_x1.25", -1.0)),
            "sure_kernel_var_x0.8_higher_cells": sum(signs(p, "var_x0.8", 1.0)),
            "sure_kernel_var_x0.8_cells": len(signs(p, "var_x0.8", 1.0)),
            "flagged_fits": [
                {"unit": r["unit"], "label": r["label"], "flags": r["flags"]}
                for r in flagged
            ],
            "failures": sum(r["status"] == "failure" for r in records),
            "phi_hat_at_cap": sum("phi_hat_at_cap" in r["flags"] for r in records),
        },
        "references": references_summary(refs),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def csv_row(rec: Record) -> dict[str, object]:
    row: dict[str, object] = {c: rec.get(c) for c in COLUMNS}
    row.update({k: inner(rec).get(k) for k in INNER_FIELDS})
    row["flags"] = ";".join(rec.get("flags") or [])
    for ref in REFS:
        row[f"distance_{ref}"] = (rec.get("distances") or {}).get(ref)
    if rec.get("kernel") is not None:
        row["kernel"] = json.dumps(np.asarray(rec["kernel"]).tolist())
    return row


Run = Callable[
    [int, str, Path, Sequence[int], int], tuple[list[Record], dict[str, NDArray]]
]
ANALYSIS: dict[str, tuple[Run, Callable[[Sequence[Record]], dict[str, Any]]]] = {
    "p300_colored_factorial": (run_factorial, factorial_summary),
    "p300_q_symmetry": (run_q, q_summary),
    "p300_phi_variance": (run_phi, phi_summary),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--central", help="the run_p300.py output folder (default: <out>/p300)"
    )
    parser.add_argument(
        "--data-dir", default=os.environ.get("MNE_DATA", str(Path.home() / "mne_data"))
    )
    parser.add_argument("--out", default="./runs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--subjects", type=int, nargs="+", default=list(protocol.SENSITIVITY_SUBJECTS)
    )
    parser.add_argument("--reps", type=int, default=heldout_p300.REPETITIONS)
    parser.add_argument(
        "--analyses", nargs="+", choices=ANALYSES, default=list(ANALYSES)
    )
    args = parser.parse_args()
    # MOABB reads its BNCI location from the MNE configuration; the environment wins.
    os.environ["MNE_DATA"] = os.environ["MNE_DATASETS_BNCI_PATH"] = args.data_dir
    central = str(Path(args.central or Path(args.out) / "p300").resolve())
    check_central(central, args.subjects, args.reps)
    for name in args.analyses:
        started = datetime.now(UTC)
        out = Path(args.out).resolve() / name
        out.mkdir(parents=True, exist_ok=True)
        run, summarize = ANALYSIS[name]
        records, arrays = run(args.workers, central, out, args.subjects, args.reps)
        with (out / "per_fit.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(csv_row(r) for r in records)
        np.savez(out / "templates.npz", **arrays)
        analysis_tools.write_json(out / "summary.json", summarize(records))
        analysis_tools.write_run_info(
            out, {**vars(args), "analysis": name, "central": central}, started
        )


if __name__ == "__main__":
    main()
