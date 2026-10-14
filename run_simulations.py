# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Simulation analyses of the paper (known true templates): ``white_sims``
(noise and event-count sweeps, with zero and lambda = inf baselines),
``calibration`` (do full-data references rank estimators as the truth does?),
``colored_factorial`` (fixed/learned kernel x white/colored noise procedure on
AR(1) noise; the paired differences of the paper's colored-noise result),
``q_symmetry`` (kernel length and symmetry; Q5 convergence) and
``known_covariance`` (true vs estimated AR(1) covariance; PD02 control).

Writes ``<out>/<analysis>/``: ``per_fit.csv``, ``templates.npz`` (keyed
``unit|label``), ``summary.json``, ``run_info.json``.
Usage: python3 -B run_simulations.py [--out DIR] [--workers N] [--reps N]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, cast

import analysis_tools
import core
import numpy as np
import protocol
import scipy
from numpy.typing import NDArray

METHODS: tuple[str, ...] = tuple(protocol.METHODS_PAIRWISE)
CELLS: tuple[str, ...] = tuple(protocol.FACTORIAL_CELLS)
EFFECTS: tuple[str, ...] = (
    "kernel_effect_white",
    "kernel_effect_colored",
    "noise_effect_fixed",
    "noise_effect_learned",
    "interaction",
)
Q5: tuple[str, ...] = ("Q5-sym", "Q5-free")
Q_CONVERGENCE: tuple[str, ...] = Q5 + ("Q3-sym", "Q3-free")
LIMIT_DEFAULT, LIMIT, RERUN_LIMIT = protocol.BFGS_LIMITS
BFGS_MAXITER_STATUS = 1  # scipy BFGS: "Maximum number of iterations has been exceeded."
PHI_CAP = 0.99  # core.estimate_phi clip
TRUTH: dict[str, NDArray] = analysis_tools.make_real_templates()
KEYS: tuple[str, ...] = tuple(sorted(TRUTH))
TRUTH_ARRAY: NDArray = np.stack([TRUTH[k] for k in KEYS])
INNER_FIELDS: tuple[str, ...] = (
    "phi_estimated",
    "phi_used",
    "noise_var_estimated",
    "noise_var_scale",
    "optimizer_status",
    "optimizer_iterations",
    "oracle_best_lambda",
)
COLUMNS: tuple[str, ...] = (
    *[
        "unit",
        "label",
        "status",
        "failure_type",
        "failure_message",
        "seconds",
        "flags",
        "warnings",
    ],
    *("lambda", "noise_var", *INNER_FIELDS, "snr_db", "distance_truth"),
    *(f"distance_{r}" for r in protocol.REFERENCES),
    "kernel",
)
Record = dict[str, Any]
Fit = tuple[Mapping[Any, NDArray], Mapping[str, object] | None]


# ---------------------------------------------------------------------------
# Generators used only here (the shared ones are in analysis_tools)
# ---------------------------------------------------------------------------


def pcn_case(
    templates: dict[str, NDArray],
    phi: float,
    rep: int,
    mean_interval: int,
    seed_offset: int,
    stationary_start: bool,
) -> tuple[NDArray, NDArray, NDArray]:
    """AR(1)-noise recording for one (mean_interval, phi, repetition).

    stationary_start=True draws the first noise sample with the marginal
    variance noise_std**2 instead of the innovation variance; the random draws
    are otherwise identical.
    """
    half_win, I, noise_std = protocol.SIM_HALF_WIN, protocol.PCN_I, protocol.PCN_NOISE
    rs = np.random.RandomState(seed_offset + 1000 * rep + int(phi * 100))
    data_white, ix, labels = core.generate_data(
        templates, I=I, noise=noise_std, mean_interval=mean_interval, rs=rs
    )
    if phi > 0:
        ar_noise = np.zeros_like(data_white)
        w = noise_std * np.sqrt(1 - phi**2) * rs.randn(*data_white.shape)
        ar_noise[0] = w[0] / np.sqrt(1 - phi**2) if stationary_start else w[0]
        for t_idx in range(1, len(data_white)):
            ar_noise[t_idx] = phi * ar_noise[t_idx - 1] + w[t_idx]
        signal = np.zeros_like(data_white)
        for ii, lbl in zip(ix, labels):
            ii = int(ii)
            sl = slice(ii - half_win, ii + half_win)
            if sl.start >= 0 and sl.stop <= len(signal):
                signal[sl] += templates[lbl]
        data = signal + ar_noise
    else:
        data = data_white
    return data, ix, labels


# ---------------------------------------------------------------------------
# Fitting a unit (one realization, several configurations)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    """kind "fit" (core.get_events_templates), "oracle" (best fixed lambda by
    SNR to the truth), "line" (lambda = inf) or "zero" (the zero template)."""

    label: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    kind: str = "fit"
    maxiter: int | None = None


@dataclass(frozen=True)
class Unit:
    name: str
    fields: Mapping[str, Any]
    data: NDArray
    ix: NDArray
    lb: NDArray
    specs: tuple[Spec, ...]
    # Events of the full-data references (OLS, EmpBayes, Average), if any.
    reference_events: tuple[NDArray, NDArray] | None = None


def templates_of(unit: Unit, **kwargs: Any) -> Fit:  # noqa: ANN401
    return core.get_events_templates(
        unit.data, unit.ix, unit.lb, half_win=protocol.SIM_HALF_WIN, **kwargs
    )


def oracle_fit(unit: Unit) -> Fit:
    """Best SNR over np.logspace(-1, 6, 100) fixed lambdas (first maximum kept)."""
    lo, hi, n = protocol.ORACLE_LAMBDAS
    best, best_t, best_lam = -np.inf, None, None
    for lam in np.logspace(lo, hi, n):
        t_est, _ = templates_of(unit, method="ls", reg_temporal=lam)
        snr = core.calculate_snr(cast(dict[str, NDArray], t_est), TRUTH)
        if snr > best:
            best, best_t, best_lam = snr, t_est, float(lam)
    if best_t is None:
        raise FloatingPointError("no finite oracle SNR over the fixed-lambda grid")
    return best_t, {"oracle_best_lambda": best_lam}


def line_fit(unit: Unit) -> Fit:
    """The lambda = inf template: the L-curve fit with its selection forced to inf."""
    with analysis_tools.forced_infinite_lambda(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t, inner = templates_of(unit, method="ls", reg_method="lcurve")
    if inner is None or inner.get("opt_val") != math.inf:
        raise RuntimeError(f"forced lambda = inf fit selected {inner}")
    return t, None


def fit_spec(unit: Unit, spec: Spec) -> tuple[Record, NDArray | None]:
    calls: dict[str, Callable[[], Fit]] = {
        "oracle": lambda: oracle_fit(unit),
        "line": lambda: line_fit(unit),
        "zero": lambda: ({k: np.zeros_like(v) for k, v in TRUTH.items()}, None),
        "fit": lambda: templates_of(unit, block_starts=None, **spec.kwargs),
    }
    failures = bool(protocol.DECISIONS["OQ6_exceptions_are_failures"])
    if spec.maxiter is None:
        record, array = analysis_tools.fit_call(calls[spec.kind], KEYS, failures)
    else:
        with analysis_tools.kernel_bfgs_maxiter(spec.maxiter) as n:
            record, array = analysis_tools.fit_call(calls[spec.kind], KEYS, failures)
        if n["unexpected"] or (record["status"] == "success" and n["patched"] != 1):
            raise RuntimeError(
                f"{unit.name} {spec.label}: BFGS limit {spec.maxiter} was applied "
                f"{n['patched']} times with {n['unexpected']} unexpected calls, not once"
            )
    if (inner(record).get("phi_estimated") or 0.0) >= PHI_CAP:
        record["flags"] = record["flags"] + ["phi_hat_at_cap"]
    return record, array


def inner(record: Record) -> Mapping[str, Any]:
    return record.get("inner") or {}


def fit_unit(unit: Unit) -> tuple[list[Record], dict[str, NDArray]]:
    """Every spec of one unit, scored against the truth and the unit's references."""
    records: list[Record] = []
    arrays: dict[str, NDArray] = {}
    references: dict[str, NDArray | None] = {}
    todo = [(unit, s) for s in unit.specs]
    if unit.reference_events is not None:
        full = Unit(unit.name, unit.fields, unit.data, *unit.reference_events, ())
        refs = [
            (full, Spec(f"reference:{r}", protocol.METHODS_PAIRWISE[r]))
            for r in protocol.REFERENCES
        ]
        todo = refs + todo
    for target, spec in todo:
        record, array = fit_spec(target, spec)
        ok = array is not None and record["status"] == "success"
        if spec.label.startswith("reference:"):
            ok = ok and not shape_failure(record, cast(NDArray, array), "reference")
            # C order, as the references were read back from disk when the
            # published distances were computed: the norm sums in memory order.
            references[spec.label[10:]] = np.ascontiguousarray(array) if ok else None
        elif ok:
            score(record, cast(NDArray, array), references)
        records.append(
            {"unit": unit.name, **unit.fields, "label": spec.label, **record}
        )
        if array is not None:
            arrays[f"{unit.name}|{spec.label}"] = array
    return records, arrays


# Errors a valid fit's scoring can raise (e.g. overflow under -W error).
SCORING_ERRORS = (ArithmeticError, RuntimeWarning, ValueError)


def shape_failure(record: Record, array: NDArray, what: str) -> bool:
    """Mark a fit whose templates are not shaped like the true ones as failed."""
    if array.shape == TRUTH_ARRAY.shape:
        return False
    record.update(status="failure", failure_type="TemplateShape")
    record["failure_message"] = (
        f"{what} template shape {array.shape} != expected {TRUTH_ARRAY.shape}"
    )
    return True


def reference_distance(array: NDArray, reference: NDArray | None) -> float | None:
    """Distance to one reference; None if the reference failed or the distance
    is not finite (that distance only is missing)."""
    if reference is None:
        return None
    try:
        d = analysis_tools.distance(array, reference)
    except SCORING_ERRORS:
        return None
    return d if math.isfinite(d) else None


def score(
    record: Record, array: NDArray, references: Mapping[str, NDArray | None]
) -> None:
    """SNR and distances of a successful fit. A template of the wrong shape or a
    nonfinite SNR or truth distance makes it a counted failure."""
    if shape_failure(record, array, "fitted"):
        return
    try:
        snr = float(core.calculate_snr(dict(zip(KEYS, array)), TRUTH))
        to_truth = analysis_tools.distance(array, TRUTH_ARRAY)
    except SCORING_ERRORS as error:
        record.update(status="failure", failure_type=type(error).__name__)
        record["failure_message"] = f"scoring against the truth: {error}"
        return
    if not (math.isfinite(snr) and math.isfinite(to_truth)):
        record.update(status="failure", failure_type="NonfiniteMetric")
        record["failure_message"] = f"SNR {snr} dB, distance to truth {to_truth}"
        return
    record.update(snr_db=snr, distance_truth=to_truth)
    if references:
        record["distances"] = {
            r: reference_distance(array, x) for r, x in references.items()
        }


def run_units(
    workers: int, units: Sequence[Unit]
) -> tuple[list[Record], dict[str, NDArray]]:
    records: list[Record] = []
    arrays: dict[str, NDArray] = {}
    for recs, arrs in analysis_tools.run_pool(workers, fit_unit, units):
        records += recs
        arrays.update(arrs)
    return records, arrays


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def snr_values(records: Iterable[Record], label: str, **match: Any) -> list[float]:  # noqa: ANN401
    """SNR of the successful fits of *label* whose fields equal *match*, in record order."""
    return [
        r["snr_db"]
        for r in records
        if r["label"] == label
        and r["status"] == "success"
        and all(r[k] == v for k, v in match.items())
    ]


def by_unit(records: Iterable[Record]) -> dict[str, dict[str, Record]]:
    out: dict[str, dict[str, Record]] = {}
    for r in records:
        out.setdefault(r["unit"], {})[r["label"]] = r
    return out


def flagged_groups(records: Iterable[Record], fields: Sequence[str]) -> dict[str, Any]:
    """Failures and flagged fits per group (unit fields + label); a group with
    more than 10% failed fits is flagged (the failure policy)."""
    groups: dict[str, dict[str, Any]] = {}
    for r in records:
        key = "|".join(str(r.get(f)) for f in fields) + "|" + str(r["label"])
        g = groups.setdefault(key, {"n": 0, "failures": 0, "flagged": 0})
        g["n"] += 1
        g["failures"] += r["status"] == "failure"
        g["flagged"] += bool(r.get("flags"))
    for g in groups.values():
        g["over_10pct"] = g["failures"] > 0.1 * g["n"]
    return {
        "groups": groups,
        "flagged": sorted(k for k, g in groups.items() if g["over_10pct"]),
        "group_fields": list(fields) + ["label"],
    }


def finite(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def interval(values: Iterable[float | None]) -> dict[str, Any]:
    """Mean and 1.96 x SE over independent units; None and nonfinite values
    dropped. A statistic that overflows (huge finite values) is None."""
    v = np.array([x for x in values if x is not None and math.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": None, "se": None, "lo": None, "hi": None}
    with np.errstate(over="ignore", invalid="ignore"):
        mean = finite(v.mean())
        se = finite(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else None
        median = finite(np.median(v))
    lo = hi = None
    if mean is not None and se is not None:
        lo, hi = finite(mean - 1.96 * se), finite(mean + 1.96 * se)
    return {
        "n": int(v.size),
        "mean": mean,
        "se": se,
        "lo": lo,
        "hi": hi,
        "median": median,
        "min": float(v.min()),
        "max": float(v.max()),
    }


def stats(values: Sequence[float | None]) -> dict[str, Any]:
    """``interval`` without the standard error: n, mean, median, min, max."""
    iv = interval(values)
    return {k: iv.get(k) for k in ("n", "mean", "median", "min", "max")}


def mean(xs: Sequence[float]) -> float | None:
    """Exact (``statistics.mean``) mean, as the published facts were computed."""
    return float(statistics.mean(xs)) if xs else None


def half_ci(xs: Sequence[float]) -> float | None:
    return 1.96 * statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None


def excludes_zero(iv: Mapping[str, Any], side: int = 0) -> bool:
    """Whether a 95% interval excludes 0 (side 1: above it, -1: below it, 0: either)."""
    if iv["lo"] is None:
        return False
    return (side >= 0 and iv["lo"] > 0) or (side <= 0 and iv["hi"] < 0)


# ---------------------------------------------------------------------------
# White-noise sweeps and their trivial baselines
# ---------------------------------------------------------------------------

GRIDS = {"noise": protocol.NOISE_SWEEP, "nevents": protocol.NEVENTS_SWEEP}


def white_units(reps: Sequence[int]) -> list[Unit]:
    specs = tuple(Spec(m, kw) for m, kw in protocol.METHODS_PAIRWISE.items())
    specs += (Spec("lambda=inf", kind="line"), Spec("zero", kind="zero"))
    cases = [
        ("noise", gi, r, case)
        # The stream is continuous: generate it in full, keep the slice.
        for gi, r, case in analysis_tools.noise_sweep(TRUTH, protocol.SIM_REPETITIONS)
        if r in reps
    ]
    cases += [
        ("nevents", gi, r, analysis_tools.nevents_case(TRUTH, count, r))
        for gi, count in enumerate(protocol.NEVENTS_SWEEP)
        for r in reps
    ]
    return [
        Unit(
            f"{sweep}_g{gi}_r{r:02d}",
            {"sweep": sweep, "grid_value": GRIDS[sweep][gi], "repetition": r},
            *case,
            specs,
        )
        for sweep, gi, r, case in cases
    ]


def white_facts(fits: Sequence[Record]) -> dict[str, Any]:
    """Methods ranked by exact mean SNR per condition, the L-curve's place,
    SURE(kernel) minus EmpBayes, and the pairwise over the true noise variance."""
    ranking, gaps, diffs = {}, [], {}
    for sweep, grid in GRIDS.items():
        ranking[sweep], diffs[sweep] = {}, {}
        for g in grid:
            means = {
                x: mean(snr_values(fits, x, sweep=sweep, grid_value=g)) for x in METHODS
            }
            if None in means.values():  # an estimator failed in every realization
                continue
            m = cast(dict[str, float], means)
            ranking[sweep][str(g)] = sorted(METHODS, key=lambda x: -m[x])
            gaps.append(max(v for k, v in m.items() if k != "L-curve") - m["L-curve"])
            diffs[sweep][str(g)] = m["SURE(kernel)"] - m["EmpBayes"]
    ranks = [r.index("L-curve") + 1 for s in ranking.values() for r in s.values()]
    ratio = {}
    for g in protocol.NOISE_SWEEP:
        v = [
            float(r["noise_var"])
            for r in fits
            if (r["sweep"], r["grid_value"], r["label"]) == ("noise", g, "SURE(kernel)")
            and r["status"] == "success"
            and r.get("noise_var") is not None
        ]
        ratio[str(g)] = float(statistics.median(v)) / g**2 if v else None
    return {
        "ranking_by_mean_snr": ranking,
        "lcurve_rank_range_of_7": [min(ranks), max(ranks)] if ranks else None,
        "lcurve_max_gap_to_best_db": float(max(gaps)) if gaps else None,
        "kernel_minus_empbayes_db": diffs,
        "pairwise_noise_var_over_true_median": ratio,
    }


def white_summary(records: Sequence[Record]) -> dict[str, Any]:
    fits = [r for r in records if r["label"] in METHODS]
    sweeps, line = {}, {}
    for sweep, grid in GRIDS.items():
        sweeps[sweep] = {
            str(g): {
                m: interval(snr_values(fits, m, sweep=sweep, grid_value=g))
                for m in METHODS
            }
            for g in grid
        }
        line[sweep] = {
            f"{g:g}": interval(
                snr_values(records, "lambda=inf", sweep=sweep, grid_value=g)
            )["mean"]
            for g in grid
        }
    return {
        "sweeps": sweeps,
        "failure_policy": flagged_groups(fits, ("sweep", "grid_value")),
        "facts": white_facts(fits),
        "trivial_baselines": {
            "lambda_inf_mean_snr_db": line,
            "zero_snr_all_exactly_0": all(
                r.get("snr_db") is not None
                and math.isclose(r["snr_db"], 0.0, rel_tol=0.0, abs_tol=0.0)
                for r in records
                if r["label"] == "zero"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Known-truth calibration of the reference rankings
# ---------------------------------------------------------------------------


def calibration_units(reps: Sequence[int]) -> list[Unit]:
    specs = tuple(Spec(m, kw) for m, kw in protocol.METHODS_PAIRWISE.items())
    units = []
    for r in reps:  # event-count recording; RandomState(42) selects K per class
        data, ix, lb = analysis_tools.nevents_case(TRUTH, protocol.CAL_I, r)
        rs = np.random.RandomState(protocol.CAL_SELECTION_SEED)
        sel = analysis_tools.rb_selection(rs, lb, protocol.CAL_K)
        units.append(
            Unit(
                f"cal_r{r:02d}",
                {"realization": r},
                data,
                ix[sel],
                lb[sel],
                specs,
                (ix, lb),
            )
        )
    return units


def distance_value(rec: Record, ref: str) -> float | None:
    if rec["status"] != "success":
        return None
    return rec["distance_truth"] if ref == "truth" else rec["distances"][ref]


def reference_rankings(units: Sequence[Mapping[str, Record]]) -> dict[str, Any]:
    """Per reference (and the truth): distance intervals, ranks, and Kendall
    tau-b of each reference's ranking vs the truth's."""
    per_reference: dict[str, Any] = {}
    for ref in ("truth", *protocol.REFERENCES):
        intervals = {
            m: interval(distance_value(fs[m], ref) for fs in units) for m in METHODS
        }
        means = {m: s["mean"] for m, s in intervals.items() if s["mean"] is not None}
        names = sorted(means)
        ranks = scipy.stats.rankdata([means[n] for n in names], method="average")
        per_reference[ref] = {
            "intervals": intervals,
            "ranks": {n: float(x) for n, x in zip(names, ranks)},
        }
    tau: dict[str, float | None] = {}
    truth_ranks = per_reference["truth"]["ranks"]
    for ref in protocol.REFERENCES:
        theirs = per_reference[ref]["ranks"]
        common = sorted(set(truth_ranks) & set(theirs))
        a, b = [truth_ranks[n] for n in common], [theirs[n] for n in common]
        t = scipy.stats.kendalltau(a, b)[0] if len(common) >= 2 else math.nan
        tau[ref] = float(t) if np.isfinite(t) else None
    return {"per_reference": per_reference, "kendall_tau_b_vs_truth": tau}


def calibration_facts(units: Sequence[Mapping[str, Record]]) -> dict[str, Any]:
    """Paired SURE(kernel) contrasts and the spread of ranks 2-4 (exact means)."""
    facts: dict[str, Any] = {}
    for other in ("SURE(lambda)", "EmpBayes"):
        for ref in ("truth", "OLS"):
            d = [
                cast(float, distance_value(fs["SURE(kernel)"], ref))
                - cast(float, distance_value(fs[other], ref))
                for fs in units
            ]
            facts[f"paired_kernel_minus_{other}_distance_{ref}"] = {
                "mean": mean(d),
                "half_ci95": half_ci(d),
                "n_kernel_closer": sum(x < 0 for x in d),
                "n": len(d),
            }
    trio = [
        cast(float, mean([fs[m]["distance_truth"] for fs in units]))
        for m in ("SURE(lambda)", "EmpBayes", "GCV")
        if units
    ]
    facts["truth_rank2_4_spread"] = float(max(trio) - min(trio)) if trio else None
    return facts


def calibration_summary(records: Sequence[Record]) -> dict[str, Any]:
    fits = [r for r in records if r["label"] in METHODS]
    refs = ("truth", *protocol.REFERENCES)
    # Common valid realizations: every method succeeded and every reference exists.
    valid = [
        fs
        for fs in by_unit(fits).values()
        if all(distance_value(fs[m], ref) is not None for m in METHODS for ref in refs)
    ]
    return {
        **reference_rankings(valid),
        "failure_policy": flagged_groups(fits, ()),
        "n_common_valid_realizations": len(valid),
        "facts": calibration_facts(valid),
    }


# ---------------------------------------------------------------------------
# AR(1) noise: colored factorial and known covariance
# ---------------------------------------------------------------------------


def pcn_units(
    mean_intervals: Sequence[int],
    phis: Sequence[float],
    reps: Sequence[int],
    specs: Callable[[float], tuple[Spec, ...]],
    seed_offset: int,
    stationary_start: bool,
) -> list[Unit]:
    return [
        Unit(
            f"sim_mi{mi}_phi{phi:g}_r{r:02d}",
            {"mean_interval": mi, "phi": phi, "repetition": r},
            *pcn_case(TRUTH, phi, r, mi, seed_offset, stationary_start),
            specs(phi),
        )
        for mi in mean_intervals
        for phi in phis
        for r in reps
    ]


def factorial_units(reps: Sequence[int]) -> list[Unit]:
    specs = tuple(Spec(c, kw) for c, kw in protocol.FACTORIAL_CELLS.items())
    specs += (Spec("Oracle", kind="oracle"),)
    return pcn_units(
        protocol.COLORED_MEAN_INTERVALS,
        protocol.FACTORIAL_PHIS,
        reps,
        lambda phi: specs,
        protocol.COLORED_SEED_OFFSET,
        protocol.COLORED_STATIONARY_START,
    )


def effects(y: Mapping[str, float]) -> dict[str, float]:
    """Kernel effect within noise, noise effect within kernel, interaction."""
    fw, fc, lw, lc = (y[c] for c in CELLS)
    return dict(
        zip(EFFECTS, (lw - fw, lc - fc, fc - fw, lc - lw, (lc - lw) - (fc - fw)))
    )


DENSE_DIFFERENCES = {  # main-text difference: (factorial effect, extreme reported)
    "fixed_colored_minus_white": ("noise_effect_fixed", "max"),
    "learned_minus_fixed_white": ("kernel_effect_white", "min"),
    "learned_minus_fixed_colored": ("kernel_effect_colored", "min"),
}


def dense_differences(sim: Mapping[str, Any]) -> dict[str, Any]:
    """The dense-case differences of the main text (dense mean interval): per phi,
    each paired difference with its 1.96 SE interval over realizations and whether
    it excludes 0; its extreme."""
    mi = protocol.COLORED_MEAN_INTERVALS[0]
    keys = {
        p: f"mi{mi}_phi{p:g}"
        for p in protocol.FACTORIAL_PHIS
        if f"mi{mi}_phi{p:g}" in sim
    }
    out: dict[str, Any] = {"mean_interval": mi, "phi": list(keys)}
    for name, (effect, tag) in DENSE_DIFFERENCES.items():
        ivs = [(p, sim[k][effect]) for p, k in keys.items()]
        out[name] = [
            {"phi": p, "mean": iv["mean"], "lo": iv["lo"], "hi": iv["hi"]}
            | {"excludes_zero": excludes_zero(iv)}
            for p, iv in ivs
        ]
        out[f"{name}_n_excluding_zero"] = sum(r["excludes_zero"] for r in out[name])
        sign = 1 if tag == "max" else -1
        defined = [r for r in out[name] if r["mean"] is not None]
        best = max(defined, key=lambda r: sign * r["mean"], default=None)
        extreme = {f"arg{tag}_phi": best["phi"]} if best else {}
        out[f"{name}_{tag}"] = best and extreme | {
            k: best[k] for k in ("mean", "lo", "hi")
        }
    return out


def factorial_summary(records: Sequence[Record]) -> dict[str, Any]:
    units = by_unit(records)
    sim: dict[str, Any] = {}
    for mi, phi in sorted({(r["mean_interval"], r["phi"]) for r in records}):
        mine = [
            fs
            for fs in units.values()
            if (fs[CELLS[0]]["mean_interval"], fs[CELLS[0]]["phi"]) == (mi, phi)
        ]
        # Common valid cells: a realization counts only if all four cells succeeded.
        eff = [
            effects({c: fs[c]["snr_db"] for c in CELLS})
            for fs in mine
            if all(
                fs[c]["status"] == "success"
                and math.isfinite(fs[c].get("snr_db", math.nan))
                for c in CELLS
            )
        ]
        cell: dict[str, Any] = {e: interval(x[e] for x in eff) for e in EFFECTS}
        cell["cells"] = {
            c: interval(snr_values(records, c, mean_interval=mi, phi=phi))
            for c in CELLS + ("Oracle",)
        }
        cell["n_valid"] = len(eff)
        sim[f"mi{mi}_phi{phi:g}"] = cell
    kernel_cells = [v[e] for v in sim.values() for e in EFFECTS[:2]]
    return {
        "sim": sim,
        "dense_paired_differences": dense_differences(sim),
        "failure_policy": {"sim": flagged_groups(records, ("mean_interval", "phi"))},
        "facts": {
            "kernel_effect_cells_excluding_zero": sum(map(excludes_zero, kernel_cells)),
            "interaction_positive_excluding_zero": sum(
                excludes_zero(v["interaction"], 1) for v in sim.values()
            ),
            "interaction_negative_excluding_zero": [
                k for k, v in sim.items() if excludes_zero(v["interaction"], -1)
            ],
        },
    }


def covariance_units(reps: Sequence[int]) -> list[Unit]:
    def specs(phi: float) -> tuple[Spec, ...]:
        known = {"ar1_phi": phi, "ar1_noise_var": protocol.COV_TRUE_VAR}
        return tuple(
            Spec(f"{m}|{cell}", {**protocol.METHODS_AR1COL[m], **extra})
            for m in ("SURE(kernel)", "SURE(lambda)")
            for cell, extra in (("estimated", {}), ("known", known))
        )

    return pcn_units(
        [protocol.COV_MEAN_INTERVAL],
        protocol.COV_PHIS,
        reps,
        specs,
        protocol.COV_SEED_OFFSET,
        protocol.COV_STATIONARY_START,
    )


def pd02_control() -> dict[str, Any]:
    """Baseline variance and phi estimates on unit-variance stationary AR(1) noise
    at P300-like window and event spacing: the PD02 control of the estimator bias."""
    c = protocol.Pd02Control()
    length, channels, phi = c.length, c.channels, c.phi
    half_win = length // 2
    ix = half_win + np.arange(c.events) * (length + c.gap)
    samples = int(ix[-1] + half_win)
    gain = np.sqrt(1 - phi**2)
    rng = np.random.RandomState(c.seed)
    variances, phis = [], []
    for _ in range(c.replicates):
        innovations = rng.standard_normal((samples, channels))
        innovations[0] /= gain
        data = cast(
            NDArray, scipy.signal.lfilter([gain], [1, -phi], innovations, axis=0)
        )
        variances.append(core.est_noise_var_baseline(data, ix, half_win))
        phis.append(core.estimate_phi(data, ix, half_win))
    return {
        **c._asdict(),
        "true_variance": 1.0,
        "variance_mean": float(np.mean(variances)),
        "phi_mean": float(np.mean(phis)),
    }


def covariance_summary(records: Sequence[Record]) -> dict[str, Any]:
    units = by_unit(records)
    sim: dict[str, Any] = {}
    for phi in sorted({r["phi"] for r in records}):
        mine = [fs for fs in units.values() if next(iter(fs.values()))["phi"] == phi]
        for m in ("SURE(kernel)", "SURE(lambda)"):
            est, kno = f"{m}|estimated", f"{m}|known"
            phis = [inner(fs[est]).get("phi_estimated") for fs in mine]
            phis = [p for p in phis if p is not None]
            sim[f"phi{phi:g}|{m}"] = {
                "estimated": interval(snr_values(records, est, phi=phi)),
                "known": interval(snr_values(records, kno, phi=phi)),
                "known_minus_estimated": interval(
                    fs[kno]["snr_db"] - fs[est]["snr_db"]
                    for fs in mine
                    if fs[kno]["status"] == "success" and fs[est]["status"] == "success"
                ),
                "phi_estimated_mean": float(np.mean(phis)) if phis else None,
            }
    lam = [
        abs(v["known_minus_estimated"]["mean"])
        for k, v in sim.items()
        if k.endswith("|SURE(lambda)") and v["known_minus_estimated"]["n"]
    ]
    return {
        "sim": sim,
        "failure_policy": {"sim": flagged_groups(records, ("phi",))},
        "sure_lambda_max_abs_known_minus_estimated_db": max(lam, default=None),
        "pd02_ar1_variance_control": pd02_control(),
    }


# ---------------------------------------------------------------------------
# Kernel length and symmetry; Q5 convergence
# ---------------------------------------------------------------------------


Q_SETTINGS: dict[str, protocol.CoreSettings] = {
    c: {**protocol.SURE_KERNEL, **q, "noise_est": protocol.Q_SIM_NOISE_EST}
    for c, q in protocol.Q_CELLS.items()
}


def q_units(
    reps: Sequence[int], limits: Callable[[int, str], Sequence[int | None]]
) -> list[Unit]:
    """One unit per realization; *limits*(repetition, cell) gives the cell's BFGS
    limits (None: core's own)."""
    units = []
    for r in reps:
        specs = tuple(
            Spec(
                cell if lim is None else f"{cell}@{lim}", Q_SETTINGS[cell], maxiter=lim
            )
            for cell in protocol.Q_CELLS
            for lim in limits(r, cell)
        )
        if specs:
            data = analysis_tools.nevents_case(TRUTH, protocol.Q_SIM_I, r)
            units.append(Unit(f"sim_r{r:02d}", {"repetition": r}, *data, specs))
    return units


def status(rec: Record | None) -> str:
    if rec is None:
        return "missing"
    if rec["status"] != "success":
        return "failed"
    s = inner(rec).get("optimizer_status")
    names: dict[object, str] = {0: "converged", BFGS_MAXITER_STATUS: "capped"}
    return names.get(s, f"status{s}")


def run_q(workers: int, reps: Sequence[int]) -> tuple[list[Record], dict[str, NDArray]]:
    """Every cell at core's limit (50) and the convergence cells at 500, then 5000
    where still capped. The 50 fits of the convergence follow-up are the plain fits."""
    records, arrays = run_units(
        workers,
        q_units(reps, lambda r, c: [None, LIMIT] if c in Q_CONVERGENCE else [None]),
    )
    capped = {(r["repetition"], r["label"]) for r in records if status(r) == "capped"}
    more, arrs = run_units(
        workers,
        q_units(
            reps, lambda r, c: [RERUN_LIMIT] if (r, f"{c}@{LIMIT}") in capped else []
        ),
    )
    arrays.update(arrs)
    return records + more, arrays


def nyquist_over_dc(kernel: NDArray | Sequence[float] | None) -> float | None:
    """|H(Nyquist)|/|H(0)|; None when |H(0)| <= threshold x max |H| (decision OQ-B1)."""
    if kernel is None:
        return None
    f = np.linspace(0.0, 0.5, protocol.KERNEL_N_FREQS)
    h = np.abs(
        np.exp(-2j * np.pi * np.outer(f, np.arange(len(kernel))))
        @ np.asarray(kernel, dtype=float)
    )
    if (
        not np.isfinite(h).all()
        or h.max() == 0
        or h[0] <= protocol.KERNEL_DC_REL_THRESHOLD * h.max()
    ):
        return None
    return float(h[-1] / h[0])


Fits = dict[str, Record | None]


def q5_fits(fs: Mapping[str, Record], cell: str) -> Fits:
    """A cell's fits at limit 50 (the plain fit), 500, 5000 (rerun) and the final one."""
    lim, rlim = fs.get(f"{cell}@{LIMIT}"), fs.get(f"{cell}@{RERUN_LIMIT}")
    return {"50": fs.get(cell), "lim": lim, "final": rlim or lim, "rerun": rlim}


def snr(rec: Record | None) -> float | None:
    return (rec or {}).get("snr_db")


def convergence_counts(f: Sequence[Fits]) -> dict[str, Any]:
    st50, st_lim = [status(x["50"]) for x in f], [status(x["lim"]) for x in f]
    final = [status(x["final"]) for x in f]
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
        f"rerun_{RERUN_LIMIT}": sum(x["rerun"] is not None for x in f),
        "other_or_failed_final": sum(s not in {"converged", "capped"} for s in final),
        "median_iterations_final_converged": (
            float(np.median(iters)) if iters else None
        ),
    }


def q5_summary(records: Sequence[Record]) -> dict[str, Any]:
    units = list(by_unit(records).values())
    base = [q5_fits(fs, "Q3-sym")["final"] for fs in units]
    out: dict[str, Any] = {
        k: {} for k in ("convergence", "kernels", "snr_change", "q5_vs_q3sym")
    }
    for cell in Q_CONVERGENCE:
        f = [q5_fits(fs, cell) for fs in units]
        out["convergence"][cell] = convergence_counts(f)
        converged = [x["final"] for x in f if status(x["final"]) == "converged"]
        ratios = [nyquist_over_dc(cast(Record, r).get("kernel")) for r in converged]
        out["kernels"][cell] = {
            "final_converged": stats(ratios),
            "final_converged_dc_near_zero": sum(v is None for v in ratios),
        }
        change = [
            (snr(x["final"]), snr(x["50"])) for x in f if status(x["50"]) == "capped"
        ]
        out["snr_change"][cell] = {
            "capped_at_50": stats(
                [a - b for a, b in change if a is not None and b is not None]
            )
        }
        if cell != "Q3-sym":
            out["q5_vs_q3sym"][cell] = {
                "converged final only": interval(
                    cast(float, snr(x["final"])) - cast(float, snr(b))
                    for x, b in zip(f, base)
                    if status(x["final"]) == "converged" and snr(b) is not None
                )
            }
    raised = [r for r in records if "@" in r["label"] and r["status"] == "success"]
    iterations = [inner(r).get("optimizer_iterations") for r in raised]
    out["max_iterations_raised_limit"] = max(
        (n for n in iterations if n is not None), default=None
    )
    grid = [
        r if "@" in r["label"] else {**r, "label": f"{r['label']}@{LIMIT_DEFAULT}"}
        for r in records
        if "@" in r["label"] or r["label"] in Q5
    ]
    out["failure_policy"] = flagged_groups(grid, ())
    return out


def q_summary(records: Sequence[Record]) -> dict[str, Any]:
    base = [r for r in records if "@" not in r["label"]]
    asym = [
        max(abs(h[i] - h[-1 - i]) for i in range(len(h))) / max(map(abs, h))
        for h in (
            r.get("kernel")
            for r in base
            if r["label"] == "Q3-free" and r["status"] == "success"
        )
        if h is not None and len(h) and max(map(abs, h)) > 0
    ]
    return {
        "q_symmetry": {
            "sim": {c: interval(snr_values(base, c)) for c in protocol.Q_CELLS},
            "failure_policy": {"sim": flagged_groups(base, ())},
            "q3free_relative_asymmetry": {
                "max": float(max(asym)) if asym else None,
                "median": float(statistics.median(asym)) if asym else None,
            },
        },
        "q5_convergence": q5_summary(records),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

FIELDS = ("sweep", "grid_value", "realization", "mean_interval", "phi", "repetition")


def csv_row(rec: Record, fields: Sequence[str]) -> dict[str, object]:
    row: dict[str, object] = {f: rec.get(f) for f in (*fields, *COLUMNS)}
    row.update({k: inner(rec).get(k) for k in INNER_FIELDS})
    row.update({k: ";".join(rec.get(k) or []) for k in ("flags", "warnings")})
    for ref in protocol.REFERENCES:
        row[f"distance_{ref}"] = (rec.get("distances") or {}).get(ref)
    if rec.get("kernel") is not None:
        row["kernel"] = json.dumps(np.asarray(rec["kernel"]).tolist())
    return row


Runner = Callable[[int, Sequence[int]], tuple[list[Record], dict[str, NDArray]]]


def units_runner(units: Callable[[Sequence[int]], list[Unit]]) -> Runner:
    return lambda workers, reps: run_units(workers, units(reps))


ANALYSIS: dict[str, tuple[Runner, Callable[[Sequence[Record]], dict[str, Any]]]] = {
    "white_sims": (units_runner(white_units), white_summary),
    "calibration": (units_runner(calibration_units), calibration_summary),
    "colored_factorial": (units_runner(factorial_units), factorial_summary),
    "q_symmetry": (run_q, q_summary),
    "known_covariance": (units_runner(covariance_units), covariance_summary),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="./runs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--reps",
        type=int,
        choices=range(1, protocol.SIM_REPETITIONS + 1),
        default=protocol.SIM_REPETITIONS,
        metavar=f"1..{protocol.SIM_REPETITIONS}",
    )
    parser.add_argument("--analyses", nargs="+", choices=ANALYSIS, default=[*ANALYSIS])
    args = parser.parse_args()
    for name in args.analyses:
        started = datetime.now(UTC)
        run, summarize = ANALYSIS[name]
        records, arrays = run(args.workers, list(range(args.reps)))
        out = Path(args.out) / name
        out.mkdir(parents=True, exist_ok=True)
        fields = [f for f in FIELDS if any(f in r for r in records)]
        with (out / "per_fit.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, [*fields, *COLUMNS], lineterminator="\n")
            writer.writeheader()
            writer.writerows(csv_row(r, fields) for r in records)
        np.savez(out / "templates.npz", allow_pickle=False, **arrays)
        analysis_tools.write_json(out / "summary.json", summarize(records))
        analysis_tools.write_run_info(out, {**vars(args), "analysis": name}, started)


if __name__ == "__main__":
    main()
