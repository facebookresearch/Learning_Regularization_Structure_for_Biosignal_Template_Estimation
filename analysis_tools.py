# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared pieces of the paper analyses.

Recording caches and selections, fitting one estimator with the failure
policy, the lambda = inf fit, the distance to a reference,
subject-level statistics, the worker pool, JSON output and the run record.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import functools
import importlib
import importlib.metadata
import json
import math
import multiprocessing
import socket
import subprocess
import sys
import time
import traceback
import warnings
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, cast, ParamSpec, Protocol, TYPE_CHECKING, TypeVar

import numpy as np
import scipy
import threadpoolctl
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    import core
    import dtw_average
    import heldout_p300
    import protocol
else:
    _PREFIX = f"{__package__}." if __package__ else ""
    core = importlib.import_module(f"{_PREFIX}core")
    dtw_average = importlib.import_module(f"{_PREFIX}dtw_average")
    heldout_p300 = importlib.import_module(f"{_PREFIX}heldout_p300")
    protocol = importlib.import_module(f"{_PREFIX}protocol")

# Errors of the machine, not of an estimator: they stop the run (decision OQ6).
INFRASTRUCTURE_ERRORS: tuple[type[BaseException], ...] = (
    MemoryError,
    OSError,
    ImportError,
)
FLAG_WORDS: tuple[str, ...] = ("success", "passed", "converged")
PACKAGES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "mne",
    "moabb",
    "wfdb",
    "matplotlib",
    "threadpoolctl",
)

Comparator = Callable[[NDArray], NDArray]


class PenaltyPath(Protocol):
    """The parts of ``core``'s L-curve penalty path that the diagnostics read."""

    rotated: NDArray
    ols: NDArray
    values: NDArray

    def log10_envelope(self, margin: int = 6) -> tuple[int, int]: ...


# The (penalty path, data, design) arguments of one ``core._select_lcurve`` call.
SelectorCall = tuple[PenaltyPath, NDArray, "NDArray | scipy.sparse.spmatrix"]
# A cell of one subject: a repetition (distances) or a (fold, repetition) pair (AUC).
Cell = TypeVar("Cell", bound=Hashable)
T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


# ---------------------------------------------------------------------------
# Recordings and selections
# ---------------------------------------------------------------------------


def save_blocks(path: Path, blocks: Sequence[heldout_p300.RecordingBlock]) -> None:
    if not blocks:
        raise ValueError(f"save_blocks({path}): no blocks to save")
    arrays: dict[str, NDArray] = {
        "channels": np.array(blocks[0].channels),
        "block_ids": np.array([block.block for block in blocks], dtype=int),
    }
    for block in blocks:
        b = block.block
        arrays[f"data_{b}"] = block.data
        arrays[f"onsets_{b}"] = block.onsets
        arrays[f"native_onsets_{b}"] = block.native_onsets
        arrays[f"labels_{b}"] = block.labels
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_blocks(path: Path, subject: int) -> list[heldout_p300.RecordingBlock]:
    """The blocks written by ``save_blocks``, in their saved order and ids."""
    with np.load(path) as z:
        if "block_ids" not in z.files:
            raise KeyError(f"{path}: no 'block_ids' array (not written by save_blocks)")
        channels = tuple(str(c) for c in z["channels"])
        blocks = []
        for b in (int(i) for i in z["block_ids"]):
            fields = ("data", "onsets", "native_onsets", "labels")
            missing = [f"{f}_{b}" for f in fields if f"{f}_{b}" not in z.files]
            if missing:
                raise KeyError(f"{path}: block {b} lacks arrays {missing}")
            blocks.append(
                heldout_p300.RecordingBlock(
                    subject=subject,
                    block=b,
                    data=z[f"data_{b}"],
                    onsets=z[f"onsets_{b}"],
                    native_onsets=z[f"native_onsets_{b}"],
                    labels=z[f"labels_{b}"],
                    channels=channels,
                )
            )
        return blocks


def whole_recording_selections(
    blocks: Sequence[heldout_p300.RecordingBlock],
    rs: np.random.RandomState,
    k: int,
    repetitions: int,
    half_win: int,
    guard: int,
) -> list[list[list[int]]]:
    """Per repetition, K eligible events of class 0 then class 1.

    Drawn from the whole recording, without replacement; each selection is a
    list of (block, event) rows sorted by block id and event. The caller passes
    a fresh ``RandomState(42)`` per subject and K (decision OQ2).
    """
    if not blocks:
        raise ValueError("whole_recording_selections: no blocks to select from")
    rows = [(b.block, e) for b in blocks for e in range(len(b.onsets))]
    eligible = np.concatenate([b.eligible(half_win, guard) for b in blocks])
    labels = np.concatenate([b.labels for b in blocks])
    out = []
    for _ in range(repetitions):
        picks = np.concatenate(
            [
                rs.choice(
                    np.flatnonzero(eligible & (labels == label)), k, replace=False
                )
                for label in np.unique(labels)
            ]
        )
        out.append(sorted([list(rows[i]) for i in picks]))
    return out


# ---------------------------------------------------------------------------
# Fitting one estimator
# ---------------------------------------------------------------------------


def plain_dtw(epochs: NDArray) -> NDArray:
    """Molina et al.'s DTW average without the low-pass prefilter."""
    return dtw_average.dtw_average(epochs, None)


@dataclass(frozen=True)
class Estimators:
    """Names mapped to fits: method settings and comparator functions."""

    methods: Mapping[str, Mapping[str, Any]]
    comparators: Mapping[str, Comparator]


def default_estimators() -> Estimators:
    """The paper's estimators: the benchmark's P300 methods and both Molina averages."""
    return Estimators(
        methods=heldout_p300.reference_methods(),
        comparators={
            protocol.FILTERED: dtw_average.molina_template,
            protocol.PLAIN: plain_dtw,
        },
    )


def failure_flags(inner: object, prefix: str = "") -> list[str]:
    """Diagnostics whose success/passed/converged entry is False."""
    flags: list[str] = []
    if isinstance(inner, Mapping):
        for k, v in inner.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if any(w in str(k).lower() for w in FLAG_WORDS) and (
                v is False or v is np.False_
            ):
                flags.append(key)
            flags.extend(failure_flags(v, key))
    elif isinstance(inner, (list, tuple)):
        for i, v in enumerate(inner):
            flags.extend(failure_flags(v, f"{prefix}[{i}]"))
    return flags


def _warning_texts(caught: Sequence[warnings.WarningMessage]) -> list[str]:
    return sorted({f"{w.category.__name__}: {w.message}" for w in caught})


def _fit(
    estimators: Estimators,
    name: str,
    training: Sequence[heldout_p300.RecordingBlock],
    rows: NDArray,
    half_win: int,
) -> tuple[Mapping[object, NDArray], dict[str, object] | None]:
    if name in estimators.methods:
        return heldout_p300.fit_templates(
            training, rows, estimators.methods[name], half_win
        )
    comparator = estimators.comparators[name]
    return heldout_p300.fit_comparator(training, rows, comparator, half_win), None


def fit_estimator(
    estimators: Estimators,
    name: str,
    training: Sequence[heldout_p300.RecordingBlock],
    rows: NDArray,
    half_win: int,
    exceptions_are_failures: bool,
) -> tuple[dict[str, Any], NDArray | None]:
    """Fit one named estimator under ``fit_call``'s failure policy."""
    if name not in estimators.methods and name not in estimators.comparators:
        raise ValueError(
            f"unknown estimator {name!r}; known: "
            f"{sorted([*estimators.methods, *estimators.comparators])}"
        )
    return fit_call(
        lambda: _fit(estimators, name, training, rows, half_win),
        ("0", "1"),
        exceptions_are_failures,
    )


def score_fit(
    record: dict[str, Any],
    array: NDArray | None,
    references: Mapping[str, NDArray | None],
    test: heldout_p300.RecordingBlock | None,
    half_win: int,
    guard: int,
) -> None:
    """Add the distance to each reference (no *test*: analysis A) or the AUC on the
    held-out *test* block (analysis B) to a fit's record; None for a failed fit."""
    ok = array is not None and record["status"] == "success"
    if test is None:
        record["distances"] = {
            r: distance(cast(NDArray, array), ref) if ok and ref is not None else None
            for r, ref in references.items()
        }
        return
    record["auc"] = None
    if ok:
        a = cast(NDArray, array)
        try:
            scores, labels = heldout_p300.score_events(
                test, {"0": a[0], "1": a[1]}, half_win, guard
            )
            record["auc"] = heldout_p300.auc(scores, labels)
        except ValueError as error:  # nonfinite scores or a one-class block
            record["status"] = "failure"
            record["failure_type"] = type(error).__name__
            record["failure_message"] = f"held-out scoring: {error}"
            return
        record["n_heldout_events"] = len(labels)
        record["n_heldout_targets"] = int(np.sum(labels == 1))


def fit_unit(
    estimators: Estimators,
    names: Sequence[str],
    training: Sequence[heldout_p300.RecordingBlock],
    rows: NDArray,
    test: heldout_p300.RecordingBlock | None,
    references: Mapping[str, NDArray | None],
    half_win: int,
    guard: int,
    exceptions_are_failures: bool,
) -> list[tuple[dict[str, Any], NDArray | None]]:
    """Fit and score every named estimator on one selection, in order."""
    out = []
    for name in names:
        record, array = fit_estimator(
            estimators, name, training, rows, half_win, exceptions_are_failures
        )
        score_fit(record, array, references, test, half_win, guard)
        out.append((record, array))
    return out


def population_template(
    references: Mapping[int, NDArray | None], subject: int
) -> NDArray | None:
    """The other subjects' mean (class, time, channel) reference (decision OQ8).

    None when there is no donor or a donor's reference failed. The subject's own
    reference is never read.
    """
    donors = [d for d in references if d != subject]
    if not donors or any(references[d] is None for d in donors):
        return None
    cohort: dict[int, Mapping[object, NDArray]] = {
        d: {"0": a[0], "1": a[1]} for d, a in references.items() if a is not None
    }
    cohort[subject] = {"0": np.zeros(1), "1": np.zeros(1)}
    t = heldout_p300.population_templates(cohort, subject)
    return np.stack([t["0"], t["1"]])


@contextlib.contextmanager
def forced_infinite_lambda() -> Iterator[list[SelectorCall]]:
    """Make ``core``'s L-curve selection return lambda = inf inside the context.

    The L-curve fit then returns the infinite-penalty limit of the
    second-difference penalty: the least-squares straight line per class and
    channel. Yields the (path, X, D) arguments of every selection call; the
    original selector is restored on exit, also after an exception.
    """
    original = core._select_lcurve  # noqa: SLF001
    calls: list[SelectorCall] = []

    def infinite(
        path: PenaltyPath,
        X: NDArray,
        D: NDArray | scipy.sparse.spmatrix,
        n_lambdas: int = 80,
    ) -> tuple[float, dict[str, object]]:
        calls.append((path, X, D))
        return math.inf, {"opt_val": math.inf, "method": "forced_inf"}

    core._select_lcurve = infinite  # noqa: SLF001
    try:
        yield calls
    finally:
        core._select_lcurve = original  # noqa: SLF001


def training_phi(
    training: Sequence[heldout_p300.RecordingBlock], rows: NDArray, half_win: int
) -> float | None:
    """``core.estimate_phi`` on the joined training blocks at the selected onsets;
    None when it is not finite (nonfinite data)."""
    by_id = {b.block: b for b in training}
    starts = np.cumsum([0] + [len(b.data) for b in training[:-1]])
    offset = {b.block: s for b, s in zip(training, starts)}
    ix = np.array([offset[int(b)] + by_id[int(b)].onsets[e] for b, e in rows])
    data = np.vstack([b.data for b in training])
    phi = float(core.estimate_phi(data, ix, half_win))
    return phi if math.isfinite(phi) else None


def distance(array: NDArray, reference: NDArray) -> float:
    """Mean over the two classes of the Frobenius norm of the difference."""
    return float(
        (
            np.linalg.norm(array[0] - reference[0])
            + np.linalg.norm(array[1] - reference[1])
        )
        / 2
    )


# ---------------------------------------------------------------------------
# Subject-level statistics
# ---------------------------------------------------------------------------


def signed_rank_test(effects: Sequence[float]) -> dict[str, object]:
    """Two-sided exact Wilcoxon signed-rank test (zeros dropped, ties averaged).

    Returns p = 1 and r = 0 when every effect is zero, and nulls with a reason
    when there is no effect at all (N = 0).
    """
    d = np.asarray(effects, dtype=float)
    if d.size == 0:
        return {
            "n": 0,
            "n_nonzero": 0,
            "p": None,
            "rank_biserial": None,
            "reason": "no subject with a valid effect",
        }
    if not np.all(np.isfinite(d)):
        raise ValueError(f"nonfinite subject effects: {d}")
    nonzero = d[d != 0]
    n = int(nonzero.size)
    if n == 0:
        return {
            "n": int(d.size),
            "n_nonzero": 0,
            "p": 1.0,
            "rank_biserial": 0.0,
            "reason": "all effects zero",
        }
    ranks = scipy.stats.rankdata(np.abs(nonzero))
    positive = float(np.sum(ranks[nonzero > 0]))
    total = float(np.sum(ranks))
    # SciPy refuses a single observation; both sign patterns of one effect tie.
    p = (
        1.0
        if n == 1
        else float(
            scipy.stats.wilcoxon(
                d,
                zero_method="wilcox",
                # The SciPy type stubs lack the PermutationMethod option.
                # pyrefly: ignore[unexpected-keyword, missing-attribute]
                method=scipy.stats.PermutationMethod(n_resamples=np.inf),
            ).pvalue
        )
    )
    return {
        "n": int(d.size),
        "n_nonzero": n,
        "p": p,
        "rank_biserial": (2 * positive - total) / total,
        "reason": None,
    }


def bootstrap_mean_ci(
    effects: Sequence[float], resamples: int, seed: int, level: float = 0.95
) -> list[float] | None:
    """Percentile interval of the mean over subjects resampled with replacement."""
    d = np.asarray(effects, dtype=float)
    if d.size == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(resamples, d.size))
    means = d[idx].mean(axis=1)
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(means, [tail, 100 - tail])
    return [float(lo), float(hi)]


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind="stable")
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def _paired_values(
    anchor: Mapping[Cell, float | None],
    other: Mapping[Cell, float | None],
    cells: Sequence[Cell],
    log_ratio: bool,
) -> tuple[list[float], list[Cell], int]:
    values: list[float] = []
    valid_cells: list[Cell] = []
    excluded_nonpositive = 0
    for cell in cells:
        a = anchor.get(cell)
        b = other.get(cell)
        if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
            continue
        if log_ratio:
            if a <= 0 or b <= 0:
                excluded_nonpositive += 1
                continue
            values.append(math.log(b / a))
        else:
            values.append(a - b)
        valid_cells.append(cell)
    return values, valid_cells, excluded_nonpositive


def paired_effects(
    anchor: Mapping[int, Mapping[Cell, float | None]],
    other: Mapping[int, Mapping[Cell, float | None]],
    log_ratio: bool,
) -> dict[int, dict[str, object]]:
    """Per subject, the mean effect over cells valid for both estimators.

    *anchor* and *other* map subject -> cell -> value (None when the cell failed).
    log_ratio: effect = log(other / anchor) (distances; positive favours the
    anchor), cells with a nonpositive distance are excluded and counted;
    otherwise effect = anchor - other (AUC; positive favours the anchor).
    """
    out: dict[int, dict[str, object]] = {}
    for subject in sorted(anchor):
        theirs = other.get(subject, {})
        cells = sorted(set(anchor[subject]) | set(theirs), key=str)
        values, valid_cells, excluded = _paired_values(
            anchor[subject], theirs, cells, log_ratio
        )
        out[subject] = {
            "effect": float(np.mean(values)) if values else None,
            "n_valid": len(values),
            "n_cells": len(cells),
            "excluded_nonpositive": excluded,
        }
        first = cells[0] if cells else None
        if isinstance(first, tuple):  # held-out cells are (fold, repetition)
            folds = sorted({c[0] for c in cells if isinstance(c, tuple)})
            out[subject]["n_valid_by_fold"] = {
                str(f): sum(
                    1 for c in valid_cells if isinstance(c, tuple) and c[0] == f
                )
                for f in folds
            }
    return out


def summarize(
    per_subject: Mapping[int, Mapping[str, object]],
    resamples: int,
    seed: int,
    test: bool,
) -> dict[str, object]:
    """Subject-level summary; the test only for inferential comparisons."""
    effects = [
        float(cast(float, v["effect"]))
        for v in per_subject.values()
        if v["effect"] is not None
    ]
    summary: dict[str, object] = {
        "n_subjects": len(effects),
        "per_subject": {str(s): v for s, v in per_subject.items()},
        "mean": float(np.mean(effects)) if effects else None,
        "median": float(np.median(effects)) if effects else None,
        "ci95": bootstrap_mean_ci(effects, resamples, seed),
    }
    if test:
        summary["test"] = signed_rank_test(effects)
    return summary


def paired_difference(
    average: Mapping[int, Mapping[Cell, float | None]],
    other: Mapping[int, Mapping[Cell, float | None]],
) -> dict[str, Any]:
    """Per subject, method minus Average over the cells both fitted (the
    failure policy); percentile CI of the mean over subjects, 10,000 resamples of
    a fresh ``default_rng(42)``. Descriptive."""
    per: dict[str, tuple[float, float, int]] = {}
    excluded: dict[str, int] = {}
    for s in sorted(average):  # numeric subject order fixes the bootstrap draws
        avg, est = average[s], other[s]
        keys = [c for c, a in avg.items() if a is not None and est.get(c) is not None]
        # Over the union, so a cell where only one of the two failed is counted.
        excluded[str(s)] = len(set(avg) | set(est)) - len(keys)
        if keys:
            m = float(np.mean([cast(float, est[c]) for c in keys]))
            a = float(np.mean([cast(float, avg[c]) for c in keys]))
            per[str(s)] = (m, a, len(keys))
    out: dict[str, Any] = {
        "n_subjects": len(per),
        "n_valid_pairs": {s: p[2] for s, p in per.items()},
        "excluded_cells_by_subject": excluded,
        "excluded_cells": sum(excluded.values()),
    }
    if not per:
        return out | {"mean_diff": None, "ci95": None, "per_subject_diff": {}}
    m = np.array([p[0] for p in per.values()])
    a = np.array([p[1] for p in per.values()])
    d = m - a
    rng = np.random.default_rng(42)
    boots = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return out | {
        "mean_auc_method": float(m.mean()),
        "mean_auc_average": float(a.mean()),
        "mean_diff": float(d.mean()),
        "ci95": [float(lo), float(hi)],
        "subjects_above_average": int((d > 0).sum()),
        "subjects_below_average": int((d < 0).sum()),
        "per_subject_diff": {s: float(x) for s, x in zip(per, d)},
    }


# ---------------------------------------------------------------------------
# Workers and the run record
# ---------------------------------------------------------------------------


def _one_blas_thread() -> None:
    threadpoolctl.threadpool_limits(1)


def run_pool(workers: int, call: Callable[[T], R], tasks: Sequence[T]) -> list[R]:
    """*call* on every task, results in task order, one BLAS thread per process.

    Workers are spawned processes, so *call* and the tasks must be picklable.
    """
    if workers <= 1:
        with threadpoolctl.threadpool_limits(1):
            return [call(task) for task in tasks]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_one_blas_thread,
    ) as pool:
        return list(pool.map(call, tasks))


def plain(value: object) -> object:
    """JSON-safe copy: arrays become lists, nonfinite floats "inf"/"-inf"/"nan"."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if math.isfinite(x):
            return x
        return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"cannot write value of type {type(value).__name__} as JSON")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), allow_nan=False, indent=1) + "\n")


def _commit() -> str | None:
    here = Path(__file__).resolve().parent
    for command in (["sl", "whereami"], ["git", "rev-parse", "HEAD"]):
        try:
            done = subprocess.run(
                command,
                cwd=here,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if done.returncode == 0 and done.stdout.strip():
            return done.stdout.strip()
    return None


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_run_info(
    out: Path, settings: Mapping[str, object], started: datetime
) -> None:
    """``out/run_info.json``: code revision, command, settings, environment, times."""
    info = {
        "commit": _commit(),
        "argv": sys.argv,
        "settings": settings,
        "python": sys.version,
        "packages": {name: _version(name) for name in PACKAGES},
        "threadpools": threadpoolctl.threadpool_info(),
        "host": socket.gethostname(),
        "start": started.isoformat(timespec="seconds"),
        "end": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    write_json(out / "run_info.json", info)


# ---------------------------------------------------------------------------
# Reference noise model and noise-corrected distance (run_reference_noise.py)
# ---------------------------------------------------------------------------


def residual_autocovariance(
    residual: NDArray, edges: Sequence[int], max_lag: int
) -> NDArray:
    """Biased per-channel autocovariance at lags 0..max_lag, summed within blocks
    (*edges*: block starts and the end) and divided by the total length."""
    acov = np.zeros((max_lag + 1, residual.shape[1]))
    for s, e in zip(edges[:-1], edges[1:]):
        r = residual[s:e]
        for k in range(max_lag + 1):
            acov[k] += np.sum(r[: len(r) - k] * r[k:], axis=0)
    acov /= len(residual)
    return acov


def bartlett_kernel(acov: NDArray) -> NDArray:
    """Bartlett-tapered autocovariance at lags -L..L (positive semidefinite)."""
    max_lag = len(acov) - 1
    tapered = acov * (1 - np.arange(max_lag + 1) / (max_lag + 1))[:, None]
    return np.concatenate([tapered[:0:-1], tapered])


def dense_rows(matrix: scipy.sparse.csc_matrix, rows: slice | NDArray) -> NDArray:
    """The selected rows of a sparse matrix as a dense array."""
    return cast("scipy.sparse.csc_matrix", matrix[rows]).toarray()


def noise_gram(
    design: scipy.sparse.csc_matrix, edges: Sequence[int], kernel: NDArray
) -> NDArray:
    """D^T Sigma_c D per channel c, symmetrized; Sigma_c is the Toeplitz covariance
    of channel c's *kernel* within blocks, none across blocks."""
    n_parameters = design.shape[1]
    g = np.zeros((kernel.shape[1], n_parameters, n_parameters))
    for s, e in zip(edges[:-1], edges[1:]):
        d = dense_rows(design, slice(s, e))
        for c in range(kernel.shape[1]):
            g[c] += d.T @ scipy.signal.fftconvolve(
                d, kernel[:, c][:, None], mode="same", axes=0
            )
    return 0.5 * (g + g.transpose(0, 2, 1))


def class_noise_power(projector: NDArray, gram: NDArray, length: int) -> NDArray:
    """Expected squared Frobenius norm per class (*length* rows) of the noise of
    theta = Q D^T y: sum_c of the class-block traces of Q G_c Q^T."""
    n_classes = projector.shape[0] // length
    power = np.zeros(n_classes)
    for c in range(len(gram)):
        cov = projector @ gram[c] @ projector.T
        for k in range(n_classes):
            power[k] += np.trace(
                cov[k * length : (k + 1) * length, k * length : (k + 1) * length]
            )
    return power


def nc_metric(
    d2: NDArray, sigma2: NDArray, extra: NDArray | None = None
) -> dict[str, Any]:
    """Noise-corrected distance sqrt(max(d2 - sigma2 + extra, 0)), mean over classes,
    fits and subjects. d2, extra (2X or None): (subjects, fits, classes); sigma2:
    (subjects, classes). A failed fit (NaN) is excluded and counted."""
    corr = d2 - sigma2[:, None, :] + (0 if extra is None else extra)
    valid_fit = ~np.any(np.isnan(corr), axis=2)
    e = np.sqrt(np.maximum(np.where(np.isnan(corr), 0.0, corr), 0))
    per_fit = np.where(valid_fit, e.mean(axis=2), np.nan)
    n_valid = valid_fit.sum(axis=1)
    with np.errstate(invalid="ignore"):
        subj = np.where(
            n_valid > 0, np.nansum(per_fit, axis=1) / np.maximum(n_valid, 1), np.nan
        )
        cm = np.where(
            n_valid[:, None] > 0,
            np.nansum(np.where(valid_fit[:, :, None], corr, 0.0), axis=1)
            / np.maximum(n_valid, 1)[:, None],
            np.nan,
        )
    rms = np.sqrt(np.maximum(cm, 0)).mean(axis=1)
    ok = ~np.isnan(subj)
    vc = np.where(valid_fit[:, :, None], corr, 0.0)
    return {
        "subject": [None if np.isnan(v) else float(v) for v in subj],
        "mean": float(subj[ok].mean()) if ok.any() else None,
        "n_subjects": int(ok.sum()),
        "rms_subject": [None if np.isnan(v) else float(v) for v in rms],
        "rms_mean": (
            float(rms[~np.isnan(rms)].mean()) if (~np.isnan(rms)).any() else None
        ),
        "trunc_cells": int(np.sum((vc < 0) & valid_fit[:, :, None])),
        "n_cells": int(2 * valid_fit.sum()),
        "trunc_fits": int(np.sum(np.any(vc < 0, axis=2) & valid_fit)),
        "n_fits": int(valid_fit.sum()),
        "excluded_failed_fits": int((~valid_fit).sum()),
    }


# ---------------------------------------------------------------------------
# Failure policy for any fit
# ---------------------------------------------------------------------------


def fit_call(
    call: Callable[[], tuple[Mapping[Any, NDArray], Mapping[str, object] | None]],
    keys: Sequence[str],
    exceptions_are_failures: bool,
) -> tuple[dict[str, Any], NDArray | None]:
    """Run one fit; return its record and its templates stacked in *keys* order.

    An exception or unusable output is a counted failure without a template
    (decision OQ6) unless it is an infrastructure error; a nonfinite template is
    a failure that keeps its template. A vector tuning value is ``kernel``, a
    scalar one ``lambda``.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        try:
            templates, inner = call()
        except INFRASTRUCTURE_ERRORS:
            raise
        except Exception as error:
            if not exceptions_are_failures:
                raise
            return _failure_record(error, start, caught), None
        seconds = time.perf_counter() - start
    try:
        return _success_record(templates, inner, keys, seconds, caught)
    except INFRASTRUCTURE_ERRORS:
        raise
    except Exception as error:  # unusable output: missing class, shapes, opt_val
        if not exceptions_are_failures:
            raise
        return _failure_record(error, start, caught), None


def _failure_record(
    error: Exception, start: float, caught: Sequence[warnings.WarningMessage]
) -> dict[str, Any]:
    return {
        "status": "failure",
        "failure_type": type(error).__name__,
        "failure_message": str(error),
        "traceback": traceback.format_exc(limit=8),
        "seconds": time.perf_counter() - start,
        "warnings": _warning_texts(caught),
        "flags": [],
        "inner": None,
    }


def _success_record(
    templates: Mapping[Any, NDArray],
    inner: Mapping[str, object] | None,
    keys: Sequence[str],
    seconds: float,
    caught: Sequence[warnings.WarningMessage],
) -> tuple[dict[str, Any], NDArray]:
    if not keys:
        raise ValueError("no class keys to stack the templates in")
    missing = [key for key in keys if key not in templates]
    if missing:
        raise KeyError(
            f"the fit returned no template for classes {missing}; "
            f"its keys are {list(templates)}"
        )
    array = np.stack([np.asarray(templates[key], dtype=float) for key in keys])
    finite = bool(np.all(np.isfinite(array)))
    record: dict[str, Any] = {
        "status": "success" if finite else "failure",
        "failure_type": None if finite else "NonfiniteTemplate",
        "failure_message": (
            None if finite else f"{int(np.sum(~np.isfinite(array)))} nonfinite values"
        ),
        "seconds": seconds,
        "warnings": _warning_texts(caught),
        "flags": failure_flags(inner),
        "inner": inner,
    }
    if record["warnings"]:
        record["flags"] = record["flags"] + ["warning"]
    value = inner.get("opt_val") if inner is not None else None
    if value is not None:
        tuning = np.asarray(cast(ArrayLike, value), dtype=float)
        if tuning.ndim > 0:
            record["kernel"] = tuning
        else:
            record["lambda"] = float(tuning)
    if inner is not None and "noise_var" in inner:
        record["noise_var"] = inner["noise_var"]
    return record, array


# ---------------------------------------------------------------------------
# SURE(kernel)'s BFGS iteration limit (decision D2: ``core`` unchanged)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def kernel_bfgs_maxiter(maxiter: int) -> Iterator[dict[str, int]]:
    """Raise SURE(kernel)'s BFGS iteration limit inside the context.

    Only ``core._optimize_kernel``'s usual ``scipy.optimize.minimize`` call gets
    ``options["maxiter"] = maxiter``; others pass through. Yields counts of
    patched, passed-through and unexpected (unpatched; the fit is invalid) calls.
    The original is restored on exit, also after an exception.
    """
    if isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter < 1:
        raise ValueError(f"maxiter must be a positive int, got {maxiter!r}")
    target = core._optimize_kernel.__code__  # noqa: SLF001
    optimize = scipy.optimize
    real: Callable[..., Any] = optimize.minimize
    calls = {"patched": 0, "passed_through": 0, "unexpected": 0}

    def wrapper(fun: Callable[..., Any], x0: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if sys._getframe(1).f_code is not target:  # noqa: SLF001
            calls["passed_through"] += 1
            return real(fun, x0, *args, **kwargs)
        if args or kwargs != {
            "jac": True,
            "method": None,
            "options": {"maxiter": 50, "disp": 0},
        }:
            calls["unexpected"] += 1
            return real(fun, x0, *args, **kwargs)
        calls["patched"] += 1
        options = {**kwargs["options"], "maxiter": maxiter}
        return real(fun, x0, *args, **{**kwargs, "options": options})

    optimize.minimize = wrapper  # pyrefly: ignore[missing-attribute]
    try:
        yield calls
    finally:
        optimize.minimize = real  # pyrefly: ignore[missing-attribute]


# ---------------------------------------------------------------------------
# Simulation generators
# ---------------------------------------------------------------------------


def make_real_templates(
    T: int = protocol.SIM_T, C: int = protocol.SIM_C
) -> dict[str, NDArray]:
    """The true templates: Ricker widths 15 and 10 times a Hanning profile."""
    return {
        "Template A": np.outer(core.ricker(T, 15), np.roll(np.hanning(C), 3)),
        "Template B": np.outer(core.ricker(T, 10), np.hanning(C)),
    }


def noise_sweep(
    templates: dict[str, NDArray], repetitions: int
) -> Iterator[tuple[int, int, tuple[NDArray, NDArray, NDArray]]]:
    """Noise sweep: one continuous RandomState(42) across levels and repetitions."""
    rs = np.random.RandomState(protocol.SIM_SEED)
    for ni, noise in enumerate(protocol.NOISE_SWEEP):
        for kk in range(repetitions):
            yield (
                ni,
                kk,
                core.generate_data(
                    templates,
                    I=protocol.NOISE_SWEEP_I,
                    noise=noise,
                    mean_interval=protocol.NOISE_SWEEP_MEAN_INTERVAL,
                    rs=rs,
                ),
            )


def nevents_case(
    templates: dict[str, NDArray], n_events: int, kk: int
) -> tuple[NDArray, NDArray, NDArray]:
    """Event-count generator: a fresh RandomState(42 + repetition) per case."""
    rs = np.random.RandomState(protocol.SIM_SEED + kk)
    noise, interval = protocol.NEVENTS_NOISE, protocol.NEVENTS_MEAN_INTERVAL
    return core.generate_data(
        templates, I=n_events, noise=noise, mean_interval=interval, rs=rs
    )


def rb_selection(rs: np.random.RandomState, labels: NDArray, k: int) -> NDArray:
    """K events per class, classes in np.unique order, returned sorted."""
    idx_sel: list[int] = []
    for label in np.unique(labels):
        label_idx = np.where(labels == label)[0]
        idx_sel.extend(rs.choice(label_idx, size=k, replace=False))
    return np.array(sorted(idx_sel), dtype=int)


# ---------------------------------------------------------------------------
# L-curve verticality
# ---------------------------------------------------------------------------


def lcurve_verticality(
    path: PenaltyPath,
    X: NDArray,
    D: NDArray | scipy.sparse.spmatrix,
) -> float:
    """Largest angle (degrees) of the L-curve from vertical, exactly arctan(mu eta / rho).

    For squared norms d(rho)/d(mu) = -mu d(eta)/d(mu), so on equally scaled log
    axes the angle from vertical is arctan(mu eta / rho); taken over 40 points per
    decade from two decades below to two above the path's lambda envelope.
    """
    energy = np.sum(path.rotated**2, axis=1)
    r0 = float(np.sum((np.asarray(X, dtype=float) - np.asarray(D @ path.ols)) ** 2))
    lower, upper = path.log10_envelope()
    lam = np.power(10.0, np.arange((lower - 2) * 40, (upper + 2) * 40 + 1) / 40.0)
    weights, shrink = core._shrinkage(path.values, lam)  # noqa: SLF001
    x = np.log(r0 + shrink**2 @ energy)
    y = np.log((path.values * weights**2) @ energy)
    return float(np.max(np.degrees(np.arctan(lam * np.exp(y - x)))))


def lcurve_deviation(calls: Sequence[SelectorCall]) -> dict[str, Any]:
    """The verticality of the selection's L-curve, or None with the reason it is
    undefined (counted as missing in the facts)."""
    if not calls:
        return {"lcurve_deviation_deg": None} | {
            "lcurve_deviation_missing": "no L-curve selection call to measure"
        }
    with np.errstate(all="ignore"):
        value = lcurve_verticality(*calls[0])
    if not math.isfinite(value):
        return {"lcurve_deviation_deg": None} | {
            "lcurve_deviation_missing": f"verticality is {value} (degenerate curve)"
        }
    return {"lcurve_deviation_deg": value}


# ---------------------------------------------------------------------------
# Timing inputs and the stage timer
# ---------------------------------------------------------------------------


def timing_templates(L: int, M: int, C: int) -> dict[str, NDArray]:
    """OQ-B15: Ricker widths 15 and 10, cycled for M > 2, times a Hanning profile."""
    widths = (15, 10)
    return {
        f"c{m}": np.outer(core.ricker(L, widths[m % 2]), np.hanning(C))
        for m in range(M)
    }


def timing_case(L: int, M: int, C: int) -> tuple[NDArray, NDArray, NDArray]:
    """One simulated recording of the timing grid (``protocol.TIMING_SIM``)."""
    decision = protocol.TIMING_SIM
    return core.generate_data(
        timing_templates(L, M, C),
        I=decision["I"],
        noise=decision["noise"],
        mean_interval=decision["mean_interval"],
        rs=np.random.RandomState(decision["seed"]),
    )


@dataclass
class _Frame:
    """An active timed call: stage, start, and time of the timed calls nested in it."""

    stage: str
    start: float
    nested: float = 0.0


class StageTimer:
    """Exclusive wall time per stage: each wrapped call's time minus the time of
    wrapped calls nested in it.

    A ``top_level_only`` function is timed only when no stage is active (else its
    time stays with the enclosing stage); an ``inside`` stage is timed only when
    called directly from that stage.
    """

    def __init__(self) -> None:
        self.exclusive: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.stack: list[_Frame] = []

    def wrap(
        self,
        stage: str,
        fn: Callable[P, T],
        top_level_only: bool = False,
        inside: str | None = None,
    ) -> Callable[P, T]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if (top_level_only and self.stack) or (
                inside is not None
                and (not self.stack or self.stack[-1].stage != inside)
            ):
                return fn(*args, **kwargs)
            frame = _Frame(stage, time.perf_counter())
            self.stack.append(frame)
            try:
                return fn(*args, **kwargs)
            finally:
                self.stack.pop()
                elapsed = time.perf_counter() - frame.start
                self.exclusive[stage] = (
                    self.exclusive.get(stage, 0.0) + elapsed - frame.nested
                )
                self.calls[stage] = self.calls.get(stage, 0) + 1
                if self.stack:
                    self.stack[-1].nested += elapsed

        return wrapper


# R2.9 rules (the post hoc analyses of run_r29.py)
# ---------------------------------------------------------------------------

R29_LC = "L-curve"
R29_SHRINK = "Average + shrinkage"
R29_METHODS: list[str] = [*protocol.ESTIMATORS, R29_SHRINK, protocol.POPULATION]
R29_PI = 1 / 6  # targets are 96 of the 576 events of every block
R29_KAPPA: float = math.log(R29_PI / (1 - R29_PI))
R29_H2: float = -(R29_PI * math.log2(R29_PI) + (1 - R29_PI) * math.log2(1 - R29_PI))
R29_LN2: float = math.log(2.0)
R29_PRIORS: dict[str, float] = {"1/2": 0.5, "1/6": 1 / 6, "1/36": 1 / 36}
R29_TAUS: tuple[float, ...] = (0.8, 0.9, 0.95)
R29_NSEQ = 8
R29_MEAN_KEYS: list[str] = [
    "auc",
    "logloss_nats",
    "IG_bits",
    "BSS",
    "gain_offset_bits",
    "gain_scale_bits",
    "gain_joint_bits",
    "IG_recal_joint_bits",
    "fail_offset",
    "fail_scale",
    "fail_joint",
    "rev_scale",
    "rev_joint",
] + [f"{k}_{p}" for p in R29_PRIORS for k in ("tpr", "tnr", "acc", "best", "regret")]
R29_MEDIAN_KEYS: Sequence[str] = [
    "log_rho_amp",
    "log_rho_var",
    "log_scale",
    "abs_mid_err",
    "mid_err",
    "b_scale",
    "b_joint",
]
R29_REFS_PAIRED: tuple[tuple[str, str], ...] = (
    ("SURE(kernel)", "vs_SK"),
    ("Average", "vs_Avg"),
)
R29_METRICS: tuple[str, ...] = ("tot2", "ec2", "ed2", "cos", "amp", "paper")
R29_SPLIT_METHODS: tuple[str, ...] = (
    *("Average", "EmpBayes", "GCV", R29_LC, "SURE(lambda)", "Molina filtered DTW"),
)
# Plain DTW nearly duplicates filtered DTW, so the cross-estimator statistics leave it out.
R29_RANK_EST: list[str] = [m for m in protocol.ESTIMATORS if m != protocol.PLAIN]

R29PerSubject = dict[tuple[int | str, str], dict[int, dict[str, float]]]


def check_r29_failures(records: Mapping[str, Mapping[str, Any]], where: str) -> None:
    """Only L-curve fits may have failed (the R2.9 failure rules)."""
    for name, rec in records.items():
        if rec["status"] != "success" and name != R29_LC:
            raise RuntimeError(
                f"{where}: {rec['key']} has status {rec['status']}; only L-curve "
                "failures are covered by the R2.9 failure rules"
            )


def ll_terms(eta: NDArray, y: NDArray) -> NDArray:
    return np.where(y == 1, np.logaddexp(0.0, -eta), np.logaddexp(0.0, eta))


def event_metrics(L: NDArray, y: NDArray) -> dict[str, float]:
    """Log loss, information gain over the 1/6 prior (bits) and Brier skill of LLRs L."""
    eta = R29_KAPPA + L
    ll = float(ll_terms(eta, y).mean())
    brier = float(np.mean((scipy.special.expit(eta) - y) ** 2))
    return {
        "logloss_nats": ll,
        "IG_bits": R29_H2 - ll / R29_LN2,
        "BSS": 1 - brier / (R29_PI * (1 - R29_PI)),
    }


def newton(
    X: NDArray, off: NDArray, y: NDArray, theta: NDArray
) -> tuple[NDArray, bool]:
    """Damped Newton logistic regression with offset; (theta, converged)."""

    def loglik(th: NDArray) -> float:
        eta = off + X @ th
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)))

    cur = loglik(theta)
    for _ in range(200):
        p = scipy.special.expit(off + X @ theta)
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        t, new = 1.0, theta + step
        val = loglik(new)
        while (
            not (np.isfinite(val) and val >= cur - 1e-12 * max(1.0, abs(cur)))
            and t > 2.0**-60
        ):
            t /= 2
            new = theta + t * step
            val = loglik(new)
        if not np.isfinite(val):
            return theta, False
        change = float(np.max(np.abs(new - theta)))
        theta, cur = new, val
        if change < 1e-10:
            return theta, True
    return theta, False


def fit_variant(
    variant: str, Lf: NDArray, yf: NDArray
) -> tuple[float, float, bool, int, float]:
    """One calibration half: (alpha, b used, converged, reversal, slope diagnostic).

    b >= 0 by the KKT check at b = 0 (concave log-likelihood): the optimum is at
    b = 0 iff the b-derivative there is <= 0 (< 0: an estimated reversal).
    Separation diverges, so it is checked; constant scores give a NaN diagnostic.
    """
    n = len(Lf)
    if variant == "offset":
        th, conv = newton(np.ones((n, 1)), R29_KAPPA + Lf, yf, np.zeros(1))
        return float(th[0]), 1.0, conv, 0, 1.0
    if variant == "scale":
        a0, conv0 = 0.0, True
    else:
        th0, conv0 = newton(np.ones((n, 1)), np.full(n, R29_KAPPA), yf, np.zeros(1))
        a0 = float(th0[0])
    if np.all(Lf == Lf[0]):
        return a0, 0.0, conv0, 0, math.nan
    dldb = float(np.sum(Lf * (yf - scipy.special.expit(R29_KAPPA + a0))))
    if dldb <= 0:
        return a0, 0.0, conv0, int(dldb < 0), 0.0
    if variant == "scale":
        sep = bool(np.all(Lf[yf == 1] > 0) and np.all(Lf[yf == 0] < 0))
    else:
        sep = bool(Lf[yf == 0].max() < Lf[yf == 1].min())
    if sep:
        return a0, 0.0, False, 0, math.nan
    if variant == "scale":
        th, conv = newton(Lf[:, None], np.full(n, R29_KAPPA), yf, np.ones(1))
        a, b = 0.0, float(th[0])
    else:
        X = np.column_stack([np.ones(n), Lf])
        th, conv = newton(X, np.full(n, R29_KAPPA), yf, np.array([0.0, 1.0]))
        a, b = float(th[0]), float(th[1])
    if b < 0:  # an invalid optimum (numerically): a counted calibration failure
        return a0, 0.0, False, 0, math.nan
    return a, b, conv, 0, (b if conv else math.nan)


def decisions(L: NDArray, y: NDArray) -> dict[str, float]:
    """Accuracy of 'target if L > log((1 - p) / p)' per prior p, the best threshold's, regret."""
    pos = y == 1
    P, N = int(pos.sum()), int((~pos).sum())
    order = np.argsort(-L, kind="mergesort")
    Ls, ys = L[order], y[order]
    last = np.r_[Ls[1:] != Ls[:-1], True]
    tp_c = np.r_[0, np.cumsum(ys == 1)[last]]
    fp_c = np.r_[0, np.cumsum(ys == 0)[last]]
    out: dict[str, float] = {}
    for name, prior in R29_PRIORS.items():
        pred = L > math.log((1 - prior) / prior)
        tpr, tnr = float(np.mean(pred[pos])), float(np.mean(~pred[~pos]))
        acc = prior * tpr + (1 - prior) * tnr
        best = float(np.max(prior * tp_c / P + (1 - prior) * (1 - fp_c / N)))
        out |= {f"tpr_{name}": tpr, f"tnr_{name}": tnr, f"acc_{name}": acc}
        out |= {f"best_{name}": best, f"regret_{name}": best - acc}
    return out


def speller_character(
    L: NDArray, mp: Mapping[str, NDArray], c: int
) -> tuple[NDArray, NDArray, NDArray]:
    """(log posterior of the true character, max posterior, correct share) per sequence."""
    sl = slice(c * 96, (c + 1) * 96)
    S = np.zeros((R29_NSEQ, 12))
    np.add.at(S, (mp["seq"][sl], mp["codes"][sl] - 1), L[sl])
    cum = np.cumsum(S, axis=0)
    A = (cum[:, :6, None] + cum[:, None, 6:]).reshape(R29_NSEQ, 36)
    logpost = A - scipy.special.logsumexp(A, axis=1, keepdims=True)
    true = (mp["trow"][c] - 1) * 6 + (mp["tcol"][c] - 7)
    mx = A.max(axis=1)
    corr = (A[:, true] == mx) / (A == mx[:, None]).sum(axis=1)
    return logpost[:, true], np.exp(logpost.max(axis=1)), corr


def speller(L: NDArray, mp: Mapping[str, NDArray]) -> dict[str, Any]:
    """Offline 6 x 6 row/column speller of one block's six characters."""
    ll, pmax_s, acc = np.zeros(R29_NSEQ), np.zeros(R29_NSEQ), np.zeros(R29_NSEQ)
    # per tau: n_trig, c_trig, p_trig, n_forced, c_forced, p_forced, sequences used
    stop: dict[str, list[Any]] = {
        str(t): [0, 0.0, 0.0, 0, 0.0, 0.0, 0] for t in R29_TAUS
    }
    for c in range(6):
        lp_true, pm, corr = speller_character(L, mp, c)
        ll += -lp_true
        pmax_s += pm
        acc += corr
        for t in R29_TAUS:
            d = stop[str(t)]
            hit = np.flatnonzero(pm >= t)
            if len(hit):
                n = int(hit[0])
                d[0], d[1], d[2] = d[0] + 1, d[1] + corr[n], d[2] + pm[n]
            else:
                n = R29_NSEQ - 1
                d[3], d[4], d[5] = d[3] + 1, d[4] + corr[n], d[5] + pm[n]
            d[6] += n + 1
    out = {"ll": ll.tolist(), "pmax": pmax_s.tolist(), "acc": acc.tolist()}
    return out | {"stop": stop}


def boot_dict(values: NDArray, n: int = 10000, seed: int = 42) -> dict[str, Any]:
    """Mean of the finite values and its percentile bootstrap interval."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": math.nan, "ci": [math.nan, math.nan], "n": 0}
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    ci = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
    return {"mean": float(v.mean()), "ci": ci, "n": len(v)}


def wolpaw(p: float, n: int = 36) -> float:
    if not np.isfinite(p):
        return math.nan
    if p <= 1 / n:
        return 0.0
    if p >= 1:
        return math.log2(n)
    return math.log2(n) + p * math.log2(p) + (1 - p) * math.log2((1 - p) / (n - 1))


def _level_metrics(recs: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in R29_MEAN_KEYS:
        vals = np.array([r[k] for r in recs], dtype=float)
        out[k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else math.nan
    for k in R29_MEDIAN_KEYS:
        vals = np.array([r.get(k, math.nan) for r in recs], dtype=float)
        out[k] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else math.nan
    out["frac_delta_nonpos"] = (
        float(np.mean([r["delta_ho"] <= 0 for r in recs if np.isfinite(r["delta_ho"])]))
        if any(np.isfinite(r["delta_ho"]) for r in recs)
        else math.nan
    )
    if "b_shrink" in recs[0]:
        out["b_shrink"] = float(np.mean([r["b_shrink"] for r in recs]))
        out["frac_b_zero"] = float(np.mean([r["b_zero"] for r in recs]))
    return out


def _speller_metrics(recs: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    nchar = 6 * len(recs)
    ll = np.sum([r["sp"]["ll"] for r in recs], axis=0) / nchar
    acc = np.sum([r["sp"]["acc"] for r in recs], axis=0) / nchar
    pm = np.sum([r["sp"]["pmax"] for r in recs], axis=0) / nchar
    for n in range(R29_NSEQ):
        out[f"sp_ll_n{n + 1}"] = float(ll[n])
        out[f"sp_gain_n{n + 1}"] = math.log2(36) - float(ll[n]) / R29_LN2
        out[f"sp_acc_n{n + 1}"] = float(acc[n])
        out[f"sp_pmax_n{n + 1}"] = float(pm[n])
    out["sp_wolpaw_n8"] = wolpaw(float(acc[-1]))
    for t in R29_TAUS:
        d = np.sum([r["sp"]["stop"][str(t)] for r in recs], axis=0)
        acc_stop = (d[1] + d[4]) / nchar
        out[f"sp_frac_trig_{t}"] = d[0] / nchar
        out[f"sp_acc_trig_{t}"] = d[1] / d[0] if d[0] else math.nan
        out[f"sp_pmax_trig_{t}"] = d[2] / d[0] if d[0] else math.nan
        out[f"sp_acc_forced_{t}"] = d[4] / d[3] if d[3] else math.nan
        out[f"sp_pmax_forced_{t}"] = d[5] / d[3] if d[3] else math.nan
        out[f"sp_mean_seq_{t}"] = d[6] / nchar
        out[f"sp_acc_stop_{t}"] = acc_stop
        out[f"sp_wolpaw_stop_{t}"] = wolpaw(acc_stop)
        out[f"sp_bits_per_seq_{t}"] = wolpaw(acc_stop) / (d[6] / nchar)
    return out


def subject_metrics(recs: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """One subject's metrics over its records; empty (no valid fit) gives {}."""
    if not recs:
        return {}
    out = _level_metrics(recs)
    out.update(_speller_metrics(recs))
    if "ov" in recs[0]:
        for k in R29_MEAN_KEYS[1:8] + [
            f"{a}_{p}" for p in R29_PRIORS for a in ("acc", "regret")
        ]:
            vals = np.array([r["ov"][k] for r in recs if "ov" in r], dtype=float)
            out[f"ov_{k}"] = (
                float(np.nanmean(vals)) if np.isfinite(vals).any() else math.nan
            )
    return out


def r29_per_subject(
    records: Sequence[dict[str, Any]],
) -> tuple[R29PerSubject, R29PerSubject]:
    """Subject metrics per (K, method) over the valid fits (R1, R2); SURE(kernel) and
    Average also on the L-curve's valid cells (R3)."""
    by: dict[tuple[int | str, str, int], list[dict[str, Any]]] = {}
    for r in records:
        if "failed" not in r:
            by.setdefault((r["K"], r["method"], r["subject"]), []).append(r)
    per: R29PerSubject = {}
    for (K, m, s), recs in by.items():
        per.setdefault((K, m), {})[s] = subject_metrics(recs)
    per_lc: R29PerSubject = {}
    for (K, m, s), recs in by.items():
        if m in {"SURE(kernel)", "Average"} and K != "pop":
            cells = {(r["fold"], r["rep"]) for r in by.get((K, R29_LC, s), [])}
            per_lc.setdefault((K, m), {})[s] = subject_metrics(
                [r for r in recs if (r["fold"], r["rep"]) in cells]
            )
    return per, per_lc


def r29_metric_entry(
    tab: Mapping[str, Mapping[int, Mapping[str, float]]],
    lc_refs: Mapping[str, Mapping[int, Mapping[str, float]]],
    metric: str,
    subj: Sequence[int],
) -> dict[str, Any]:
    """Per method: bootstrap of the subject values and of the paired differences."""

    def column(src: Mapping[int, Mapping[str, float]]) -> NDArray:
        return np.array([src.get(s, {}).get(metric, math.nan) for s in subj], float)

    ent: dict[str, Any] = {}
    for m in R29_METHODS:
        vals = column(tab[m])
        e = boot_dict(vals)
        e["per_subject"] = vals.tolist()
        for ref, name in R29_REFS_PAIRED:
            src = lc_refs[ref] if m == R29_LC else tab[ref]
            if m != ref:
                e[name] = boot_dict(vals - column(src))
        ent[m] = e
    return ent


def template_parts(T: NDArray) -> tuple[NDArray, NDArray]:
    """Common part and difference; T is (2, L, C) with index 0 = NonTarget, 1 = Target."""
    return 0.5 * (T[0] + T[1]), T[1] - T[0]


def cosine(a: NDArray, b: NDArray) -> float:
    return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def fit_metrics(T: NDArray, R: NDArray) -> list[float]:
    """R29_METRICS of T against R (tot2 = 2 ec2 + 0.5 ed2)."""
    dT = T - R
    tot2 = float(np.sum(dT**2))
    C, D = template_parts(T)
    CR, DR = template_parts(R)
    ec2 = float(np.sum((C - CR) ** 2))
    ed2 = float(np.sum((D - DR) ** 2))
    paper = 0.5 * (np.linalg.norm(dT[0]) + np.linalg.norm(dT[1]))
    amp = float(np.linalg.norm(D) / np.linalg.norm(DR))
    return [tot2, ec2, ed2, cosine(D, DR), amp, float(paper)]


def amp_direction_split(
    A: NDArray, dref2: Sequence[float], k_list: Sequence[int]
) -> dict[str, Any]:
    """||D - D_R||^2 = |D_R|^2 [(1 - c+^2) + (a^2 - 2ac + c+^2)], a = |D|/|D_R|,
    c = cos(D, D_R), c+ = max(c, 0): direction (the infimum under positive
    rescaling) plus scale; gaps to SURE(kernel) in 0.5 ||dD||^2 units, OLS reference.
    A: (k, s, r, estimator, reference, R29_METRICS), NaN at excluded fits.
    """
    mi = {m: i for i, m in enumerate(protocol.ESTIMATORS)}
    ri_, nan = protocol.REFERENCES.index("OLS"), math.nan
    dr2 = np.array(dref2)[None, :, None, None]  # broadcast over (k, s, r, m)
    c = A[:, :, :, :, ri_, 3]
    a = A[:, :, :, :, ri_, 4]
    cp = np.maximum(c, 0.0)
    direction = dr2 * (1 - cp**2)
    amplitude = dr2 * (a**2 - 2 * a * c + cp**2)
    base = mi[protocol.ANCHOR]
    out: dict[str, Any] = {"norm_D_ref_uV": np.sqrt(dr2.ravel()).tolist()}
    if k_list and k_list[0] == 5:
        out["negative_cos_fits_K05"] = {
            m: int(np.sum(c[0, :, :, mi[m]] < 0)) for m in protocol.ESTIMATORS
        }
    for ki, K in enumerate(k_list):
        for m in R29_SPLIT_METHODS:
            dd = direction[ki, :, :, mi[m]] - direction[ki, :, :, base]
            da = amplitude[ki, :, :, mi[m]] - amplitude[ki, :, :, base]
            # R3: the L-curve over its valid repetitions (NaN at its excluded fits).
            gd = 0.5 * (np.nanmean(dd, axis=1) if m == R29_LC else np.mean(dd, axis=1))
            ga = 0.5 * (np.nanmean(da, axis=1) if m == R29_LC else np.mean(da, axis=1))
            ok = np.isfinite(gd) & np.isfinite(ga)  # subjects with a valid fit
            out[f"K{K:02d}|{m}"] = {
                "gap_D_direction_part_mean": float(gd[ok].mean()) if ok.any() else nan,
                "gap_D_amplitude_part_mean": float(ga[ok].mean()) if ok.any() else nan,
                "n_subjects": int(ok.sum()),
                "subjects_direction_part_positive": int(np.sum(gd > 0)),
                "subjects_amplitude_part_positive": int(np.sum(ga > 0)),
                "per_subject_direction": gd.tolist(),
                "per_subject_amplitude": ga.tolist(),
            }
    return out


def spearman_r5b(
    b: NDArray, idx: list[int], col: int, subj_mean: NDArray
) -> list[float]:
    """R5(b): subject means over the cells common to the estimator set.
    b: (s, f, r, estimator, field), field 0 = AUC."""
    rho = []
    for si in range(len(b)):
        x, y = b[si][:, :, idx, col], b[si][:, :, idx, 0]
        pair = np.isfinite(x) & np.isfinite(y)  # (f, r, m)
        cm = subj_mean[si] if pair.all() else b[si][pair.all(axis=2)].mean(axis=0)
        rho.append(float(scipy.stats.spearmanr(cm[idx, col], cm[idx, 0])[0]))
    return rho


def pearson_r4a(b: NDArray, idx: list[int], col: int) -> tuple[list[float], list[int]]:
    """R4(a): centre over the estimators valid in each cell (one joint mask for both)."""
    within, nobs = [], []
    for si in range(len(b)):
        x = b[si][:, :, idx, col]
        y = b[si][:, :, idx, 0]
        ok = np.isfinite(x) & np.isfinite(y)
        if not ok.all():
            x, y = np.where(ok, x, np.nan), np.where(ok, y, np.nan)
            x = (x - np.nanmean(x, axis=2, keepdims=True)).ravel()
            y = (y - np.nanmean(y, axis=2, keepdims=True)).ravel()
            keep = ok.ravel()
            x, y = x[keep], y[keep]
        else:
            x = (x - x.mean(axis=2, keepdims=True)).ravel()
            y = (y - y.mean(axis=2, keepdims=True)).ravel()
        nobs.append(int(x.size))
        within.append(float(np.corrcoef(x, y)[0, 1]))
    return within, nobs
