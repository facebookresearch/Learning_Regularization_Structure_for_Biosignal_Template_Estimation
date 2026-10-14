# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Held-out recording-block evaluation of P300 templates (BNCI2014-009).

Each subject's stored blocks are held out in turn. Templates are fitted on K
selected events per class from the other blocks, joined with ``block_starts``,
and every eligible held-out window is scored against them. Preprocessing is
done per block by ``data_loaders_p300.load_bnci2014_009_blocks``; this module
needs no MNE or MOABB.
"""

from __future__ import annotations

import copy
import importlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np
import scipy
from numpy.typing import NDArray

if TYPE_CHECKING:
    import core
    import dtw_average
    import protocol
else:
    _PREFIX = f"{__package__}." if __package__ else ""
    core = importlib.import_module(f"{_PREFIX}core")
    dtw_average = importlib.import_module(f"{_PREFIX}dtw_average")
    protocol = importlib.import_module(f"{_PREFIX}protocol")

HALF_WIN = 51
# Half support of the 845-tap 1-20 Hz FIR at 256 Hz, in 128 Hz samples.
GUARD = 211
K = 5
REPETITIONS = 20
SEED = 42
LABELS: dict[str, int] = {"NonTarget": 0, "Target": 1}

# One class's epochs (K, 2 * half_win, C) -> its template (2 * half_win, C).
Comparator = Callable[[NDArray], NDArray]


@dataclass(frozen=True)
class RecordingBlock:
    """One separately preprocessed recording block at the analysis rate.

    ``onsets`` index ``data`` rows; ``native_onsets`` are the same events at
    the acquisition rate. ``labels`` are 1 for Target and 0 for NonTarget.
    """

    subject: int
    block: int
    data: NDArray
    onsets: NDArray
    native_onsets: NDArray
    labels: NDArray
    channels: tuple[str, ...]

    def __post_init__(self) -> None:
        where = f"subject {self.subject}, block {self.block}"
        if self.data.ndim != 2:
            raise ValueError(f"{where}: data must be 2-D, got {self.data.shape}")
        if len(self.channels) != self.data.shape[1]:
            raise ValueError(
                f"{where}: {len(self.channels)} channel names for "
                f"{self.data.shape[1]} data columns"
            )
        for name in ("onsets", "native_onsets", "labels"):
            values = getattr(self, name)
            if values.shape != (len(self.onsets),):
                raise ValueError(
                    f"{where}: {name} has shape {values.shape}, expected "
                    f"({len(self.onsets)},)"
                )
            if not np.issubdtype(values.dtype, np.integer):
                raise ValueError(
                    f"{where}: {name} must be integers, got {values.dtype}"
                )
        if not np.all(np.isin(self.labels, (0, 1))):
            raise ValueError(f"{where}: labels must be 0 or 1, got {set(self.labels)}")

    def eligible(self, half_win: int = HALF_WIN, guard: int = GUARD) -> NDArray:
        return eligible_events(self.onsets, len(self.data), half_win, guard)


@dataclass(frozen=True)
class FoldFit:
    """One method's fit for one subject, held-out block, and repetition."""

    subject: int
    held_out: int
    repetition: int
    method: str
    selection: NDArray
    templates: Mapping[object, NDArray]
    inner: Mapping[str, object] | None
    auc: float

    @property
    def cell(self) -> tuple[int, int, int, str, float]:
        return self.subject, self.held_out, self.repetition, self.method, self.auc


def reference_methods() -> dict[str, dict[str, Any]]:
    """The P300 method settings of the reference benchmark (a fresh copy)."""
    return copy.deepcopy(protocol.METHODS_AR1COL)


def comparator_methods() -> dict[str, Comparator]:
    """The paper's comparators, fitted beside ``reference_methods()``.

    Molina et al. (2024)'s filtered DTW average with its frozen settings.
    """
    return {"Molina": dtw_average.molina_template}


def event_labels(codes: NDArray, event_id: Mapping[str, int]) -> NDArray:
    """Map event codes to Target = 1 and NonTarget = 0 by their names.

    The names must be exactly Target and NonTarget; numeric codes carry no
    meaning, so there is no largest-code fallback.
    """
    names = {str(name): int(code) for name, code in event_id.items()}
    unknown = sorted(set(names) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown event names {unknown} in {names}")
    missing = sorted(set(LABELS) - set(names))
    if missing:
        raise ValueError(f"Missing event names {missing} in {names}")
    if names["Target"] == names["NonTarget"]:
        raise ValueError(f"Target and NonTarget share one code in {names}")
    lookup = {names[name]: label for name, label in LABELS.items()}
    values = np.asarray(codes).tolist()
    unnamed = sorted(set(values) - set(lookup))
    if unnamed:
        raise ValueError(f"Event codes {unnamed} have no name in {names}")
    return np.array([lookup[code] for code in values], dtype=int)


def eligible_events(
    onsets: NDArray, n_samples: int, half_win: int = HALF_WIN, guard: int = GUARD
) -> NDArray:
    """Complete windows at least ``guard`` samples from both block edges."""
    # Compare without onset arithmetic, so unsigned and narrow integers cannot wrap.
    onsets = np.asarray(onsets)
    return (onsets >= half_win + guard) & (onsets <= n_samples - half_win - guard)


def training_blocks(
    blocks: Sequence[RecordingBlock], held_out: int
) -> list[RecordingBlock]:
    """The other blocks of one subject, in block order."""
    subjects = {block.subject for block in blocks}
    if len(subjects) != 1:
        raise ValueError(f"Blocks must come from one subject, got {sorted(subjects)}")
    ids = [block.block for block in blocks]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Block ids must be unique, got {ids}")
    if held_out not in ids:
        raise ValueError(f"Held-out block {held_out} is not among {ids}")
    training = sorted(
        (block for block in blocks if block.block != held_out),
        key=lambda block: block.block,
    )
    if not training:
        raise ValueError(f"No training block remains after holding out {held_out}")
    return training


def select_events(
    training: Sequence[RecordingBlock],
    k: int = K,
    repetitions: int = REPETITIONS,
    seed: int = SEED,
    half_win: int = HALF_WIN,
    guard: int = GUARD,
) -> list[NDArray]:
    """Draw K eligible events per class from all training blocks together.

    A fresh ``RandomState(seed)`` serves every repetition of one fold; class 0
    is drawn before class 1, without replacement, from candidates ordered by
    block and event. Each selection is an int array of (block, event index)
    rows, class 0 first.
    """
    pool = [
        (block.block, int(event), int(block.labels[event]))
        for block in training
        for event in np.flatnonzero(block.eligible(half_win, guard))
    ]
    candidates = {
        label: np.array([(b, e) for b, e, lab in pool if lab == label], dtype=int)
        for label in (0, 1)
    }
    for label, rows in candidates.items():
        if len(rows) < k:
            raise ValueError(
                f"Only {len(rows)} eligible class-{label} training events for K={k}"
            )
    rng = np.random.RandomState(seed)
    selections = []
    for _ in range(repetitions):
        picks = [
            candidates[label][rng.choice(len(candidates[label]), k, replace=False)]
            for label in (0, 1)
        ]
        selections.append(np.vstack(picks))
    return selections


def _join_selection(
    training: Sequence[RecordingBlock], selection: NDArray, half_win: int
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Joined training data, selected onsets and labels, and block starts."""
    lengths = [len(block.data) for block in training]
    starts = np.cumsum([0] + lengths[:-1])
    by_id = {block.block: (block, start) for block, start in zip(training, starts)}
    unknown = sorted({int(b) for b in selection[:, 0]} - set(by_id))
    if unknown:
        raise ValueError(f"Selected blocks {unknown} are not training blocks")
    for b, e in selection:
        block = by_id[b][0]
        if block.onsets[e] < half_win:
            raise ValueError(f"Window of block {b} event {e} starts before its block")
        if block.onsets[e] > len(block.data) - half_win:
            raise ValueError(f"Window of block {b} event {e} ends after its block")
    ix = np.array([by_id[b][1] + by_id[b][0].onsets[e] for b, e in selection])
    lb = np.array([by_id[b][0].labels[e] for b, e in selection])
    data = np.vstack([block.data for block in training])
    return data, ix, lb, starts


def fit_templates(
    training: Sequence[RecordingBlock],
    selection: NDArray,
    method_kwargs: Mapping[str, Any],
    half_win: int = HALF_WIN,
) -> tuple[dict[object, NDArray], dict[str, object] | None]:
    """Fit one method on the training blocks joined as separate domains."""
    data, ix, lb, starts = _join_selection(training, selection, half_win)
    return core.get_events_templates(
        data, ix, lb, half_win=half_win, block_starts=starts, **method_kwargs
    )


def fit_comparator(
    training: Sequence[RecordingBlock],
    selection: NDArray,
    comparator: Comparator,
    half_win: int = HALF_WIN,
) -> dict[object, NDArray]:
    """Apply a comparator per class to exactly the epochs Average averages.

    The epochs are the selected training windows ``[onset - half_win, onset +
    half_win)``, cut as ``method='average'`` cuts them and kept in selection
    order.
    """
    data, ix, lb, _ = _join_selection(training, selection, half_win)
    epochs = core.epoch_data(data, ix, -half_win, half_win)
    return {str(label): comparator(epochs[lb == label]) for label in np.unique(lb)}


def score_events(
    block: RecordingBlock,
    templates: Mapping[object, NDArray],
    half_win: int = HALF_WIN,
    guard: int = GUARD,
) -> tuple[NDArray, NDArray]:
    """Score eligible windows: ||Z - T_nontarget||^2 - ||Z - T_target||^2."""
    shape = (2 * half_win, block.data.shape[1])
    for key in ("0", "1"):
        if key not in templates:
            raise ValueError(
                f"Template for label {key} is missing: {sorted(templates, key=str)}"
            )
        if templates[key].shape != shape:
            raise ValueError(
                f"Template {key} has shape {templates[key].shape}, expected {shape}"
            )
    events = np.flatnonzero(block.eligible(half_win, guard))
    windows = np.stack(
        [
            block.data[onset - half_win : onset + half_win]
            for onset in block.onsets[events]
        ]
    )
    scores = np.sum((windows - templates["0"]) ** 2, axis=(1, 2)) - np.sum(
        (windows - templates["1"]) ** 2, axis=(1, 2)
    )
    if not np.all(np.isfinite(scores)):
        raise ValueError(
            f"{int(np.sum(~np.isfinite(scores)))} nonfinite scores in subject "
            f"{block.subject}, block {block.block}"
        )
    return scores, block.labels[events]


def auc(scores: NDArray, labels: NDArray) -> float:
    """ROC AUC as the Mann-Whitney statistic with average ranks for ties."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels)
    if scores.shape != labels.shape:
        raise ValueError(f"{scores.shape} scores for {labels.shape} labels")
    if not np.all(np.isfinite(scores)):
        raise ValueError("AUC scores must be finite")
    if not np.all(np.isin(labels, (0, 1))):
        raise ValueError(f"AUC labels must be 0 or 1, got {set(labels.tolist())}")
    positives = int(np.sum(labels == 1))
    negatives = len(labels) - positives
    if positives == 0:
        raise ValueError("AUC needs at least one label-1 event")
    if negatives == 0:
        raise ValueError("AUC needs at least one label-0 event")
    rank_sum = float(np.sum(scipy.stats.rankdata(scores)[labels == 1]))
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def evaluate_fold(
    blocks: Sequence[RecordingBlock],
    held_out: int,
    methods: Mapping[str, Mapping[str, Any]],
    k: int = K,
    repetitions: int = REPETITIONS,
    seed: int = SEED,
    half_win: int = HALF_WIN,
    guard: int = GUARD,
    *,
    comparators: Mapping[str, Comparator] | None = None,
) -> list[FoldFit]:
    """Fit and score every method on selections shared across methods.

    *methods* maps names to ``get_events_templates`` settings, normally
    ``reference_methods()``. *comparators*, normally ``comparator_methods()``,
    are fitted by ``fit_comparator`` on the same selections, after the methods
    of each repetition.
    """
    comparators = comparators or {}
    shared = sorted(set(methods) & set(comparators))
    if shared:
        raise ValueError(f"Comparator names {shared} are also method names")
    training = training_blocks(blocks, held_out)
    test = next(block for block in blocks if block.block == held_out)
    selections = select_events(training, k, repetitions, seed, half_win, guard)
    fits = []
    for repetition, selection in enumerate(selections):
        fitted: dict[str, tuple[dict[object, NDArray], dict[str, object] | None]]
        fitted = {
            method: fit_templates(training, selection, kwargs, half_win)
            for method, kwargs in methods.items()
        }
        for name, comparator in comparators.items():
            fitted[name] = (
                fit_comparator(training, selection, comparator, half_win),
                None,
            )
        for method, (templates, inner) in fitted.items():
            scores, labels = score_events(test, templates, half_win, guard)
            fits.append(
                FoldFit(
                    test.subject,
                    held_out,
                    repetition,
                    method,
                    selection,
                    templates,
                    inner,
                    auc(scores, labels),
                )
            )
    return fits


def subject_mean_aucs(
    cells: Iterable[tuple[int, int, int, str, float]],
    subjects: Sequence[int],
    methods: Sequence[str],
    folds: Sequence[int] = (0, 1, 2),
    repetitions: int = REPETITIONS,
) -> dict[tuple[int, str], float]:
    """Average each subject's fold and repetition AUCs equally, per method.

    Every (subject, fold, repetition, method) cell must appear exactly once
    with a finite AUC, so a mean always covers the complete grid.
    """
    seen: dict[tuple[int, int, int, str], float] = {}
    for subject, fold, repetition, method, value in cells:
        key = (subject, fold, repetition, method)
        if key in seen:
            raise ValueError(f"Duplicate AUC cell {key}")
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite AUC {value} in cell {key}")
        seen[key] = value
    expected = {
        (s, f, r, m)
        for s in subjects
        for f in folds
        for r in range(repetitions)
        for m in methods
    }
    unexpected = sorted(set(seen) - expected)
    if unexpected:
        raise ValueError(
            f"{len(unexpected)} AUC cells outside the grid: {unexpected[:3]}"
        )
    missing = sorted(expected - set(seen))
    if missing:
        raise ValueError(f"{len(missing)} AUC cells are missing: {missing[:3]}")
    return {
        (s, m): float(
            np.mean([seen[s, f, r, m] for f in folds for r in range(repetitions)])
        )
        for s in subjects
        for m in methods
    }


def population_templates(
    references: Mapping[int, Mapping[object, NDArray]], subject: int
) -> dict[object, NDArray]:
    """Leave-one-subject-out population template for *subject*, per class.

    The mean, per class, sample, and channel, of the other subjects' reference
    templates. The caller supplies the complete cohort of independently fitted
    full-data OLS references, with one channel order, unit, and time grid;
    *subject*'s own entry is never read. A descriptive control: it enters no
    test family.
    """
    if subject not in references:
        raise ValueError(f"Subject {subject} has no reference templates")
    donors = sorted(s for s in references if s != subject)
    if not donors:
        raise ValueError(f"No other subject's references remain for subject {subject}")
    first = references[donors[0]]
    if not first:
        raise ValueError(f"Subject {donors[0]} has an empty reference mapping")
    keys = sorted(first, key=str)
    shape = np.shape(first[keys[0]])
    for donor in donors:
        templates = references[donor]
        if sorted(templates, key=str) != keys:
            raise ValueError(
                f"Subject {donor} has classes {sorted(templates, key=str)}, "
                f"expected {keys}"
            )
        for key in keys:
            values = np.asarray(templates[key])
            if values.ndim != 2 or values.size == 0:
                raise ValueError(
                    f"Subject {donor} class {key} template must be nonempty 2-D, "
                    f"got shape {values.shape}"
                )
            if values.shape != shape:
                raise ValueError(
                    f"Subject {donor} class {key} template has shape "
                    f"{values.shape}, expected {shape}"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Subject {donor} class {key} template is nonfinite")
    return {
        key: np.mean(
            np.stack([np.asarray(references[d][key], dtype=float) for d in donors]),
            axis=0,
        )
        for key in keys
    }


def population_cells(
    blocks: Sequence[RecordingBlock],
    templates: Mapping[object, NDArray],
    name: str = "Population",
    half_win: int = HALF_WIN,
    guard: int = GUARD,
) -> list[tuple[int, int, int, str, float]]:
    """Score every block of one subject against one fixed template.

    Gives one ``(subject, block, 0, name, auc)`` cell per block, on the windows
    the held-out evaluation scores. The template uses no data of the subject,
    so it is the same for every repetition:
    ``subject_mean_aucs(cells, subjects, [name], repetitions=1)`` averages each
    subject's block AUCs equally.
    """
    subjects = {block.subject for block in blocks}
    if len(subjects) != 1:
        raise ValueError(
            f"Population blocks must come from one subject, got {sorted(subjects)}"
        )
    ids = [block.block for block in blocks]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Population block ids must be unique, got {ids}")
    cells = []
    for block in blocks:
        scores, labels = score_events(block, templates, half_win, guard)
        cells.append((block.subject, block.block, 0, name, auc(scores, labels)))
    return cells
