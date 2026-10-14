# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Held-out block evaluation: train/test isolation, selection, and scoring."""

from __future__ import annotations

import importlib
import unittest
from collections.abc import Callable
from typing import Any, TYPE_CHECKING
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from numpy.typing import NDArray

if TYPE_CHECKING:
    import core
    import dtw_average
    import heldout_p300 as heldout
else:
    core = importlib.import_module("core")
    dtw_average = importlib.import_module("dtw_average")
    heldout = importlib.import_module("heldout_p300")

N_SAMPLES = 1200
CHANNELS = ("C1", "C2", "C3")
# Mostly identity with a long tail, so a joined filter reaches eligible windows.
FILTER = np.hanning(801) / np.hanning(801).sum() * 0.2
FILTER[400] += 1.0

RawBlock = tuple[NDArray, NDArray, NDArray]


def raw_blocks() -> list[RawBlock]:
    """Three recordings with overlapping responses and correlated noise."""
    rs = np.random.RandomState(5)
    response = np.outer(np.hanning(102), [1.0, 0.6, -0.3])
    blocks = []
    for _ in range(3):
        noise = np.cumsum(rs.normal(size=(N_SAMPLES, 3)), axis=0) * 0.05
        noise -= np.linspace(noise[0], noise[-1], N_SAMPLES)
        onsets = np.arange(270, N_SAMPLES - 270, 20)
        labels = (rs.uniform(size=len(onsets)) < 0.3).astype(int)
        data = noise + 0.2 * rs.normal(size=noise.shape)
        for onset, label in zip(onsets, labels):
            data[onset - 51 : onset + 51] += (1.0 + label) * response
        blocks.append((data, onsets, labels))
    return blocks


def filtered(data: NDArray) -> NDArray:
    return np.column_stack(
        [np.convolve(column, FILTER, mode="same") for column in data.T]
    )


def recording_block(
    index: int, data: NDArray, onsets: NDArray, labels: NDArray
) -> heldout.RecordingBlock:
    return heldout.RecordingBlock(
        subject=1,
        block=index,
        data=data,
        onsets=onsets,
        native_onsets=2 * onsets,
        labels=labels,
        channels=CHANNELS,
    )


def preprocess_separately(raw: list[RawBlock]) -> list[heldout.RecordingBlock]:
    return [
        recording_block(index, filtered(data), onsets, labels)
        for index, (data, onsets, labels) in enumerate(raw)
    ]


def preprocess_joined(raw: list[RawBlock]) -> list[heldout.RecordingBlock]:
    """Deliberately unsafe: filter the concatenated recording, then split."""
    joined = filtered(np.vstack([data for data, _, _ in raw]))
    return [
        recording_block(
            index, joined[index * N_SAMPLES : (index + 1) * N_SAMPLES], o, lb
        )
        for index, (_, o, lb) in enumerate(raw)
    ]


class Run:
    """One held-out fold through the real orchestration, with fitter inputs."""

    def __init__(self, blocks: list[heldout.RecordingBlock], held_out: int) -> None:
        self.blocks = blocks
        self.held_out = held_out
        self.calls: list[dict[str, Any]] = []
        original = core.get_events_templates

        def record(*args: Any, **kwargs: Any) -> Any:
            data, ix, lb = args
            self.calls.append({"data": data, "ix": ix, "lb": lb, **kwargs})
            return original(*args, **kwargs)

        with patch.object(heldout.core, core.get_events_templates.__name__, new=record):
            self.fits: list[heldout.FoldFit] = heldout.evaluate_fold(
                blocks,
                held_out,
                heldout.reference_methods(),
                repetitions=2,
                comparators=heldout.comparator_methods(),
            )

    def by_key(self) -> dict[tuple[int, int, int, str], heldout.FoldFit]:
        return {fit.cell[:4]: fit for fit in self.fits}

    def aucs(self, method: str) -> list[float]:
        return [fit.auc for fit in self.fits if fit.method == method]


def assert_same_training(first: Run, second: Run) -> None:
    """Everything learned from training data must be identical."""
    for block_a, block_b in zip(first.blocks, second.blocks):
        if block_a.block != first.held_out:
            assert_array_equal(block_a.data, block_b.data)
    names = [*heldout.reference_methods(), *heldout.comparator_methods()]
    expected = {(1, first.held_out, r, name) for r in range(2) for name in names}
    fits_a, fits_b = first.by_key(), second.by_key()
    for run, fits in ((first, fits_a), (second, fits_b)):
        if len(fits) != len(run.fits):
            raise AssertionError(f"{len(run.fits)} fits share {len(fits)} keys")
        if set(fits) != expected:
            raise AssertionError(f"Fit keys {sorted(fits)} differ from {expected}")
    for key in sorted(expected):
        fit_a, fit_b = fits_a[key], fits_b[key]
        assert_array_equal(fit_a.selection, fit_b.selection)
        for label in ("0", "1"):
            assert_array_equal(fit_a.templates[label], fit_b.templates[label])
        inner_a, inner_b = fit_a.inner or {}, fit_b.inner or {}
        for name in ("opt_val", "noise_var", "cost"):
            assert_array_equal(
                np.asarray(inner_a.get(name, np.nan)),
                np.asarray(inner_b.get(name, np.nan)),
            )
    n_calls = 2 * len(heldout.reference_methods())
    if not len(first.calls) == len(second.calls) == n_calls:
        raise AssertionError(
            f"{len(first.calls)} and {len(second.calls)} core calls, expected {n_calls}"
        )
    for call_a, call_b in zip(first.calls, second.calls):
        for key in ("data", "ix", "lb", "block_starts"):
            assert_array_equal(call_a[key], call_b[key])
        assert core.estimate_phi(call_a["data"], call_a["ix"], 51) == core.estimate_phi(
            call_b["data"], call_b["ix"], 51
        ), "AR(1) estimate differs between runs"


class LeakageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = raw_blocks()

    def changed(
        self, change: Callable[[RawBlock], RawBlock], held_out: int = 1
    ) -> list[RawBlock]:
        return [
            change(block) if index == held_out else block
            for index, block in enumerate(self.raw)
        ]

    def test_held_out_signal_does_not_reach_training(self) -> None:
        rs = np.random.RandomState(9)
        perturbed = self.changed(
            lambda b: (b[0] + 5 * rs.normal(size=b[0].shape), b[1], b[2])
        )
        baseline = Run(preprocess_separately(self.raw), 1)
        changed = Run(preprocess_separately(perturbed), 1)
        assert_same_training(baseline, changed)
        # The perturbation does reach the held-out scores, Molina's included.
        self.assertNotEqual(
            [fit.auc for fit in baseline.fits], [fit.auc for fit in changed.fits]
        )
        self.assertNotEqual(baseline.aucs("Molina"), changed.aucs("Molina"))

    def test_held_out_labels_do_not_reach_training(self) -> None:
        rs = np.random.RandomState(4)
        permuted = self.changed(lambda b: (b[0], b[1], rs.permutation(b[2])))
        baseline = Run(preprocess_separately(self.raw), 1)
        changed = Run(preprocess_separately(permuted), 1)
        assert_same_training(baseline, changed)
        self.assertNotEqual(
            [fit.auc for fit in baseline.fits], [fit.auc for fit in changed.fits]
        )
        self.assertNotEqual(baseline.aucs("Molina"), changed.aucs("Molina"))

    def test_unsafe_joined_preprocessing_is_detected(self) -> None:
        rs = np.random.RandomState(9)
        perturbed = self.changed(
            lambda b: (b[0] + 5 * rs.normal(size=b[0].shape), b[1], b[2])
        )
        baseline = Run(preprocess_joined(self.raw), 1)
        changed = Run(preprocess_joined(perturbed), 1)
        with self.assertRaises(AssertionError):
            assert_same_training(baseline, changed)

    def test_fitter_receives_only_training_blocks(self) -> None:
        blocks = preprocess_separately(self.raw)
        for held_out in range(3):
            run = Run(blocks, held_out)
            training = [block for block in blocks if block.block != held_out]
            self.assertEqual(len(run.calls), 2 * len(heldout.reference_methods()))
            for call in run.calls:
                assert_array_equal(call["data"], np.vstack([b.data for b in training]))
                assert_array_equal(call["block_starts"], [0, N_SAMPLES])
                self.assertEqual(call["half_win"], heldout.HALF_WIN)


class SelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.blocks = preprocess_separately(raw_blocks())

    def test_folds_are_disjoint_and_cover_all_blocks(self) -> None:
        held_out = []
        for block in self.blocks:
            training = heldout.training_blocks(self.blocks, block.block)
            self.assertEqual(
                [b.block for b in training],
                sorted(set(range(3)) - {block.block}),
            )
            held_out.append(block.block)
            for selection in heldout.select_events(training):
                self.assertNotIn(block.block, selection[:, 0])
        self.assertEqual(sorted(held_out), [0, 1, 2])

    def test_exactly_k_unique_eligible_events_per_class(self) -> None:
        training = heldout.training_blocks(self.blocks, 0)
        by_id = {block.block: block for block in training}
        selections = heldout.select_events(training)
        self.assertEqual(len(selections), heldout.REPETITIONS)
        for selection in selections:
            labels = [by_id[b].labels[e] for b, e in selection]
            self.assertEqual(labels, [0] * heldout.K + [1] * heldout.K)
            self.assertEqual(len({tuple(row) for row in selection}), 2 * heldout.K)
            self.assertTrue(all(by_id[b].eligible()[e] for b, e in selection))

    def test_seed_protocol_is_fresh_per_fold(self) -> None:
        # Fresh RandomState(42) per fold, class 0 then class 1, candidates
        # ordered by block and event across both training blocks.
        training = heldout.training_blocks(self.blocks, 2)
        pools = {
            label: [
                (b.block, e)
                for b in training
                for e in np.flatnonzero(b.eligible())
                if b.labels[e] == label
            ]
            for label in (0, 1)
        }
        rng = np.random.RandomState(42)
        for selection in heldout.select_events(training):
            expected = [
                pools[label][i]
                for label in (0, 1)
                for i in rng.choice(len(pools[label]), 5, replace=False)
            ]
            assert_array_equal(selection, expected)
        assert_array_equal(
            np.vstack(heldout.select_events(training)),
            np.vstack(heldout.select_events(training)),
        )

    def test_eligibility_at_and_beyond_each_bound(self) -> None:
        low = heldout.GUARD + heldout.HALF_WIN
        high = N_SAMPLES - heldout.GUARD - heldout.HALF_WIN
        onsets = np.array([low, high, low - 1, high + 1])
        assert_array_equal(
            heldout.eligible_events(onsets, N_SAMPLES), [True, True, False, False]
        )
        # Unsigned and narrow onsets must not wrap around either bound.
        assert_array_equal(
            heldout.eligible_events(np.array([0, 50], dtype=np.uint16), N_SAMPLES),
            [False, False],
        )
        assert_array_equal(
            heldout.eligible_events(np.array([32760], dtype=np.int16), 32800), [False]
        )


class LabelTest(unittest.TestCase):
    def test_maps_names_not_numeric_order(self) -> None:
        codes = np.array([1, 2, 2, 1])
        assert_array_equal(
            heldout.event_labels(codes, {"Target": 1, "NonTarget": 2}), [1, 0, 0, 1]
        )
        assert_array_equal(
            heldout.event_labels(codes, {"Target": 2, "NonTarget": 1}), [0, 1, 1, 0]
        )

    def test_rejects_unknown_and_missing_names(self) -> None:
        cases = (
            ({"Target": 2, "NonTarget": 1, "Rest": 3}, "Unknown event names"),
            ({"Target": 2}, "Missing event names"),
            ({"Target": 1, "NonTarget": 1}, "share one code"),
            ({"Target": 2, "NonTarget": 3}, r"Event codes \[1\] have no name"),
        )
        for event_id, message in cases:
            with (
                self.subTest(event_id=event_id),
                self.assertRaisesRegex(ValueError, message),
            ):
                heldout.event_labels(np.array([1, 2]), event_id)


class ScoreTest(unittest.TestCase):
    def test_score_favors_target(self) -> None:
        rs = np.random.RandomState(1)
        templates: dict[object, NDArray] = {
            "0": rs.normal(size=(102, 3)),
            "1": rs.normal(size=(102, 3)),
        }
        data = np.zeros((N_SAMPLES, 3))
        onsets = np.array([300, 600])
        data[249:351] = templates["1"]
        data[549:651] = templates["0"]
        block = recording_block(0, data, onsets, np.array([1, 0]))
        scores, labels = heldout.score_events(block, templates)
        gap = np.sum((templates["1"] - templates["0"]) ** 2)
        np.testing.assert_allclose(scores, [gap, -gap])
        assert_array_equal(labels, [1, 0])

    def test_auc_equals_pairwise_count_with_ties(self) -> None:
        rs = np.random.RandomState(2)
        scores = rs.randint(0, 6, size=40).astype(float)
        labels = rs.randint(0, 2, size=40)
        pos, neg = scores[labels == 1], scores[labels == 0]
        expected = (
            np.sum(pos[:, None] > neg[None, :])
            + 0.5 * np.sum(pos[:, None] == neg[None, :])
        ) / (len(pos) * len(neg))
        self.assertAlmostEqual(heldout.auc(scores, labels), expected, places=12)
        self.assertEqual(heldout.auc(np.array([0.0, 1.0]), np.array([0, 1])), 1.0)
        self.assertEqual(heldout.auc(np.array([1.0, 0.0]), np.array([0, 1])), 0.0)

    def test_auc_rejects_nonfinite_scores_and_single_class(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be finite"):
            heldout.auc(np.array([0.0, np.nan]), np.array([0, 1]))
        with self.assertRaisesRegex(ValueError, "at least one label-0 event"):
            heldout.auc(np.array([0.0, 1.0]), np.array([1, 1]))

    def test_score_rejects_nonfinite_templates(self) -> None:
        block = recording_block(
            0, np.zeros((N_SAMPLES, 3)), np.array([300]), np.array([1])
        )
        templates: dict[object, NDArray] = {
            "0": np.zeros((102, 3)),
            "1": np.full((102, 3), np.inf),
        }
        with self.assertRaisesRegex(ValueError, "1 nonfinite scores"):
            heldout.score_events(block, templates)


class AggregationTest(unittest.TestCase):
    def grid(self) -> list[tuple[int, int, int, str, float]]:
        return [
            (subject, fold, rep, method, 0.5 + 0.01 * rep + 0.1 * subject)
            for subject in (1, 2)
            for fold in range(3)
            for rep in range(heldout.REPETITIONS)
            for method in ("OLS", "GCV")
        ]

    def test_means_cover_all_sixty_aucs(self) -> None:
        means = heldout.subject_mean_aucs(self.grid(), [1, 2], ["OLS", "GCV"])
        self.assertAlmostEqual(means[1, "OLS"], 0.6 + 0.01 * 9.5, places=12)
        self.assertEqual(len(means), 4)


def selected_windows(
    training: list[heldout.RecordingBlock], selection: NDArray, label: int
) -> NDArray:
    """The selected class-*label* windows, cut from their own blocks in order."""
    by_id = {block.block: block for block in training}
    return np.stack(
        [
            by_id[b].data[by_id[b].onsets[e] - 51 : by_id[b].onsets[e] + 51]
            for b, e in selection
            if by_id[b].labels[e] == label
        ]
    )


class ComparatorTest(unittest.TestCase):
    def setUp(self) -> None:
        blocks = preprocess_separately(raw_blocks())
        # Unequal block lengths: shorten block 0, dropping events near its end.
        short = blocks[0]
        keep = short.onsets <= 1000 - 51 - heldout.GUARD
        blocks[0] = recording_block(
            0, short.data[:1000], short.onsets[keep], short.labels[keep]
        )
        self.blocks = blocks

    def test_comparator_receives_the_selected_training_windows(self) -> None:
        seen: list[NDArray] = []

        def mean(epochs: NDArray) -> NDArray:
            seen.append(epochs.copy())
            return epochs.mean(0)

        fits = heldout.evaluate_fold(
            self.blocks,
            2,
            {"Average": {"method": "average"}},
            repetitions=3,
            comparators={"Mean": mean},
        )
        training = heldout.training_blocks(self.blocks, 2)
        selections = heldout.select_events(training, repetitions=3)
        self.assertEqual(len(seen), 2 * len(selections))
        for repetition, selection in enumerate(selections):
            for label in (0, 1):
                assert_array_equal(
                    seen[2 * repetition + label],
                    selected_windows(training, selection, label),
                )
        # Exactly the epochs Average averages: the mean comparator reproduces it.
        by_key = {fit.cell[:4]: fit for fit in fits}
        for repetition in range(3):
            for label in ("0", "1"):
                assert_array_equal(
                    by_key[1, 2, repetition, "Mean"].templates[label],
                    by_key[1, 2, repetition, "Average"].templates[label],
                )
            self.assertIsNone(by_key[1, 2, repetition, "Mean"].inner)

    def test_molina_runs_through_the_orchestration_on_a_complete_grid(self) -> None:
        methods = heldout.reference_methods()
        cells = []
        for held_out in range(3):
            fits = heldout.evaluate_fold(
                self.blocks,
                held_out,
                methods,
                repetitions=2,
                comparators=heldout.comparator_methods(),
            )
            cells += [fit.cell for fit in fits]
            training = heldout.training_blocks(self.blocks, held_out)
            for fit in fits:
                if fit.method != "Molina":
                    continue
                for label in (0, 1):
                    windows = selected_windows(training, fit.selection, label)
                    assert_array_equal(
                        fit.templates[str(label)],
                        dtw_average.molina_template(windows),
                    )
        names = [*methods, "Molina"]
        means = heldout.subject_mean_aucs(cells, [1], names, repetitions=2)
        self.assertEqual(len(cells), 3 * 2 * len(names))
        self.assertEqual(sorted(means), sorted((1, name) for name in names))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in means.values()))


def references(subjects: range, seed: int = 3) -> dict[int, dict[object, NDArray]]:
    rs = np.random.RandomState(seed)
    return {
        s: {"0": rs.normal(size=(102, 3)), "1": rs.normal(size=(102, 3))}
        for s in subjects
    }


class PopulationTest(unittest.TestCase):
    def test_mean_of_the_other_subjects(self) -> None:
        refs = references(range(1, 11))
        population = heldout.population_templates(refs, 4)
        donors = [1, 2, 3, 5, 6, 7, 8, 9, 10]
        for label in ("0", "1"):
            assert_array_equal(
                population[label],
                np.mean(np.stack([refs[d][label] for d in donors]), 0),
            )
            total = sum(refs[s][label] for s in range(1, 11))
            np.testing.assert_allclose(
                population[label], (total - refs[4][label]) / 9, rtol=1e-12
            )

    def test_subject_entry_is_never_read(self) -> None:
        refs = references(range(1, 11))
        population = heldout.population_templates(refs, 4)
        refs[4] = {"other": np.full((7,), np.nan)}
        unchanged = heldout.population_templates(refs, 4)
        for label in ("0", "1"):
            assert_array_equal(unchanged[label], population[label])
        # A donor's reference does reach the template.
        refs[5]["1"] = refs[5]["1"] + 1.0
        self.assertFalse(
            np.array_equal(heldout.population_templates(refs, 4)["1"], population["1"])
        )

    def test_donor_order_is_fixed_and_inputs_are_not_mutated(self) -> None:
        # Donor 2, 3, 4 values 1e16, 1, -1e16: the mean depends on summation order.
        values = {1: 0.0, 2: 1e16, 3: 1.0, 4: -1e16}
        refs: dict[int, dict[object, NDArray]] = {
            s: {"0": np.full((2, 2), v)} for s, v in values.items()
        }
        shuffled: dict[int, dict[object, NDArray]] = {s: refs[s] for s in (4, 1, 2, 3)}
        copies = {s: refs[s]["0"].copy() for s in refs}
        result = heldout.population_templates(refs, 1)["0"]
        assert_array_equal(heldout.population_templates(shuffled, 1)["0"], result)
        assert_array_equal(result, np.zeros((2, 2)))  # ((1e16 + 1) - 1e16) / 3
        insertion_order = np.mean(np.stack([refs[s]["0"] for s in (2, 4, 3)]), 0)
        self.assertFalse(np.array_equal(insertion_order, result))
        for s in refs:
            assert_array_equal(refs[s]["0"], copies[s])

    def test_cells_average_block_aucs_equally(self) -> None:
        blocks = preprocess_separately(raw_blocks())
        # Unequal event counts: block 2 keeps only its first third of events.
        third = len(blocks[2].onsets) // 3
        blocks[2] = recording_block(
            2, blocks[2].data, blocks[2].onsets[:third], blocks[2].labels[:third]
        )
        response = np.outer(np.hanning(102), [1.0, 0.6, -0.3])
        templates: dict[object, NDArray] = {"0": response, "1": 2 * response}
        cells = heldout.population_cells(blocks, templates)
        self.assertEqual(
            [cell[:4] for cell in cells], [(1, b, 0, "Population") for b in range(3)]
        )
        scored = [heldout.score_events(block, templates) for block in blocks]
        block_aucs = [heldout.auc(scores, labels) for scores, labels in scored]
        self.assertEqual([cell[4] for cell in cells], block_aucs)
        self.assertGreater(len(set(block_aucs)), 1)
        mean = heldout.subject_mean_aucs(cells, [1], ["Population"], repetitions=1)
        self.assertAlmostEqual(
            mean[1, "Population"], float(np.mean(block_aucs)), places=15
        )
        pooled = heldout.auc(
            np.concatenate([s for s, _ in scored]),
            np.concatenate([lb for _, lb in scored]),
        )
        self.assertNotAlmostEqual(mean[1, "Population"], pooled, places=6)
