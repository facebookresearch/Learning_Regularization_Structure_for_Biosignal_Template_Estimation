# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Statistics and the failure policy of the paper analyses."""

from __future__ import annotations

import dataclasses
import importlib
import itertools
import math
import tempfile
import time
import unittest
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast, TYPE_CHECKING
from unittest import mock

import numpy as np
import scipy
from numpy.typing import NDArray

if TYPE_CHECKING:
    import analysis_tools
    import core
    import heldout_p300
    import protocol
else:
    analysis_tools = importlib.import_module("analysis_tools")
    core = importlib.import_module("core")
    heldout_p300 = importlib.import_module("heldout_p300")
    protocol = importlib.import_module("protocol")


def brute_force_p(effects: Sequence[float]) -> float:
    """Exact two-sided p over all 2^n sign patterns, zeros dropped, ties averaged."""
    nonzero = [x for x in effects if x != 0]
    magnitudes = [abs(x) for x in nonzero]
    # Twice the average rank: 2 * (number smaller) + (number equal) + 1.
    doubled = [
        2 * sum(m < v for m in magnitudes) + sum(m == v for m in magnitudes) + 1
        for v in magnitudes
    ]
    total = sum(doubled)
    observed = abs(2 * sum(r for r, x in zip(doubled, nonzero) if x > 0) - total)
    hits = sum(
        abs(2 * sum(r for r, s in zip(doubled, signs) if s) - total) >= observed
        for signs in itertools.product((0, 1), repeat=len(nonzero))
    )
    return hits / 2 ** len(nonzero)


def p_value(effects: Sequence[float]) -> float:
    return cast(float, analysis_tools.signed_rank_test(effects)["p"])


class SignedRankTest(unittest.TestCase):
    def test_matches_brute_force_on_continuous_and_tied_effects(self) -> None:
        rng = np.random.RandomState(0)
        cases = [(rng.randn(10) + 0.3).tolist() for _ in range(10)]
        cases += [np.round(rng.randn(n) + 0.3, 1).tolist() for n in range(2, 12)]
        for d in cases:
            with self.subTest(effects=d):
                self.assertEqual(p_value(d), brute_force_p(d))

    def test_zeros_are_dropped_and_counted(self) -> None:
        d = [0.5, -0.5, 1.0, 1.0, 0.0, 2.0, -1.0, 0.5, 3.0, 0.0]
        out = analysis_tools.signed_rank_test(d)
        self.assertEqual((out["n"], out["n_nonzero"]), (10, 8))
        self.assertEqual(out["p"], brute_force_p(d))

    def test_edge_cases_keep_their_defined_results(self) -> None:
        # SciPy cannot be called for these; the results are the exact test's.
        empty = analysis_tools.signed_rank_test([])
        self.assertEqual(
            (empty["n"], empty["p"], empty["rank_biserial"]), (0, None, None)
        )
        zero = analysis_tools.signed_rank_test([0.0, 0.0])
        self.assertEqual((zero["p"], zero["rank_biserial"]), (1.0, 0.0))
        single = analysis_tools.signed_rank_test([0.0, -2.5])
        self.assertEqual(
            (single["n"], single["n_nonzero"], single["p"], single["rank_biserial"]),
            (2, 1, 1.0, -1.0),
        )
        self.assertEqual(
            analysis_tools.signed_rank_test([4.0])["p"], brute_force_p([4.0])
        )
        two = [1.0, 2.0]
        self.assertEqual(p_value(two), brute_force_p(two))

    def test_nonfinite_effects_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonfinite subject effects"):
            analysis_tools.signed_rank_test([1.0, math.nan])

    def test_rank_biserial_sign_and_value(self) -> None:
        # Ranks 1..4, positives {2, 4}: (6 - 4) / 10.
        out = analysis_tools.signed_rank_test([-1, 2, -3, 4])
        self.assertAlmostEqual(cast(float, out["rank_biserial"]), 0.2, places=15)
        self.assertEqual(
            analysis_tools.signed_rank_test([1, 2, 3])["rank_biserial"], 1.0
        )


class HolmTest(unittest.TestCase):
    def test_known_examples(self) -> None:
        self.assertEqual(analysis_tools.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(analysis_tools.holm([0.5, 1.0, 0.001]), [1.0, 1.0, 0.003])


class PairedEffectsTest(unittest.TestCase):
    def test_common_valid_cells_and_sign_convention(self) -> None:
        anchor: dict[int, dict[int, float | None]] = {1: {0: 1.0, 1: 2.0, 2: None}}
        other: dict[int, dict[int, float | None]] = {1: {0: 2.0, 1: 0.0, 2: 3.0}}
        out = analysis_tools.paired_effects(anchor, other, log_ratio=True)
        # Only cell 0 is valid and positive for both; other farther -> positive.
        self.assertEqual(out[1]["effect"], math.log(2.0))
        self.assertEqual((out[1]["n_valid"], out[1]["excluded_nonpositive"]), (1, 1))
        auc = analysis_tools.paired_effects(
            {1: {0: 0.8}}, {1: {0: 0.7}}, log_ratio=False
        )
        self.assertEqual(auc[1]["effect"], 0.8 - 0.7)


def synthetic_blocks() -> list[heldout_p300.RecordingBlock]:
    rs = np.random.RandomState(3)
    t = np.arange(102)
    shapes = {
        0: np.sin(t / 8.0)[:, None] * np.array([1.0, 2.0]),
        1: 3 * np.exp(-(((t - 70) / 9.0) ** 2))[:, None] * np.array([2.0, 1.0]),
    }
    blocks = []
    for b in range(3):
        onsets = np.arange(280, 1220, 30)
        labels = (rs.rand(len(onsets)) < 0.3).astype(np.int64)
        data = rs.randn(1500, 2)
        for onset, label in zip(onsets, labels):
            data[onset - 51 : onset + 51] += shapes[int(label)]
        blocks.append(
            heldout_p300.RecordingBlock(
                subject=1,
                block=b,
                data=data,
                onsets=onsets,
                native_onsets=2 * onsets,
                labels=labels,
                channels=("C1", "C2"),
            )
        )
    return blocks


class FailurePolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.blocks = synthetic_blocks()
        self.rows = heldout_p300.select_events(self.blocks, k=5, repetitions=1)[0]

    def fit(
        self,
        comparator: analysis_tools.Comparator,
        exceptions_are_failures: bool = True,
    ) -> tuple[dict[str, object], NDArray | None]:
        estimators = analysis_tools.Estimators(
            methods={}, comparators={"probe": comparator}
        )
        return analysis_tools.fit_estimator(
            estimators, "probe", self.blocks, self.rows, 51, exceptions_are_failures
        )

    def test_comparator_receives_exactly_the_epochs_average_averages(self) -> None:
        seen: list[int] = []

        def mean(epochs: NDArray) -> NDArray:
            seen.append(len(epochs))
            return epochs.mean(axis=0)

        record, array = self.fit(mean)
        _, average = analysis_tools.fit_estimator(
            analysis_tools.default_estimators(),
            "Average",
            self.blocks,
            self.rows,
            51,
            True,
        )
        self.assertEqual((record["status"], seen), ("success", [5, 5]))
        assert array is not None and average is not None
        np.testing.assert_allclose(array, average, rtol=1e-13, atol=0)

    def test_nonfinite_template_is_a_failure_that_keeps_its_template(self) -> None:
        calls: list[int] = []

        def probe(epochs: NDArray) -> NDArray:
            # Classes are fitted in label order: the first call is class 0.
            calls.append(len(epochs))
            mean = epochs.mean(axis=0)
            return np.full_like(mean, np.nan) if len(calls) == 1 else mean

        record, array = self.fit(probe)
        self.assertEqual(
            (record["status"], record["failure_type"], record["failure_message"]),
            ("failure", "NonfiniteTemplate", "204 nonfinite values"),
        )
        assert array is not None
        self.assertTrue(np.isnan(array[0]).all() and np.isfinite(array[1]).all())

    def test_estimator_exception_is_a_counted_failure_without_template(self) -> None:
        def singular(epochs: NDArray) -> NDArray:
            raise np.linalg.LinAlgError("singular")

        record, array = self.fit(singular)
        self.assertEqual(
            (record["status"], record["failure_type"], record["failure_message"]),
            ("failure", "LinAlgError", "singular"),
        )
        self.assertIsNone(array)
        self.assertIn("LinAlgError", cast(str, record["traceback"]))
        with self.assertRaisesRegex(np.linalg.LinAlgError, "singular"):
            self.fit(singular, exceptions_are_failures=False)

    def test_infrastructure_error_and_unknown_name_stop_the_run(self) -> None:
        def out_of_memory(epochs: NDArray) -> NDArray:
            raise MemoryError

        with self.assertRaises(MemoryError):
            self.fit(out_of_memory)
        with self.assertRaisesRegex(ValueError, "unknown estimator 'Ridge'"):
            analysis_tools.fit_estimator(
                analysis_tools.default_estimators(),
                "Ridge",
                self.blocks,
                self.rows,
                51,
                True,
            )

    def test_warnings_and_failed_diagnostics_are_flagged_not_failures(self) -> None:
        def warn(epochs: NDArray) -> NDArray:
            warnings.warn("probe warning", RuntimeWarning, stacklevel=2)
            return epochs.mean(axis=0)

        record, _ = self.fit(warn)
        self.assertEqual(
            (record["status"], record["flags"], record["warnings"]),
            ("success", ["warning"], ["RuntimeWarning: probe warning"]),
        )
        inner = {
            "optimizer_success": False,
            "lcurve": {"converged": np.False_, "passed": True},
            "steps": [{"success": True}, {"success": False}],
        }
        self.assertEqual(
            analysis_tools.failure_flags(inner),
            ["optimizer_success", "lcurve.converged", "steps[1].success"],
        )


class BlockCacheTest(unittest.TestCase):
    def test_saved_blocks_load_back_with_their_own_ids(self) -> None:
        # Held-out folds are addressed by block id, so non-contiguous ids must
        # survive the cache unchanged.
        blocks = [
            dataclasses.replace(block, block=b)
            for block, b in zip(synthetic_blocks(), (2, 5, 7))
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.npz"
            analysis_tools.save_blocks(path, blocks)
            loaded = analysis_tools.load_blocks(path, subject=1)
        self.assertEqual([b.block for b in loaded], [2, 5, 7])
        for saved, back in zip(blocks, loaded):
            np.testing.assert_array_equal(back.data, saved.data)
            np.testing.assert_array_equal(back.onsets, saved.onsets)
            np.testing.assert_array_equal(back.native_onsets, saved.native_onsets)
            np.testing.assert_array_equal(back.labels, saved.labels)
            self.assertEqual(back.channels, saved.channels)


class SelectionTest(unittest.TestCase):
    def test_whole_recording_selections_follow_the_protocol_and_nest_across_k(
        self,
    ) -> None:
        # The K curve's K = 5 selections are the central batch's; the
        # manuscript states K = 5 within 10 within 20 within 50.
        blocks = synthetic_blocks()
        guard = heldout_p300.GUARD

        def draw(k: int) -> list[list[list[int]]]:
            rs = np.random.RandomState(42)
            return analysis_tools.whole_recording_selections(
                blocks, rs, k, 3, 51, guard
            )

        selections = {k: draw(k) for k in (5, 10, 15)}
        self.assertEqual(draw(5), selections[5])  # a fresh generator per call
        for k, reps in selections.items():
            for rows in reps:
                labels = [int(blocks[b].labels[e]) for b, e in rows]
                self.assertEqual((labels.count(0), labels.count(1)), (k, k))
                self.assertEqual(rows, sorted(rows))
                self.assertTrue(all(blocks[b].eligible(51, guard)[e] for b, e in rows))
        for small, large in ((5, 10), (10, 15)):
            for a, b in zip(selections[small], selections[large]):
                self.assertLessEqual({tuple(x) for x in a}, {tuple(x) for x in b})

    def test_selections_address_blocks_by_id_not_position(self) -> None:
        # The same recording with block ids 2, 5, 7 gives the same draws, with
        # the rows naming the blocks by their ids.
        blocks = synthetic_blocks()
        renamed = [
            dataclasses.replace(block, block=b) for block, b in zip(blocks, (2, 5, 7))
        ]
        guard = heldout_p300.GUARD
        drawn = [
            analysis_tools.whole_recording_selections(
                bs, np.random.RandomState(42), 5, 3, 51, guard
            )
            for bs in (blocks, renamed)
        ]
        ids = {0: 2, 1: 5, 2: 7}
        self.assertEqual(
            [[[ids[b], e] for b, e in rows] for rows in drawn[0]], drawn[1]
        )


class UnitTest(unittest.TestCase):
    def test_unit_loop_equals_evaluate_fold(self) -> None:
        # The analyses fit and score one selection at a time; that must give
        # the held-out evaluation the method diff defines.
        blocks = synthetic_blocks()
        methods = {
            name: heldout_p300.reference_methods()[name]
            for name in ("Average", "OLS", "SURE(lambda)")
        }

        def mean(epochs: NDArray) -> NDArray:
            return epochs.mean(axis=0)

        estimators = analysis_tools.Estimators(
            methods=methods, comparators={"probe": mean}
        )
        names = [*methods, "probe"]
        guard = heldout_p300.GUARD
        for held_out in range(3):
            expected = {
                (fit.repetition, fit.method): fit.auc
                for fit in heldout_p300.evaluate_fold(
                    blocks, held_out, methods, 5, 2, comparators={"probe": mean}
                )
            }
            training = heldout_p300.training_blocks(blocks, held_out)
            test = blocks[held_out]
            for rep, rows in enumerate(heldout_p300.select_events(training, 5, 2)):
                fits = analysis_tools.fit_unit(
                    estimators, names, training, rows, test, {}, 51, guard, True
                )
                for name, (record, _) in zip(names, fits):
                    self.assertEqual(record["auc"], expected[rep, name])


class PopulationTest(unittest.TestCase):
    def test_own_reference_never_enters_own_population(self) -> None:
        rs = np.random.RandomState(0)
        refs: dict[int, NDArray | None] = {s: rs.randn(2, 6, 3) for s in (1, 2, 3)}
        population = analysis_tools.population_template(refs, 1)
        assert population is not None, "three subjects with references leave two donors"
        np.testing.assert_allclose(
            population,
            (cast(NDArray, refs[2]) + cast(NDArray, refs[3])) / 2,
            rtol=1e-15,
        )
        changed = dict(refs) | {1: None}
        np.testing.assert_array_equal(
            analysis_tools.population_template(changed, 1), population
        )
        self.assertIsNone(analysis_tools.population_template(refs | {3: None}, 1))


class PairedDifferenceTest(unittest.TestCase):
    def test_differences_use_the_cells_both_estimators_fitted(self) -> None:
        average: dict[int, dict[tuple[int, int], float | None]] = {
            1: {(0, 0): 0.6, (0, 1): 0.7, (1, 0): None},
            2: {(0, 0): 0.5, (0, 1): 0.5, (1, 0): 0.8},
        }
        method: dict[int, dict[tuple[int, int], float | None]] = {
            1: {(0, 0): None, (0, 1): 0.9, (1, 0): 0.9},
            2: {(0, 0): 0.6, (0, 1): 0.6, (1, 0): 0.8},
        }
        out = analysis_tools.paired_difference(average, method)
        # Subject 1: only cell (0, 1) is valid for both; each failed cell counts once.
        self.assertEqual(out["per_subject_diff"]["1"], 0.9 - 0.7)
        self.assertEqual(out["excluded_cells_by_subject"], {"1": 2, "2": 0})
        self.assertEqual(out["n_valid_pairs"], {"1": 1, "2": 3})
        subject_2 = float(np.mean([0.6, 0.6, 0.8])) - float(np.mean([0.5, 0.5, 0.8]))
        self.assertEqual(out["per_subject_diff"]["2"], subject_2)
        self.assertEqual(out["subjects_above_average"], 2)


def line_fit(data: NDArray, ix: NDArray, lb: NDArray, hw: int) -> dict[str, NDArray]:
    """Least squares with a constant and a slope per class and channel (overlapping
    windows add up, as in the full design)."""
    t = np.arange(2 * hw, dtype=float)
    basis = np.stack([np.ones_like(t), t - t.mean()], axis=1)
    classes = np.unique(lb)
    x = np.zeros((len(data), 2 * len(classes)))
    for i, label in zip(ix, lb):
        c = int(np.flatnonzero(classes == label)[0])
        x[i - hw : i + hw, 2 * c : 2 * c + 2] += basis
    coef, *_ = np.linalg.lstsq(x, data, rcond=None)
    return {
        str(label): basis @ coef[2 * c : 2 * c + 2] for c, label in enumerate(classes)
    }


class ForcedInfiniteLambdaTest(unittest.TestCase):
    def test_forced_fit_is_the_least_squares_line_and_the_selector_is_restored(
        self,
    ) -> None:
        # The lambda = inf baseline is the penalty's null-space fit.
        rs = np.random.RandomState(3)
        ix = np.sort(rs.choice(np.arange(25, 2975), 60, replace=False))
        lb = rs.randint(0, 2, size=len(ix))
        data = rs.randn(3000, 2)
        for i, label in zip(ix, lb):
            data[i - 20 : i + 20] += np.linspace(-1.0, 2.0, 40)[:, None] * (1 + label)
        original = core._select_lcurve  # noqa: SLF001
        with analysis_tools.forced_infinite_lambda() as calls:
            templates, inner = core.get_events_templates(
                data, ix, lb, half_win=20, method="ls", reg_method="lcurve"
            )
        self.assertIs(core._select_lcurve, original)  # noqa: SLF001
        self.assertEqual(
            (len(calls), cast(dict[str, object], inner)["opt_val"]), (1, math.inf)
        )
        for label, line in line_fit(data, ix, lb, 20).items():
            np.testing.assert_allclose(templates[label], line, rtol=1e-9, atol=1e-9)
        with (
            self.assertRaisesRegex(RuntimeError, "raised inside the forced context"),
            analysis_tools.forced_infinite_lambda(),
        ):
            raise RuntimeError("raised inside the forced context")
        self.assertIs(core._select_lcurve, original)  # noqa: SLF001


class ModelNoiseTest(unittest.TestCase):
    def test_noise_power_equals_the_explicit_covariance_oracle(self) -> None:
        # sum_c tr_class(Q D^T Sigma_c D Q) with Sigma_c written out densely: the
        # tapered residual autocovariance, Toeplitz within blocks, zero across them.
        rng = np.random.default_rng(3)
        hw, lag, edges = 10, 8, [0, 300, 600, 900]
        data = rng.standard_normal((900, 2)).cumsum(axis=0) * 0.1
        windows = [range(s + 15, e - 15) for s, e in zip(edges, edges[1:])]
        ix = np.concatenate([np.sort(rng.choice(w, 25, False)) for w in windows])
        lb = rng.permutation(np.arange(len(ix)) % 2)
        design = core.get_design_matrix(data, ix, lb, hw, np.array([0, 1]), True)
        sparse = scipy.sparse.csc_matrix(design)
        d = sparse.toarray()
        q = np.linalg.inv(d.T @ d)
        residual = data - d @ (q @ d.T @ data)
        acov = analysis_tools.residual_autocovariance(residual, edges, lag)
        gram = analysis_tools.noise_gram(
            sparse, edges, analysis_tools.bartlett_kernel(acov)
        )
        taper = (1 - np.arange(lag + 1) / (lag + 1)) / len(data)
        dist = np.abs(np.subtract.outer(np.arange(300), np.arange(300)))
        expected = np.zeros(2)
        for c in range(2):
            r = [residual[s:e, c] for s, e in zip(edges, edges[1:])]
            gamma = taper * [
                sum(x[: len(x) - k] @ x[k:] for x in r) for k in range(lag + 1)
            ]
            toeplitz = np.where(dist <= lag, gamma[np.minimum(dist, lag)], 0.0)
            cov = q @ d.T @ scipy.linalg.block_diag(*[toeplitz] * 3) @ d @ q
            expected += [
                np.trace(cov[k * 20 : (k + 1) * 20, k * 20 : (k + 1) * 20])
                for k in (0, 1)
            ]
        power = analysis_tools.class_noise_power(q, gram, 2 * hw)
        np.testing.assert_allclose(power, expected, rtol=1e-9)


class NoiseCorrectedMetricTest(unittest.TestCase):
    def test_truncates_at_zero_and_excludes_failed_fits(self) -> None:
        # One subject, two fits; the second failed (NaN). Class 0 of the first is
        # truncated (4 - 5 < 0), class 1 gives sqrt(9 - 1); 2X = 2 shifts both.
        d2, sigma2 = np.array([[[4.0, 9.0], [np.nan, np.nan]]]), np.array([[5.0, 1.0]])
        out = analysis_tools.nc_metric(d2, sigma2)
        self.assertAlmostEqual(cast(float, out["mean"]), math.sqrt(8.0) / 2)
        counts = ("trunc_cells", "n_cells", "n_fits", "excluded_failed_fits")
        self.assertEqual(tuple(out[x] for x in counts), (1, 2, 1, 1))
        adjusted = analysis_tools.nc_metric(d2, sigma2, np.full((1, 2, 2), 2.0))
        self.assertAlmostEqual(cast(float, adjusted["mean"]), (1 + math.sqrt(10)) / 2)


def assert_same_case(
    got: Sequence[NDArray], want: Sequence[NDArray], case: str
) -> None:
    for name, a, b in zip(("data", "onsets", "labels"), got, want, strict=True):
        np.testing.assert_array_equal(a, b, err_msg=f"{case}: {name} differ")


class SimulationSeedProtocolTest(unittest.TestCase):
    """Why: the published sweeps used these two seed schemes; another scheme
    gives other realizations and other SNR tables."""

    def test_event_sweep_reseeds_per_repetition(self) -> None:
        truth = analysis_tools.make_real_templates()
        for count, rep in ((15, 1), (4, 0)):  # out of order: no state carries over
            expected = core.generate_data(
                truth,
                I=count,
                noise=protocol.NEVENTS_NOISE,
                mean_interval=protocol.NEVENTS_MEAN_INTERVAL,
                rs=np.random.RandomState(protocol.SIM_SEED + rep),
            )
            case = analysis_tools.nevents_case(truth, count, rep)
            assert_same_case(case, expected, f"K={count} repetition {rep}")

    def test_noise_sweep_keeps_one_continuous_stream(self) -> None:
        truth = analysis_tools.make_real_templates()
        rs = np.random.RandomState(protocol.SIM_SEED)
        expected = [
            core.generate_data(
                truth,
                I=protocol.NOISE_SWEEP_I,
                noise=noise,
                mean_interval=protocol.NOISE_SWEEP_MEAN_INTERVAL,
                rs=rs,
            )
            for noise in protocol.NOISE_SWEEP
            for _ in range(2)
        ]
        observed = list(analysis_tools.noise_sweep(truth, 2))
        self.assertEqual(
            [(g, r) for g, r, _ in observed],
            [(g, r) for g in range(len(protocol.NOISE_SWEEP)) for r in range(2)],
        )
        for (g, r, case), want in zip(observed, expected, strict=True):
            assert_same_case(case, want, f"noise level {g} repetition {r}")


class KernelBfgsMaxiterTest(unittest.TestCase):
    """Why: the Q5 convergence result holds only if the override raises the
    kernel optimizer's iteration limit and changes nothing else."""

    def fit(self) -> tuple[bytes, dict[str, object]]:
        truth = analysis_tools.make_real_templates()
        data, ix, labels = analysis_tools.nevents_case(truth, protocol.Q_SIM_I, 0)
        kwargs: protocol.CoreSettings = {
            **protocol.SURE_KERNEL,
            **protocol.Q_CELLS["Q5-free"],
            "noise_est": protocol.Q_SIM_NOISE_EST,
        }
        templates, inner = core.get_events_templates(
            data,
            ix,
            labels,
            half_win=protocol.SIM_HALF_WIN,
            **kwargs,
        )
        if inner is None:
            self.fail("SURE(kernel) returned no optimizer details (inner is None)")
        array = np.stack([templates[k] for k in sorted(templates, key=str)])
        return array.tobytes(), inner

    def test_limit_50_reproduces_the_unpatched_fit_bitwise(self) -> None:
        plain, plain_inner = self.fit()
        with analysis_tools.kernel_bfgs_maxiter(50) as calls:
            patched, patched_inner = self.fit()
        self.assertEqual(calls["patched"], 1)
        self.assertEqual(patched, plain)
        for key in ("opt_val", "optimizer_iterations", "cost", "noise_var"):
            self.assertEqual(
                np.asarray(patched_inner[key]).tobytes(),
                np.asarray(plain_inner[key]).tobytes(),
                key,
            )

    def test_raised_limit_goes_further_and_the_original_is_restored(self) -> None:
        real = scipy.optimize.minimize
        _, capped = self.fit()
        # On this realization the default limit binds.
        self.assertEqual(
            (capped["optimizer_status"], capped["optimizer_iterations"]), (1, 50)
        )
        with analysis_tools.kernel_bfgs_maxiter(500):
            _, raised = self.fit()
        self.assertGreater(cast(int, raised["optimizer_iterations"]), 50)
        self.assertLessEqual(cast(float, raised["cost"]), cast(float, capped["cost"]))
        with (
            self.assertRaisesRegex(KeyError, "inside"),
            analysis_tools.kernel_bfgs_maxiter(500),
        ):
            raise KeyError("inside")
        self.assertIs(scipy.optimize.minimize, real)


class FitCallOutputTest(unittest.TestCase):
    """Why: an estimator that returns unusable output must be a counted failure
    like one that raises, not a crash of the whole batch (decision OQ6)."""

    def test_unusable_output_is_a_counted_failure(self) -> None:
        good = np.zeros((3, 2))
        Fit = tuple[dict[str, NDArray], dict[str, object] | None]
        cases: list[tuple[type[Exception], Callable[[], Fit], tuple[str, ...]]] = [
            (KeyError, lambda: ({"0": good}, None), ("0", "1")),
            (ValueError, lambda: ({"0": good, "1": np.zeros(4)}, None), ("0", "1")),
            (TypeError, lambda: ({"0": good}, {"opt_val": {}}), ("0",)),
        ]
        for error, call, keys in cases:
            with self.subTest(error=error.__name__):
                record, array = analysis_tools.fit_call(call, keys, True)
                self.assertEqual(
                    (record["status"], record["failure_type"], array),
                    ("failure", error.__name__, None),
                )
                with self.assertRaises(error):
                    analysis_tools.fit_call(call, keys, False)

    def test_missing_tuning_value_is_not_recorded(self) -> None:
        record, array = analysis_tools.fit_call(
            lambda: ({"0": np.ones((3, 2))}, {"opt_val": None}), ("0",), True
        )
        self.assertEqual(record["status"], "success")
        self.assertNotIn("lambda", record)
        self.assertNotIn("kernel", record)
        self.assertEqual(cast(NDArray, array).shape, (1, 3, 2))


class StageTimerTest(unittest.TestCase):
    def test_nested_time_counts_once_in_the_innermost_timed_stage(self) -> None:
        # The stage split (R2.10) gives the kernel objective's share as the sum of
        # exclusive stage times; a nested call must not be counted in its caller
        # as well.
        now = [0.0]

        def spend(seconds: float) -> None:
            now[0] += seconds

        timer = analysis_tools.StageTimer()
        decomposition = timer.wrap("decomp", lambda: spend(1.0), inside="objective")
        final = timer.wrap("final", lambda: spend(2.0), top_level_only=True)

        def objective_body() -> None:
            spend(3.0)
            decomposition()
            final()  # inside a stage: its time stays with the objective

        objective = timer.wrap("objective", objective_body)
        with mock.patch.object(time, time.perf_counter.__name__, lambda: now[0]):
            objective()
            final()  # top level: timed
            decomposition()  # not called from the objective: untimed
        self.assertEqual(
            timer.exclusive, {"decomp": 1.0, "objective": 5.0, "final": 2.0}
        )
        self.assertEqual(timer.calls, {"decomp": 1, "objective": 1, "final": 1})


# R2.9 rules (the post hoc analyses of run_r29.py) on synthetic data.


def r29_record(
    K: int | str, s: int, f: int, r: int, m: str, v: float
) -> dict[str, object]:
    """An amplitude-value record whose every metric equals `v`."""
    rec: dict[str, object] = {"K": K, "subject": s, "fold": f, "rep": r, "method": m}
    rec.update(dict.fromkeys(analysis_tools.R29_MEAN_KEYS, v), delta_ho=v)
    rec.update(dict.fromkeys(analysis_tools.R29_MEDIAN_KEYS, v), b_shrink=0.5, b_zero=0)
    stop = {str(t): [1, 1.0, 0.95, 0, 0.0, 0.0, 3] for t in analysis_tools.R29_TAUS}
    rec["sp"] = {"ll": [v] * 8, "acc": [1.0] * 8, "pmax": [0.9] * 8, "stop": stop}
    return rec


class R29FailureRulesTest(unittest.TestCase):
    """A failed L-curve fit carries no metric (R1); levels use each estimator's own
    valid fits (R2); paired L-curve comparisons use the cells valid for both (R3)."""

    def test_excluded_fit_is_dropped_and_the_reference_uses_the_common_cells(
        self,
    ) -> None:
        lc, sk = analysis_tools.R29_LC, "SURE(kernel)"
        records = []
        # Two subjects, 3 folds x 2 repetitions; subject 1's L-curve fit at fold 0,
        # repetition 1 failed, where SURE(kernel) is 100. Other values: 0, L-curve 1.
        for s in (1, 2):
            for f in range(3):
                records.append(r29_record("pop", s, f, 0, protocol.POPULATION, 0.0))
                for r in range(2):
                    bad = (s, f, r) == (1, 0, 1)
                    for m in [*protocol.ESTIMATORS, analysis_tools.R29_SHRINK]:
                        if m == lc and bad:
                            failed: dict[str, object] = {"K": 5, "subject": s}
                            failed |= {"fold": f, "rep": r, "method": m, "failed": "k"}
                            records.append(failed)
                            continue
                        v = 1.0 if m == lc else (100.0 if m == sk and bad else 0.0)
                        records.append(r29_record(5, s, f, r, m, v))
        per, per_lc = analysis_tools.r29_per_subject(records)
        tab = {m: per[5, m] for m in analysis_tools.R29_METHODS[:-1]}
        tab[protocol.POPULATION] = per["pop", protocol.POPULATION]
        refs = {ref: per_lc[5, ref] for ref, _ in analysis_tools.R29_REFS_PAIRED}
        S = analysis_tools.r29_metric_entry(tab, refs, "IG_bits", [1, 2])
        self.assertEqual(S[lc]["per_subject"], [1.0, 1.0])
        # SURE(kernel)'s own level keeps its 100 in subject 1; the paired difference does not.
        self.assertAlmostEqual(S[sk]["per_subject"][0], 100.0 / 6)
        self.assertEqual(S[lc]["vs_SK"]["mean"], 1.0)
        # Estimators without exclusions are paired with SURE(kernel) on all cells.
        self.assertAlmostEqual(S["Average"]["vs_SK"]["mean"], -100.0 / 12)
        # A subject without any valid L-curve fit drops out of the L-curve's cohort mean.
        del tab[lc][2]
        S = analysis_tools.r29_metric_entry(tab, refs, "IG_bits", [1, 2])
        self.assertEqual((S[lc]["mean"], S[lc]["n"]), (1.0, 1))

    def test_only_lcurve_failures_are_accepted(self) -> None:
        check = analysis_tools.check_r29_failures
        check({analysis_tools.R29_LC: {"status": "failure", "key": "k1"}}, "unit")
        with self.assertRaisesRegex(RuntimeError, "only L-curve failures"):
            check({"OLS": {"status": "failure", "key": "k3"}}, "unit")


class R29AmplitudeValueTest(unittest.TestCase):
    """The probability, decision and speller metrics define the value of the amplitude."""

    def test_recalibration_slope_stays_nonnegative(self) -> None:
        rs = np.random.RandomState(0)
        y = (rs.rand(200) < 1 / 6).astype(int)
        # Scores against the labels: the constrained optimum is b = 0, a reversal.
        fit = analysis_tools.fit_variant
        _, b, _, rev, diag = fit("scale", -(y - 0.5) * 2 + 0.1 * rs.randn(200), y)
        self.assertEqual((b, rev, diag), (0.0, 1, 0.0))
        # Informative scores: a positive slope, converged.
        _, b, conv, rev, diag = fit("joint", (y - 0.5) * 2 + rs.randn(200), y)
        self.assertTrue(conv, f"joint fit did not converge (b = {b})")
        self.assertGreater(b, 0.0)
        self.assertEqual(rev, 0)
        self.assertEqual(diag, b)
        # Separated scores: the slope diverges and the fit is flagged as not converged.
        _, b, conv, _, diag = fit("scale", np.where(y == 1, 1.0, -1.0), y)
        self.assertFalse(conv)
        self.assertTrue(math.isnan(diag))
        # A numerically negative optimum is a counted calibration failure, not an error.
        y = (np.arange(288) % 6 == 0).astype(int)
        L = 1e7 + np.random.RandomState(0).normal(size=288) + 0.5 * y
        _, b, conv, rev, diag = fit("joint", L, y)
        self.assertEqual((b, conv, rev), (0.0, False, 0))
        self.assertTrue(math.isnan(diag))

    def test_best_threshold_accuracy_equals_a_brute_force_scan(self) -> None:
        rs = np.random.RandomState(1)
        y = (rs.rand(300) < 0.3).astype(int)
        L = y + rs.randn(300)
        out = analysis_tools.decisions(L, y)
        for name, prior in analysis_tools.R29_PRIORS.items():
            best = max(
                prior * np.mean(L[y == 1] >= c) + (1 - prior) * np.mean(L[y == 0] < c)
                for c in np.r_[np.inf, np.sort(L)]
            )
            self.assertAlmostEqual(out[f"best_{name}"], best)
            self.assertGreaterEqual(out[f"regret_{name}"], 0.0)

    def test_information_gain_is_zero_for_prior_only_probabilities(self) -> None:
        y = np.r_[np.ones(10), np.zeros(50)].astype(int)
        ig = analysis_tools.event_metrics(np.zeros(60), y)["IG_bits"]
        self.assertAlmostEqual(ig, 0.0)

    def test_speller_finds_the_flashed_character(self) -> None:
        # 6 characters x 8 sequences x 12 flashes; character c's target codes are 1+c, 7+c.
        codes = np.tile(np.arange(1, 13), 48)
        idx = np.arange(576)
        trow, tcol = np.arange(1, 7), np.arange(7, 13)
        mp = {"codes": codes, "char": idx // 96, "seq": (idx % 96) // 12}
        mp |= {"trow": trow, "tcol": tcol}
        target = (codes == trow[idx // 96]) | (codes == tcol[idx // 96])
        sp = analysis_tools.speller(np.where(target, 1.0, -1.0), mp)
        self.assertEqual(sp["acc"], [6.0] * 8)  # every character, every sequence
        self.assertLess(float(np.max(np.diff(sp["ll"]))), 0.0)  # evidence accumulates
        # One sequence: row + column scores are 2 for the true cell, 0 for its 10
        # row/column neighbours and -2 for the other 25 cells.
        p = scipy.special.softmax(np.r_[2.0, np.zeros(10), np.full(25, -2.0)])
        self.assertAlmostEqual(sp["pmax"][0] / 6, float(p.max()))


class R29DirectionCheckTest(unittest.TestCase):
    """The direction/amplitude reading rests on two exact identities and on the
    R4(a)/R5(b) handling of excluded fits."""

    def test_split_components_match_the_closed_form(self) -> None:
        est, at = protocol.ESTIMATORS, analysis_tools
        # D_R = 2 e1; a fit's D = 2 a (c e1 + sqrt(1 - c^2) e2), c < 0 for five estimators.
        R = np.zeros((2, 102, 16))
        R[1, 0, 0] = 2.0
        cases = {m: [(1 + i / 4, 0.9 - 0.2 * i)] * 2 for i, m in enumerate(est)}
        cases["SURE(kernel)"] = [(1.5, -0.3), (0.5, 0.6)]  # differs between repetitions
        A = np.full((1, 1, 2, len(est), len(protocol.REFERENCES), 6), np.nan)
        for m, reps in cases.items():
            for r, (a, c) in enumerate(reps):
                T = np.zeros((2, 102, 16))
                T[1, 0, 0], T[1, 1, 0] = 2 * a * c, 2 * a * math.sqrt(1 - c * c)
                A[0, 0, r, est.index(m)] = at.fit_metrics(T, R)
        A[0, 0, 1, est.index(at.R29_LC)] = np.nan  # an excluded L-curve fit
        out = at.amp_direction_split(A, [4.0], [5])

        def parts(a: float, c: float) -> NDArray:
            cp = max(c, 0.0)  # direction, scale; c < 0: the whole |D_R|^2 is direction
            return np.array([4 * (1 - cp**2), 4 * (a * a - 2 * a * c + cp**2)])

        for m in at.R29_SPLIT_METHODS:
            reps = (
                [0] if m == at.R29_LC else [0, 1]
            )  # R3: SURE(kernel) on the same reps
            sk = np.mean([parts(*cases["SURE(kernel)"][r]) for r in reps], axis=0)
            want = 0.5 * (parts(*cases[m][0]) - sk)
            e = out[f"K05|{m}"]
            got = [e["gap_D_direction_part_mean"], e["gap_D_amplitude_part_mean"]]
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
            self.assertEqual(e["n_subjects"], 1)
        # The split adds up to ||D - D_R||^2, and tot2 = 2 ec2 + 0.5 ed2.
        tot2, ec2, ed2 = A[0, 0, 0, :, 0, :3].T
        np.testing.assert_allclose(tot2, 2 * ec2 + 0.5 * ed2, rtol=1e-12)
        for i, m in enumerate(est):
            self.assertAlmostEqual(float(parts(*cases[m][0]).sum()), ed2[i], places=10)

    def test_r4a_centres_over_the_valid_estimators_of_each_cell(self) -> None:
        est = protocol.ESTIMATORS
        b = np.random.RandomState(3).rand(1, 3, 2, len(est), 5)
        b[0, 0, 0, est.index(analysis_tools.R29_LC)] = np.nan
        idx = [est.index(m) for m in analysis_tools.R29_RANK_EST]
        (r,), (n,) = analysis_tools.pearson_r4a(b, idx, 1)
        xs, ys = [], []
        for f in range(3):
            for rep in range(2):
                ok = [j for j in idx if np.isfinite(b[0, f, rep, j, 0])]
                xs += list(b[0, f, rep, ok, 1] - b[0, f, rep, ok, 1].mean())
                ys += list(b[0, f, rep, ok, 0] - b[0, f, rep, ok, 0].mean())
        self.assertEqual(n, 6 * len(idx) - 1)
        self.assertAlmostEqual(r, float(np.corrcoef(xs, ys)[0, 1]))

    def test_r5b_uses_the_cells_common_to_the_estimator_set(self) -> None:
        est = protocol.ESTIMATORS
        b = np.random.RandomState(4).rand(1, 3, 2, len(est), 5)
        b[0, 2, 1, est.index(analysis_tools.R29_LC)] = np.nan
        idx = [est.index(m) for m in analysis_tools.R29_RANK_EST]
        subj_mean = np.nanmean(b, axis=(1, 2))
        (rho,) = analysis_tools.spearman_r5b(b, idx, 1, subj_mean)
        common = b[0].reshape(6, len(est), 5)[:5].mean(axis=0)
        want = scipy.stats.spearmanr(common[idx, 1], common[idx, 0])[0]
        self.assertAlmostEqual(rho, float(want))
