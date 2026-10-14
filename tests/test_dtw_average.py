# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Filtered DTW average (Molina et al. 2024): DTW, warp, filter, and estimator."""

from __future__ import annotations

import importlib
import itertools
import unittest
from typing import TYPE_CHECKING

import numpy as np
import scipy
from numpy.typing import NDArray

if TYPE_CHECKING:
    import dtw_average as m
else:
    m = importlib.import_module("dtw_average")

PathList = list[tuple[int, int]]


def all_paths(n: int, k: int) -> list[PathList]:
    """Every monotone path (0,0)->(n-1,k-1) with steps (1,1), (1,0), (0,1)."""
    out: list[PathList] = []

    def rec(path: PathList) -> None:
        i, j = path[-1]
        if (i, j) == (n - 1, k - 1):
            out.append(list(path))
            return
        for di, dj in ((1, 1), (1, 0), (0, 1)):
            if i + di < n and j + dj < k:
                path.append((i + di, j + dj))
                rec(path)
                path.pop()

    rec([(0, 0)])
    return out


def path_cost(c: NDArray, path: PathList | NDArray) -> float:
    return float(sum(c[i, j] for i, j in path))


def as_tuples(path: NDArray) -> PathList:
    return [(int(i), int(j)) for i, j in path]


def bump(n: int, center: float, width: float) -> NDArray:
    t = np.arange(n, dtype=float)
    return np.exp(-0.5 * ((t - center) / width) ** 2)


def frozen_design() -> NDArray:
    return m.design_lowpass(128.0, 20.0, 26.6, 60.0)


class DTWPathTest(unittest.TestCase):
    def test_matches_brute_force_on_short_signals(self) -> None:
        # The DP must find the true minimum over all admissible paths, for both
        # costs and unequal lengths; otherwise the warped trials are not Molina's.
        rng = np.random.default_rng(1)
        for n, k in itertools.product(range(1, 6), range(1, 6)):
            paths = all_paths(n, k)
            for cost in ("absolute", "squared"):
                for _ in range(5):
                    r, s = rng.standard_normal(n), rng.standard_normal(k)
                    c = m.local_cost(r, s, cost)
                    best = min(path_cost(c, p) for p in paths)
                    path, total = m.dtw_path(r, s, cost)
                    self.assertAlmostEqual(total, best, places=12)
                    self.assertAlmostEqual(path_cost(c, path), best, places=12)
                    self.assertIn(as_tuples(path), paths)

    def test_path_properties(self) -> None:
        rng = np.random.default_rng(2)
        for _ in range(20):
            r, s = rng.standard_normal(102), rng.standard_normal(102)
            path, _ = m.dtw_path(r, s)
            self.assertEqual(tuple(path[0]), (0, 0))
            self.assertEqual(tuple(path[-1]), (101, 101))
            steps = {tuple(d) for d in np.diff(path, axis=0)}
            self.assertLessEqual(steps, {(1, 1), (1, 0), (0, 1)})
            # Every reference and trial index is visited.
            np.testing.assert_array_equal(np.unique(path[:, 0]), np.arange(102))
            np.testing.assert_array_equal(np.unique(path[:, 1]), np.arange(102))

    def test_tie_break_is_backtracking_priority(self) -> None:
        # Among all minimum-cost paths, the chosen one is greedy backtracking
        # with priority diagonal > (1,0) > (0,1): brute force on integer
        # signals, which create many exact ties.
        rng = np.random.default_rng(3)
        prio = {(1, 1): 0, (1, 0): 1, (0, 1): 2}

        def backward_priorities(p: PathList) -> list[int]:
            return [
                prio[p[t][0] - p[t - 1][0], p[t][1] - p[t - 1][1]]
                for t in range(len(p) - 1, 0, -1)
            ]

        for _ in range(200):
            n, k = int(rng.integers(2, 6)), int(rng.integers(2, 6))
            r = rng.integers(0, 3, n).astype(float)
            s = rng.integers(0, 3, k).astype(float)
            c = m.local_cost(r, s, "absolute")
            costs = [(path_cost(c, p), p) for p in all_paths(n, k)]
            best = min(v for v, _ in costs)
            optimal = [p for v, p in costs if v == best]
            expected = min(optimal, key=backward_priorities)
            path, _ = m.dtw_path(r, s)
            self.assertEqual(as_tuples(path), expected)


class RestrictedWarpTest(unittest.TestCase):
    def test_keeps_first_match_per_reference_index(self) -> None:
        # (0,1) and (2,3) are reached by (0,1) steps and are dropped.
        path = np.array([(0, 0), (0, 1), (1, 2), (2, 2), (2, 3), (3, 4), (4, 4)])
        s = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        np.testing.assert_array_equal(
            m.restricted_warp(s, path, 5), [10.0, 12.0, 12.0, 14.0, 14.0]
        )

    def test_output_length_is_reference_length(self) -> None:
        rng = np.random.default_rng(4)
        for _ in range(50):
            r, s = rng.standard_normal(102), rng.standard_normal(102)
            path, _ = m.dtw_path(r, s)
            w = m.restricted_warp(s, path, 102)
            self.assertEqual(w.shape, (102,))
            # The warped trial is made of trial samples only.
            self.assertLessEqual(set(w.tolist()), set(s.tolist()))

    def test_invalid_path_fails_loudly(self) -> None:
        s = np.array([10.0, 20.0, 30.0])
        bad = (
            ([(0, 0), (0, 1), (0, 2)], "path must run from"),  # misses i = 2
            ([(0, -1), (1, -1), (2, -1)], "path must run from"),  # negative j
            ([(0, 0), (1, 2), (2, 2)], "a step other than"),  # (1,2) jump
            ([(0, 0), (1, 1)], "path must run from"),  # stops early
            ([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)], "integer array"),
        )
        for path, message in bad:
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, message):
                m.restricted_warp(s, np.array(path), 3)

    def test_unrestricted_path_long_excursion(self) -> None:
        # No global window: a pulse at index 20 in the reference and at 80 in
        # the trial aligns with zero cost (60-sample excursion); a banded DTW
        # would not.
        r = np.zeros(102)
        s = np.zeros(102)
        pulse = np.hanning(9)[1:-1]
        r[17:24] = pulse
        s[77:84] = pulse
        path, total = m.dtw_path(r, s)
        self.assertEqual(total, 0.0)
        self.assertEqual(int(np.abs(path[:, 0] - path[:, 1]).max()), 60)
        np.testing.assert_array_equal(m.restricted_warp(s, path, 102), r)

    def test_integer_shift_is_undone_away_from_edges(self) -> None:
        # Exact positive control. A compact pulse on an exactly zero background,
        # delayed by d samples: a zero-cost path exists, and every zero-cost
        # path maps each reference index to a trial sample of equal value, so
        # the warped trial equals the reference exactly.
        r = np.zeros(102)
        r[44:57] = np.hanning(15)[1:-1]  # nonzero only on 44..56
        for d in (-6, -3, 3, 6):
            s = np.roll(r, d)  # the wrapped samples are zeros
            path, total = m.dtw_path(r, s)
            self.assertEqual(total, 0.0)
            np.testing.assert_array_equal(m.restricted_warp(s, path, 102), r)
            self.assertGreater(np.abs(s - r).max(), 0.1)  # the shift was real


class FilterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.taps = np.asarray(m.TAPS)

    def test_frozen_taps_equal_the_scipy_design(self) -> None:
        # Frozen reconstruction: firwin(73, 23.3 Hz) with the Kaiser beta of
        # kaiserord(60 dB, 6.6 Hz transition at 128 Hz).
        numtaps, beta = scipy.signal.kaiserord(60.0, 6.6 / 64.0)
        self.assertEqual(numtaps, 72)  # raised to 73 for an integer delay
        self.assertAlmostEqual(beta / m.KAISER_BETA, 1.0, delta=1e-6)
        ref = scipy.signal.firwin(73, 23.3, window=("kaiser", beta), fs=128.0)
        np.testing.assert_allclose(self.taps, ref, rtol=0, atol=1e-15)
        designed = m.design_lowpass(
            m.FS_HZ,
            m.PASSBAND_EDGE_HZ,
            m.STOPBAND_EDGE_HZ,
            m.STOPBAND_ATTENUATION_DB,
        )
        np.testing.assert_allclose(self.taps, designed, rtol=0, atol=1e-15)

    def test_reflections_match_numpy_pad(self) -> None:
        x = np.random.default_rng(7).standard_normal(102)
        odd = np.pad(x, (36, 36), mode="reflect", reflect_type="odd")
        np.testing.assert_allclose(
            m.zero_phase_fir(x, self.taps),
            np.convolve(odd, self.taps, mode="valid"),
            rtol=0,
            atol=1e-14,
        )
        even = np.pad(x, (36, 36), mode="reflect")
        np.testing.assert_allclose(
            m.zero_phase_fir(x, self.taps, "even_reflection"),
            np.convolve(even, self.taps, mode="valid"),
            rtol=0,
            atol=1e-14,
        )

    def test_zero_phase(self) -> None:
        # A symmetric bump stays centred (no group delay).
        x = bump(102, 50, 5)
        y = m.zero_phase_fir(x, self.taps)
        self.assertEqual(int(np.argmax(y)), 50)
        np.testing.assert_allclose(y[20:81], x[20:81], atol=1e-3)

    def test_removes_high_frequency(self) -> None:
        n = np.arange(102)
        y = m.zero_phase_fir(np.sin(2 * np.pi * 40 * n / 128), self.taps)
        self.assertLess(np.abs(y[36:66]).max(), 2e-3)


class DTWAverageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.taps = np.asarray(m.TAPS)
        self.epochs = np.random.default_rng(5).standard_normal((5, 102, 3))

    def test_identical_trials_return_reference(self) -> None:
        n = np.arange(102)
        x = np.sin(2 * np.pi * 3 * n / 128) + 0.5 * np.sin(
            2 * np.pi * 11 * n / 128 + 0.3
        )
        epochs = np.repeat(x[None, :, None], 5, axis=0)
        # Identity warp: the plain result is the reference, the filtered one F(it).
        np.testing.assert_allclose(
            m.dtw_average(epochs, None)[:, 0], x, rtol=0, atol=1e-14
        )
        np.testing.assert_allclose(
            m.molina_template(epochs)[:, 0],
            m.zero_phase_fir(x, self.taps),
            rtol=0,
            atol=1e-14,
        )

    def test_single_trial_returns_filtered_trial(self) -> None:
        e = self.epochs[:1]
        np.testing.assert_array_equal(m.dtw_average(e, None), e[0])
        expected = np.stack(
            [m.zero_phase_fir(e[0, :, c], self.taps) for c in range(e.shape[2])],
            axis=1,
        )
        np.testing.assert_array_equal(m.molina_template(e), expected)

    def test_molina_template_applies_the_frozen_settings(self) -> None:
        # The comparator is the filtered average with the absolute cost, the
        # diagonal-first tie order, tie_rtol 1e-10, and odd-reflection edges.
        e = np.round(np.random.default_rng(12).standard_normal((5, 102, 2)), 1)
        explicit = m.dtw_average(
            e,
            self.taps,
            cost="absolute",
            tie_break=("diagonal", "reference_only", "trial_only"),
            edge="odd_reflection",
            tie_rtol=1e-10,
        )
        np.testing.assert_array_equal(m.molina_template(e), explicit)
        np.testing.assert_allclose(
            m.molina_template(e),
            m.dtw_average(e, frozen_design()),
            rtol=0,
            atol=1e-14,
        )
        # Each setting matters on these rounded (tie-rich) epochs.
        variants = {
            "cost": m.dtw_average(e, self.taps, cost="squared"),
            "tie_break": m.dtw_average(
                e, self.taps, tie_break=("trial_only", "reference_only", "diagonal")
            ),
            "tie_rtol": m.dtw_average(e, self.taps, tie_rtol=0.05),
            "edge": m.dtw_average(e, self.taps, edge="zero"),
        }
        for setting, result in variants.items():
            with self.subTest(setting=setting):
                self.assertFalse(np.array_equal(result, explicit))

    def test_equivariances(self) -> None:
        # Data are in volts (~1e-5): the result must not depend on units, sign,
        # offset, or trial order.
        base = m.molina_template(self.epochs)
        for changed, expected in (
            (-self.epochs, -base),
            (self.epochs + 2.5, base + 2.5),
            (self.epochs[[3, 0, 4, 1, 2]], base),
        ):
            np.testing.assert_allclose(
                m.molina_template(changed), expected, rtol=0, atol=1e-12
            )
        exact = m.molina_template(self.epochs * 2.0**-17)
        np.testing.assert_array_equal(exact, base * 2.0**-17)
        np.testing.assert_allclose(
            m.molina_template(self.epochs * 1e-5), base * 1e-5, rtol=0, atol=1e-17
        )
        np.testing.assert_allclose(
            m.molina_template(self.epochs * 1e6), base * 1e6, rtol=0, atol=1e-6
        )

    def test_channels_are_independent(self) -> None:
        full = m.molina_template(self.epochs)
        one = m.molina_template(self.epochs[:, :, 1:2])
        np.testing.assert_array_equal(full[:, 1:2], one)

    def test_jitter_attenuates_less_with_dtw(self) -> None:
        # The paper's qualitative claim: with latency jitter, the DTW average
        # keeps the peak amplitude better than the conventional average.
        epochs = np.stack([bump(102, 50 + d, 4) for d in (-6, -3, 0, 3, 6)])
        plain = epochs.mean(axis=0)
        dtw_f = m.molina_template(epochs[:, :, None])[:, 0]
        self.assertLess(plain.max(), 0.8)
        self.assertGreater(dtw_f.max(), plain.max() + 0.1)
