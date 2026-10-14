# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Independent full-covariance references for the focused C3 correction."""

from __future__ import annotations

import importlib
import unittest
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import scipy.linalg as la
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray

if TYPE_CHECKING:
    import core
else:
    core = importlib.import_module("core")


def explicit_design(
    n: int, half_win: int, ix: NDArray, labels: NDArray, events: NDArray
) -> NDArray:
    length = 2 * half_win
    design = np.zeros((n, len(events) * length))
    for center, label in zip(ix, labels):
        if center < half_win or center + half_win > n or label not in events:
            continue
        column = list(events).index(label) * length
        design[center - half_win : center + half_win, column : column + length] += (
            np.eye(length)
        )
    return design


def stationary_acf(coefficients: NDArray, n: int, variance: float) -> NDArray:
    """Use a Lyapunov covariance and state propagation, not production recursion."""
    order = len(coefficients)
    transition = np.zeros((order, order))
    transition[0] = coefficients
    transition[1:, :-1] = np.eye(order - 1)
    innovation = np.zeros((order, order))
    innovation[0, 0] = 1.0
    stationary = la.solve_discrete_lyapunov(transition, innovation)
    state = stationary[:, 0].copy()
    result = np.empty(n)
    for lag in range(n):
        result[lag] = state[0]
        state = transition @ state
    return result * (variance / result[0])


class TestStructuredCovariance(unittest.TestCase):
    def test_ar1_matches_full_reference_across_cutoff_and_boundaries(self) -> None:
        for n, half_win, centers in (
            (96, 6, [6, 13, 55, 62, 90, 91]),
            (760, 50, [50, 110, 440, 510, 710, 711]),
        ):
            ix = np.array(centers)
            labels = np.array(["A", "B", "A", "B", "A", "B"])
            events = np.array(["B", "A"])
            design = explicit_design(n, half_win, ix, labels, events)
            for phi in (0.0, 0.5, 0.98, 0.995, 0.999, -0.6):
                with self.subTest(length=2 * half_win, phi=phi):
                    sigma = 0.7 * la.toeplitz(phi ** np.arange(n))
                    expected = design.T @ sigma @ design
                    actual = core.build_DtSigmaD_ar1(
                        ix, labels, events, half_win, phi, 0.7, data_length=n
                    )
                    assert_allclose(actual, expected, rtol=5e-12, atol=5e-14)

    def test_persistent_ar1_is_psd_at_small_and_large_variance(self) -> None:
        ix = np.array([6, 13, 55, 62, 90])
        labels = np.array(["A", "B", "A", "B", "A"])
        events = np.array(["A", "B"])
        for variance in (1e-12, 1.0, 1e12):
            matrix = core.build_DtSigmaD_ar1(
                ix, labels, events, 6, 0.995, variance, data_length=96
            )
            eigenvalues = la.eigvalsh(matrix)
            tolerance = len(matrix) * np.finfo(float).eps * eigenvalues[-1]
            self.assertGreaterEqual(eigenvalues[0], -tolerance)
            # Transposed event-pair sums can differ at floating-point roundoff.
            assert_allclose(matrix, matrix.T, rtol=0, atol=tolerance)

    def test_ar1_does_not_allocate_a_recording_length_lookup(self) -> None:
        ix = np.array([3, 1_000_000_003])
        labels = np.array(["A", "A"])
        events = np.array(["A"])
        phi = 0.999999999
        # Only these 12 rows of a billion-sample recording have nonzero D.
        times = (ix[:, None] - 3 + np.arange(6)).ravel()
        covariance = phi ** np.abs(times[:, None] - times[None, :])
        design = np.vstack([np.eye(6), np.eye(6)])
        expected = design.T @ covariance @ design
        actual = core.build_DtSigmaD_ar1(
            ix, labels, events, 3, phi, 1.0, data_length=1_000_000_006
        )
        assert_allclose(actual, expected, rtol=5e-12, atol=5e-14)

    def test_event_order_class_order_and_single_class(self) -> None:
        ix = np.array([6, 13, 55, 62, 90, 91])
        labels = np.array(["A", "B", "A", "B", "A", "B"])
        for events in (np.array(["A", "B"]), np.array(["B", "A"]), np.array(["A"])):
            design = explicit_design(96, 6, ix, labels, events)
            expected = design.T @ la.toeplitz(0.995 ** np.arange(96)) @ design
            for order in (np.arange(len(ix)), np.arange(len(ix))[::-1]):
                actual = core.build_DtSigmaD_ar1(
                    ix[order], labels[order], events, 6, 0.995, 1.0, data_length=96
                )
                assert_allclose(actual, expected, rtol=5e-12, atol=5e-14)

    def test_empty_and_excluded_events_give_zero_covariance(self) -> None:
        for ix in (np.array([], dtype=int), np.array([95, 96])):
            labels = np.full(len(ix), "A")
            actual = core.build_DtSigmaD_ar1(
                ix, labels, np.array(["A"]), 6, 0.995, 1.0, data_length=96
            )
            assert_array_equal(actual, np.zeros((12, 12)))

    def test_ar3_full_horizon_matches_stationary_reference(self) -> None:
        ix = np.array([6, 13, 55, 62, 90, 91])
        labels = np.array(["A", "B", "A", "B", "A", "B"])
        events = np.array(["A", "B"])
        design = explicit_design(96, 6, ix, labels, events)
        for coefficients in ([0.7, 0.2, 0.08], [0.9, -0.5, 0.12]):
            acf = stationary_acf(np.array(coefficients), 96, 1.3)
            expected = design.T @ la.toeplitz(acf) @ design
            actual = core.build_DtSigmaD_acf(ix, labels, events, 6, acf, data_length=96)
            assert_allclose(actual, expected, rtol=5e-12, atol=5e-14)

    def test_finite_support_acf_retains_exact_zero_padding(self) -> None:
        ix = np.array([6, 13, 55, 62, 90, 91])
        labels = np.array(["A", "B", "A", "B", "A", "B"])
        events = np.array(["B", "A"])
        coefficients = np.array([1.0, 0.5, -0.2])
        acf = np.correlate(coefficients, coefficients, mode="full")[2:]
        full = np.pad(acf, (0, 96 - len(acf)))
        design = explicit_design(96, 6, ix, labels, events)
        expected = design.T @ la.toeplitz(full) @ design
        actual = core.build_DtSigmaD_acf(ix, labels, events, 6, acf, data_length=96)
        assert_array_equal(actual, expected)


class TestArHorizon(unittest.TestCase):
    def test_extension_preserves_fitted_moments_and_default_prefix(self) -> None:
        data = np.random.RandomState(19).standard_normal((96, 3))
        ix = np.array([6, 13, 55, 62, 90])
        default = core.estimate_acf_ar(data, ix, 6)
        expanded = core.estimate_acf_ar(data, ix, 6, max_lag=95)
        self.assertEqual(len(default), 25)
        self.assertEqual(len(expanded), 96)
        assert_array_equal(expanded[:25], default)

        # The original fit uses seven-sample, per-channel demeaned baselines.
        moments = np.zeros(4)
        count = 0
        for center in ix:
            segment = data[center - 6 : center + 1]
            for channel in segment.T:
                centered = channel - channel.mean()
                moments += np.array(
                    [np.dot(centered[: 7 - lag], centered[lag:]) for lag in range(4)]
                )
                count += 7
        assert_allclose(default[:4], moments / count, rtol=5e-15, atol=5e-16)
        coefficients = la.solve(la.toeplitz(default[:3]), default[1:4])
        expected = stationary_acf(coefficients, 96, float(default[0]))
        assert_allclose(expanded, expected, rtol=5e-12, atol=5e-14)

    def test_both_sure_callers_request_all_retained_event_lags(self) -> None:
        data = np.random.RandomState(31).standard_normal((120, 2))
        ix = np.array([16, 23, 89, 114, 115])
        labels = np.array(["A", "B", "A", "B", "A"])
        original = core.estimate_acf_ar
        requested: list[int | None] = []

        def observe(
            data: NDArray,
            ix: NDArray,
            half_win: int,
            ar_order: int = 3,
            *,
            max_lag: int | None = None,
        ) -> NDArray:
            requested.append(max_lag)
            return original(data, ix, half_win, ar_order, max_lag=max_lag)

        for vec_mode in (False, True):
            requested.clear()
            with patch.object(core, core.estimate_acf_ar.__name__, side_effect=observe):
                core.get_events_templates(
                    data,
                    ix,
                    labels,
                    half_win=6,
                    method="ls",
                    opt_reg_temporal=True,
                    vec_mode=vec_mode,
                    vlen=3,
                    noise_est="ar3_colored",
                )
            # The last retained center is 114, not 115; 114 - 16 + 12 - 1 = 109.
            self.assertEqual(requested, [109])


class TestColoredSureReference(unittest.TestCase):
    def test_corrected_covariance_matches_observation_space_risk(self) -> None:
        ix = np.array([6, 13, 55, 62, 90])
        labels = np.array(["A", "B", "A", "B", "A"])
        events = np.array(["A", "B"])
        design = explicit_design(96, 6, ix, labels, events)
        sigma = la.toeplitz(0.995 ** np.arange(96))
        covariance = core.build_DtSigmaD_ar1(
            ix, labels, events, 6, 0.995, 1.0, data_length=96
        )
        gram = design.T @ design
        data = np.random.RandomState(42).standard_normal((96, 3))
        ols = la.solve(gram, design.T, assume_a="pos")
        theta_ml = ols @ data
        difference = np.diff(np.eye(12), n=2, axis=0)
        penalty = np.kron(np.eye(2), difference.T @ difference)
        mu, basis = la.eigh(gram, gram + penalty)
        rotated = basis.T @ design.T @ data
        precomp = core.precompute_colored_trace(gram, covariance, basis, mu)
        for strength in (0.01, 1.0, 100.0):
            estimator = la.solve(gram + strength * penalty, design.T, assume_a="pos")
            fitted = estimator @ data
            expected = (
                0.5 * np.sum(fitted**2)
                - np.sum(fitted * theta_ml)
                + data.shape[1] * np.sum((estimator @ sigma) * ols)
            )
            scalar = core.SURE_cost_colored(
                strength,
                mu,
                rotated,
                basis,
                theta_ml,
                covariance,
                len(data),
                data.shape[1],
                _precomp=precomp,
            )
            kernel, _ = core._KernelSUREObjective(
                gram, theta_ml, data, 1.0, 2, DtSigmaD=covariance
            ).value_gradient(np.sqrt(strength) * np.array([1.0, -2.0, 1.0]))
            assert_allclose([scalar, kernel], expected, rtol=5e-10, atol=5e-10)


if __name__ == "__main__":
    unittest.main()
