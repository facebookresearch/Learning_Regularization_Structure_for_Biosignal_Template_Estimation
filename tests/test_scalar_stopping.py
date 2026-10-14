# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for scalar SURE's design-scaled candidate search."""

from __future__ import annotations

import importlib
import unittest
from typing import Any, cast, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

if TYPE_CHECKING:
    import core
else:
    core = importlib.import_module("core")


def scalar_inputs(noise: float = 0.1) -> tuple[NDArray, NDArray, NDArray]:
    template = np.array([0.0, 0.2, 0.7, 1.0, 0.6, -0.2, -0.4, 0.0])
    return core.generate_data(
        {"A": np.column_stack([template, 0.7 * template])},
        I=3,
        noise=noise,
        mean_interval=12,
        rs=np.random.RandomState(19),
    )


def fit_scalar(
    noise_est: str = "pairwise",
    noise: float = 0.1,
    omega: float = 0.5,
) -> tuple[dict[object, NDArray], dict[str, Any]]:
    data, ix, labels = scalar_inputs(noise)
    estimates, info = core.get_events_templates(
        data,
        ix,
        labels,
        half_win=4,
        method="ls",
        opt_reg_temporal=True,
        noise_est=noise_est,
        omega=omega,
    )
    if info is None:
        raise AssertionError("Scalar SURE must return stopping diagnostics")
    return estimates, info


def fit_epoch_pair(
    mean: NDArray, deviation: NDArray
) -> tuple[dict[object, NDArray], dict[str, Any]]:
    """Two nonoverlapping epochs with a known mean and opposite deviations."""
    length, channels = mean.shape
    data = np.vstack([mean + deviation, np.zeros((4, channels)), mean - deviation])
    estimates, info = core.get_events_templates(
        data,
        np.array([length // 2, 3 * length // 2 + 4]),
        np.array(["A", "A"]),
        half_win=length // 2,
        method="ls",
        opt_reg_temporal=True,
    )
    if info is None:
        raise AssertionError("Paired-epoch scalar fit must return diagnostics")
    return estimates, info


def second_difference_penalty(length: int) -> NDArray:
    difference = np.diff(np.eye(length), n=2, axis=1)
    return difference @ difference.T


def legacy_cost(noise_est: str, omega: float, lam: float, sigsq: float) -> float:
    """SURE from the original generalized-eigenvector evaluation of scalar_inputs."""
    data, ix, labels = scalar_inputs()
    design = core.get_design_matrix(data, ix, labels, half_win=4)
    gram, cross = design.T @ design, design.T @ data
    mu, basis = linalg.eigh(gram, gram + second_difference_penalty(8))
    mu = np.clip(mu, 0, 1)
    rotated, ols = basis.T @ cross, np.linalg.solve(gram, cross)
    shape = (len(data), data.shape[1])
    if noise_est == "pairwise":
        return core.SURE_cost_fast(lam, mu, rotated, basis, ols, sigsq, *shape, omega)
    phi = core.estimate_phi(data, ix, 4)
    covariance = core.build_DtSigmaD_ar1(
        ix, labels, np.unique(labels), 4, phi, sigsq, data_length=len(data)
    )
    precomputed = core.precompute_colored_trace(gram, covariance, basis, mu)
    return core.SURE_cost_colored(
        lam, mu, rotated, basis, ols, covariance, *shape, omega, None, None, precomputed
    )


class TestScalarCandidateSearch(unittest.TestCase):
    def test_two_wells_recover_the_lower_minimum(self) -> None:
        # Two signal modes give the actual SURE objective two separated minima.
        # Opposite epoch deviations give pairwise noise variance one and DtD=2I.
        length = 16
        penalty = second_difference_penalty(length)
        _, modes = np.linalg.eigh(penalty)
        mean = np.column_stack(
            [
                np.sqrt(1.01) * modes[:, 2] + np.sqrt(500) * modes[:, -1],
                np.zeros(length),
            ]
        )
        deviation = (
            np.tile([1.0, -1.0], length // 2)[:, None] * np.ones((1, 2)) / np.sqrt(2)
        )
        estimates, info = fit_epoch_pair(mean, deviation)
        lam = cast(float, info["opt_val"])
        base = [row for row in info["search_refinements"] if row["search"] == "base"]
        self.assertEqual(len(base), 2)
        self.assertLess(min(row["lambda"] for row in base), 0.1)
        self.assertGreater(max(row["lambda"] for row in base), 1000)
        self.assertLess(lam, 0.1)
        self.assertLess(info["cost"], max(row["cost"] for row in base) - 200)
        inverse = np.linalg.inv(2 * np.eye(length) + lam * penalty)
        theta = inverse @ (2 * mean)
        score = 0.5 * np.sum(theta**2) - np.sum(mean * theta) + 2 * np.trace(inverse)
        self.assertAlmostEqual(cast(float, info["cost"]), score, places=9)
        np.testing.assert_allclose(estimates["A"], theta, atol=1e-12)
        self.assertIn(
            "global optimum not certified", cast(str, info["optimizer_message"])
        )

    def test_noiseless_epochs_select_the_exact_zero_limit(self) -> None:
        signal = np.tile([0.0, 1.0, -1.0, 0.0], (2, 1)).T
        estimates, info = fit_epoch_pair(signal, np.zeros_like(signal))
        self.assertEqual(info["opt_val"], 0.0)
        self.assertEqual(info["search_selected_from"], "zero")
        self.assertTrue(info["search_near_lower_bound"])
        self.assertEqual(info["cost"], info["search_limit_scores"]["zero"])
        np.testing.assert_allclose(estimates["A"], signal, atol=1e-12)

    def test_infinite_limit_keeps_the_penalty_null_space(self) -> None:
        # A linear mean lies in the penalty null space, so the infinite limit
        # keeps it exactly instead of returning a zero template.
        t = np.arange(8.0)
        mean = np.column_stack([0.5 + 0.1 * t, -0.2 + 0.3 * t])
        deviation = np.tile([1.0, -1.0], 4)[:, None] * np.ones((1, 2)) / np.sqrt(2)
        estimates, info = fit_epoch_pair(mean, deviation)
        self.assertEqual(info["opt_val"], np.inf)
        self.assertEqual(info["search_selected_from"], "infinite")
        self.assertTrue(info["search_near_upper_bound"])
        self.assertEqual(info["cost"], info["search_limit_scores"]["infinite"])
        np.testing.assert_allclose(estimates["A"], mean, atol=1e-12)

    def test_infinite_limit_keeps_each_class_trend(self) -> None:
        t = np.arange(8.0)
        means = {
            "A": np.column_stack([0.5 + 0.1 * t, -0.2 + 0.3 * t]),
            "B": np.column_stack([1.0 - 0.2 * t, 0.4 + 0.05 * t]),
        }
        deviation = np.tile([1.0, -1.0], 4)[:, None] * np.ones((1, 2)) / np.sqrt(2)
        gap = np.zeros((4, 2))
        data = np.vstack(
            [
                means["A"] + deviation,
                gap,
                means["B"] + deviation,
                gap,
                means["A"] - deviation,
                gap,
                means["B"] - deviation,
            ]
        )
        estimates, info = core.get_events_templates(
            data,
            np.array([4, 16, 28, 40]),
            np.array(["A", "B", "A", "B"]),
            half_win=4,
            method="ls",
            opt_reg_temporal=True,
        )
        if info is None:
            raise AssertionError("Scalar SURE must return stopping diagnostics")
        self.assertEqual(info["opt_val"], np.inf)
        for label, mean in means.items():
            with self.subTest(label=label):
                np.testing.assert_allclose(estimates[label], mean, atol=1e-12)

    def test_bounds_extend_six_decades_beyond_penalty_transitions(self) -> None:
        data, ix, labels = scalar_inputs()
        design = core.get_design_matrix(data, ix, labels, half_win=4)
        values = linalg.eigh(
            second_difference_penalty(8), design.T @ design, eigvals_only=True
        )
        positive = values[values > 1e-10 * np.max(values)]
        expected = (
            int(np.floor(-np.log10(np.max(positive)) - 6)),
            int(np.ceil(-np.log10(np.min(positive)) + 6)),
        )
        _, info = fit_scalar()
        self.assertEqual(info["search_log10_bounds"], expected)
        self.assertEqual(
            len(info["search_grid_lambdas"]), 10 * (expected[1] - expected[0]) + 1
        )
        self.assertTrue(info["search_stability_passed"])
        for name in ("denser", "outer"):
            row = info["search_stability"][name]
            self.assertLessEqual(row["score_change_over_ols_energy"], 1e-8)
            self.assertLessEqual(row["template_change_over_ols_norm"], 1e-5)

    def test_objective_matches_the_original_white_and_colored_scores(self) -> None:
        for noise_est in ("pairwise", "ar1_colored"):
            for omega in (0.5, 0.7):
                with self.subTest(noise_est=noise_est, omega=omega):
                    _, info = fit_scalar(noise_est, omega=omega)
                    lambdas = np.r_[0.0, info["search_grid_lambdas"]]
                    scores = np.r_[
                        info["search_limit_scores"]["zero"], info["search_grid_scores"]
                    ]
                    moderate = lambdas <= 1e2
                    expected = [
                        legacy_cost(noise_est, omega, lam, info["noise_var"])
                        for lam in lambdas[moderate]
                    ]
                    np.testing.assert_allclose(scores[moderate], expected, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
