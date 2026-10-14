# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Independent corner, range, and stability checks for the L-curve selector."""

from __future__ import annotations

import importlib
import unittest
import warnings
from typing import cast, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

if TYPE_CHECKING:
    import core
else:
    core = importlib.import_module("core")


def direct_curvature(
    gram: NDArray,
    penalty: NDArray,
    data: NDArray,
    design: NDArray,
    lam: float,
    step: float = 1e-3,
) -> float:
    """Signed curvature (counterclockwise positive) from direct solves and
    five-point log-lambda differences."""
    residual, regularity = [], []
    for value in np.exp(np.log(lam) + step * np.arange(-2, 3)):
        theta = np.linalg.solve(gram + value * penalty, design.T @ data)
        residual.append(np.sum((data - design @ theta) ** 2))
        regularity.append(np.sum(theta * (penalty @ theta)))
    slopes = []
    for f in (np.log(residual), np.log(regularity)):
        first = (f[0] - 8 * f[1] + 8 * f[3] - f[4]) / (12 * step)
        second = (-f[0] + 16 * f[1] - 30 * f[2] + 16 * f[3] - f[4]) / (12 * step**2)
        slopes.append((first, second))
    (dx, ddx), (dy, ddy) = slopes
    return float((dx * ddy - dy * ddx) / (dx**2 + dy**2) ** 1.5)


def unit_ridge_curvature(lam: float) -> float:
    """Signed curvature of the D = S = X = 1 L-curve: negative (concave) for every lambda."""
    return -lam * (1 + lam) / (2 * (1 + lam**2) ** 1.5)


def three_component_problem(
    scale: float = 1.0,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Standard-form Tikhonov on singular values 1, 0.1, 0.01 with Picard coefficients
    and 1e-2 white noise: a convex L-corner near lambda = 6.08e-4 / scale."""
    design = np.diag([1.0, 0.1, 0.01])
    rng = np.random.RandomState(0)
    data = design @ np.sqrt(np.diag(design))[:, None] + 0.01 * rng.randn(3, 1)
    return design.T @ design, design.T @ data, scale * np.eye(3), data


def dense_signed_curvature(
    gram: NDArray, penalty: NDArray, data: NDArray, design: NDArray, grid: NDArray
) -> NDArray:
    return np.array(
        [direct_curvature(gram, penalty, data, design, lam) for lam in grid]
    )


def smoothing_problem(
    relative_noise: float, seed: int
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """An ill-posed first-kind problem written from scratch: a Gaussian smoothing
    integral operator (width 0.04, midpoint rule, 32 points), a smooth solution,
    white noise of the given relative size, and a second-difference penalty."""
    n = 32
    t = (np.arange(n) + 0.5) / n
    width = 0.04
    operator = np.exp(-((t[:, None] - t[None, :]) ** 2) / (2 * width**2)) / (
        n * np.sqrt(2 * np.pi) * width
    )
    solution = np.sin(np.pi * t) + 0.5 * np.sin(3 * np.pi * t)
    exact = operator @ solution
    noise = np.random.RandomState(seed).normal(size=n)
    noise *= relative_noise * np.linalg.norm(exact) / np.linalg.norm(noise)
    difference = np.diff(np.eye(n), n=2, axis=0)
    return operator, (exact + noise)[:, None], difference.T @ difference, solution


class TestLcurveSelection(unittest.TestCase):
    def test_identity_ridge_is_concave_and_has_no_corner(self) -> None:
        # rho = mu^2/(1+mu)^2 and eta = 1/(1+mu)^2 give a log-log curve that is
        # concave for every mu (Reginska 1996): its largest bend, at mu = 1, is
        # clockwise, and no convex corner exists.
        grid = np.logspace(-6, 6, 241)
        self.assertTrue(all(unit_ridge_curvature(lam) < 0 for lam in grid))
        one = np.ones((1, 1))
        self.assertAlmostEqual(
            direct_curvature(one, one, one, one, 1.0),
            unit_ridge_curvature(1.0),
            places=6,
        )
        with self.assertRaisesRegex(ValueError, "no positive interior curvature peak"):
            core.select_lambda_lcurve(one, one, one, one, one)

    def test_three_component_problem_selects_its_convex_corner(self) -> None:
        gram, cross, penalty, data = three_component_problem()
        design = np.sqrt(gram)
        lam, info = core.select_lambda_lcurve(gram, cross, penalty, data, design)
        grid = np.logspace(-8, 1, 901)
        curvature = dense_signed_curvature(gram, penalty, data, design, grid)
        self.assertLess(abs(np.log10(lam / grid[np.argmax(curvature)])), 0.01)
        self.assertGreater(np.max(curvature), 0)
        self.assertAlmostEqual(
            cast(float, info["selected_curvature"])
            / direct_curvature(gram, penalty, data, design, lam),
            1.0,
            places=6,
        )
        self.assertEqual(info["grid_log10_envelope"], (-10, 6))
        self.assertTrue(info["positive_corner"])
        self.assertTrue(info["stability_passed"])

    def test_corner_moves_with_penalty_scale_and_stays_interior(self) -> None:
        # The legacy fixed grid placed these corners outside or at its edges.
        gram, cross, penalty, data = three_component_problem()
        reference, _ = core.select_lambda_lcurve(
            gram, cross, penalty, data, np.sqrt(gram)
        )
        # 1e-9 puts the corner at 6.1e5, above the legacy 1e5 limit.
        for scale, envelope in ((1e3, (-13, 3)), (1e-6, (-4, 12)), (1e-9, (-1, 15))):
            with self.subTest(scale=scale):
                lam, info = core.select_lambda_lcurve(
                    gram, cross, scale * penalty, data, np.sqrt(gram)
                )
                self.assertAlmostEqual(lam * scale / reference, 1.0, places=6)
                self.assertEqual(info["grid_log10_envelope"], envelope)
                self.assertFalse(info["at_first_eligible"])
                self.assertFalse(info["at_last_eligible"])

    def test_noncommuting_system_matches_independent_direct_solves(self) -> None:
        rng = np.random.RandomState(19)
        design = rng.normal(size=(18, 4))
        coefficients = np.array([[1.0, 0.5], [0.8, -0.3], [0.2, 0.1], [0.05, 0.02]])
        data = design @ coefficients + 0.2 * rng.normal(size=(18, 2))
        gram = design.T @ design
        penalty = np.diag([0.0, 0.2, 1.0, 5.0])
        lam, info = core.select_lambda_lcurve(
            gram, design.T @ data, penalty, data, design
        )
        expected = direct_curvature(gram, penalty, data, design, lam)
        self.assertAlmostEqual(
            cast(float, info["selected_curvature"]) / expected, 1.0, places=6
        )
        # Refinement never returns less curvature than the positive grid peak
        # it starts from, so the stability ratio has a positive denominator.
        self.assertGreater(cast(float, info["selected_grid_curvature"]), 0)
        self.assertGreaterEqual(
            cast(float, info["selected_curvature"]),
            cast(float, info["selected_grid_curvature"]),
        )
        result = optimize.minimize_scalar(
            lambda t: -direct_curvature(gram, penalty, data, design, np.exp(t)),
            bounds=(np.log(lam) - 0.5, np.log(lam) + 0.5),
            method="bounded",
            options={"xatol": 1e-8},
        )
        # Stencil roundoff perturbs the argmax; reuse the curvature tolerance above.
        self.assertGreaterEqual(expected, -float(result.fun) * (1 - 5e-7))
        self.assertTrue(info["stability_passed"])

    def test_clockwise_end_bend_does_not_win(self) -> None:
        # Toward large lambda the curve bends clockwise into its vertical
        # asymptote (the template approaching the penalty null space). That
        # concave bend is negative here and must never be selected.
        gram, cross, penalty, data = three_component_problem()
        design = np.sqrt(gram)
        grid = np.logspace(-8, 1, 901)
        curvature = dense_signed_curvature(gram, penalty, data, design, grid)
        corner, bend = grid[np.argmax(curvature)], grid[np.argmin(curvature)]
        self.assertLess(np.min(curvature), 0)
        self.assertGreater(bend, 10 * corner)
        lam, info = core.select_lambda_lcurve(gram, cross, penalty, data, design)
        self.assertLess(abs(np.log(lam / corner)), 0.01)
        self.assertGreater(abs(np.log(lam / bend)), 1.0)
        self.assertGreater(cast(float, info["selected_curvature"]), 0)

    def test_tiny_data_give_the_same_corner_without_warnings(self) -> None:
        # Scaling the data scales both norms; the log-log curve only shifts.
        gram, cross, penalty, data = three_component_problem()
        design = np.sqrt(gram)
        reference, _ = core.select_lambda_lcurve(gram, cross, penalty, data, design)
        tiny = 1.1e-15
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lam, info = core.select_lambda_lcurve(
                gram, tiny * cross, penalty, tiny * data, design
            )
        self.assertAlmostEqual(lam / reference, 1.0, places=5)
        self.assertEqual(info["nonfinite_eligible_curvatures"], 0)

    def test_all_nonfinite_curvatures_raise(self) -> None:
        one = np.ones((1, 1))
        data = 1e308 * one
        with (
            np.errstate(over="ignore", invalid="ignore"),
            self.assertRaisesRegex(ValueError, "no finite curvature among 35 eligible"),
        ):
            core.select_lambda_lcurve(one, data, one, data, one, n_lambdas=30)

    def test_fit_without_penalized_variation_has_no_corner(self) -> None:
        # Every lambda gives the same estimate; roundoff in the penalized
        # modes would otherwise produce a plausible-looking corner. At 128
        # samples the eigendecomposed penalty leaks about 5e-10 of the trend
        # into penalized modes.
        for length in (8, 128):
            design = np.eye(length)
            difference = np.diff(np.eye(length), n=2, axis=1)
            trend = (0.5 + np.arange(length) / (2 * length))[:, None]
            for name, data in (("linear trend", trend), ("zero data", 0 * trend)):
                with (
                    self.subTest(length=length, data=name),
                    self.assertRaisesRegex(ValueError, "L-curve is undefined"),
                ):
                    core.select_lambda_lcurve(
                        design,
                        design.T @ data,
                        difference @ difference.T,
                        data,
                        design,
                    )

    def test_variation_threshold_separates_resolved_and_constant_paths(self) -> None:
        # An unpenalized unit coordinate next to the three-component problem,
        # whose OLS template is scaled to 2e-5 of the template norm: a resolvable convex corner at
        # the unscaled problem's lambda. At 5e-6 every lambda gives the same
        # template within the 1e-5 tolerance.
        gram3, _, penalty3, data3 = three_component_problem()
        reference, _ = core.select_lambda_lcurve(
            gram3, np.sqrt(gram3) @ data3, penalty3, data3, np.sqrt(gram3)
        )
        design = np.diag([1.0, 1.0, 0.1, 0.01])
        penalty = np.diag([0.0, 1.0, 1.0, 1.0])
        for size, resolvable in ((2e-5, True), (5e-6, False)):
            # Scaled so that the penalized coordinates' OLS template has norm `size`.
            ols3 = np.linalg.solve(gram3, np.sqrt(gram3) @ data3)
            data = np.vstack([np.ones((1, 1)), size * data3 / np.linalg.norm(ols3)])
            args = (design.T @ design, design.T @ data, penalty, data, design)
            with self.subTest(size=size):
                if resolvable:
                    lam, info = core.select_lambda_lcurve(*args)
                    self.assertAlmostEqual(lam / reference, 1.0, places=5)
                    self.assertTrue(info["stability_passed"])
                else:
                    with self.assertRaisesRegex(ValueError, "L-curve is undefined"):
                        core.select_lambda_lcurve(*args)

    def test_endpoint_cancellation_does_not_hide_an_interior_corner(self) -> None:
        # The zero and infinite limits differ by 1e-6 of the template norm,
        # but intermediate lambdas move it by up to 8e-5.
        design = np.array([[1.0, -100.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]])
        factor = np.array([[1.0, -100.0, 0.0], [0.0, 1000.0, 0.0]])
        data = np.array([[1e-4], [-1e-4], [1.0]])
        gram, penalty = design.T @ design, factor.T @ factor
        endpoints = np.linalg.solve(gram, design.T @ data) - np.array([[0], [0], [1]])
        self.assertLess(np.linalg.norm(endpoints), 1e-5)
        lam, info = core.select_lambda_lcurve(
            gram, design.T @ data, penalty, data, design
        )
        expected = direct_curvature(gram, penalty, data, design, lam)
        self.assertAlmostEqual(
            cast(float, info["selected_curvature"]) / expected, 1.0, places=4
        )
        self.assertTrue(info["stability_passed"])

    def test_estimator_templates_match_a_direct_solve_at_the_corner(self) -> None:
        template = np.array([0.0, 0.3, 0.9, 1.0, 0.4, -0.3, -0.2, 0.0])
        data, ix, labels = core.generate_data(
            {"A": np.column_stack([template, 0.5 * template])},
            I=16,
            noise=0.2,
            mean_interval=12,
            rs=np.random.RandomState(7),
        )
        estimates, info = core.get_events_templates(
            data, ix, labels, half_win=4, method="ls", reg_method="lcurve"
        )
        if info is None:
            raise AssertionError("L-curve must return selection diagnostics")
        design = core.get_design_matrix(data, ix, labels, half_win=4)
        difference = np.diff(np.eye(8), n=2, axis=1)
        system = design.T @ design + cast(float, info["opt_val"]) * (
            difference @ difference.T
        )
        direct = np.linalg.solve(system, design.T @ data)
        np.testing.assert_allclose(estimates["A"], direct, rtol=1e-8, atol=1e-10)


class TestIllPosedCorner(unittest.TestCase):
    """The convex corner on an independent first-kind problem with a known optimum."""

    grid: NDArray = np.logspace(-10, 4, 561)

    def errors(
        self, operator: NDArray, data: NDArray, penalty: NDArray, solution: NDArray
    ) -> NDArray:
        gram, cross = operator.T @ operator, operator.T @ data
        return np.array(
            [
                np.linalg.norm(
                    np.linalg.solve(gram + lam * penalty, cross)[:, 0] - solution
                )
                for lam in self.grid
            ]
        )

    def test_selected_lambda_is_the_convex_corner_near_the_optimal_error(self) -> None:
        for relative_noise, bound in ((0.01, 1.25), (0.05, 1.1)):
            for seed in range(5):
                with self.subTest(noise=relative_noise, seed=seed):
                    operator, data, penalty, solution = smoothing_problem(
                        relative_noise, seed
                    )
                    gram = operator.T @ operator
                    lam, info = core.select_lambda_lcurve(
                        gram, operator.T @ data, penalty, data, operator
                    )
                    curvature = dense_signed_curvature(
                        gram, penalty, data, operator, self.grid
                    )
                    self.assertLess(
                        abs(np.log10(lam / self.grid[np.argmax(curvature)])), 0.03
                    )
                    self.assertTrue(info["stability_passed"])
                    estimate = np.linalg.solve(gram + lam * penalty, operator.T @ data)[
                        :, 0
                    ]
                    best = self.errors(operator, data, penalty, solution).min()
                    self.assertLess(np.linalg.norm(estimate - solution) / best, bound)

    def test_clockwise_rule_picks_the_large_lambda_end_bend(self) -> None:
        # Negative control: maximizing the clockwise (negative) curvature, the
        # orientation used before this fix, selects the bend into the vertical
        # asymptote at large lambda and over-smooths by an order of magnitude.
        for seed in range(5):
            with self.subTest(seed=seed):
                operator, data, penalty, solution = smoothing_problem(0.01, seed)
                gram = operator.T @ operator
                curvature = dense_signed_curvature(
                    gram, penalty, data, operator, self.grid
                )
                corner, bend = np.argmax(curvature), np.argmin(curvature)
                self.assertGreater(self.grid[bend], 100 * self.grid[corner])
                errors = self.errors(operator, data, penalty, solution)
                self.assertGreater(errors[bend] / errors.min(), 10.0)
                self.assertLess(errors[corner] / errors.min(), 1.25)
