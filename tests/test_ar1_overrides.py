# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit AR(1) inputs to colored SURE, for the phi and variance sensitivity.

The paper perturbs phi to phi_hat^(1/f) (up to 0.991 for phi_hat = 0.982), so a
supplied phi must reach the covariance unclipped, and the known-covariance
simulation supplies the true phi and marginal variance. Each override is
checked against D^T Sigma D built here from the design and the AR(1)
covariance, with no covariance between separately recorded blocks.
"""

from __future__ import annotations

import importlib
import unittest
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray
from scipy import linalg

if TYPE_CHECKING:
    import core
else:
    core = importlib.import_module("core")

HALF_WIN = 5
T = 2 * HALF_WIN
BLOCK_LENGTHS = (170, 150)
SURE_PATHS: dict[str, dict[str, Any]] = {
    "scalar": {},
    "kernel": {"vec_mode": True, "symm": True, "vlen": 3},
}
AR1_FIELDS = {"phi_estimated", "phi_used", "noise_var_estimated", "noise_var_scale"}


def ar1_noise(rs: np.random.RandomState, length: int, phi: float) -> NDArray:
    noise = np.zeros((length, 2))
    innovations = rs.normal(scale=0.3, size=(length, 2))
    noise[0] = innovations[0] / np.sqrt(1 - phi**2)
    for t in range(1, length):
        noise[t] = phi * noise[t - 1] + innovations[t]
    return noise


def two_blocks() -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Two classes, two channels, two separately recorded blocks, joined."""
    rs = np.random.RandomState(5)
    shape = np.sin(np.linspace(0, np.pi, T))
    templates = {
        "A": np.column_stack([shape, 0.5 * shape]),
        "B": np.column_stack([-0.6 * shape, shape]),
    }
    parts, onsets, labels = [], [], []
    start = 0
    for length in BLOCK_LENGTHS:
        block = ar1_noise(rs, length, 0.9)
        local = np.arange(12, length - HALF_WIN, 17)
        for k, onset in enumerate(local):
            label = "AB"[k % 2]
            block[onset - HALF_WIN : onset + HALF_WIN] += templates[label]
            labels.append(label)
        parts.append(block)
        onsets.extend(start + local)
        start += length
    starts = np.cumsum([0, *BLOCK_LENGTHS[:-1]])
    return np.vstack(parts), np.array(onsets), np.array(labels), starts


def fit(
    inputs: tuple[NDArray, NDArray, NDArray, NDArray],
    path: str,
    noise_est: str = "ar1_colored",
    **overrides: Any,
) -> tuple[NDArray, dict[str, Any]]:
    data, ix, lb, starts = inputs
    templates, inner = core.get_events_templates(
        data,
        ix,
        lb,
        half_win=HALF_WIN,
        method="ls",
        opt_reg_temporal=True,
        noise_est=noise_est,
        block_starts=starts,
        **SURE_PATHS[path],
        **overrides,
    )
    if inner is None:
        raise AssertionError("SURE fits must return diagnostics")
    return np.stack([templates[label] for label in ("A", "B")]), inner


def design_and_covariance(
    inputs: tuple[NDArray, NDArray, NDArray, NDArray],
    phi: float,
    noise_var: float,
    *,
    blockwise: bool = True,
) -> tuple[NDArray, NDArray]:
    """D, and D^T Sigma D with Sigma_ij = noise_var * phi^|i-j| within a block."""
    data, ix, lb, starts = inputs
    n = len(data)
    design = np.zeros((n, 2 * T))
    for onset, label in zip(ix, lb):
        column = "AB".index(label) * T
        design[onset - HALF_WIN + np.arange(T), column + np.arange(T)] = 1.0
    samples = np.arange(n)
    sigma = noise_var * np.power(phi, np.abs(samples[:, None] - samples[None, :]))
    if blockwise:
        block = np.searchsorted(starts, samples, side="right") - 1
        sigma = sigma * (block[:, None] == block[None, :])
    return design, design.T @ sigma @ design


def penalty(kernel: NDArray) -> NDArray:
    """Block-diagonal R R^T of the template penalty for both classes."""
    R = linalg.convolution_matrix(kernel, T - len(kernel) + 1)
    return linalg.block_diag(R @ R.T, R @ R.T)


def stacked(solution: NDArray) -> NDArray:
    return solution.T.reshape(2, 2, T).transpose(1, 2, 0)


def scalar_score(
    data: NDArray, design: NDArray, covariance: NDArray, lam: float
) -> float:
    gram, cross = design.T @ design, design.T @ data
    S = penalty(np.array([1.0, -2.0, 1.0]))
    mu, basis = linalg.eigh(gram, gram + S)
    mu = np.clip(mu, 0, 1)
    ols = np.linalg.solve(gram, cross)
    return core.SURE_cost_colored(
        lam,
        mu,
        basis.T @ cross,
        basis,
        ols,
        covariance,
        len(data),
        data.shape[1],
        0.5,
        gram,
        S,
    )


def kernel_score(
    data: NDArray, design: NDArray, covariance: NDArray, vec: NDArray, scale: float
) -> float:
    gram = design.T @ design
    ols = np.linalg.solve(gram, design.T @ data)
    objective = core._KernelSUREObjective(
        gram, ols, data, 1.0, 2, True, 0.5, scale, DtSigmaD=covariance
    )
    return objective.value_gradient(vec)[0]


class OverrideCovarianceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = two_blocks()
        data, ix, _, _ = self.inputs
        self.phi_hat = float(core.estimate_phi(data, ix, HALF_WIN))
        self.var_hat = float(core.est_noise_var_baseline(data, ix, HALF_WIN))
        # One case per override, plus the known-covariance pair.
        self.cases: dict[str, tuple[dict[str, Any], float, float]] = {
            "phi_above_cap": ({"ar1_phi": 0.995}, 0.995, self.var_hat),
            "absolute_variance": ({"ar1_noise_var": 0.37}, self.phi_hat, 0.37),
            "scaled_variance": (
                {"noise_var_scale": 1.25},
                self.phi_hat,
                1.25 * self.var_hat,
            ),
            "known_covariance": (
                {"ar1_phi": 0.93, "ar1_noise_var": 0.52},
                0.93,
                0.52,
            ),
        }

    def test_scalar_objective_and_templates_use_the_supplied_covariance(
        self,
    ) -> None:
        data = self.inputs[0]
        for name, (overrides, phi, noise_var) in self.cases.items():
            with self.subTest(case=name):
                templates, inner = fit(self.inputs, "scalar", **overrides)
                design, covariance = design_and_covariance(self.inputs, phi, noise_var)
                lambdas = np.asarray(inner["search_grid_lambdas"])
                moderate = (lambdas >= 1e-3) & (lambdas <= 1e2)
                expected = [
                    scalar_score(data, design, covariance, lam)
                    for lam in lambdas[moderate]
                ]
                assert_allclose(
                    np.asarray(inner["search_grid_scores"])[moderate],
                    expected,
                    rtol=1e-9,
                )
                lam = inner["opt_val"]
                self.assertTrue(0 < lam < np.inf, f"{name}: lambda={lam}")
                assert_allclose(
                    inner["cost"],
                    scalar_score(data, design, covariance, lam),
                    rtol=1e-9,
                )
                gram = design.T @ design
                solution = np.linalg.solve(
                    gram + lam * penalty(np.array([1.0, -2.0, 1.0])),
                    design.T @ data,
                )
                # Two direct solves agree to the conditioning of the system.
                expected = stacked(solution)
                assert_allclose(
                    templates, expected, rtol=0, atol=1e-7 * np.abs(expected).max()
                )

    def test_kernel_objective_and_templates_use_the_supplied_covariance(
        self,
    ) -> None:
        data = self.inputs[0]
        x0 = np.array([1.0, -3.0])  # The symmetric seed [1, -3, 3, -1][:2].
        for name, (overrides, phi, noise_var) in self.cases.items():
            with self.subTest(case=name):
                templates, inner = fit(self.inputs, "kernel", **overrides)
                design, covariance = design_and_covariance(self.inputs, phi, noise_var)
                scale = inner["optimizer_cost_scale"]
                at_seed = kernel_score(data, design, covariance, x0, 1.0)
                assert_allclose(scale, max(abs(at_seed), 1e-30), rtol=1e-9)
                kernel = np.asarray(inner["opt_val"])
                assert_allclose(
                    inner["cost"],
                    kernel_score(data, design, covariance, kernel[:2], scale),
                    rtol=1e-9,
                )
                solution = np.linalg.solve(
                    design.T @ design + penalty(kernel), design.T @ data
                )
                # Two direct solves agree to the conditioning of the system.
                expected = stacked(solution)
                assert_allclose(
                    templates, expected, rtol=0, atol=1e-7 * np.abs(expected).max()
                )

    def test_reference_covariance_distinguishes_clipping_and_block_joins(
        self,
    ) -> None:
        # Guards the two tests above: a clipped phi or covariance across the
        # join would give a measurably different objective.
        data = self.inputs[0]
        _, inner = fit(self.inputs, "scalar", ar1_phi=0.995)
        lam = inner["opt_val"]
        for name, phi, blockwise in (
            ("clipped", 0.99, True),
            ("joined", 0.995, False),
        ):
            with self.subTest(variant=name):
                design, covariance = design_and_covariance(
                    self.inputs, phi, self.var_hat, blockwise=blockwise
                )
                other = scalar_score(data, design, covariance, lam)
                self.assertGreater(abs(other - inner["cost"]), 1e-6 * abs(other))

    def test_inner_records_estimated_and_used_values(self) -> None:
        data, ix, lb, _ = self.inputs
        for path in SURE_PATHS:
            for name, (overrides, phi, noise_var) in self.cases.items():
                with self.subTest(path=path, case=name):
                    _, inner = fit(self.inputs, path, **overrides)
                    self.assertEqual(inner["phi_estimated"], self.phi_hat)
                    self.assertEqual(inner["phi_used"], phi)
                    self.assertEqual(inner["noise_var_estimated"], self.var_hat)
                    self.assertEqual(
                        inner["noise_var_scale"], overrides.get("noise_var_scale", 1.0)
                    )
                    assert_allclose(inner["noise_var"], noise_var, rtol=1e-15)
            with self.subTest(path=path, noise_est="ar3_colored"):
                _, inner = fit(self.inputs, path, "ar3_colored")
                span = int(ix.max() - ix.min()) + T - 1
                assert_array_equal(
                    inner["ar3_acf"],
                    core.estimate_acf_ar(data, ix, HALF_WIN, ar_order=3, max_lag=span),
                )
                self.assertFalse(AR1_FIELDS & set(inner))


class DefaultAR1Test(unittest.TestCase):
    def test_estimates_given_explicitly_reproduce_the_default_fit(self) -> None:
        # The override path with phi_hat and the estimated variance is the
        # default computation, bit for bit, on both SURE paths.
        inputs = two_blocks()
        for path in SURE_PATHS:
            with self.subTest(path=path):
                templates, inner = fit(inputs, path)
                explicit, explicit_inner = fit(
                    inputs,
                    path,
                    ar1_phi=inner["phi_estimated"],
                    ar1_noise_var=inner["noise_var_estimated"],
                )
                self.assertEqual(templates.tobytes(), explicit.tobytes())
                for key in ("opt_val", "cost", "noise_var", "phi_used"):
                    self.assertEqual(
                        np.asarray(inner[key]).tobytes(),
                        np.asarray(explicit_inner[key]).tobytes(),
                    )
                self.assertEqual(inner["noise_var"], inner["noise_var_estimated"])
                self.assertEqual(inner["phi_used"], inner["phi_estimated"])

    def test_unit_scale_keeps_a_float32_variance_unchanged(self) -> None:
        # A float64 unit scale must not promote the estimated variance, alone
        # or beside a supplied phi.
        data, ix, lb, starts = two_blocks()
        inputs = (data.astype(np.float32), ix, lb, starts)
        for path in SURE_PATHS:
            phis: tuple[dict[str, Any], ...] = ({}, {"ar1_phi": 0.9})
            for phi in phis:
                with self.subTest(path=path, phi=phi):
                    templates, inner = fit(inputs, path, **phi)
                    scaled, scaled_inner = fit(
                        inputs, path, noise_var_scale=np.float64(1.0), **phi
                    )
                    self.assertEqual(templates.tobytes(), scaled.tobytes())
                    self.assertEqual(type(inner["noise_var"]), np.float32)
                    self.assertEqual(type(scaled_inner["noise_var"]), np.float32)
                    self.assertEqual(inner["cost"], scaled_inner["cost"])

    def test_paths_without_ar1_covariance_record_no_ar1_fields(self) -> None:
        inputs = two_blocks()
        cases: tuple[tuple[str, dict[str, Any]], ...] = (
            ("white_scalar", {"noise_est": "pairwise"}),
            ("white_kernel", {"noise_est": "pairwise", "vec_mode": True}),
            ("gcv", {"reg_method": "gcv"}),
        )
        for name, kwargs in cases:
            with self.subTest(case=name):
                data, ix, lb, _ = inputs
                _, inner = core.get_events_templates(
                    data,
                    ix,
                    lb,
                    half_win=HALF_WIN,
                    method="ls",
                    opt_reg_temporal=True,
                    **kwargs,
                )
                self.assertIsNotNone(inner)
                self.assertFalse((AR1_FIELDS | {"ar3_acf"}) & set(inner or {}))


if __name__ == "__main__":
    unittest.main()
