# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Independent restricted-evidence references and EB regression tests."""

from __future__ import annotations

import importlib
import unittest
from collections.abc import Callable
from typing import cast, TYPE_CHECKING

import numpy as np
import scipy.linalg as la
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

if TYPE_CHECKING:
    import core
else:
    core = importlib.import_module("core")


def penalty_basis(S: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    values, basis = la.eigh(S)
    positive = values > len(S) * np.finfo(float).eps * np.max(np.abs(values))
    return values[positive], basis[:, positive], basis[:, ~positive]


def case_data(name: str) -> tuple[NDArray, NDArray, NDArray]:
    specs = {
        "ridge": (42, 60, 12, 1),
        "proper_nonidentity": (17, 72, 12, 3),
        "second_difference_one_class": (43, 80, 12, 1),
        "second_difference_two_classes": (44, 96, 16, 3),
    }
    seed, n, p, channels = specs[name]
    rng = np.random.RandomState(seed)
    D = rng.standard_normal((n, p)) / np.sqrt(n)
    if name == "ridge":
        S = np.eye(p)
    elif name == "proper_nonidentity":
        rotation, _ = la.qr(rng.standard_normal((p, p)))
        S = (rotation * np.geomspace(0.2, 5.0, p)) @ rotation.T
    else:
        classes = 2 if name.endswith("two_classes") else 1
        differences = np.diff(np.eye(p // classes), n=2, axis=0)
        S = la.block_diag(*([differences.T @ differences] * classes))
    values, positive, null = penalty_basis(S)
    theta = (positive / np.sqrt(values)) @ rng.standard_normal((len(values), channels))
    theta += null @ rng.standard_normal((null.shape[1], channels))
    X = D @ theta + 0.7 * rng.standard_normal((n, channels))
    return D, X, S


def observation_evidence(
    D: NDArray, X: NDArray, S: NDArray
) -> tuple[float, float, Callable[[float, float], float]]:
    """Integrate in observation space, independently of the production update.

    QR removes the identifiable flat-prior nuisance components. A grid locates
    every interior local minimum before bounded refinement in log lambda.
    This reference does not assume that every evidence problem is unimodal.
    """
    values, positive, null = penalty_basis(S)
    n, channels = X.shape
    q = null.shape[1]
    if q:
        nuisance = D @ null
        assert np.linalg.matrix_rank(nuisance) == q
        rotation, _ = la.qr(nuisance, mode="full")
        contrast = rotation[:, q:]
    else:
        contrast = np.eye(n)
    Y = contrast.T @ X
    B = (contrast.T @ D @ positive) / np.sqrt(values)
    covariance = B @ B.T
    n_eff = n - q

    def profile(loglam: float) -> tuple[float, float]:
        factor = la.cho_factor(np.eye(n_eff) + covariance / np.exp(loglam), lower=True)
        variance = float(np.sum(Y * la.cho_solve(factor, Y)) / (channels * n_eff))
        logdet = 2 * np.log(np.diag(factor[0])).sum()
        return float(channels / 2 * (logdet + n_eff * np.log(variance))), variance

    def joint(logalpha: float, logbeta: float) -> float:
        covariance_full = (
            np.exp(-logbeta) * np.eye(n_eff) + np.exp(-logalpha) * covariance
        )
        factor = la.cho_factor(covariance_full, lower=True)
        quadratic = np.sum(Y * la.cho_solve(factor, Y))
        logdet = 2 * np.log(np.diag(factor[0])).sum()
        return float((channels * logdet + quadratic) / 2)

    grid = np.linspace(-14.0, 14.0, 141)
    costs = np.array([profile(float(point))[0] for point in grid])
    candidates = [(float(grid[0]), costs[0]), (float(grid[-1]), costs[-1])]
    for index in range(1, len(grid) - 1):
        if costs[index] <= min(costs[index - 1], costs[index + 1]):
            result = minimize_scalar(
                lambda point: profile(point)[0],
                bounds=(grid[index - 1], grid[index + 1]),
                method="bounded",
                options={"xatol": 1e-10},
            )
            assert result.success
            candidates.append((float(result.x), float(result.fun)))
    best, _ = min(candidates, key=lambda item: item[1])
    assert grid[0] < best < grid[-1], "Reference optimum reached the search boundary"
    return float(np.exp(best)), profile(best)[1], joint


def select(
    D: NDArray, X: NDArray, S: NDArray, *, tight: bool = False
) -> tuple[float, dict[str, object]]:
    if tight:
        return core.select_lambda_empbayes(
            D.T @ D, D.T @ X, S, X, D, max_iter=2000, tol=1e-11
        )
    return core.select_lambda_empbayes(D.T @ D, D.T @ X, S, X, D)


def fitted(D: NDArray, X: NDArray, S: NDArray, lam: float) -> NDArray:
    return la.solve(D.T @ D + lam * S, D.T @ X, assume_a="pos")


def event_case() -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    truth = {
        "A": np.outer(core.ricker(12, 2.0), [0.7, 1.0, 0.4]),
        "B": np.outer(core.ricker(12, 3.0), [0.4, 0.7, 1.0]),
    }
    X, ix, labels = core.generate_data(
        truth, I=5, noise=0.2, mean_interval=8.0, rs=np.random.RandomState(42)
    )
    D = np.asarray(core.get_design_matrix(X, ix, labels, half_win=6), dtype=float)
    differences = np.diff(np.eye(12), n=2, axis=0)
    S = la.block_diag(differences.T @ differences, differences.T @ differences)
    return D, X, S, ix, labels


class TestEmpBayes(unittest.TestCase):
    def test_reference_and_channel_replication(self) -> None:
        for name in (
            "ridge",
            "proper_nonidentity",
            "second_difference_one_class",
            "second_difference_two_classes",
        ):
            D, original, S = case_data(name)
            reference, variance, joint = observation_evidence(D, original, S)
            q = penalty_basis(S)[2].shape[1]
            for copies in (1, 2, 8):
                with self.subTest(case=name, copies=copies):
                    X = np.tile(original, (1, copies))
                    lam, info = select(D, X, S)
                    self.assertTrue(info["converged"])
                    self.assertLessEqual(cast(int, info["n_iter"]), 30)
                    assert_allclose(lam, reference, rtol=1e-3)
                    tight, detail = select(D, X, S, tight=True)
                    self.assertTrue(detail["converged"])
                    assert_allclose(tight, reference, rtol=2e-6)
                    assert_allclose(detail["noise_var"], variance, rtol=2e-6)
                    self.assertEqual(detail["penalty_nullity"], q)
                    gamma = cast(float, detail["gamma"])
                    self.assertEqual(detail["gamma_total"], X.shape[1] * gamma)
                    self.assertEqual(detail["gamma_penalized"], gamma - q)
                    self.assertEqual(
                        detail["gamma_penalized_total"], X.shape[1] * (gamma - q)
                    )
            logalpha, logbeta = np.log(reference / variance), -np.log(variance)
            step = 1e-4
            derivatives = [
                (joint(logalpha + step, logbeta) - joint(logalpha - step, logbeta))
                / (2 * step),
                (joint(logalpha, logbeta + step) - joint(logalpha, logbeta - step))
                / (2 * step),
            ]
            assert_allclose(np.array(derivatives) / original.shape[1], 0.0, atol=1e-5)

    def test_unpenalized_signal_is_preserved(self) -> None:
        D, X, S = case_data("second_difference_two_classes")
        null = penalty_basis(S)[2]
        coefficients = np.arange(null.shape[1] * X.shape[1]).reshape(null.shape[1], -1)
        shift = null @ coefficients
        shifted = X + D @ shift
        lam, _ = select(D, X, S, tight=True)
        changed, _ = select(D, shifted, S, tight=True)
        assert_allclose(changed, lam, rtol=1e-8)
        assert_allclose(
            fitted(D, shifted, S, changed) - fitted(D, X, S, lam), shift, atol=1e-9
        )

    def test_scaling_and_coordinate_permutation(self) -> None:
        D, X, S = case_data("second_difference_two_classes")
        lam, _ = select(D, X, S, tight=True)
        theta = fitted(D, X, S, lam)
        for scale in (1e-6, 1e6):
            with self.subTest(data_scale=scale):
                changed, _ = select(D, scale * X, S, tight=True)
                assert_allclose(changed, lam, rtol=1e-8)
        for scale in (1e-3, 1e3):
            with self.subTest(penalty_scale=scale):
                changed, _ = select(D, X, scale * S, tight=True)
                assert_allclose(scale * changed, lam, rtol=1e-8)
                assert_allclose(
                    fitted(D, X, scale * S, changed), theta, rtol=1e-8, atol=1e-9
                )
        permutation = np.random.RandomState(6).permutation(len(S))
        changed, _ = select(
            D[:, permutation], X, S[np.ix_(permutation, permutation)], tight=True
        )
        assert_allclose(changed, lam, rtol=1e-8)

    def test_overlapping_event_design_with_complete_right_boundary(self) -> None:
        D, X, S, ix, labels = event_case()
        self.assertEqual(np.max(ix) + 6, len(X))
        self.assertTrue(np.any(np.diff(np.sort(ix)) < 12))
        reference, _, _ = observation_evidence(D, X, S)
        lam, info = select(D, X, S)
        self.assertTrue(info["converged"])
        assert_allclose(lam, reference, rtol=1e-3)
        estimates, detail = core.get_events_templates(
            X, ix, labels, half_win=6, method="ls", reg_method="empbayes"
        )
        self.assertIsNotNone(detail)
        assert_allclose(
            np.concatenate([estimates["A"], estimates["B"]]),
            fitted(D, X, S, lam),
            rtol=1e-10,
            atol=1e-12,
        )
