# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""References and regressions for the kernel-only numerical implementation."""

from __future__ import annotations

import hashlib
import importlib
import json
import unittest
from pathlib import Path
from typing import cast, ClassVar, TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import scipy.linalg as la
import scipy.sparse
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray

if TYPE_CHECKING:
    import core
else:
    core = importlib.import_module("core")


def dense_cost(
    vec: NDArray,
    A: NDArray,
    theta: NDArray,
    X: NDArray,
    sigsq: float,
    M: int,
    symm: bool,
    omega: float,
    scale: float,
    cov: NDArray | None,
) -> float:
    """Dense inverse reference, with an independently built convolution matrix."""
    h = np.concatenate((vec, vec[-2::-1])) if symm else vec
    T = len(A) // M
    R = np.zeros((T, T - len(h) + 1))
    for j in range(R.shape[1]):
        R[j : j + len(h), j] = h
    S = np.kron(np.eye(M), R @ R.T)
    P = np.linalg.inv(A + S)
    Phi = P @ A @ theta
    C = X.shape[1]
    if cov is None:
        trace = sigsq * np.trace(P)
        correction_variance = sigsq
    else:
        trace = np.trace(np.linalg.inv(A) @ P @ cov)
        correction_variance = np.trace(cov) / len(A)
    correction = (
        (2 * omega - 1) * correction_variance * np.trace(P)
        if abs(omega - 0.5) > 1e-5
        else 0.0
    )
    return float(
        (
            omega * (np.sum(Phi**2) + 2 * C * trace)
            - 2 * omega * np.sum(theta * Phi)
            - C * correction
        )
        / scale
    )


def spectral_cost(
    vec: NDArray,
    A: NDArray,
    theta: NDArray,
    X: NDArray,
    sigsq: float,
    M: int,
    symm: bool,
    omega: float,
    scale: float,
    cov: NDArray | None,
) -> float:
    """Retain the old kernel evaluation using the unchanged scalar SURE helpers.

    This small reference replaces a dependency on a complete archived core.py.
    The separate dense reference also checks the underlying trace expression.
    """
    kernel = core.gen_corefilt(vec, symm)
    R = la.convolution_matrix(kernel, len(A) // M - len(kernel) + 1)
    S = la.block_diag(*([R @ R.T] * M))
    mu, V = la.eigh(A, A + S)
    mu = np.clip(mu, 0, 1)
    rotated = V.T @ (A @ theta)
    if cov is None:
        cost = core.SURE_cost_fast(
            1.0, mu, rotated, V, theta, sigsq, len(X), X.shape[1], omega
        )
    else:
        precomp = core.precompute_colored_trace(A, cov, V, mu)
        cost = core.SURE_cost_colored(
            1.0,
            mu,
            rotated,
            V,
            theta,
            cov,
            len(X),
            X.shape[1],
            omega,
            _precomp=precomp,
        )
    return float(cost / scale)


def input_digest(*arrays: NDArray) -> str:
    checksum = hashlib.sha256()
    for array in arrays:
        checksum.update(json.dumps([array.dtype.str, array.shape]).encode())
        checksum.update(np.ascontiguousarray(array).tobytes())
    return checksum.hexdigest()


class TestKernelObjective(unittest.TestCase):
    def test_objective_and_gradient_against_independent_references(self) -> None:
        rng = np.random.RandomState(819)
        for M in (1, 2):
            p = 7 * M
            D = rng.normal(size=(4 * p, p))
            A = D.T @ D
            covariance = 0.3 * la.toeplitz(0.6 ** np.arange(len(D)))
            cov = D.T @ covariance @ D
            for C in (1, 3):
                X = rng.normal(size=(len(D), C))
                theta = np.linalg.solve(A, D.T @ X)
                for symm, vec in (
                    (True, np.array([0.7, -1.3])),
                    (False, np.array([0.7, -1.3, 0.5])),
                ):
                    for omega in (0.3, 0.5, 0.8, 0.499995, 0.500005, 0.50002):
                        for colored in (False, True):
                            with self.subTest(
                                M=M, C=C, symm=symm, omega=omega, colored=colored
                            ):
                                covariance_arg = cov if colored else None
                                args = (
                                    A,
                                    theta,
                                    X,
                                    0.3,
                                    M,
                                    symm,
                                    omega,
                                    2.3,
                                    covariance_arg,
                                )
                                objective = core._KernelSUREObjective(*args)
                                value, gradient = objective.value_gradient(vec)
                                expected = dense_cost(vec, *args)
                                old_value = spectral_cost(vec, *args)
                                assert_allclose(value, expected, atol=1e-11, rtol=1e-11)
                                assert_allclose(
                                    value, old_value, atol=1e-11, rtol=1e-11
                                )
                                step = 1e-5
                                finite_difference = np.array(
                                    [
                                        (
                                            dense_cost(vec + step * e, *args)
                                            - dense_cost(vec - step * e, *args)
                                        )
                                        / (2 * step)
                                        for e in np.eye(len(vec))
                                    ]
                                )
                                assert_allclose(
                                    gradient, finite_difference, atol=1e-7, rtol=1e-7
                                )

    def test_cost_scaling_scales_gradient_without_changing_inputs(self) -> None:
        A = np.diag(np.arange(1.0, 8.0))
        theta = np.ones((7, 2))
        X = np.ones((10, 2))
        vec = np.array([0.7, -1.3])
        inputs = [array.copy() for array in (A, theta, X, vec)]
        unscaled = core._KernelSUREObjective(A, theta, X, 0.3, 1, True)
        scaled = core._KernelSUREObjective(A, theta, X, 0.3, 1, True, _cost_scale=2.3)
        value, gradient = unscaled.value_gradient(vec)
        scaled_value, scaled_gradient = scaled.value_gradient(vec)
        assert_allclose(scaled_value, value / 2.3, atol=0, rtol=0)
        assert_allclose(scaled_gradient, gradient / 2.3, atol=0, rtol=0)
        for actual, expected in zip((A, theta, X, vec), inputs):
            assert_array_equal(actual, expected)

    def test_indefinite_or_singular_penalized_matrix_is_not_hidden(self) -> None:
        for A in (np.diag([-1.0, 1.0, 1.0]), np.diag([0.0, 1.0, 1.0])):
            with self.subTest(A=A), self.assertRaises(np.linalg.LinAlgError):
                core._KernelSUREObjective(
                    A, np.ones((3, 1)), np.ones((5, 1)), 0.3, 1
                ).value_gradient(np.zeros(1))


class TestSavedKernelFailure(unittest.TestCase):
    templates: ClassVar[dict[str, NDArray]]
    data: ClassVar[NDArray]
    ix: ClassVar[NDArray]
    labels: ClassVar[NDArray]

    @classmethod
    def setUpClass(cls) -> None:
        # Keep the actual failed input byte-identical across NumPy/BLAS builds.
        # The original seed protocol and input checksum are recorded below.
        with np.load(
            Path(__file__).with_name("kernel_failure_case.npz"), allow_pickle=False
        ) as fixture:
            cls.templates = {
                "Template A": fixture["template_a"],
                "Template B": fixture["template_b"],
            }
            cls.data, cls.ix, cls.labels = (
                fixture["data"],
                fixture["ix"],
                fixture["labels"],
            )
        for array in (cls.data, cls.ix, cls.labels):
            array.setflags(write=False)

    def fit_kernel(self) -> tuple[dict[str, NDArray], dict[str, object]]:
        estimates, info = core.get_events_templates(
            self.data,
            self.ix,
            self.labels,
            half_win=50,
            method="ls",
            reg_temporal=500,
            opt_reg_temporal=True,
            vec_mode=True,
            symm=True,
            vlen=3,
            noise_est="pairwise",
        )
        if info is None:
            raise AssertionError("Kernel optimization must return diagnostics")
        labeled_estimates: dict[str, NDArray] = {}
        for label, estimate in estimates.items():
            if not isinstance(label, str):
                raise TypeError("The saved fixture must have string event labels")
            labeled_estimates[label] = estimate
        return labeled_estimates, info

    def test_saved_candidate_uses_cholesky_and_matches_dense_reference(self) -> None:
        self.assertEqual(
            input_digest(self.data, self.ix, self.labels),
            "b2c3a41b2572b705c7992173106dbc7da81f11ad39886f9ae538c644b9f25c41",
        )
        D = cast(
            scipy.sparse.csr_matrix,
            core.get_design_matrix(
                self.data, self.ix, self.labels, 50, sparse_out=True
            ),
        )
        A = (D.T @ D).toarray()
        theta = np.linalg.inv(A) @ (D.T @ self.data)
        vec = np.array([77.85512553347937, -154.46816900816924])
        args = (
            A,
            theta,
            self.data,
            0.04186437818652037,
            2,
            True,
            0.5,
            1.7370617953732328,
            None,
        )
        with patch.object(
            core.sp.linalg,
            core.sp.linalg.eigh.__name__,
            side_effect=AssertionError("Kernel must not call eigh"),
        ):
            value, gradient = core._KernelSUREObjective(*args).value_gradient(vec)
        self.assertTrue(np.all(np.isfinite(gradient)))
        assert_allclose(value, dense_cost(vec, *args), atol=1e-10, rtol=1e-10)

    def test_previously_failed_fit_converges_with_unchanged_settings(self) -> None:
        with (
            patch.object(
                core.sp.optimize,
                core.sp.optimize.minimize.__name__,
                wraps=core.sp.optimize.minimize,
            ) as minimize,
            patch.object(
                core.sp.linalg,
                core.sp.linalg.eigh.__name__,
                side_effect=AssertionError("Kernel must not call eigh"),
            ),
        ):
            estimates, info = self.fit_kernel()
        self.assertEqual(minimize.call_count, 1)
        kwargs = minimize.call_args.kwargs
        assert_array_equal(kwargs["x0"], [1, -3])
        self.assertEqual(kwargs["options"], {"maxiter": 50, "disp": 0})
        self.assertIsNone(kwargs["method"])
        self.assertIs(kwargs["jac"], True)
        self.assertNotIn("tol", kwargs)
        self.assertNotIn("bounds", kwargs)
        self.assertTrue(info["optimizer_success"])
        self.assertEqual(info["optimizer_status"], 0)
        self.assertLessEqual(cast(int, info["optimizer_iterations"]), 50)
        self.assertLess(cast(float, info["optimizer_gradient_max_abs"]), 1e-5)
        self.assertGreater(cast(int, info["optimizer_function_evaluations"]), 0)
        self.assertEqual(
            info["optimizer_function_evaluations"],
            info["optimizer_gradient_evaluations"],
        )
        self.assertTrue(info["optimizer_message"])
        assert_allclose(info["noise_var"], 0.04186437818652037, atol=1e-15, rtol=0)
        assert_allclose(
            info["optimizer_cost_scale"], 1.7370617953732328, atol=1e-12, rtol=1e-12
        )
        self.assertTrue(np.isfinite(core.calculate_snr(estimates, self.templates)))
        self.assertEqual(
            input_digest(self.data, self.ix, self.labels),
            "b2c3a41b2572b705c7992173106dbc7da81f11ad39886f9ae538c644b9f25c41",
        )


if __name__ == "__main__":
    unittest.main()
