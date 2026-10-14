# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""Unit tests for the core template estimation module."""

from __future__ import annotations

import importlib
import unittest
from typing import TYPE_CHECKING

import numpy as np
import scipy as sp
import scipy.sparse
from numpy.testing import assert_allclose
from numpy.typing import NDArray

if TYPE_CHECKING:
    import core
else:
    core = importlib.import_module("core")


class TestRicker(unittest.TestCase):
    """Tests for the Ricker (Mexican hat) wavelet generator."""

    def test_symmetry(self) -> None:
        w: NDArray = core.ricker(51, 10)
        assert_allclose(w, w[::-1], atol=1e-14)

    def test_peak_at_center(self) -> None:
        w: NDArray = core.ricker(51, 10)
        self.assertEqual(np.argmax(w), 25)

    def test_shape(self) -> None:
        w: NDArray = core.ricker(100, 15)
        self.assertEqual(w.shape, (100,))


class TestGenerateData(unittest.TestCase):
    """Tests for synthetic data generation."""

    def test_output_shapes(self) -> None:
        T = 50
        C = 2
        n_events = 5
        real_templates = {
            "A": np.outer(core.ricker(T, 10), np.hanning(C)),
            "B": np.outer(core.ricker(T, 8), np.ones(C)),
        }
        rs = np.random.RandomState(np.array([42]))
        data, ix, labels = core.generate_data(
            real_templates, I=n_events, noise=0.1, rs=rs
        )

        self.assertEqual(data.ndim, 2)
        self.assertEqual(data.shape[1], C)
        self.assertEqual(len(ix), len(labels))
        self.assertEqual(len(ix), n_events * len(real_templates))

    def test_deterministic(self) -> None:
        real_templates = {
            "A": np.outer(core.ricker(50, 10), np.hanning(2)),
        }
        data1, ix1, _lb1 = core.generate_data(
            real_templates, I=5, noise=0.1, rs=np.random.RandomState(np.array([0]))
        )
        data2, ix2, _lb2 = core.generate_data(
            real_templates, I=5, noise=0.1, rs=np.random.RandomState(np.array([0]))
        )
        assert_allclose(data1, data2)
        assert_allclose(ix1, ix2)

    def test_noiseless_forward_model_includes_final_event(self) -> None:
        template = np.arange(1.0, 21.0).reshape(10, 2)
        real_templates = {"A": template, "B": -template}
        events = np.array(["B", "A"])
        stacked_templates = np.vstack([real_templates[event] for event in events])

        for mean_interval in (2.0, 20.0):
            for seed in (0, 7):
                data, ix, labels = core.generate_data(
                    real_templates,
                    I=3,
                    noise=0,
                    mean_interval=mean_interval,
                    rs=np.random.RandomState(seed),
                )
                self.assertEqual(ix[-1] + 5, len(data))

                for sparse_out in (False, True):
                    with self.subTest(
                        mean_interval=mean_interval, seed=seed, sparse_out=sparse_out
                    ):
                        design = core.get_design_matrix(
                            data, ix, labels, 5, events=events, sparse_out=sparse_out
                        )
                        assert_allclose(design @ stacked_templates, data, atol=1e-14)
                        counts = np.asarray(design[:, ::10].sum(axis=0)).ravel()
                        assert_allclose(counts, [3, 3], atol=0, rtol=0)


class TestEventBoundaries(unittest.TestCase):
    """Check exact right-boundary inclusion against independent matrices."""

    def _boundary_problem(
        self,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        data = np.zeros((20, 1))
        ix = np.array([3, 5, 17, 18])
        labels = np.array(["A", "B", "A", "B"])
        events = np.array(["B", "A"])

        # The complete epochs occupy [0, 6), [2, 8), and [14, 20).
        # The event at 18 would end at 21 and must remain excluded.
        expected_design = np.zeros((20, 12))
        expected_design[2:8, :6] = np.eye(6)
        expected_design[:6, 6:] = np.eye(6)
        expected_design[14:20, 6:] = np.eye(6)
        return data, ix, labels, events, expected_design

    def test_design_includes_complete_boundary_epochs(self) -> None:
        data, ix, labels, events, expected = self._boundary_problem()
        designs = []
        for sparse_out in (False, True):
            with self.subTest(sparse_out=sparse_out):
                design = core.get_design_matrix(
                    data, ix, labels, 3, events=events, sparse_out=sparse_out
                )
                if sp.sparse.issparse(design):
                    design = design.toarray()
                designs.append(design)
                assert_allclose(design, expected, atol=0, rtol=0)
                assert_allclose(design[:, ::6].sum(axis=0), [1, 2], atol=0, rtol=0)
        assert_allclose(designs[0], designs[1], atol=0, rtol=0)

    def test_design_excludes_incomplete_right_boundary_epochs(self) -> None:
        data = np.zeros((20, 1))
        ix = np.array([18, 19, 20])
        labels = np.array(["A", "A", "A"])
        for sparse_out in (False, True):
            with self.subTest(sparse_out=sparse_out):
                design = core.get_design_matrix(
                    data, ix, labels, 3, sparse_out=sparse_out
                )
                if sp.sparse.issparse(design):
                    design = design.toarray()
                assert_allclose(design, np.zeros((20, 6)), atol=0, rtol=0)

    def test_ar1_covariance_matches_explicit_at_boundary(self) -> None:
        data, ix, labels, events, design = self._boundary_problem()
        for phi in (0.0, 0.6):
            with self.subTest(phi=phi):
                sigsq = 2.5
                acf = sigsq * phi ** np.arange(len(data))
                sigma = sp.linalg.toeplitz(acf)
                expected = design.T @ sigma @ design
                actual = core.build_DtSigmaD_ar1(
                    ix, labels, events, 3, phi, sigsq, data_length=len(data)
                )
                assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_acf_covariance_matches_explicit_at_boundary(self) -> None:
        data, ix, labels, events, design = self._boundary_problem()
        lags = np.arange(len(data))
        acf = 0.7 * 0.65**lags + 0.3 * (-0.25) ** lags
        sigma = sp.linalg.toeplitz(acf)
        expected = design.T @ sigma @ design
        actual = core.build_DtSigmaD_acf(
            ix, labels, events, 3, acf, data_length=len(data)
        )
        assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


class TestCalculateSNR(unittest.TestCase):
    """Tests for SNR computation."""

    def test_perfect_match(self) -> None:
        t = {"A": np.ones((10, 2))}
        snr: float = core.calculate_snr(t, t)
        self.assertEqual(snr, np.inf)

    def test_known_snr(self) -> None:
        signal = np.ones((10, 1))
        noisy = signal + 0.1 * np.ones((10, 1))
        snr: float = core.calculate_snr({"A": noisy}, {"A": signal})
        self.assertAlmostEqual(snr, 20.0, places=5)


class TestSURECostConsistency(unittest.TestCase):
    """Verify SURE_cost_fast matches SURE written with explicit matrices."""

    def test_fast_matches_explicit(self) -> None:
        T = 30
        C = 2
        half_win = T // 2
        real_templates = {"A": np.outer(core.ricker(T, 5), np.ones(C))}
        rs = np.random.RandomState(np.array([42]))
        data, ix, labels = core.generate_data(
            real_templates, I=8, noise=0.1, mean_interval=40, rs=rs
        )

        events = np.unique(labels)
        M = len(events)
        design = core.get_design_matrix(data, ix, labels, half_win, events=events)
        D = design
        DtD = D.T @ D
        DtX = D.T @ data

        R = sp.linalg.convolution_matrix([1, -2, 1], T - 2)
        RR = R @ R.T
        S = sp.linalg.block_diag(*([RR] * M))

        DtD_dense = np.asarray(DtD)
        S_dense = np.asarray(S)
        DtX_dense = np.asarray(DtX)

        theta_ml = np.linalg.solve(DtD_dense, DtX_dense)
        Dpinv = np.linalg.solve(DtD_dense, D.T)
        DpinvX = Dpinv @ data
        sigsq = 0.01

        mu, V = sp.linalg.eigh(DtD_dense, DtD_dense + S_dense)
        mu = np.clip(mu, 0, 1)
        DtX_rot = V.T @ DtX_dense

        n_obs = data.shape[0]
        n_cols = data.shape[1]

        for lam in [0.1, 1.0, 10.0, 100.0]:
            # SURE = ||G X||^2 / 2 + C sigma^2 tr(G^T D^+) - tr(X^T G^T D^+ X),
            # with G = (D^T D + lam S)^{-1} D^T.
            G = np.linalg.solve(DtD_dense + lam * S_dense, D.T)
            GX = G @ data
            cost_explicit = (
                0.5 * np.sum(GX**2)
                + n_cols * sigsq * np.sum(G * Dpinv)
                - np.trace(GX.T @ DpinvX)
            )
            cost_fast = core.SURE_cost_fast(
                lam,
                mu,
                DtX_rot,
                V,
                theta_ml,
                sigsq,
                n_obs,
                n_cols,
            )
            assert_allclose(
                cost_fast,
                cost_explicit,
                rtol=1e-6,
                err_msg=f"Mismatch at lambda={lam}",
            )


class TestEmpBayesConvergence(unittest.TestCase):
    """Test that Empirical Bayes converges to a finite lambda."""

    def test_convergence(self) -> None:
        T = 30
        C = 2
        half_win = T // 2
        real_templates = {"A": np.outer(core.ricker(T, 5), np.ones(C))}
        rs = np.random.RandomState(np.array([42]))
        data, ix, labels = core.generate_data(
            real_templates, I=10, noise=0.1, mean_interval=40, rs=rs
        )

        events = np.unique(labels)
        M = len(events)
        D = core.get_design_matrix(data, ix, labels, half_win, events=events)
        DtD = D.T @ D
        DtX = D.T @ data

        R = sp.linalg.convolution_matrix([1, -2, 1], T - 2)
        RR = R @ R.T
        S = sp.sparse.block_diag([RR] * M, format="csr")

        lam, info = core.select_lambda_empbayes(DtD, DtX, S, data, D)

        self.assertGreater(lam, 0)
        self.assertLess(lam, 1e6)
        n_iter: int = int(info["n_iter"])  # pyre-ignore[6]
        self.assertLess(n_iter, 30)

        S_dense = np.asarray(S.todense())
        DtD_dense = np.asarray(DtD)
        DtX_dense = np.asarray(DtX)
        Phi = np.linalg.solve(DtD_dense + lam * S_dense, DtX_dense)
        self.assertTrue(np.all(np.isfinite(Phi)), "Solution contains non-finite values")
        self.assertGreater(np.linalg.norm(Phi), 0, "Solution is zero")


class TestGetEventsTemplatesAverage(unittest.TestCase):
    """Test that method='average' matches manual epoch averaging."""

    def test_matches_manual_average(self) -> None:
        T = 30
        C = 2
        half_win = T // 2
        real_templates = {"A": np.outer(core.ricker(T, 5), np.ones(C))}
        rs = np.random.RandomState(np.array([42]))
        data, ix, labels = core.generate_data(
            real_templates, I=10, noise=0.05, mean_interval=60, rs=rs
        )

        templates, _ = core.get_events_templates(
            data, ix, labels, half_win=half_win, method="average"
        )

        for label in np.unique(labels):
            mask = labels == label
            epochs = core.epoch_data(data, ix[mask], -half_win, half_win)
            manual_avg: NDArray = epochs.mean(axis=0)
            assert_allclose(templates[label], manual_avg, atol=1e-14)


class TestLSImprovesOverAverage(unittest.TestCase):
    """Test that regularized LS produces better templates than averaging."""

    def test_ls_improves_over_average(self) -> None:
        T, C = 50, 2
        half_win = T // 2
        real_templates = {"A": np.outer(core.ricker(T, 10), np.hanning(C))}
        rs = np.random.RandomState(np.array([42]))
        data, ix, labels = core.generate_data(
            real_templates, I=5, noise=0.5, mean_interval=60, rs=rs
        )
        avg_t, _ = core.get_events_templates(
            data, ix, labels, half_win=half_win, method="average"
        )
        ls_t, _ = core.get_events_templates(
            data, ix, labels, half_win=half_win, method="ls", reg_temporal=500
        )
        for key in real_templates:
            avg_err = np.linalg.norm(avg_t[key] - real_templates[key])
            ls_err = np.linalg.norm(ls_t[key] - real_templates[key])
            self.assertLess(
                ls_err,
                avg_err,
                f"LS should have lower error than averaging for template {key}",
            )


class TestGenCorefilt(unittest.TestCase):
    """Tests for kernel symmetrization."""

    def test_symmetric_output(self) -> None:
        vec = np.array([1.0, -2.0])
        out: NDArray = core.gen_corefilt(vec, symm=True)
        self.assertEqual(len(out), 3)
        assert_allclose(out, [1.0, -2.0, 1.0])

    def test_nonsymmetric_passthrough(self) -> None:
        vec = np.array([1.0, -2.0, 3.0])
        out: NDArray = core.gen_corefilt(vec, symm=False)
        assert_allclose(out, vec)


class TestNoiseEstimation(unittest.TestCase):
    """Tests for noise variance estimators."""

    def _make_noise_data(self, sigma: float = 0.5) -> tuple[NDArray, NDArray, NDArray]:
        rs = np.random.RandomState(np.array([42]))
        T_len = 1000
        data: NDArray = sigma * rs.randn(T_len, 1)
        half_win = 20
        ix = np.arange(half_win, T_len - half_win, 3 * half_win)
        lb = np.zeros(len(ix), dtype=int)
        return data, ix, lb

    def test_pairwise_known_noise(self) -> None:
        sigma = 0.5
        data, ix, lb = self._make_noise_data(sigma)
        est: float = core.est_noise_var_pairwise(data, ix, lb, half_win=20)
        self.assertAlmostEqual(est, sigma**2, delta=0.1)

    def test_hpf_known_noise(self) -> None:
        sigma = 0.5
        data, _ix, _lb = self._make_noise_data(sigma)
        est: float = core.est_noise_var_hpf(data)
        self.assertGreater(est, 0)
        self.assertLess(est, sigma**2 * 5)

    def test_baseline_known_noise(self) -> None:
        sigma = 0.5
        data, ix, _lb = self._make_noise_data(sigma)
        est: float = core.est_noise_var_baseline(data, ix, half_win=20)
        self.assertAlmostEqual(est, sigma**2, delta=0.15)


class TestGCVAndLCurve(unittest.TestCase):
    """Test GCV and L-curve lambda selection."""

    def _setup_problem(
        self,
    ) -> tuple[NDArray, NDArray, scipy.sparse.csr_matrix, NDArray, NDArray]:
        T, C = 30, 2
        half_win = T // 2
        real_templates = {"A": np.outer(core.ricker(T, 5), np.ones(C))}
        rs = np.random.RandomState(np.array([42]))
        data, ix, labels = core.generate_data(
            real_templates, I=10, noise=0.1, mean_interval=40, rs=rs
        )
        events = np.unique(labels)
        M = len(events)
        D = np.asarray(
            core.get_design_matrix(data, ix, labels, half_win, events=events)
        )
        DtD: NDArray = D.T @ D
        DtX: NDArray = D.T @ data
        R = sp.linalg.convolution_matrix([1, -2, 1], T - 2)
        RR = R @ R.T
        S = scipy.sparse.block_diag([RR] * M, format="csr")
        return DtD, DtX, S, data, D

    def test_gcv_convergence(self) -> None:
        DtD, DtX, S, data, D = self._setup_problem()
        lam, info = core.select_lambda_gcv(DtD, DtX, S, data, D)
        self.assertGreater(lam, 0)
        self.assertTrue(np.isfinite(lam))
        S_dense = np.asarray(S.todense())
        DtD_dense = np.asarray(DtD)
        DtX_dense = np.asarray(DtX)
        Phi = np.linalg.solve(DtD_dense + lam * S_dense, DtX_dense)
        self.assertTrue(np.all(np.isfinite(Phi)))
        self.assertGreater(np.linalg.norm(Phi), 0)

    def test_lcurve_convergence(self) -> None:
        DtD, DtX, S, data, D = self._setup_problem()
        lam, info = core.select_lambda_lcurve(DtD, DtX, S, data, D)
        self.assertGreater(lam, 0)
        self.assertTrue(np.isfinite(lam))
        S_dense = np.asarray(S.todense())
        DtD_dense = np.asarray(DtD)
        DtX_dense = np.asarray(DtX)
        Phi = np.linalg.solve(DtD_dense + lam * S_dense, DtX_dense)
        self.assertTrue(np.all(np.isfinite(Phi)))
        self.assertGreater(np.linalg.norm(Phi), 0)


class TestSUREKernelOptimization(unittest.TestCase):
    """Test the SURE kernel optimization path (SURE(kernel) — paper's main contribution)."""

    def test_sure_kernel_converges(self) -> None:
        T, C = 50, 2
        half_win = T // 2
        real_templates = {"A": np.outer(core.ricker(T, 10), np.hanning(C))}
        rs = np.random.RandomState(np.array([42]))
        data, ix, labels = core.generate_data(
            real_templates, I=10, noise=0.1, mean_interval=60, rs=rs
        )
        templates, inner = core.get_events_templates(
            data,
            ix,
            labels,
            half_win=half_win,
            method="ls",
            reg_temporal=500,
            opt_reg_temporal=True,
            vec_mode=True,
            symm=True,
            vlen=3,
        )
        self.assertIsNotNone(inner)
        assert inner is not None
        kernel = inner["opt_val"]
        self.assertTrue(
            np.all(np.isfinite(kernel)),  # pyre-ignore[6]
            "Kernel contains non-finite values",
        )
        for key, tmpl in templates.items():
            self.assertTrue(np.all(np.isfinite(tmpl)), f"Template {key} is non-finite")

        ls_t, _ = core.get_events_templates(
            data, ix, labels, half_win=half_win, method="ls", reg_temporal=500
        )
        for key in real_templates:
            kernel_err = np.linalg.norm(templates[key] - real_templates[key])
            ls_err = np.linalg.norm(ls_t[key] - real_templates[key])
            self.assertLessEqual(kernel_err, ls_err * 1.1)


class TestErrorHandling(unittest.TestCase):
    """Test that invalid inputs raise appropriate errors."""

    def test_unknown_method_raises(self) -> None:
        data = np.zeros((100, 2))
        ix = np.array([50])
        lb = np.array(["A"])
        with self.assertRaises(ValueError):
            core.get_events_templates(data, ix, lb, half_win=10, method="nonexistent")

    def test_unknown_noise_method_raises(self) -> None:
        data = np.zeros((100, 2))
        with self.assertRaises(ValueError):
            core.est_noise_var(data, method="nonexistent")


if __name__ == "__main__":
    unittest.main()
