# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""AR diagnostics: order recovery, calibration of the residual test, segment and block
boundaries, the long-series fits, the OLS residual and the EEG/ECG trigger rule."""

from __future__ import annotations

import importlib
import unittest
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import scipy
from numpy.typing import NDArray

if TYPE_CHECKING:
    import ar_tools
    import core
else:
    ar_tools = importlib.import_module("ar_tools")
    core = importlib.import_module("core")

ORDERS = (0, 1, 2, 3)
SETTINGS = ar_tools.Bootstrap(drop=3, resid_lags=5, n_boot=39, burn_in=300)
CORRECTION = ar_tools.Correction(pilot=20, iterations=2, tol=0.02)
PILOT = ar_tools.Correction(pilot=50, iterations=3, tol=0.005)


def ar_series(
    coef: Sequence[float], n: int, rs: np.random.RandomState, burn: int = 500
) -> NDArray:
    a = np.concatenate([[1.0], -np.asarray(coef)])
    return scipy.signal.lfilter([1.0], a, rs.randn(n + burn))[burn:]


def placement(
    n_events: int = 120, gap: int = 45, hw: int = 20, n_b: int = 12
) -> ar_tools.Placement:
    """Baseline segments of onsets every `gap` samples."""
    onsets = hw + np.arange(n_events) * gap + 5
    n = int(onsets[-1] + hw + 5)
    return ar_tools.Placement(ar_tools.segment_starts(onsets, hw, n_b, n), n_b, n, None)


def diagnostics(
    x: NDArray, at: ar_tools.Placement, orders: Sequence[int]
) -> ar_tools.ChannelDiagnostics:
    return ar_tools.channel_diagnostics(
        x,
        at,
        orders,
        SETTINGS,
        5,
        np.random.RandomState(0),
        CORRECTION,
        np.random.RandomState(1),
    )


class OrderTest(unittest.TestCase):
    """Why: the supplement's AR-order statements rest on the calibrated residual test,
    because per-segment centering biases the raw BIC toward higher orders."""

    def test_the_smallest_order_not_rejected_is_the_true_order(self) -> None:
        at = placement(n_events=200, gap=32, hw=51, n_b=30)
        for coef in ([0.8], [0.9, -0.5]):
            x = ar_series(coef, at.n_samples, np.random.RandomState(1))
            out = diagnostics(x, at, ORDERS)["orders"]
            pvals = [out[str(p)]["residual_p_calibrated"] for p in ORDERS]
            chosen = next(
                p for p, v in zip(ORDERS, pvals) if v is not None and v >= 0.05
            )
            self.assertEqual(chosen, len(coef), (coef, pvals))

    def test_raw_bic_prefers_higher_orders_on_centered_segments(self) -> None:
        at = placement(n_events=800, gap=50, hw=30, n_b=30)
        x = ar_series([0.8], at.n_samples, np.random.RandomState(1))
        seg = ar_tools.segments(x, at.starts, at.n_b)
        self.assertGreater(ar_tools.bic_gain([seg], 1, [2, 3], 3), 10.0)
        raw = x[at.starts[:, None] + np.arange(at.n_b)[None, :]]
        self.assertLess(ar_tools.bic_gain([raw], 1, [2, 3], 3), 0.0)

    def test_without_bootstrap_sets_every_order_stays_uncalibrated(self) -> None:
        at = placement()
        x = ar_series([0.6], at.n_samples, np.random.RandomState(3))
        rs, pilot = np.random.RandomState(0), np.random.RandomState(1)
        out = ar_tools.channel_diagnostics(
            x, at, ORDERS, SETTINGS._replace(n_boot=0), 5, rs, CORRECTION, pilot
        )
        for e in out["orders"].values():
            self.assertEqual(
                (e["residual_p_calibrated"], e["calibration_valid"]), (None, False)
            )
        # Neither RandomState was drawn from.
        self.assertEqual(rs.randn(), np.random.RandomState(0).randn())
        self.assertEqual(pilot.randn(), np.random.RandomState(1).randn())

    def test_undefined_statistics_raise_instead_of_returning_inf(self) -> None:
        # A zero or overflowing residual leaves BIC undefined, and constant or
        # overflowing segments the ACF; the script records either as a failed channel.
        with self.assertRaisesRegex(ValueError, r"AR\(2\) .* is 0.0 over 100 "):
            ar_tools.bic(0.0, 100, 2)
        with self.assertRaisesRegex(ValueError, r"AR\(1\) .* is inf over 100 "):
            ar_tools.bic(float("inf"), 100, 1)
        at = placement()
        with self.assertRaisesRegex(ValueError, "pooled ACF undefined: zero energy"):
            diagnostics(np.ones(at.n_samples), at, ORDERS)
        huge = 1e155 * (-1.0) ** np.arange(at.n_samples)  # finite, squares overflow
        with (
            np.errstate(over="ignore"),
            self.assertRaisesRegex(ValueError, "pooled ACF undefined: energy inf"),
        ):
            diagnostics(huge, at, ORDERS)

    def test_pacf_of_an_ar1_acf_cuts_off_after_lag_one(self) -> None:
        pacf = ar_tools.pacf_from_acf(0.7 ** np.arange(1, 8))
        np.testing.assert_allclose(pacf, [0.7] + [0.0] * 6, atol=1e-12)


class CalibrationTest(unittest.TestCase):
    """Why: the bootstrap residual test must hold its level under the true model,
    reject a misspecified order, and be calibrated from a bias-corrected model."""

    def test_level_under_the_true_model_and_power_against_a_wrong_order(self) -> None:
        at = placement(n_events=80, gap=32, hw=51, n_b=30)
        rs = np.random.RandomState(11)
        pvals, wrong = [], []
        for _ in range(40):
            seg = ar_tools.segments(
                ar_series([0.9, -0.4], at.n_samples, rs), at.starts, at.n_b
            )
            for p, out in ((2, pvals), (0, wrong)):
                coef, rss, n, resid = ar_tools.cls_fit([seg], p, 3)
                q = ar_tools.residual_test(resid, 5, p)["q"]
                b = ar_tools.bootstrap_pvalues(
                    coef, rss / n, at, p, [], SETTINGS, rs, q
                )
                out.append(b["residual_p"])
        self.assertLess(scipy.stats.kstest(pvals, "uniform")[0], 0.25)
        self.assertLessEqual(np.mean(np.array(pvals) < 0.05), 0.1)
        self.assertGreaterEqual(np.mean(np.array(wrong) < 0.05), 0.9)

    def test_bias_correction_recovers_a_high_phi(self) -> None:
        at = placement(n_events=600, gap=32, hw=51, n_b=30)
        x = ar_series([0.98], at.n_samples, np.random.RandomState(5), burn=3000)
        coef = ar_tools.cls_fit([ar_tools.segments(x, at.starts, at.n_b)], 1, 3)[0]
        self.assertLess(coef[0], 0.93)  # CLS on centered 30-sample segments
        theta, info = ar_tools.bias_corrected_coef(
            coef, at, 1, 3, np.random.RandomState(1), PILOT, 2000
        )
        self.assertAlmostEqual(theta[0], 0.98, delta=0.015)
        self.assertTrue(info["valid"], info.get("reason"))

    def test_a_correction_reaching_the_stationarity_boundary_is_invalid(self) -> None:
        at = placement(n_events=300, gap=32, hw=51, n_b=30)
        theta, info = ar_tools.bias_corrected_coef(
            np.array([0.97]), at, 1, 3, np.random.RandomState(2), PILOT, 2000
        )
        self.assertLess(ar_tools.max_root_modulus(theta), 0.999)
        self.assertGreaterEqual(info["boundary_events"], 1)
        self.assertFalse(info["valid"])


class BoundariesTest(unittest.TestCase):
    """Why: no lag product, regressor or residual may cross a segment boundary."""

    def test_baseline_statistics_ignore_segment_offsets_and_order(self) -> None:
        rs = np.random.RandomState(0)
        at = placement()
        x = ar_series([0.6], at.n_samples, rs)
        y = x.copy()
        for s in at.starts:
            y[s : s + at.n_b] += rs.randn() * 100.0
        a, b = diagnostics(x, at, (0, 1, 2)), diagnostics(y, at, (0, 1, 2))
        np.testing.assert_allclose(a["acf"], b["acf"], rtol=1e-8)
        for p in ("0", "1", "2"):
            self.assertAlmostEqual(
                a["orders"][p]["bic"], b["orders"][p]["bic"], places=5
            )
        seg = ar_tools.segments(x, at.starts, at.n_b)
        acf = ar_tools.pooled_acf([seg], 3)
        np.testing.assert_allclose(acf, ar_tools.pooled_acf([seg[::-1]], 3))
        np.testing.assert_allclose(acf, ar_tools.pooled_acf(list(seg), 3), rtol=1e-12)

    def test_long_series_statistics_stay_inside_each_block(self) -> None:
        rs = np.random.RandomState(2)
        a, b = ar_series([0.5], 3000, rs) + 50.0, ar_series([0.5], 3000, rs) - 50.0
        joint = ar_tools.series_diagnostics([a, b], (1,), 3, 3, 3)
        sep = ar_tools.center([a, b])
        num = sum(float(np.sum(x[:-1] * x[1:])) for x in sep)
        den = sum(float(np.sum(x**2)) for x in sep)
        self.assertAlmostEqual(joint["acf"][0], num / den, places=12)
        y = np.concatenate([x[3:] for x in sep])
        z = np.concatenate([x[2:-1] for x in sep])
        coef = joint["orders"]["1"]["coef"][0]
        self.assertAlmostEqual(coef, float(z @ y / (z @ z)), places=12)
        whole = ar_tools.series_diagnostics([np.concatenate([a, b])], (1,), 3, 3, 3)
        self.assertNotAlmostEqual(joint["acf"][0], whole["acf"][0], places=3)

    def test_a_segment_leaving_its_block_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[90, 102\) crosses the end 100"):
            ar_tools.segment_starts(np.array([25, 110]), 20, 12, 200, [0, 100])
        for onset in (10, 230):  # starts -10 and 210: outside the 200-sample recording
            with self.assertRaisesRegex(ValueError, "outside the recording"):
                ar_tools.segment_starts(np.array([onset]), 20, 12, 200, [0, 100])
        starts = ar_tools.segment_starts(np.array([25, 125]), 20, 12, 200, [0, 100])
        np.testing.assert_array_equal(starts, [5, 105])

    def test_bootstrap_blocks_are_independent_and_segments_centered(self) -> None:
        at = ar_tools.Placement(np.array([5, 30, 105]), 10, 200, [0, 100])
        coef = np.array([0.5])
        sims = ar_tools.simulate_segments(
            coef, 1.0, at, 4, np.random.RandomState(0), 50
        )
        np.testing.assert_allclose(sims.mean(axis=2), 0.0, atol=1e-12)
        # Each block is simulated over its own span: moving the second block (and its
        # segment) later leaves every replicate unchanged.
        moved = ar_tools.Placement(np.array([5, 30, 1105]), 10, 1200, [0, 1100])
        again = ar_tools.simulate_segments(
            coef, 1.0, moved, 4, np.random.RandomState(0), 50
        )
        np.testing.assert_array_equal(sims, again)

    def test_overlap_fraction(self) -> None:
        # Windows [0, 20) and [10, 30); baselines [0, 5) and [10, 15): only the second
        # lies in another event's window.
        f = ar_tools.overlap_fraction(np.array([10, 20]), 10, np.array([0, 10]), 5, 40)
        self.assertEqual(f, 0.5)
        # Windows [0, 20) and [12, 32); baselines [0, 14) and [12, 26) have 2 and 8
        # samples inside the other window.
        f = ar_tools.overlap_fraction(np.array([10, 22]), 10, np.array([0, 12]), 14, 40)
        self.assertAlmostEqual(f, 10 / 28, places=15)


class LongSeriesTest(unittest.TestCase):
    """Why: the residual-series table reports CLS coefficients and BIC-best orders."""

    def test_cls_recovers_a_known_ar2_and_bic_picks_it(self) -> None:
        rs = np.random.RandomState(1)
        segs = [ar_series([1.2, -0.5], n, rs) for n in (20000, 15000, 25000)]
        d = ar_tools.series_diagnostics(segs, ORDERS, 3, 10, 5)
        np.testing.assert_allclose(d["orders"]["2"]["coef"], [1.2, -0.5], atol=0.02)
        self.assertEqual(min(d["orders"], key=lambda p: d["orders"][p]["bic"]), "2")
        p1, p2 = d["orders"]["1"]["p_asymptotic"], d["orders"]["2"]["p_asymptotic"]
        if p1 is None or p2 is None:
            self.fail(f"asymptotic p undefined: order 1 {p1}, order 2 {p2}")
        self.assertLess(p1, 1e-6)
        self.assertGreater(p2, 1e-3)


class OlsResidualTest(unittest.TestCase):
    """Why: the residual series must be X - D theta_hat of the full-data OLS fit, and
    its window-matched variance measured exactly as the noise estimator measures it."""

    def test_residual_is_orthogonal_to_the_design_and_equals_window_subtraction(
        self,
    ) -> None:
        rs = np.random.RandomState(6)
        n, hw, ch = 6000, 20, 3
        ix = np.sort(rs.choice(np.arange(hw + 5, n - hw - 5), 300, replace=False))
        lb = rs.randint(0, 2, size=len(ix))
        shapes = [
            np.sin(np.arange(2 * hw) / 4.0)[:, None] * np.ones(ch),
            np.hanning(2 * hw)[:, None] * np.array([1.0, 2.0, 3.0]),
        ]
        data = rs.randn(n, ch) * 0.5
        for i, label in zip(ix, lb):
            data[i - hw : i + hw] += shapes[label]
        templates, _ = core.get_events_templates(data, ix, lb, half_win=hw, method="ls")
        ols = np.stack([np.asarray(templates[k]) for k in ("0", "1")])
        resid, ortho, p = ar_tools.ols_residual(data, ix, lb, hw, ols)
        self.assertLess(ortho, 1e-8)
        self.assertEqual(p, 2 * 2 * hw)
        manual = data.copy()
        for i, label in zip(ix, lb):
            manual[i - hw : i + hw] -= ols[label]
        np.testing.assert_allclose(resid, manual, atol=1e-12)

    def test_window_matched_variance_equals_the_estimators_formula(self) -> None:
        rs = np.random.RandomState(9)
        x = ar_series([0.95], 12011, rs)
        bounds = [0, 6013, 12011]
        ix = np.sort(rs.choice(np.arange(60, 12011 - 60), 200, replace=False))
        x[6000:6013] = 1e3  # dropped remainders of the two blocks
        x[12011 - 28 :] = -1e3
        ix = ix[[not (5900 < i < 6200 or i > 11800) for i in ix]]
        for hw in (2, 4, 6, 51):  # 1-3 samples, where the AR minimum of 4 differs
            n_est = ar_tools.baseline_length(hw, 0.3, 1)
            wm = ar_tools.window_matched_variances(x, bounds, ix, hw, n_est)
            want = core.est_noise_var_baseline(x[:, None], ix, hw)
            self.assertAlmostEqual(wm["event_baselines"], float(want), places=14)
        tiles = [x[0:6000].reshape(-1, 30), x[6013 : 6013 + 5970].reshape(-1, 30)]
        manual = np.mean(np.concatenate([np.var(t, axis=1) for t in tiles]))
        self.assertAlmostEqual(wm["tiled"], float(manual), places=14)


class TriggerTest(unittest.TestCase):
    """Why: the trigger decides whether the EEG/ECG noise-model comparison runs."""

    def rec(
        self, bics: Sequence[float], p: float | None
    ) -> ar_tools.ChannelDiagnostics:
        orders: dict[str, ar_tools.OrderDiagnostics] = {
            str(k): {
                "coef": np.zeros(k),
                "bic": v,
                "residual_acf": np.zeros(5),
                "residual_q": 0.0,
                "stationary": True,
                "bic_gain_best_higher": None,
                "residual_p_calibrated": p,
                "calibration_valid": p is not None,
            }
            for k, v in enumerate(bics)
        }
        return {"acf": np.zeros(5), "pacf": np.zeros(5), "orders": orders}

    def decide(
        self, channels: Sequence[ar_tools.ChannelDiagnostics | None]
    ) -> ar_tools.TriggerDecision:
        return ar_tools.trigger_decision(channels, ORDERS, 1, 10.0, 0.05, 0.5)

    def test_bic_margin_and_residual_majority(self) -> None:
        d = self.decide([self.rec([0, -100, -111, -105], 0.5)])
        self.assertTrue(d.get("bic_rule") and d["run_performance"])
        d = self.decide([self.rec([0, -100, -109, -105], 0.5)])
        self.assertFalse(d["run_performance"])
        recs = [self.rec([0, -100, -101, -101], p) for p in (0.01, 0.01, 0.5)]
        self.assertTrue(self.decide(recs).get("residual_rule"))
        self.assertFalse(self.decide(recs[1:]).get("residual_rule"))  # 1 of 2

    def test_failed_channels_and_unavailable_calibrations_do_not_reject(self) -> None:
        reject = self.rec([0.0] * 4, 0.01)
        bad = None  # a failed channel
        self.assertTrue(self.decide([reject, bad, reject]).get("residual_rule"))
        self.assertFalse(self.decide([reject, bad, bad, reject]).get("residual_rule"))
        d = self.decide([self.rec([0.0] * 4, None), reject])
        self.assertEqual(
            (d.get("residual_rejections"), d.get("n_calibration_unavailable")), (1, 1)
        )
        d = self.decide([reject, bad, bad, bad])
        self.assertEqual((d["status"], d["run_performance"]), ("indeterminate", False))
