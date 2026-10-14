# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""AR(0)-AR(3) diagnostics of noise segments.

A series is a list of segments: 1-D arrays (the blocks of a long residual series) or
2-D arrays whose rows are equal-length segments (the noise estimator's baseline
segments). Each segment is centered separately and nothing crosses a segment
boundary. Conditional least squares (CLS) drops the first ``drop`` samples of every
segment for every order, so BIC uses the same n across orders.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from typing import NamedTuple, NotRequired, TYPE_CHECKING, TypedDict

import numpy as np
import scipy
from numpy.typing import NDArray

if TYPE_CHECKING:
    import core
else:
    _PREFIX = f"{__package__}." if __package__ else ""
    core = importlib.import_module(f"{_PREFIX}core")


class Placement(NamedTuple):
    """One channel's baseline segments: starts, length, recording length, blocks."""

    starts: NDArray
    n_b: int
    n_samples: int
    block_starts: Sequence[int] | None


class Bootstrap(NamedTuple):
    """CLS samples dropped per segment, residual-test lags, bootstrap sets, burn-in."""

    drop: int
    resid_lags: int
    n_boot: int
    burn_in: int


class Correction(NamedTuple):
    """Bias correction: pilot sets per iteration, iterations, convergence tolerance."""

    pilot: int
    iterations: int
    tol: float


class CorrectionInfo(TypedDict):
    boundary_events: int
    rejected_steps: int
    valid: bool
    reason: NotRequired[str]


class ResidualTest(TypedDict):
    q: float
    p_asymptotic: float | None
    residual_acf: NDArray


class Calibration(TypedDict):
    residual_p: float
    bic_gain_null_q95: NotRequired[float]


class OrderDiagnostics(TypedDict):
    coef: NDArray
    bic: float
    residual_acf: NDArray
    residual_q: float
    stationary: bool
    bic_gain_best_higher: float | None
    residual_p_calibrated: float | None
    calibration_valid: bool
    coef_bias_corrected: NotRequired[NDArray]
    bias_correction: NotRequired[CorrectionInfo]
    plugin_residual_p: NotRequired[float]
    bic_gain_null_q95: NotRequired[float | None]


class ChannelDiagnostics(TypedDict):
    acf: NDArray
    pacf: NDArray
    orders: dict[str, OrderDiagnostics]


class SeriesOrder(TypedDict):
    coef: NDArray
    sigma2: float
    bic: float
    stationary: bool
    q: float
    p_asymptotic: float | None
    residual_acf: NDArray


class SeriesDiagnostics(TypedDict):
    marginal_variance: float
    acf: NDArray
    pacf: NDArray
    orders: dict[str, SeriesOrder]


class TriggerDecision(TypedDict):
    baseline_order: int
    status: str
    n_channels: int
    n_failed_channels: int
    run_performance: bool
    median_bic: NotRequired[dict[int, float]]
    best_higher_order: NotRequired[int]
    median_bic_gain: NotRequired[float]
    bic_rule: NotRequired[bool]
    residual_rejections: NotRequired[int]
    residual_rule: NotRequired[bool]
    n_calibration_unavailable: NotRequired[int]


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------


def baseline_length(half_win: int, frac: float, minimum: int) -> int:
    """Baseline samples of a window; minimum 1 is ``core.est_noise_var_baseline``'s."""
    return max(minimum, int(2 * half_win * frac))


def segment_starts(
    onsets: NDArray,
    half_win: int,
    n_b: int,
    n_samples: int,
    block_starts: Sequence[int] | None = None,
) -> NDArray:
    """Start of each event's baseline segment; each must lie inside one block."""
    starts = np.asarray(onsets, dtype=np.int64) - half_win
    bounds = list(block_starts or [0]) + [n_samples]
    if bounds[0] != 0:
        raise ValueError(f"block_starts must begin at 0, got {bounds[0]}")
    for s in starts:
        if not 0 <= s < n_samples:
            raise ValueError(
                f"baseline segment start {s} outside the recording [0, {n_samples})"
            )
        b = int(np.searchsorted(bounds, s, side="right")) - 1
        if s + n_b > bounds[b + 1]:
            raise ValueError(
                f"baseline segment [{s}, {s + n_b}) crosses the end {bounds[b + 1]} "
                f"of its block (block bounds {bounds})"
            )
    return starts


def segments(x: NDArray, starts: NDArray, n_b: int) -> NDArray:
    """(n_segments, n_b) matrix of one channel's segments, each centered."""
    seg = x[starts[:, None] + np.arange(n_b)[None, :]]
    return seg - seg.mean(axis=1, keepdims=True)


def center(series: Sequence[NDArray]) -> list[NDArray]:
    return [np.asarray(s, dtype=float) - float(np.mean(s)) for s in series]


def overlap_fraction(
    onsets: NDArray, half_win: int, starts: NDArray, n_b: int, n_samples: int
) -> float:
    """Fraction of baseline samples inside another event's window [o - hw, o + hw)."""
    cover = np.zeros(n_samples + 1, dtype=np.int64)
    np.add.at(cover, np.clip(np.asarray(onsets) - half_win, 0, n_samples), 1)
    np.add.at(cover, np.clip(np.asarray(onsets) + half_win, 0, n_samples), -1)
    count = np.cumsum(cover)[:n_samples]
    idx = starts[:, None] + np.arange(n_b)[None, :]
    return float(np.mean(count[idx] >= 2))


# ---------------------------------------------------------------------------
# Fits and statistics (lags run along the last axis of every array)
# ---------------------------------------------------------------------------


def cls_fit(
    segs: Sequence[NDArray], p: int, drop: int
) -> tuple[NDArray, float, int, list[NDArray]]:
    """AR(p) CLS on common observations: coef, RSS, n, residuals per array."""
    if drop < p:
        raise ValueError(f"AR({p}) CLS needs drop >= p to form lags, got drop={drop}")
    ys = [s[..., drop:] for s in segs]
    if p == 0:
        return (
            np.zeros(0),
            float(sum(np.sum(y**2) for y in ys)),
            int(sum(y.size for y in ys)),
            ys,
        )
    lags = [
        np.stack([s[..., drop - k : s.shape[-1] - k] for k in range(1, p + 1)], axis=-1)
        for s in segs
    ]
    a = np.concatenate([lag.reshape(-1, p) for lag in lags])
    y = np.concatenate([yy.reshape(-1) for yy in ys])
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    resid = [yy - lag @ coef for yy, lag in zip(ys, lags)]
    return coef, float(sum(np.sum(r**2) for r in resid)), int(y.size), resid


def bic(rss: float, n: int, p: int) -> float:
    """BIC of a Gaussian AR(p) fit; undefined (ValueError) for a zero or nonfinite
    residual."""
    if not math.isfinite(rss):
        raise ValueError(
            f"AR({p}) residual sum of squares is {rss} over {n} observations: BIC "
            "undefined (overflow or nonfinite data)"
        )
    if not rss > 0:
        raise ValueError(
            f"AR({p}) residual sum of squares is {rss} over {n} observations: BIC "
            "undefined (constant or exactly predictable segments)"
        )
    base = n * math.log(rss / n)
    return base + p * math.log(n)


def pooled_acf(segs: Sequence[NDArray], lags: int) -> NDArray:
    """Pooled within-segment autocorrelation at lags 1..lags."""
    denom = float(sum(np.sum(s**2) for s in segs))
    n = sum(s.size for s in segs)
    if not math.isfinite(denom):
        raise ValueError(
            f"pooled ACF undefined: energy {denom} over {n} samples (overflow or "
            "nonfinite data)"
        )
    if not denom > 0:
        raise ValueError(f"pooled ACF undefined: zero energy over {n} samples")
    return np.array(
        [
            sum(float(np.sum(s[..., :-k] * s[..., k:])) for s in segs) / denom
            for k in range(1, lags + 1)
        ]
    )


def pacf_from_acf(r: NDArray) -> NDArray:
    """Durbin-Levinson partial autocorrelations from autocorrelations r[0] = lag 1."""
    rho = np.concatenate([[1.0], r])
    out, phi = [], np.zeros(0)
    for k in range(1, len(rho)):
        if k == 1:
            new = np.array([rho[1]])
        else:
            num = rho[k] - np.dot(phi, rho[1:k][::-1])
            den = 1.0 - np.dot(phi, rho[1:k])
            if not (np.isfinite(den) and abs(den) > 0.0):
                raise ValueError(
                    f"PACF undefined at lag {k}: Durbin-Levinson denominator {den}"
                )
            last = num / den
            new = np.concatenate([phi - last * phi[::-1], [last]])
        phi = new
        out.append(phi[-1])
    return np.array(out)


def residual_test(resid: Sequence[NDArray], lags: int, p: int) -> ResidualTest:
    """Box-Pierce-type Q = n * sum r_k^2 of the pooled residual ACF, with its
    asymptotic chi-square(lags - p) p-value."""
    r = pooled_acf(resid, lags)
    q = float(sum(x.size for x in resid) * np.sum(r**2))
    df = lags - p
    return {
        "q": q,
        "p_asymptotic": float(scipy.stats.chi2.sf(q, df)) if df > 0 else None,
        "residual_acf": r,
    }


def max_root_modulus(coef: NDArray) -> float:
    """Largest modulus of the AR characteristic roots; 0 for AR(0)."""
    if len(coef) == 0:
        return 0.0
    return float(np.max(np.abs(np.roots(np.concatenate([[1.0], -np.asarray(coef)])))))


def bic_gain(
    segs: Sequence[NDArray], p: int, higher: Sequence[int], drop: int
) -> float:
    """BIC(p) minus the smallest BIC of the higher orders (positive: a higher order
    is preferred)."""
    if not higher:
        raise ValueError(f"BIC gain of AR({p}) needs a higher order, got none")
    b = {q: bic(*cls_fit(segs, q, drop)[1:3], q) for q in [p, *higher]}
    return b[p] - min(b[q] for q in higher)


# ---------------------------------------------------------------------------
# Baseline segments: bootstrap calibration and bias correction
# ---------------------------------------------------------------------------


def simulate_segments(
    coef: NDArray,
    sigma2: float,
    at: Placement,
    n_rep: int,
    rs: np.random.RandomState,
    burn_in: int,
) -> NDArray:
    """(n_rep, n_segments, n_b) replicates at the same placements, each segment
    re-centered: an AR(p) series per block (blocks independent) over the span its
    segments cover, after a burn-in."""
    starts, n_b = at.starts, at.n_b
    bounds = list(at.block_starts or [0]) + [at.n_samples]
    out = np.empty((n_rep, len(starts), n_b))
    block_of = np.searchsorted(bounds, starts, side="right") - 1
    a = np.concatenate([[1.0], -np.asarray(coef)])
    for b in np.unique(block_of):
        idx = np.flatnonzero(block_of == b)
        lo, hi = int(starts[idx].min()), int(starts[idx].max()) + n_b
        e = rs.randn(n_rep, burn_in + hi - lo) * math.sqrt(sigma2)
        x = scipy.signal.lfilter([1.0], a, e, axis=1)[:, burn_in:]
        pos = (starts[idx] - lo)[:, None] + np.arange(n_b)[None, :]
        seg = x[:, pos]
        out[:, idx, :] = seg - seg.mean(axis=2, keepdims=True)
    return out


def bias_corrected_coef(
    coef: NDArray,
    at: Placement,
    p: int,
    drop: int,
    rs: np.random.RandomState,
    correction: Correction,
    burn_in: int,
    max_modulus: float = 0.999,
) -> tuple[NDArray, CorrectionInfo]:
    """Iterated bootstrap bias correction of CLS on centered segments: theta whose
    replicates have mean fitted coefficients equal to the observed ones. Invalid
    (info["valid"] False, with info["reason"]) when the observed fit is outside the
    stationary region (largest root modulus >= max_modulus), when a step reaches that
    boundary (it is halved toward the current theta), or when a final pilot batch at
    the returned theta misses the observed fit by more than the tolerance."""
    n_pilot, iterations, tol = correction
    target = np.asarray(coef, dtype=float)
    theta = target.copy()
    info: CorrectionInfo = {"boundary_events": 0, "rejected_steps": 0, "valid": True}
    if max_root_modulus(target) >= max_modulus:
        info["valid"] = False
        info["reason"] = "observed fit outside the stationary region"
        return target, info

    def mean_fit(th: NDArray) -> NDArray:
        sims = simulate_segments(th, 1.0, at, n_pilot, rs, burn_in)
        return np.mean([cls_fit([s], p, drop)[0] for s in sims], axis=0)

    for _ in range(iterations):
        step = theta + (target - mean_fit(theta))
        halvings = 0
        while max_root_modulus(step) >= max_modulus and halvings < 60:
            step = theta + 0.5 * (step - theta)
            halvings += 1
        if max_root_modulus(step) >= max_modulus:
            info["rejected_steps"] += 1
            step = theta
        info["boundary_events"] += int(halvings > 0)
        theta = step
    final = target - mean_fit(theta)
    worst = float(np.max(np.abs(final)))
    if info["boundary_events"] or info["rejected_steps"]:
        info["valid"] = False
        info["reason"] = "correction reaches the stationarity boundary"
    elif worst > tol:
        info["valid"] = False
        info["reason"] = f"not converged (max |residual| {worst:.4f} > {tol})"
    return theta, info


def bootstrap_pvalues(
    gen: NDArray,
    sigma2: float,
    at: Placement,
    p: int,
    higher: Sequence[int],
    settings: Bootstrap,
    rs: np.random.RandomState,
    q_obs: float,
) -> Calibration:
    """Under the generating model `gen`: the parametric-bootstrap p-value of Q and the
    95th percentile of the best-higher-order BIC gain."""
    drop, resid_lags, n_boot, burn_in = settings
    qb: list[float] = []
    gb: list[float] = []
    for done in range(0, n_boot, 50):  # chunks bound memory on long recordings
        sims = simulate_segments(gen, sigma2, at, min(50, n_boot - done), rs, burn_in)
        qb += [
            residual_test(cls_fit([s], p, drop)[3], resid_lags, p)["q"] for s in sims
        ]
        if higher:
            gb += [bic_gain([s], p, higher, drop) for s in sims]
    out: Calibration = {
        "residual_p": float((1 + np.sum(np.array(qb) >= q_obs)) / (n_boot + 1))
    }
    if higher:
        out["bic_gain_null_q95"] = float(np.percentile(gb, 95))
    return out


def channel_diagnostics(
    x: NDArray,
    at: Placement,
    orders: Sequence[int],
    settings: Bootstrap,
    acf_lags: int,
    rs: np.random.RandomState,
    correction: Correction,
    rs_pilot: np.random.RandomState,
) -> ChannelDiagnostics:
    """Every diagnostic of one channel's baseline segments.

    Per order: CLS fit, BIC, residual test, BIC gain of the best higher order and,
    for a stationary fit, its bootstrap calibration. CLS on short centered segments is
    biased toward zero at high phi, so the plug-in model rejects the true model; the
    calibration uses the bias-corrected model (its replicates follow the plug-in
    ones in `rs`). An invalid correction leaves the order uncalibrated.
    """
    seg = segments(x, at.starts, at.n_b)
    acf = pooled_acf([seg], acf_lags)
    out: ChannelDiagnostics = {"acf": acf, "pacf": pacf_from_acf(acf), "orders": {}}
    for p in orders:
        higher = [h for h in orders if h > p]
        out["orders"][str(p)] = _order_diagnostics(
            seg, at, p, higher, settings, rs, correction, rs_pilot
        )
    return out


def _order_diagnostics(
    seg: NDArray,
    at: Placement,
    p: int,
    higher: Sequence[int],
    settings: Bootstrap,
    rs: np.random.RandomState,
    correction: Correction,
    rs_pilot: np.random.RandomState,
) -> OrderDiagnostics:
    coef, rss, n, resid = cls_fit([seg], p, settings.drop)
    test = residual_test(resid, settings.resid_lags, p)
    entry: OrderDiagnostics = {
        "coef": coef,
        "bic": bic(rss, n, p),
        "residual_acf": test["residual_acf"],
        "residual_q": test["q"],
        "stationary": max_root_modulus(coef) < 1.0,
        "bic_gain_best_higher": (
            bic_gain([seg], p, higher, settings.drop) if higher else None
        ),
        "residual_p_calibrated": None,
        "calibration_valid": False,
    }
    if not entry["stationary"] or settings.n_boot <= 0:
        return entry
    q = entry["residual_q"]
    plug = bootstrap_pvalues(coef, rss / n, at, p, higher, settings, rs, q)
    chosen: Calibration | None = plug
    if p > 0:
        theta, info = bias_corrected_coef(
            coef, at, p, settings.drop, rs_pilot, correction, settings.burn_in
        )
        entry["coef_bias_corrected"] = theta
        entry["bias_correction"] = info
        chosen = None
        if info["valid"]:  # these replicates follow the plug-in ones in rs
            chosen = bootstrap_pvalues(theta, rss / n, at, p, higher, settings, rs, q)
    entry["plugin_residual_p"] = plug["residual_p"]
    entry["calibration_valid"] = chosen is not None
    if chosen is not None:
        entry["residual_p_calibrated"] = chosen["residual_p"]
        entry["bic_gain_null_q95"] = chosen.get("bic_gain_null_q95")
    return entry


def trigger_decision(
    channels: Sequence[ChannelDiagnostics | None],
    orders: Sequence[int],
    baseline_order: int,
    margin: float,
    alpha: float,
    fraction: float,
) -> TriggerDecision:
    """Whether a modality's noise-model comparison runs (OQ-B17).

    BIC rule: the median BIC of the best higher order beats the baseline order's by
    more than margin. Residual rule: more than fraction of all channels reject the
    baseline order at alpha. Failed channels and unavailable calibrations count as
    not rejecting; with no usable channel, or more than half failed, the decision is
    "indeterminate" (no comparison). `channels`: None for a failed channel.
    """
    if baseline_order not in orders:
        raise ValueError(
            f"baseline order {baseline_order} is not among orders {list(orders)}"
        )
    higher = [p for p in orders if p > baseline_order]
    if not higher:
        raise ValueError(
            f"no order above the baseline order {baseline_order} in {list(orders)}"
        )
    ok = [c for c in channels if c is not None]
    n_all = len(channels)
    if not ok or len(ok) < n_all - fraction * n_all:
        return {
            "baseline_order": baseline_order,
            "status": "indeterminate",
            "n_channels": n_all,
            "n_failed_channels": n_all - len(ok),
            "run_performance": False,
        }
    med_bic = {
        p: float(np.median([c["orders"][str(p)]["bic"] for c in ok])) for p in orders
    }
    best = min(higher, key=lambda p: med_bic[p])
    gain = med_bic[baseline_order] - med_bic[best]
    base = [c["orders"][str(baseline_order)] for c in ok]
    pvals = [e["residual_p_calibrated"] for e in base]
    rejects = sum(1 for p in pvals if p is not None and p < alpha)
    return {
        "baseline_order": baseline_order,
        "median_bic": med_bic,
        "best_higher_order": best,
        "median_bic_gain": gain,
        "bic_rule": gain > margin,
        "residual_rejections": rejects,
        "n_channels": n_all,
        "n_failed_channels": n_all - len(ok),
        "residual_rule": rejects > fraction * n_all,
        "status": "evaluated",
        "n_calibration_unavailable": sum(1 for p in pvals if p is None),
        "run_performance": bool(gain > margin or rejects > fraction * n_all),
    }


# ---------------------------------------------------------------------------
# Long residual series
# ---------------------------------------------------------------------------


def ols_residual(
    data: NDArray,
    ix: NDArray,
    lb: NDArray,
    half_win: int,
    ols: NDArray,
) -> tuple[NDArray, float, int]:
    """r = X - D theta_hat for OLS templates stacked in np.unique(lb) order; returns
    r, ||D^T r||_F / ||D^T X||_F (0 for an exact OLS solution), and P = columns of D."""
    events = np.unique(lb)
    design = core.get_design_matrix(
        data, ix, lb, half_win, events=events, sparse_out=True
    )
    theta = np.vstack([ols[i] for i in range(len(events))])
    resid = data - design @ theta
    scale = float(np.linalg.norm(design.T @ data))
    if not (math.isfinite(scale) and scale > 0.0):
        raise ValueError(f"OLS residual orthogonality undefined: ||D^T X|| = {scale}")
    ortho = float(np.linalg.norm(design.T @ resid)) / scale
    return resid, ortho, int(design.shape[1])


def window_matched_variances(
    x: NDArray, bounds: Sequence[int], ix: NDArray, half_win: int, n_b: int
) -> dict[str, float]:
    """Variance as the noise estimator measures it, np.var (mean removed) over n_b
    samples: "tiled" over windows tiling every block (remainder dropped);
    "event_baselines" over the baseline windows [onset - half_win, + n_b) of all
    events."""
    tiled = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        m = (b - a) // n_b * n_b
        tiled.append(np.var(x[a : a + m].reshape(-1, n_b), axis=1))
    if not sum(t.size for t in tiled):
        raise ValueError(f"no {n_b}-sample window fits in the blocks {list(bounds)}")
    starts = np.asarray(ix, dtype=np.int64) - half_win
    starts = starts[(starts >= 0) & (starts + n_b <= len(x))]
    if not starts.size:
        raise ValueError(
            f"no baseline window [onset - {half_win}, + {n_b}) lies inside the "
            f"{len(x)}-sample series ({len(ix)} events)"
        )
    ev = np.var(x[starts[:, None] + np.arange(n_b)[None, :]], axis=1)
    return {
        "tiled": float(np.mean(np.concatenate(tiled))),
        "event_baselines": float(np.mean(ev)),
    }


def series_diagnostics(
    series: Sequence[NDArray],
    orders: Sequence[int],
    drop: int,
    acf_lags: int,
    resid_lags: int,
) -> SeriesDiagnostics:
    """Marginal variance, ACF/PACF, and per order the CLS fit, BIC and residual test."""
    segs = center(series)
    acf = pooled_acf(segs, acf_lags)
    out: SeriesDiagnostics = {
        "marginal_variance": float(
            sum(np.sum(s**2) for s in segs) / sum(s.size for s in segs)
        ),
        "acf": acf,
        "pacf": pacf_from_acf(acf),
        "orders": {},
    }
    for p in orders:
        coef, rss, n, resid = cls_fit(segs, p, drop)
        test = residual_test(resid, resid_lags, p)
        out["orders"][str(p)] = {
            "coef": coef,
            "sigma2": rss / n,
            "bic": bic(rss, n, p),
            "stationary": max_root_modulus(coef) < 1.0,
            "q": test["q"],
            "p_asymptotic": test["p_asymptotic"],
            "residual_acf": test["residual_acf"],
        }
    return out


def bootstrap_residual_test(
    coef: NDArray,
    sigma2: float,
    lengths: Sequence[int],
    p: int,
    settings: Bootstrap,
    rs: np.random.RandomState,
    q_obs: float,
) -> float | None:
    """Parametric-bootstrap p-value (1 + #{Q* >= Q}) / (1 + B) of the order-p residual
    test under the fitted AR(p): one independent series per block length, each
    centered; None for a nonstationary fit."""
    if max_root_modulus(coef) >= 1.0:
        return None
    drop, lags, n_boot, burn_in = settings
    a = np.concatenate([[1.0], -np.asarray(coef)])
    qs: list[float] = []
    for _ in range(n_boot):
        sim = [
            scipy.signal.lfilter(
                [1.0], a, rs.randn(burn_in + int(n)) * math.sqrt(sigma2)
            )[burn_in:]
            for n in lengths
        ]
        qs.append(residual_test(cls_fit(center(sim), p, drop)[3], lags, p)["q"])
    return float((1 + np.sum(np.array(qs) >= q_obs)) / (1 + n_boot))
