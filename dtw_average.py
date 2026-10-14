# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Filtered DTW "enhanced average" of Molina et al. (2024).

M. Molina et al., "Enhanced average for event-related potential analysis using
dynamic time warping", Biomed. Signal Process. Control 87:105531 (2024),
section 3.4. The authors released no code; this is an independent
implementation, the comparator for the P300 held-out evaluation.

Per class and per channel, for K epochs s_t[n] of length N:

1. reference r[n] = mean of the K epochs;
2. DTW between r (rows, index i) and s_t (columns, index j), local cost
   |r[i] - s_t[j]|, steps (1,1), (1,0), (0,1) with unit weights, path from
   (0, 0) to (N-1, N-1), no global window;
3. restricted path: drop every path element reached by a (0,1) step (a step
   that does not advance the reference index), which keeps the first matched
   trial sample per reference index, giving the warped trial s_w[i];
4. low-pass filter each warped trial (minimum-order Kaiser-window FIR, 60 dB);
5. template = mean over t of the filtered warped trials.

The frozen settings are the module constants; ``molina_template`` applies
them. Choices the paper leaves open (tie order and tolerance, transition
band, odd-reflection edges) were fixed before any real-data look. Indices are
0-based.
"""

from __future__ import annotations

import math

import numpy as np
import scipy
from numpy.typing import NDArray

# Step codes: (delta_i, delta_j), i = reference index, j = trial index.
DIAGONAL = 0  # (1, 1)
REFERENCE_ONLY = 1  # (1, 0): reference advances, trial sample repeated
TRIAL_ONLY = 2  # (0, 1): trial advances, dropped by the restricted path
_STEP_DELTAS: dict[int, tuple[int, int]] = {
    DIAGONAL: (1, 1),
    REFERENCE_ONLY: (1, 0),
    TRIAL_ONLY: (0, 1),
}
_STEP_NAMES: dict[str, int] = {
    "diagonal": DIAGONAL,
    "reference_only": REFERENCE_ONLY,
    "trial_only": TRIAL_ONLY,
}

LOCAL_COST = "absolute"
TIE_BREAK: tuple[str, str, str] = ("diagonal", "reference_only", "trial_only")
# Predecessor costs within this fraction of the total path cost count as tied:
# with the absolute cost exact ties are common and rounding must not decide
# them. In the pilot's sample, rounding-level gaps were <= 1e-14 of the cost
# and genuine gaps >= 2e-7.
TIE_RTOL = 1e-10
FS_HZ = 128.0
PASSBAND_EDGE_HZ = 20.0
# 20 + 0.15 * (64 - 20): a reconstruction of MATLAB lowpass() FIR mode at its
# default steepness; the paper gives no transition width.
STOPBAND_EDGE_HZ = 26.6
STOPBAND_ATTENUATION_DB = 60.0
KAISER_BETA = 5.65326
EDGE = "odd_reflection"
# design_lowpass(FS_HZ, PASSBAND_EDGE_HZ, STOPBAND_EDGE_HZ,
# STOPBAND_ATTENUATION_DB): kaiserord gives 72 taps, raised to 73 (type I).
TAPS: tuple[float, ...] = (
    -5.904833369713041e-05,
    0.0001997791549402377,
    0.0003660962140646108,
    2.3843190663176215e-05,
    -0.0006369155635456473,
    -0.0007221447756965553,
    0.0002839656462764123,
    0.0014316205029825232,
    0.001022478471093764,
    -0.0011071576081873554,
    -0.0025949968257444962,
    -0.0009756311040068379,
    0.0027005960942925976,
    0.00398227891630494,
    0.000148492773938973,
    -0.005256470999058796,
    -0.005233386678039765,
    0.002005438939959412,
    0.008823372714178334,
    0.005731799304296038,
    -0.006112746165038812,
    -0.013244808452018903,
    -0.004558850319136813,
    0.012909561360582394,
    0.018141494209027314,
    0.0003336340337072468,
    -0.02352401460659004,
    -0.022951533675953387,
    0.009500757686082445,
    0.040753669004197975,
    0.027025962839866458,
    -0.032401988146767646,
    -0.0763481444230959,
    -0.029759333823770846,
    0.11902873895653547,
    0.28908362642792923,
    0.36397993011885876,
    0.28908362642792923,
    0.11902873895653547,
    -0.029759333823770846,
    -0.0763481444230959,
    -0.032401988146767646,
    0.027025962839866458,
    0.040753669004197975,
    0.009500757686082445,
    -0.022951533675953387,
    -0.02352401460659004,
    0.0003336340337072468,
    0.018141494209027314,
    0.012909561360582394,
    -0.004558850319136813,
    -0.013244808452018903,
    -0.006112746165038812,
    0.005731799304296038,
    0.008823372714178334,
    0.002005438939959412,
    -0.005233386678039765,
    -0.005256470999058796,
    0.000148492773938973,
    0.00398227891630494,
    0.0027005960942925976,
    -0.0009756311040068379,
    -0.0025949968257444962,
    -0.0011071576081873554,
    0.001022478471093764,
    0.0014316205029825232,
    0.0002839656462764123,
    -0.0007221447756965553,
    -0.0006369155635456473,
    2.3843190663176215e-05,
    0.0003660962140646108,
    0.0001997791549402377,
    -5.904833369713041e-05,
)


def _check_signal(x: NDArray, name: str) -> NDArray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size < 1:
        raise ValueError(f"{name} must be a non-empty 1-D array, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values")
    return x


def local_cost(r: NDArray, s: NDArray, cost: str) -> NDArray:
    """Local cost matrix C[i, j] between reference r[i] and trial s[j]."""
    diff = r[:, None] - s[None, :]
    if cost == "absolute":  # Molina et al. (2024), eq. (1)
        return np.abs(diff)
    if cost == "squared":  # EEG2ERP reimplementation (appendix H.2)
        return diff * diff
    raise ValueError(f"unknown cost {cost!r}; expected 'absolute' or 'squared'")


def accumulated_cost(c: NDArray) -> NDArray:
    """D[i, j] = c[i, j] + min(D[i-1, j-1], D[i-1, j], D[i, j-1]), D[0, 0] = c[0, 0].

    Returned with an extra leading row and column (shape (N+1, M+1)): +inf
    except the virtual origin D_out[0, 0] = 0, so D_out[i+1, j+1] is the cost
    of the best path from (0, 0) to (i, j). Backtracking stops at (0, 0) and
    never uses the virtual origin.
    """
    n, m = c.shape
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    cl = c.tolist()
    rows = d.tolist()
    for i in range(1, n + 1):
        prev = rows[i - 1]
        cur = rows[i]
        ci = cl[i - 1]
        for j in range(1, m + 1):
            cur[j] = ci[j - 1] + min(prev[j - 1], prev[j], cur[j - 1])
    return np.array(rows)


def backtrack(
    d: NDArray,
    tie_break: tuple[str, ...] = TIE_BREAK,
    tie_rtol: float = TIE_RTOL,
) -> NDArray:
    """Tolerance-selected path from (0, 0) to (N-1, M-1) given the padded cost.

    At each point the predecessor with the smallest accumulated cost is taken.
    Predecessors within ``tie_rtol * D(N-1, M-1)`` of the smallest are treated
    as tied, and ties go to the first step in *tie_break* (names of the step
    that led to the current point). With the absolute cost, exact ties between
    different paths are common (|a - x| + |x - b| = |a - b| for x between a
    and b), so without a tolerance floating-point rounding (e.g. the summation
    order of the reference) would decide them. Returns an (L, 2) int array of
    (i, j).
    """
    order = [_STEP_NAMES[name] for name in tie_break]
    if sorted(order) != [DIAGONAL, REFERENCE_ONLY, TRIAL_ONLY]:
        raise ValueError(
            f"tie_break must be a permutation of {TIE_BREAK}, got {tie_break}"
        )
    if not (tie_rtol >= 0.0 and math.isfinite(tie_rtol)):
        raise ValueError(f"tie_rtol must be finite and >= 0, got {tie_rtol}")
    tol = tie_rtol * abs(float(d[-1, -1]))
    i, j = d.shape[0] - 1, d.shape[1] - 1  # padded coordinates of (N-1, M-1)
    path = [(i - 1, j - 1)]
    while (i, j) != (1, 1):
        vals = {
            code: float(d[i - di, j - dj]) for code, (di, dj) in _STEP_DELTAS.items()
        }
        best_val = min(vals.values())
        if not math.isfinite(best_val):
            raise AssertionError(f"no finite predecessor at {(i - 1, j - 1)}")
        code = next(c for c in order if vals[c] <= best_val + tol)
        di, dj = _STEP_DELTAS[code]
        i, j = i - di, j - dj
        path.append((i - 1, j - 1))
    return np.array(path[::-1], dtype=np.int64)


def dtw_path(
    r: NDArray,
    s: NDArray,
    cost: str = LOCAL_COST,
    tie_break: tuple[str, ...] = TIE_BREAK,
    tie_rtol: float = TIE_RTOL,
) -> tuple[NDArray, float]:
    """Return (selected path, optimal DP cost C*).

    The path is the tolerance-selected optimal path from ``backtrack``; its
    own cost can exceed C* = D(N-1, M-1) by at most (L-1) * tie_rtol * C*
    (exactly C* when tie_rtol = 0 or when no near-tie is merged).
    """
    r = _check_signal(r, "reference")
    s = _check_signal(s, "trial")
    d = accumulated_cost(local_cost(r, s, cost))
    if not math.isfinite(float(d[-1, -1])):
        raise ValueError("accumulated DTW cost is not finite (overflow)")
    return backtrack(d, tie_break, tie_rtol), float(d[-1, -1])


def restricted_warp(s: NDArray, path: NDArray, n: int) -> NDArray:
    """Warped trial: drop path elements reached by a (0,1) step, read s on the rest.

    Equivalent to s_w[i] = s[first j matched to reference index i]. Molina's
    rule to repeat the last sample when the result is shorter than the
    reference cannot trigger for a valid path (every reference index appears
    in it), so a length mismatch raises instead of being padded.
    """
    s = _check_signal(s, "trial")
    path = np.asarray(path)
    if (
        path.ndim != 2
        or path.shape[1] != 2
        or len(path) < 1
        or not np.issubdtype(path.dtype, np.integer)
    ):
        raise ValueError(
            f"path must be an (L, 2) integer array, got {path.shape} {path.dtype}"
        )
    if tuple(path[0]) != (0, 0) or tuple(path[-1]) != (n - 1, len(s) - 1):
        raise ValueError(f"path must run from (0, 0) to ({n - 1}, {len(s) - 1})")
    steps = np.diff(path, axis=0)
    if not all(tuple(d) in {(1, 1), (1, 0), (0, 1)} for d in steps):
        raise ValueError("path has a step other than (1,1), (1,0), (0,1)")
    keep = np.ones(len(path), dtype=bool)
    keep[1:] = ~((steps[:, 0] == 0) & (steps[:, 1] == 1))
    kept = path[keep]
    if len(kept) != n or not np.array_equal(kept[:, 0], np.arange(n)):
        raise AssertionError(
            "restricted path must hold exactly one element per reference index; "
            f"got {len(kept)} for n={n}"
        )
    return s[kept[:, 1]]


def design_lowpass(
    fs_hz: float,
    passband_edge_hz: float,
    stopband_edge_hz: float,
    stopband_attenuation_db: float,
    force_odd: bool = True,
) -> NDArray:
    """Minimum-order Kaiser-window low-pass FIR (cutoff at the transition midpoint).

    The order comes from Kaiser's formula (``scipy.signal.kaiserord``). With
    *force_odd*, an even tap count is increased by one so that the filter is
    type I with an integer group delay, which allows exact delay compensation.
    """
    nyq = fs_hz / 2.0
    if not 0.0 < passband_edge_hz < stopband_edge_hz < nyq:
        raise ValueError(
            "need 0 < passband edge < stopband edge < Nyquist, got "
            f"{passband_edge_hz}, {stopband_edge_hz}, {nyq}"
        )
    numtaps, beta = scipy.signal.kaiserord(
        stopband_attenuation_db, (stopband_edge_hz - passband_edge_hz) / nyq
    )
    if force_odd and numtaps % 2 == 0:
        numtaps += 1
    cutoff = 0.5 * (passband_edge_hz + stopband_edge_hz)
    return scipy.signal.firwin(
        numtaps, cutoff, window=("kaiser", beta), fs=fs_hz, scale=True
    )


def zero_phase_fir(x: NDArray, taps: NDArray, edge: str = EDGE) -> NDArray:
    """Apply a symmetric odd-length FIR once, delay-compensated (zero phase).

    ``edge='odd_reflection'``: extend each end by the filter half-length h
    with ``2*x[0] - x[h:0:-1]`` and ``2*x[-1] - x[-2:-h-2:-1]`` (point
    reflection about the end sample), filter, crop. Leaves constants and
    linear ramps unchanged. ``edge='even_reflection'`` (``x[h:0:-1]``,
    ``x[-2:-h-2:-1]``) and ``edge='zero'`` (zero padding) are
    boundary-sensitivity diagnostics only.
    """
    x = _check_signal(x, "signal")
    taps = np.asarray(taps, dtype=np.float64)
    if taps.ndim != 1 or len(taps) % 2 != 1 or not np.all(np.isfinite(taps)):
        raise ValueError(
            "taps must be a finite 1-D filter with an odd number of taps, got "
            f"shape {taps.shape}"
        )
    if not np.allclose(taps, taps[::-1], rtol=0, atol=1e-15):
        raise ValueError("taps must be symmetric")
    h = len(taps) // 2
    if edge in {"odd_reflection", "even_reflection"} and len(x) < h + 1:
        raise ValueError(
            f"signal length {len(x)} is too short for {edge} of length {h}"
        )
    if edge == "odd_reflection":
        left = 2.0 * x[0] - x[h:0:-1]
        right = 2.0 * x[-1] - x[-2 : -h - 2 : -1]
    elif edge == "even_reflection":
        left = x[h:0:-1]
        right = x[-2 : -h - 2 : -1]
    elif edge == "zero":
        left = right = np.zeros(h)
    else:
        raise ValueError(f"unknown edge {edge!r}")
    padded = np.concatenate([left, x, right])
    y = np.convolve(padded, taps, mode="valid")
    if y.shape != x.shape:
        raise AssertionError(f"filtered shape {y.shape} differs from {x.shape}")
    return y


def warp_epochs(
    epochs: NDArray,
    cost: str = LOCAL_COST,
    tie_break: tuple[str, ...] = TIE_BREAK,
    tie_rtol: float = TIE_RTOL,
) -> NDArray:
    """Warped trials s_w for epochs of shape (K, N, C); returns (K, N, C)."""
    epochs = np.asarray(epochs, dtype=np.float64)
    if epochs.ndim != 3 or min(epochs.shape) < 1:
        raise ValueError(f"epochs must have shape (K, N, C), got {epochs.shape}")
    if not np.all(np.isfinite(epochs)):
        raise ValueError("epochs contain non-finite values")
    k, n, c = epochs.shape
    reference = epochs.mean(axis=0)
    warped = np.empty_like(epochs)
    for ch in range(c):
        for t in range(k):
            path, _ = dtw_path(
                reference[:, ch], epochs[t, :, ch], cost, tie_break, tie_rtol
            )
            warped[t, :, ch] = restricted_warp(epochs[t, :, ch], path, n)
    return warped


def dtw_average(
    epochs: NDArray,
    taps: NDArray | None,
    cost: str = LOCAL_COST,
    tie_break: tuple[str, ...] = TIE_BREAK,
    edge: str = EDGE,
    tie_rtol: float = TIE_RTOL,
) -> NDArray:
    """Template (N, C) from epochs (K, N, C).

    ``taps=None`` gives the plain DTW-based average r^w; otherwise the
    filtered DTW-based average r^f (Molina et al. 2024, eq. 3).
    """
    warped = warp_epochs(epochs, cost, tie_break, tie_rtol)
    if taps is not None:
        k, _, c = warped.shape
        for t in range(k):
            for ch in range(c):
                warped[t, :, ch] = zero_phase_fir(warped[t, :, ch], taps, edge)
    template = warped.mean(axis=0)
    if not np.all(np.isfinite(template)):
        raise AssertionError("non-finite template")
    return template


def molina_template(epochs: NDArray) -> NDArray:
    """The comparator: filtered DTW average of one class's epochs (K, N, C).

    Applies the frozen settings (absolute cost, ``TIE_BREAK``, ``TIE_RTOL``,
    the 73 ``TAPS`` with odd-reflection edges) and returns an (N, C) template.
    """
    return dtw_average(
        epochs,
        np.asarray(TAPS, dtype=np.float64),
        cost=LOCAL_COST,
        tie_break=TIE_BREAK,
        edge=EDGE,
        tie_rtol=TIE_RTOL,
    )
