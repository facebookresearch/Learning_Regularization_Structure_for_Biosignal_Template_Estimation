# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

import warnings
from typing import NamedTuple

import scipy
import scipy as sp
from scipy import signal as sig

"""
Core template estimation methods.

This module implements:
- OLS template estimation
- Regularized LS with fixed Tikhonov regularization (Reg. LS)
- SURE-optimized lambda with fixed 2nd-order differences (SURE(lambda))
- SURE-optimized convolution kernel (SURE(kernel))
- GCV-selected lambda (GCV)
- L-curve-selected lambda (L-curve)
- Empirical Bayes / Evidence Maximization (EmpBayes)
- Data generation utilities for simulations
"""


import numpy as np
from numpy.typing import NDArray


def ricker(points: int, a: float) -> NDArray:
    """Compute a Ricker wavelet (Mexican hat).

    Replaces ``scipy.signal.ricker`` which was removed in scipy >= 1.15.

    Parameters
    ----------
    points : int
        Number of samples in the output wavelet.
    a : float
        Width parameter of the wavelet.

    Returns
    -------
    NDArray
        1-D array of shape ``(points,)`` containing the wavelet samples.
    """
    A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
    wsq = a**2
    vec = np.arange(0, points) - (points - 1.0) / 2
    xsq = vec**2
    mod = 1 - xsq / wsq
    gauss = np.exp(-xsq / (2 * wsq))
    return A * mod * gauss


def epoch_data(data: NDArray, indices: NDArray, tmin: int, tmax: int) -> NDArray:
    """Extract fixed-length epochs from continuous data around event indices.

    Parameters
    ----------
    data : NDArray
        Continuous data array of shape ``(T, C)`` where *T* is the number of
        time samples and *C* is the number of channels.
    indices : NDArray
        1-D array of integer event-onset indices into *data*.
    tmin : int
        Start of the epoch window relative to each event index (inclusive).
    tmax : int
        End of the epoch window relative to each event index (exclusive).

    Returns
    -------
    NDArray
        3-D array of shape ``(n_events, tmax - tmin, C)`` containing the
        extracted epochs.  Epochs that would exceed the data boundaries are
        left as zeros.
    """
    epochs = np.zeros((len(indices), tmax - tmin, data.shape[1]))
    for ii, idx in enumerate(indices):
        sl = slice(idx + tmin, idx + tmax)
        if sl.start >= 0 and sl.stop <= len(data):
            epochs[ii] = data[sl]
    return epochs


def get_design_matrix(
    data: NDArray,
    ix: NDArray,
    lb: NDArray,
    half_win: int,
    events: NDArray | None = None,
    sparse_out: bool = False,
) -> NDArray | scipy.sparse.csr_matrix:
    """Build the event-indicator design matrix.

    Each column is a shifted indicator for one time-sample of one event type.
    The matrix can be returned as dense or sparse depending on size and the
    *sparse_out* flag. Complete windows ending exactly at the data length
    are included.

    Parameters
    ----------
    data : NDArray
        Continuous data array of shape ``(T, C)``.  Only its length is used.
    ix : NDArray
        1-D integer array of event-onset indices.
    lb : NDArray
        1-D array of event labels corresponding to *ix*.
    half_win : int
        Half-width of the template window (full window is ``2 * half_win``).
    events : NDArray | None
        Unique event labels to include.  If *None*, derived from ``np.unique(lb)``.
    sparse_out : bool
        If *True*, always return a sparse ``csr_matrix``.

    Returns
    -------
    NDArray | scipy.sparse.csr_matrix
        Design matrix of shape ``(T, n_events * 2 * half_win)``.
    """

    if events is None:
        events = np.unique(lb)

    T = len(data)
    # pyrefly: ignore [bad-argument-type]
    n_cols = len(events) * 2 * half_win

    if sparse_out or T * n_cols > 50_000_000:
        rows, cols = [], []
        # pyrefly: ignore [not-iterable]
        for ii, ev in enumerate(events):
            sel = (lb == ev) & (ix <= (T - half_win))
            if sel.sum() > 0:
                sel_ix = ix[sel]
                for kk in range(2 * half_win):
                    r = sel_ix + kk - half_win
                    valid = (r >= 0) & (r < T)
                    rows.extend(r[valid])
                    cols.extend([kk + ii * (2 * half_win)] * valid.sum())
        data_vals = np.ones(len(rows), dtype=np.float64)
        return sp.sparse.csr_matrix((data_vals, (rows, cols)), shape=(T, n_cols))

    design = np.zeros((T, n_cols))
    # pyrefly: ignore [not-iterable]
    for ii, ev in enumerate(events):
        sel = (lb == ev) & (ix <= (T - half_win))
        if sel.sum() > 0:
            for kk in range(2 * half_win):
                design[ix[sel] + kk - half_win, kk + ii * (2 * half_win)] = 1

    return design


def generate_data(
    real_templates: dict[str, NDArray],
    I: int = 50,
    noise: float = 2e-2,
    mean_interval: float = 100.0,
    rs: np.random.RandomState | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Generate simulated continuous data with embedded event templates.

    Events are placed at random intervals drawn from an exponential
    distribution, with additive Gaussian noise.

    Parameters
    ----------
    real_templates : dict[str, NDArray]
        Mapping from event-type name to template array of shape
        ``(2 * half_win, C)``.
    I : int
        Number of repetitions per event type.
    noise : float
        Standard deviation of the additive Gaussian noise.
    mean_interval : float
        Mean inter-event interval in samples.
    rs : np.random.RandomState | None
        Random state for reproducibility.  A new one is created if *None*.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        ``(data, ix, labels)`` where *data* has shape ``(T, C)``, *ix* is a
        1-D array of event-onset indices, and *labels* is a 1-D array of
        event-type strings.
    """
    if rs is None:
        rs = np.random.RandomState()

    t_names = list(real_templates.keys())
    half_win = real_templates[t_names[0]].shape[0] // 2
    C = real_templates[t_names[0]].shape[1]

    labels = rs.permutation(t_names * I)
    ix = []
    intervals = mean_interval + rs.exponential(mean_interval, len(labels))
    count = half_win

    for kk in intervals:
        kk = int(kk)
        ix.append(count + kk)
        count += kk

    ix = np.array(ix)
    L = np.max(ix) + half_win
    data = noise * rs.randn(L, C)

    for ii, l in zip(ix, labels):
        ii = int(ii)
        sl = slice(ii - half_win, ii + half_win)
        data[sl] += real_templates[l]

    return data, ix, labels


def calculate_snr(
    templates: dict[str, NDArray],
    real_templates: dict[str, NDArray],
) -> float:
    """Compute the average SNR (dB) between estimated and ground-truth templates.

    For each event type the SNR is ``10 * log10(mean(real^2) / mean((est - real)^2))``,
    and the result is averaged over all event types.

    Parameters
    ----------
    templates : dict[str, NDArray]
        Estimated templates keyed by event-type name.
    real_templates : dict[str, NDArray]
        Ground-truth templates keyed by event-type name.

    Returns
    -------
    float
        Mean SNR in decibels across all event types.
    """
    snr = 0
    for t in list(templates.keys()):
        s = np.mean((real_templates[t]) ** 2) / np.mean(
            (templates[t] - real_templates[t]) ** 2
        )
        s = 10 * np.log10(s)
        snr += s
    return snr / len(templates)


# --- SURE cost functions ---


def SURE_cost_fast(
    regval: float,
    mu: NDArray,
    DtX_rot: NDArray,
    V: NDArray,
    theta_ml: NDArray,
    sigsq: float,
    n_obs: int,
    n_cols: int,
    omega: float = 0.5,
) -> float:
    """Fast SURE cost using precomputed generalized eigendecomposition.

    V from ``eigh(DtD, DtD+S)`` satisfies ``V^T (DtD+S) V = I``,
    ``V^T DtD V = diag(mu)``.  V is B-orthonormal, NOT orthonormal -- all
    norms use V explicitly.

    The regularized solution is
    ``Phi_hat = V diag(g) V^T D^T X = V diag(g) DtX_rot``
    where ``g_i = 1 / (mu_i + lam * (1 - mu_i))``.

    Parameters
    ----------
    regval : float
        Regularization parameter (lambda).
    mu : NDArray
        Generalized eigenvalues of shape ``(p,)``.
    DtX_rot : NDArray
        Rotated data ``V^T D^T X`` of shape ``(p, C)``.
    V : NDArray
        Generalized eigenvectors of shape ``(p, p)``.
    theta_ml : NDArray
        ML (OLS) estimate of shape ``(p, C)``.
    sigsq : float
        Noise variance estimate.
    n_obs : int
        Number of observations (rows in *X*).
    n_cols : int
        Number of data channels (columns in *X*).
    omega : float
        SURE weighting parameter (default 0.5).

    Returns
    -------
    float
        Scalar SURE cost.
    """
    g = 1.0 / (mu + regval * (1 - mu) + 1e-30)

    # Phi_hat in original basis: V @ diag(g) @ DtX_rot
    Phi_hat = V @ (g[:, None] * DtX_rot)

    # ||Phi_hat||^2 (need V, can't use diagonal formula)
    Phi_hat_norm_sq = np.sum(Phi_hat**2)

    # tr(G^T D^+) = tr((DtD + lam*S)^{-1}) = tr(V diag(g) V^T) = sum_i g_i ||V_i||^2
    V_col_norms_sq = np.sum(V**2, axis=0)
    trace_GtDp = np.sum(g * V_col_norms_sq)
    trace_term = 2 * sigsq * trace_GtDp

    # X^T G^T D^+ X = theta_ml^T Phi_hat
    cross_term = np.sum(theta_ml * Phi_hat)

    # tr(G^T G) for omega != 0.5
    G_correction = 0
    if abs(omega - 0.5) > 1e-5:
        GG_trace = np.sum(g**2 * V_col_norms_sq)
        G_correction = (2 * omega - 1) * sigsq * GG_trace

    norm_vec_sum = (
        omega * (Phi_hat_norm_sq + trace_term * n_cols) - G_correction * n_cols
    )
    cost = norm_vec_sum - omega * 2 * cross_term

    return cost


def estimate_phi(data: NDArray, ix: NDArray, half_win: int) -> float:
    """Estimate AR(1) coefficient from pre-stimulus baselines.

    Parameters
    ----------
    data : NDArray
        Continuous data of shape ``(T,)`` or ``(T, C)``.
    ix : NDArray
        1-D integer array of event-onset indices.
    half_win : int
        Half-width of the template window.

    Returns
    -------
    float
        Median AR(1) coefficient clipped to ``[0, 0.99]``.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n_baseline = max(4, int(2 * half_win * 0.3))
    phi_estimates = []
    for idx in ix:
        start = idx - half_win
        end = start + n_baseline
        if start < 0 or end > len(data):
            continue
        segment = data[start:end]
        for ch in range(segment.shape[1]):
            x = segment[:, ch] - segment[:, ch].mean()
            r0 = np.mean(x**2)
            if r0 < 1e-30:
                continue
            r1 = np.mean(x[:-1] * x[1:])
            phi_estimates.append(r1 / r0)
    if not phi_estimates:
        return 0.0
    return np.clip(np.median(phi_estimates), 0.0, 0.99)


def SURE_cost_colored(
    regval: float,
    mu: NDArray,
    DtX_rot: NDArray,
    V: NDArray,
    theta_ml: NDArray,
    DtSigmaD: NDArray,
    n_obs: int,
    n_cols: int,
    omega: float = 0.5,
    _DtD: NDArray | None = None,
    _S: NDArray | None = None,
    _precomp: dict[str, NDArray | float] | None = None,
) -> float:
    """SURE cost with colored (AR1) noise covariance.

    Uses ``2 tr(M D^T Sigma_W D)`` where
    ``M = (D^T D)^{-1} (D^T D + lam S)^{-1}``.

    When *_precomp* dict is provided, uses precomputed trace coefficients
    for O(p) evaluation (no matrix solves per lambda).

    Parameters
    ----------
    regval : float
        Regularization parameter (lambda).
    mu : NDArray
        Generalized eigenvalues of shape ``(p,)``.
    DtX_rot : NDArray
        Rotated data ``V^T D^T X`` of shape ``(p, C)``.
    V : NDArray
        Generalized eigenvectors of shape ``(p, p)``.
    theta_ml : NDArray
        ML (OLS) estimate of shape ``(p, C)``.
    DtSigmaD : NDArray
        ``D^T Sigma_W D`` matrix of shape ``(p, p)``.
    n_obs : int
        Number of observations.
    n_cols : int
        Number of data channels.
    omega : float
        SURE weighting parameter (default 0.5).
    _DtD : NDArray | None
        Gram matrix for the slow-path fallback.
    _S : NDArray | None
        Penalty matrix for the slow-path fallback.
    _precomp : dict[str, NDArray | float] | None
        Precomputed trace coefficients from :func:`precompute_colored_trace`.

    Returns
    -------
    float
        Scalar colored-SURE cost.
    """
    g = 1.0 / (mu + regval * (1 - mu) + 1e-30)

    Phi_hat = V @ (g[:, None] * DtX_rot)
    Phi_hat_norm_sq = np.sum(Phi_hat**2)

    p = len(mu)

    if _precomp is not None:
        # Fast path: tr(M @ DtSigmaD) = sum(trace_coeffs * g)
        # where trace_coeffs = diag(R) with R = V^T DtSigmaD DtD^{-1} V
        trace_term = 2 * np.sum(_precomp["trace_coeffs"] * g)
    elif _DtD is not None and _S is not None:
        # Slow path: direct solve (for backward compat)
        DtD_plus_lS = _DtD + regval * _S
        inv_DtD_plus_lS = np.linalg.solve(DtD_plus_lS, np.eye(p))
        M = np.linalg.solve(_DtD, inv_DtD_plus_lS)
        trace_term = 2 * np.trace(M @ DtSigmaD)
    else:
        # Fallback: use V-basis
        VtV = V.T @ V
        M_rot = np.diag(1.0 / (mu + 1e-30)) @ VtV @ np.diag(g)
        M = V @ M_rot @ V.T
        trace_term = 2 * np.trace(M @ DtSigmaD)

    cross_term = np.sum(theta_ml * Phi_hat)

    G_correction = 0
    if abs(omega - 0.5) > 1e-5:
        V_col_norms_sq = (
            _precomp["V_col_norms_sq"] if _precomp is not None else np.sum(V**2, axis=0)
        )
        GG_trace = np.sum(g**2 * V_col_norms_sq)
        sigsq_approx = (
            _precomp["sigsq_approx"] if _precomp is not None else np.trace(DtSigmaD) / p
        )
        G_correction = (2 * omega - 1) * sigsq_approx * GG_trace

    norm_vec_sum = (
        omega * (Phi_hat_norm_sq + trace_term * n_cols) - G_correction * n_cols
    )
    cost = norm_vec_sum - omega * 2 * cross_term

    return cost


def precompute_colored_trace(
    DtD: NDArray,
    DtSigmaD: NDArray,
    V: NDArray,
    mu: NDArray,
) -> dict[str, NDArray | float]:
    """Precompute trace coefficients for fast :func:`SURE_cost_colored` evaluation.

    ``tr(M @ DtSigmaD) = tr(DtD^{-1} (DtD + lam*S)^{-1} DtSigmaD)``

    With ``(DtD + lam*S)^{-1} = V diag(g) V^T``, this becomes::

        tr(DtD^{-1} V diag(g) V^T DtSigmaD) = sum_i g_i [V^T DtSigmaD DtD^{-1} V]_ii

    We precompute ``R = V^T DtSigmaD DtD^{-1} V`` and return ``diag(R)``
    as *trace_coeffs*.

    Parameters
    ----------
    DtD : NDArray
        Gram matrix ``D^T D`` of shape ``(p, p)``.
    DtSigmaD : NDArray
        ``D^T Sigma_W D`` matrix of shape ``(p, p)``.
    V : NDArray
        Generalized eigenvectors of shape ``(p, p)``.
    mu : NDArray
        Generalized eigenvalues of shape ``(p,)``.

    Returns
    -------
    dict[str, NDArray | float]
        Dictionary with keys ``'trace_coeffs'``, ``'V_col_norms_sq'``, and
        ``'sigsq_approx'``.
    """
    p = len(mu)
    # DtD^{-1} V  (one solve, done once)
    DtD_inv_V = np.linalg.solve(DtD, V)
    # R = V^T DtSigmaD DtD^{-1} V = (DtSigmaD^T V)^T DtD^{-1} V element-wise
    R = V.T @ DtSigmaD @ DtD_inv_V
    trace_coeffs = np.diag(R)

    V_col_norms_sq = np.sum(V**2, axis=0)
    sigsq_approx = np.trace(DtSigmaD) / p

    return {
        "trace_coeffs": trace_coeffs,
        "V_col_norms_sq": V_col_norms_sq,
        "sigsq_approx": sigsq_approx,
    }


def _resolve_acf_max_lag(half_win: int, max_lag: int | None) -> int:
    """Validate the output horizon independently of the fitted AR moments."""
    if max_lag is None:
        return 4 * half_win
    if not isinstance(max_lag, (int, np.integer)):
        raise ValueError(  # noqa: TRY004 - Preserve the public ValueError contract.
            "max_lag must be an integer or None, "
            f"got {max_lag!r} (type={type(max_lag).__name__})"
        )
    if max_lag < 0:
        raise ValueError(f"max_lag must be nonnegative, got {max_lag!r}")
    return int(max_lag)


def estimate_acf_ar(
    data: NDArray,
    ix: NDArray,
    half_win: int,
    ar_order: int = 3,
    *,
    max_lag: int | None = None,
) -> NDArray:
    """Estimate noise autocorrelation function via AR(p) model from baselines.

    Fits an AR(p) model to pre-stimulus baseline segments via Yule-Walker,
    then computes the ACF from the AR parameters.

    Parameters
    ----------
    data : NDArray
        Continuous data of shape ``(T,)`` or ``(T, C)``.
    ix : NDArray
        1-D integer array of event-onset indices.
    half_win : int
        Half-width of the template window.
    ar_order : int
        Order of the AR model to fit (default 3).
    max_lag : int | None
        Largest returned lag, defaulting to ``4 * half_win``. Only the
        continuation length changes; baseline samples and fitted AR parameters
        do not depend on this value.

    Returns
    -------
    NDArray
        Autocovariance ``R(k)`` through *max_lag*. If no baseline is available,
        the existing white-noise fallback returns only the zero-lag variance.
    """
    output_max_lag = _resolve_acf_max_lag(half_win, max_lag)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_baseline = max(ar_order + 4, int(2 * half_win * 0.3))
    segments = []
    for idx in ix:
        start = idx - half_win
        end = start + n_baseline
        if start < 0 or end > len(data):
            continue
        segments.append(data[start:end])

    if not segments:
        return np.array([np.mean(np.var(data, axis=0))])

    # Keep the fitted moments independent of the requested continuation length.
    fit_max_lag = 4 * half_win
    r = np.zeros(fit_max_lag + 1)
    count = 0
    for seg in segments:
        for ch in range(seg.shape[1]):
            x = seg[:, ch] - seg[:, ch].mean()
            n = len(x)
            for k in range(min(n, fit_max_lag + 1)):
                r[k] += np.sum(x[: n - k] * x[k:n])
            count += n

    if count == 0:
        return np.array([np.mean(np.var(data, axis=0))])

    r /= count

    # Fit AR(p) via Yule-Walker
    p = min(ar_order, len(r) - 1)
    if p < 1 or r[0] < 1e-30:
        acf = np.zeros(output_max_lag + 1)
        acf[0] = r[0]
        return acf

    # Build Yule-Walker system: [R(0)...R(p-1); ... ; R(p-1)...R(0)] phi = [R(1)...R(p)]
    R_mat = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            R_mat[i, j] = r[abs(i - j)]

    r_vec = r[1 : p + 1]
    try:
        ar_coeffs = np.linalg.solve(R_mat, r_vec)
    except np.linalg.LinAlgError:
        acf = np.zeros(output_max_lag + 1)
        acf[0] = r[0]
        return acf

    # Innovation variance
    sigma2_e = r[0] - np.dot(ar_coeffs, r[1 : p + 1])
    sigma2_e = max(sigma2_e, 1e-30)

    # Generate ACF from AR model: R(k) = sum_i phi_i R(k-i) for k > p
    acf = np.zeros(output_max_lag + 1)
    acf[: min(p + 1, output_max_lag + 1)] = r[: min(p + 1, output_max_lag + 1)]
    for k in range(p + 1, output_max_lag + 1):
        acf[k] = sum(ar_coeffs[i] * acf[k - 1 - i] for i in range(p))

    return acf


def _event_block_labels(ix: NDArray, event_blocks: NDArray | None) -> NDArray:
    if event_blocks is None:
        return np.zeros(len(ix), dtype=int)
    blocks = np.asarray(event_blocks)
    if blocks.shape != np.shape(ix):
        raise ValueError(
            f"event_blocks must have the shape of ix {np.shape(ix)}, got {blocks.shape}"
        )
    return blocks


def build_DtSigmaD_acf(
    ix: NDArray,
    lb: NDArray,
    events: NDArray,
    half_win: int,
    acf: NDArray,
    data_length: int | None = None,
    event_blocks: NDArray | None = None,
) -> NDArray:
    """Build D^T Sigma_W D from a general autocorrelation function.

    ``acf[k] = R(k)``, the noise autocovariance at lag *k*.
    Lags beyond the supplied array are treated as zero. For an AR model,
    supply all lags required by the event windows. A shorter array is exact
    only when its omitted covariance is zero, as for a finite-support model.

    Parameters
    ----------
    ix : NDArray
        1-D integer array of event-onset indices.
    lb : NDArray
        1-D array of event labels corresponding to *ix*.
    events : NDArray
        Unique event labels to include.
    half_win : int
        Half-width of the template window.
    acf : NDArray
        Autocorrelation function of shape ``(max_lag + 1,)``.
    data_length : int | None
        Total length of the data array. Excludes events whose windows extend
        past the end; windows ending exactly at this length are included.
    event_blocks : NDArray | None
        Recording-block label of each event in *ix*. Event pairs from
        different blocks contribute no covariance. *None* treats all events
        as one block.

    Returns
    -------
    NDArray
        Symmetric matrix of shape ``(p, p)`` where ``p = len(events) * 2 * half_win``.
    """
    M = len(events)
    T = 2 * half_win
    p = M * T
    blocks = _event_block_labels(ix, event_blocks)

    max_lag = len(acf) - 1
    offsets = np.arange(T)[:, None] - np.arange(T)[None, :]

    DtSigmaD = np.zeros((p, p))

    for ii, ev_i in enumerate(events):
        sel_i = lb == ev_i
        if data_length is not None:
            sel_i = sel_i & (ix <= data_length - half_win)
        ix_i = ix[sel_i]
        for jj, ev_j in enumerate(events):
            sel_j = lb == ev_j
            if data_length is not None:
                sel_j = sel_j & (ix <= data_length - half_win)
            ix_j = ix[sel_j]
            blocks_j = blocks[sel_j]
            block = np.zeros((T, T))
            for ti, block_i in zip(ix_i, blocks[sel_i]):
                for tj in ix_j[blocks_j == block_i]:
                    dt_base = ti - tj
                    abs_lags = np.abs(dt_base + offsets)
                    valid = abs_lags <= max_lag
                    contrib = np.where(valid, acf[np.minimum(abs_lags, max_lag)], 0.0)
                    block += contrib
            DtSigmaD[ii * T : (ii + 1) * T, jj * T : (jj + 1) * T] = block

    return DtSigmaD


def _required_covariance_lag(
    ix: NDArray,
    lb: NDArray,
    events: NDArray,
    half_win: int,
    data_length: int,
) -> int:
    """Largest lag for retained event pairs, assuming complete left windows."""
    selected = np.isin(lb, events) & (ix <= data_length - half_win)
    times = ix[selected]
    if times.size == 0:
        return 0
    return int(times.max() - times.min()) + 2 * half_win - 1


def build_DtSigmaD_ar1(
    ix: NDArray,
    lb: NDArray,
    events: NDArray,
    half_win: int,
    phi: float,
    sigsq: float,
    data_length: int | None = None,
    event_blocks: NDArray | None = None,
) -> NDArray:
    """Build D^T Sigma_W D for AR(1) noise: ``(Sigma_W)_ij = sigsq * phi^|i-j|``.

    For the design matrix D with event indicator columns, entry ``(a, b)`` of
    the ``(ev_i, ev_j)`` block equals::

        sum over event pairs (ti, tj):  sigsq * phi^|ti + a - tj - b|

    where *a*, *b* are template sample offsets ``(0..T-1)``, *ti*/*tj* are
    event times.

    Uses vectorized Toeplitz structure: each event pair contributes a
    Toeplitz matrix ``sigsq * phi^|dt_base + a - b|`` which depends only on
    the inter-event time difference ``dt_base = ti - tj``.

    Parameters
    ----------
    ix : NDArray
        1-D integer array of event-onset indices.
    lb : NDArray
        1-D array of event labels corresponding to *ix*.
    events : NDArray
        Unique event labels to include.
    half_win : int
        Half-width of the template window.
    phi : float
        Stationary AR(1) coefficient, including negative values (``abs(phi) < 1``).
    sigsq : float
        Marginal noise variance, equal to innovation variance divided by
        ``1 - phi**2``.
    data_length : int | None
        Total data length. Excludes events whose windows extend past the end;
        windows ending exactly at this length are included.
    event_blocks : NDArray | None
        Recording-block label of each event in *ix*. Event pairs from
        different blocks contribute no covariance. *None* treats all events
        as one block.

    Returns
    -------
    NDArray
        Symmetric matrix of shape ``(p, p)`` where ``p = len(events) * 2 * half_win``.
    """
    M = len(events)
    T = 2 * half_win
    p = M * T
    blocks = _event_block_labels(ix, event_blocks)

    # Each event-pair block has only 2*T-1 distinct lags. Evaluate these
    # directly without a tail cutoff or a lookup table spanning the recording.
    # Negative phi retains its alternating-sign correlations.
    lag_offsets = np.arange(1 - T, T)
    offset_indices = np.arange(T)[:, None] - np.arange(T)[None, :] + T - 1

    DtSigmaD = np.zeros((p, p))

    for ii, ev_i in enumerate(events):
        sel_i = lb == ev_i
        if data_length is not None:
            sel_i = sel_i & (ix <= data_length - half_win)
        ix_i = ix[sel_i]
        for jj, ev_j in enumerate(events):
            sel_j = lb == ev_j
            if data_length is not None:
                sel_j = sel_j & (ix <= data_length - half_win)
            ix_j = ix[sel_j]
            blocks_j = blocks[sel_j]
            block = np.zeros((T, T))
            for ti, block_i in zip(ix_i, blocks[sel_i]):
                for tj in ix_j[blocks_j == block_i]:
                    dt_base = ti - tj
                    correlations = sigsq * np.power(phi, np.abs(dt_base + lag_offsets))
                    block += correlations[offset_indices]
            DtSigmaD[ii * T : (ii + 1) * T, jj * T : (jj + 1) * T] = block

    return DtSigmaD


def gen_corefilt(vec: NDArray, symm: bool) -> NDArray:
    """Optionally symmetrise a 1-D filter kernel.

    When *symm* is True, the output is ``[vec, flip(vec)[1:]]`` giving a
    symmetric (zero-phase) kernel of length ``2 * len(vec) - 1``.

    Parameters
    ----------
    vec : NDArray
        1-D filter kernel.
    symm : bool
        If *True*, mirror *vec* to create a symmetric kernel.

    Returns
    -------
    NDArray
        The (possibly symmetrised) 1-D kernel.
    """
    if symm:
        out = np.zeros(2 * len(vec) - 1)
        out[: len(vec)] = vec
        out[len(vec) :] = np.flip(vec)[1:]
    else:
        out = vec
    return out


class _KernelSUREObjective:
    """Evaluate the existing kernel SURE objective and its gradient.

    At lambda=1 the generalized spectral weights are all one, so the
    objective can use P = inverse(DtD + S) through Cholesky solves. The
    covariance and omega correction retain the scalar objective's behavior.
    No diagonal jitter or fallback is applied to invalid matrices.
    """

    def __init__(
        self,
        DtD: NDArray,
        theta_ml: NDArray,
        X: NDArray,
        sigsq: float,
        M: int,
        symm: bool = False,
        omega: float = 0.5,
        _cost_scale: float | None = None,
        DtSigmaD: NDArray | None = None,
    ) -> None:
        design_gram = np.asarray(DtD)
        ml_templates = np.asarray(theta_ml)
        self.DtD = design_gram
        self.theta_ml = ml_templates
        self.DtX = design_gram @ ml_templates
        self.M = M
        self.T = design_gram.shape[0] // M
        self.symm = symm
        self.omega = omega
        self.cost_scale = 1.0 if _cost_scale is None else _cost_scale
        self.identity = np.eye(design_gram.shape[0])
        n_cols = X.shape[1] if X.ndim > 1 else 1

        if DtSigmaD is None:
            trace_matrix = sigsq * self.identity
            correction_variance = sigsq
        else:
            # Only the symmetric part contributes to trace(P @ trace_matrix).
            trace_matrix = np.linalg.solve(self.DtD, np.asarray(DtSigmaD)).T
            trace_matrix = (trace_matrix + trace_matrix.T) / 2
            correction_variance = np.trace(DtSigmaD) / self.DtD.shape[0]
        self.Q = 2 * omega * n_cols * trace_matrix
        if abs(omega - 0.5) > 1e-5:
            self.Q -= (2 * omega - 1) * n_cols * correction_variance * self.identity

    def value_gradient(self, vec: NDArray) -> tuple[float, NDArray]:
        """Return the scaled cost and derivatives in independent kernel entries."""
        kernel = gen_corefilt(vec, self.symm)
        R = sp.linalg.convolution_matrix(kernel, self.T - len(kernel) + 1)
        S = sp.linalg.block_diag(*([R @ R.T] * self.M))
        factor = sp.linalg.cho_factor(self.DtD + S, lower=True)
        P = sp.linalg.cho_solve(factor, self.identity)
        Phi = sp.linalg.cho_solve(factor, self.DtX)
        cost = (
            self.omega * np.sum(Phi**2)
            - 2 * self.omega * np.sum(self.theta_ml * Phi)
            + np.sum(P * self.Q.T)
        )

        # dPhi = -P @ dS @ Phi. Differentiate the trace term as well.
        Z = P @ (Phi - self.theta_ml)
        grad_S = -self.omega * (Z @ Phi.T + Phi @ Z.T) - P @ self.Q @ P
        block = sum(
            grad_S[j * self.T : (j + 1) * self.T, j * self.T : (j + 1) * self.T]
            for j in range(self.M)
        )
        grad_R = block @ R
        gradient = np.empty(len(vec))
        for k in range(len(vec)):
            basis = np.zeros(len(vec))
            basis[k] = 1.0
            dR = sp.linalg.convolution_matrix(
                gen_corefilt(basis, self.symm), self.T - len(kernel) + 1
            )
            gradient[k] = 2 * np.sum(grad_R * dR)
        return float(cost / self.cost_scale), gradient / self.cost_scale


# --- Noise estimation ---


def calc_vrf(
    order: int,
    wn: float,
) -> tuple[NDArray, NDArray, float]:
    """Compute the variance reduction factor for a digital highpass filter.

    Parameters
    ----------
    order : int
        Filter order for the Butterworth highpass.
    wn : float
        Normalised cutoff frequency (0 to 1, where 1 is Nyquist).

    Returns
    -------
    tuple[NDArray, NDArray, float]
        ``(numerator, denominator, vrf)`` -- the filter coefficients and the
        variance reduction factor.
    """

    b, a = sig.butter(order, wn, btype="highpass")
    bt = sig.dlti(b, a)
    _, y = sig.dimpulse(bt, n=1000)
    vrf = np.linalg.norm(y) ** 2
    return b, a, vrf


def hpf(x: NDArray, wn: float, order: int) -> tuple[NDArray, float]:
    """Apply a causal Butterworth highpass filter to multi-channel data.

    Parameters
    ----------
    x : NDArray
        Input data of shape ``(T,)`` or ``(T, C)``.
    wn : float
        Normalised cutoff frequency (0 to 1, where 1 is Nyquist).
    order : int
        Butterworth filter order.

    Returns
    -------
    tuple[NDArray, float]
        ``(filtered_data, vrf)`` -- the filtered data of shape ``(T, C)``
        and the variance reduction factor.
    """

    b, a, vrf = calc_vrf(order, wn)
    zi = sig.lfilter_zi(b, a)
    dims = x.shape
    if len(dims) > 1:
        nCols = dims[1]
        A = x
    else:
        nCols = 1
        A = x.reshape(dims[0], 1)

    M = np.zeros_like(A)
    for ii in range(nCols):
        M[:, ii], _ = sig.lfilter(b, a, A[:, ii], axis=0, zi=zi * A[0, ii])

    return M, vrf


def est_noise_var_hpf(data: NDArray) -> float:
    """Estimate noise variance using a highpass filter.

    Applies a 4th-order Butterworth highpass at normalised cutoff 0.6 and
    corrects for the filter's variance reduction factor.

    Parameters
    ----------
    data : NDArray
        Continuous data of shape ``(T,)`` or ``(T, C)``.

    Returns
    -------
    float
        Mean noise variance across channels.
    """
    M, vrf = hpf(data, 0.6, 4)
    var_vec = np.var(M, axis=0) / vrf
    return np.mean(var_vec)


def est_noise_var_pairwise(
    data: NDArray,
    ix: NDArray,
    lb: NDArray,
    half_win: int,
) -> float:
    """Estimate noise variance from pairwise epoch differences.

    For each event type, take all pairs of epochs and compute their difference.
    The common template cancels exactly, leaving
    ``d_ij = (W_i - W_j) + (delta_i - delta_j)``.
    Under the assumption that trial-to-trial template variations are small
    relative to sensing noise, ``Var(d_ij) ~= 2 * sigma^2``.

    Parameters
    ----------
    data : NDArray
        Continuous data of shape ``(T, C)``.
    ix : NDArray
        1-D integer array of event-onset indices.
    lb : NDArray
        1-D array of event labels corresponding to *ix*.
    half_win : int
        Half-width of the template window.

    Returns
    -------
    float
        Scalar noise variance estimate.
    """
    epochs_by_label = {}
    for label in np.unique(lb):
        mask = lb == label
        label_ix = ix[mask]
        valid = (label_ix - half_win >= 0) & (label_ix + half_win <= len(data))
        label_ix = label_ix[valid]
        if len(label_ix) >= 2:
            ep = np.stack([data[i - half_win : i + half_win] for i in label_ix])
            epochs_by_label[label] = ep

    if not epochs_by_label:
        return est_noise_var_hpf(data)

    var_estimates = []
    for _label, epochs in epochs_by_label.items():
        n_epochs = epochs.shape[0]
        for i in range(n_epochs):
            for j in range(i + 1, n_epochs):
                diff = epochs[i] - epochs[j]
                var_estimates.append(np.var(diff) / 2.0)

    return np.mean(var_estimates)


def est_noise_var_baseline(
    data: NDArray,
    ix: NDArray,
    half_win: int,
    baseline_frac: float = 0.3,
) -> float:
    """Estimate noise variance from the pre-stimulus baseline of each epoch.

    For EEG/ERP data the period before event onset contains no evoked response,
    so its variance reflects only ongoing noise.  The first *baseline_frac* of
    each epoch window (the pre-stimulus portion) is used.

    Parameters
    ----------
    data : NDArray
        Continuous data of shape ``(T, C)``.
    ix : NDArray
        1-D integer array of event-onset indices.
    half_win : int
        Half-width of the template window.
    baseline_frac : float
        Fraction of the epoch window to use as baseline (default 0.3).

    Returns
    -------
    float
        Scalar noise variance estimate averaged across channels and epochs.
    """
    n_baseline = max(1, int(2 * half_win * baseline_frac))
    var_per_epoch = []

    for idx in ix:
        start = idx - half_win
        end = start + n_baseline
        if start >= 0 and end <= len(data):
            segment = data[start:end]
            var_per_epoch.append(np.var(segment, axis=0))

    if not var_per_epoch:
        return est_noise_var_hpf(data)

    return np.mean(var_per_epoch)


def _est_noise_colored(
    data: NDArray,
    ix: NDArray | None,
    half_win: int | None,
    method: str = "",
    **_kwargs: object,
) -> float:
    if ix is None:
        raise ValueError(f"{method} noise estimation requires ix (got None)")
    if half_win is None:
        raise ValueError(f"{method} noise estimation requires half_win (got None)")
    return est_noise_var_baseline(data, ix, half_win)


def _est_noise_pairwise(
    data: NDArray,
    ix: NDArray | None,
    lb: NDArray | None,
    half_win: int | None,
    **_kwargs: object,
) -> float:
    if ix is None:
        raise ValueError("Pairwise noise estimation requires ix (got None)")
    if lb is None:
        raise ValueError("Pairwise noise estimation requires lb (got None)")
    if half_win is None:
        raise ValueError("Pairwise noise estimation requires half_win (got None)")
    return est_noise_var_pairwise(data, ix, lb, half_win)


def _est_noise_baseline(
    data: NDArray,
    ix: NDArray | None,
    half_win: int | None,
    **_kwargs: object,
) -> float:
    if ix is None:
        raise ValueError("Baseline noise estimation requires ix (got None)")
    if half_win is None:
        raise ValueError("Baseline noise estimation requires half_win (got None)")
    return est_noise_var_baseline(data, ix, half_win)


def _est_noise_hpf(data: NDArray, **_kwargs: object) -> float:
    return est_noise_var_hpf(data)


# pyre-ignore[3]: return type intentionally broad for dispatch dict
_NOISE_ESTIMATORS: dict[str, object] = {
    "ar1_colored": _est_noise_colored,
    "ar3_colored": _est_noise_colored,
    "pairwise": _est_noise_pairwise,
    "baseline": _est_noise_baseline,
    "hpf": _est_noise_hpf,
}


def est_noise_var(
    data: NDArray,
    method: str = "pairwise",
    ix: NDArray | None = None,
    lb: NDArray | None = None,
    half_win: int | None = None,
) -> float:
    """Estimate noise variance with a configurable method.

    Parameters
    ----------
    data : NDArray
        Continuous data of shape ``(T, C)``.
    method : str
        Noise estimation method.  One of ``'pairwise'``, ``'baseline'``,
        ``'ar1_colored'``, ``'ar3_colored'``, or ``'hpf'``.
    ix : NDArray | None
        Event-onset indices.  Required for ``'pairwise'``, ``'baseline'``,
        and colored variants.
    lb : NDArray | None
        Event labels.  Required for ``'pairwise'``.
    half_win : int | None
        Half-width of the template window.  Required for ``'pairwise'``,
        ``'baseline'``, and colored variants.

    Returns
    -------
    float
        Estimated noise variance.
    """
    estimator = _NOISE_ESTIMATORS.get(method)
    if estimator is None:
        raise ValueError(f"Unknown noise estimation method: {method}")
    # pyre-ignore[29]: dispatch dict callable
    return estimator(
        data=data,
        method=method,
        ix=ix,
        lb=lb,
        half_win=half_win,
    )


# --- Stable regularization path for design-scaled lambda searches ---
# Scalar SURE and the L-curve search six decades beyond the penalty's
# transition scales, where direct solves of D^T D + lam S lose accuracy.


def _dense(matrix: NDArray | scipy.sparse.spmatrix) -> NDArray:
    return np.asarray(
        matrix.todense() if sp.sparse.issparse(matrix) else matrix, dtype=float
    )


def _relative_change(change: float, scale: float) -> float:
    """Return change / scale; with a zero scale only an exact zero passes."""
    if scale > 0:
        return change / scale
    return 0.0 if change == 0 else np.inf


def _penalty_factor(penalty: NDArray) -> NDArray:
    """Return ``F`` with ``F^T F = penalty``, dropping roundoff-level modes."""
    eigenvalues, vectors = np.linalg.eigh(penalty)
    tolerance = len(eigenvalues) * np.finfo(float).eps * np.max(np.abs(eigenvalues))
    if eigenvalues[0] < -tolerance:
        raise ValueError(
            "Penalty matrix is not positive semidefinite: smallest eigenvalue "
            f"{eigenvalues[0]:.3e} is below -{tolerance:.3e}"
        )
    keep = eigenvalues > tolerance
    if not np.any(keep):
        raise ValueError(
            "Penalty matrix has no eigenvalue above the roundoff tolerance "
            f"{tolerance:.3e}; every lambda gives the same estimate"
        )
    return np.sqrt(eigenvalues[keep])[:, None] * vectors[:, keep].T


def _shrinkage(values: NDArray, lam: float | NDArray) -> tuple[NDArray, NDArray]:
    """Return weights ``1 / (1 + lam v)`` and ``1 - weights`` without cancellation."""
    scaled = np.multiply.outer(lam, values)
    return 1.0 / (1.0 + scaled), scaled / (1.0 + scaled)


class _PenaltyPath:
    """Solutions of ``(D^T D + lam S) theta = D^T X`` for every lam in [0, inf].

    ``factor`` is a full-row-rank ``F`` with ``F^T F = S``. Whitening with the
    Cholesky factor of ``D^T D`` and an SVD of the whitened ``F`` give
    ``B^T D^T D B = I`` and ``B^T S B = diag(values)``. Directions that ``F``
    does not penalize keep values of exactly zero, so the infinite-penalty
    limit is the least-squares fit within that null space. ``D^T D`` must be
    conditioned well enough that normal-equation roundoff, about
    ``p * eps * cond(D^T D)`` of the template, stays below 1e-7.
    """

    def __init__(self, gram: NDArray, cross: NDArray, factor: NDArray) -> None:
        spectrum = np.linalg.eigvalsh(gram)
        limit = 1e-7 / (len(gram) * np.finfo(float).eps)
        if spectrum[0] * limit <= spectrum[-1]:
            raise np.linalg.LinAlgError(
                "D^T D is too ill-conditioned for the design-scaled search: "
                f"eigenvalues span [{spectrum[0]:.3e}, {spectrum[-1]:.3e}], "
                f"beyond the condition-number limit {limit:.3e}; every template "
                "sample must be well observed by event windows"
            )
        chol = sp.linalg.cholesky(gram, lower=True)
        whitened = sp.linalg.solve_triangular(chol, factor.T, lower=True).T
        _, singular, right = sp.linalg.svd(whitened, full_matrices=True)
        rank_tolerance = len(gram) * np.finfo(float).eps * singular[0]
        if len(singular) < len(factor) or singular[-1] <= rank_tolerance:
            raise ValueError(
                f"Penalty factor of shape {factor.shape} does not have full row "
                f"rank: smallest whitened singular value {singular[-1]:.3e}"
            )
        self.chol: NDArray = chol
        self.values: NDArray = np.zeros(len(gram))
        self.values[: len(singular)] = singular**2
        self.null: NDArray = np.arange(len(gram)) >= len(singular)
        self.basis: NDArray = sp.linalg.solve_triangular(chol.T, right.T, lower=False)
        self.rotated: NDArray = self.basis.T @ cross
        self.ols: NDArray = sp.linalg.cho_solve((chol, True), cross)

    def weights(self, lam: float) -> NDArray:
        if np.isposinf(lam):
            return self.null.astype(float)
        return _shrinkage(self.values, lam)[0]

    def templates(self, lam: float) -> NDArray:
        return self.basis @ (self.weights(lam)[:, None] * self.rotated)

    def template_change(self, first: float, second: float) -> float:
        """Template difference between two lambdas over the OLS template norm."""
        change = np.linalg.norm(self.templates(first) - self.templates(second))
        return _relative_change(float(change), float(np.linalg.norm(self.ols)))

    def variation_bound(self) -> float:
        """Upper bound on any template difference along the path over the OLS norm.

        Penalized weights stay in [0, 1], so no two lambdas can differ by more
        than the sum of the penalized modes' template norms. Endpoint
        differences can cancel because ``B`` is not orthonormal.
        """
        penalized = ~self.null
        bound = np.sum(
            np.linalg.norm(self.basis[:, penalized], axis=0)
            * np.linalg.norm(self.rotated[penalized], axis=1)
        )
        return _relative_change(float(bound), float(np.linalg.norm(self.ols)))

    def log10_envelope(self, margin: int = 6) -> tuple[int, int]:
        """Integer log10 bounds ``margin`` decades beyond every penalty transition."""
        positive = self.values[~self.null]
        lower = int(np.floor(-np.log10(np.max(positive)) - margin))
        upper = int(np.ceil(-np.log10(np.min(positive)) + margin))
        return lower, upper


# --- GCV, L-curve, Empirical Bayes lambda selection ---
# GCV uses a joint eigendecomposition of the normal equations; the L-curve
# uses the stable path above.
# EB uses Cholesky solves and an explicit penalty-nullspace convention.


def select_lambda_gcv(
    DtD: NDArray | scipy.sparse.spmatrix,
    DtX: NDArray | scipy.sparse.spmatrix,
    S: NDArray | scipy.sparse.spmatrix,
    X: NDArray,
    D: NDArray | scipy.sparse.spmatrix,
    n_lambdas: int = 80,
) -> tuple[float, dict[str, object]]:
    """Select regularization lambda via Generalized Cross-Validation.

    Uses generalized eigendecomposition.  V from ``eigh(DtD, DtD+S)`` is
    B-orthonormal (``V^T (DtD+S) V = I``), **not** orthonormal.  All
    computations account for this by working in the original basis via V.

    Parameters
    ----------
    DtD : NDArray | scipy.sparse.spmatrix
        Gram matrix ``D^T D`` of shape ``(p, p)``.
    DtX : NDArray | scipy.sparse.spmatrix
        Cross-product ``D^T X`` of shape ``(p, C)``.
    S : NDArray | scipy.sparse.spmatrix
        Penalty matrix of shape ``(p, p)``.
    X : NDArray
        Data matrix of shape ``(n, C)``.
    D : NDArray | scipy.sparse.spmatrix
        Design matrix of shape ``(n, p)``.
    n_lambdas : int
        Number of candidate lambda values on the log-spaced grid.

    Returns
    -------
    tuple[float, dict[str, object]]
        Best lambda and a diagnostics dict with keys ``'opt_val'`` and
        ``'method'``.
    """

    DtD_dense = np.asarray(
        DtD.todense() if sp.sparse.issparse(DtD) else DtD, dtype=float
    )
    S_dense = np.asarray(S.todense() if sp.sparse.issparse(S) else S, dtype=float)
    DtX_dense = np.asarray(
        DtX.todense() if sp.sparse.issparse(DtX) else DtX, dtype=float
    )

    n_obs = X.shape[0]
    XtX_sum = np.sum(X**2)

    mu, V = sp.linalg.eigh(DtD_dense, DtD_dense + S_dense)
    mu = np.clip(mu, 0, 1)
    DtX_rot = V.T @ DtX_dense

    lambdas = np.logspace(-3, 5, n_lambdas)
    best_lam, best_score = lambdas[0], np.inf

    for lam in lambdas:
        g = 1.0 / (mu + lam * (1 - mu) + 1e-30)
        # trace(H) = trace(D G) where G=(DtD+lam*S)^{-1}D^T
        # = trace(DtD (DtD+lam*S)^{-1}) = trace(DtD V diag(g) V^T)
        # = trace(V^T DtD V diag(g)) = sum(mu_i * g_i)
        trace_H = np.sum(mu * g)
        denom = n_obs - trace_H
        if denom <= 0:
            continue
        # Phi_hat = V diag(g) DtX_rot
        Phi_hat = V @ (g[:, None] * DtX_rot)
        # ||X - D Phi_hat||^2 = XtX - 2 theta_ml^T Phi_hat + Phi_hat^T DtD Phi_hat
        # But simpler: = XtX - 2 DtX^T Phi_hat + Phi_hat^T DtD Phi_hat
        cross = np.sum(DtX_dense * Phi_hat)
        quad = np.sum(Phi_hat * (DtD_dense @ Phi_hat))
        residual_ss = XtX_sum - 2 * cross + quad
        score = residual_ss / (denom**2) * n_obs
        if score < best_score:
            best_score = score
            best_lam = lam

    return best_lam, {"opt_val": best_lam, "method": "gcv"}


def _lcurve_grid(lower: int, upper: int, n_lambdas: int, density: int) -> NDArray:
    """Extend the legacy [1e-3, 1e5] log grid with its own anchor and spacing."""
    segments = (n_lambdas - 1) * density
    step = 8.0 / segments
    first = min(0, int(np.floor((lower + 3.0) / step)))
    last = max(segments, int(np.ceil((upper + 3.0) / step)))
    grid = np.power(10.0, -3.0 + step * np.arange(first, last + 1))
    # first <= 0, so 1e-3 sits at this nonnegative index; it is 0 when the
    # envelope starts at or above 1e-3, leaving nothing below to overwrite.
    legacy_start = -first
    if density == 1:
        grid[legacy_start : legacy_start + n_lambdas] = np.logspace(-3, 5, n_lambdas)
    return grid


def _lcurve_curvature(
    path: _PenaltyPath, energy: NDArray, residual_zero: float, lambdas: NDArray
) -> NDArray:
    """Signed L-curve curvature from analytic log-lambda derivatives.

    With horizontal log-residual, vertical log-penalty, and increasing lambda,
    the L-curve corner (Hansen and O'Leary 1993) is the convex bend from the
    steep small-lambda branch to the flat large-lambda branch: a
    counterclockwise turn, positive here (up to a positive factor, the
    negative of Hansen's ``lcfun``, which his ``l_corner`` minimizes).
    Clockwise bends are concave and negative here; under the conditions of
    Reginska (1996) and Hansen (2000, section 6) they are the ends of the
    curve. They must not be maximized, and neither must an absolute value
    that also rewards them.
    """
    weights, shrink = _shrinkage(path.values, lambdas)
    dw = -weights * shrink
    d2w = dw * (weights - shrink)
    residual = residual_zero + shrink**2 @ energy
    residual_d1 = -2.0 * (shrink * dw) @ energy
    residual_d2 = 2.0 * (dw**2 - shrink * d2w) @ energy
    penalty = (path.values * weights**2) @ energy
    penalty_d1 = 2.0 * (path.values * weights * dw) @ energy
    penalty_d2 = 2.0 * (path.values * (dw**2 + weights * d2w)) @ energy
    with np.errstate(divide="ignore", invalid="ignore"):
        dx = residual_d1 / residual
        dy = penalty_d1 / penalty
        ddx = residual_d2 / residual - dx**2
        ddy = penalty_d2 / penalty - dy**2
        return (dx * ddy - dy * ddx) / (dx**2 + dy**2) ** 1.5


class _LcurveCandidate(NamedTuple):
    grid_index: int
    lam: float
    curvature: float
    success: bool
    iterations: int
    function_evaluations: int

    def as_dict(self) -> dict[str, object]:
        return {
            "grid_index": self.grid_index,
            "lambda": self.lam,
            "curvature": self.curvature,
            "success": self.success,
            "iterations": self.iterations,
            "function_evaluations": self.function_evaluations,
        }


class _LcurveCorner(NamedTuple):
    lambdas: NDArray
    curvature: NDArray
    candidates: list[_LcurveCandidate]
    selected: _LcurveCandidate
    nonfinite: int


def _lcurve_peaks(curvature: NDArray) -> NDArray:
    """Indices of positive interior local maxima, excluding five points per end."""
    middle, left, right = curvature[5:-5], curvature[4:-6], curvature[6:-4]
    finite = np.isfinite(left) & np.isfinite(middle) & np.isfinite(right)
    with np.errstate(invalid="ignore"):
        peak = finite & (middle > 0) & (middle >= left) & (middle >= right)
    return np.flatnonzero(peak) + 5


def _refine_lcurve_peak(
    path: _PenaltyPath,
    energy: NDArray,
    residual_zero: float,
    lambdas: NDArray,
    curvature: NDArray,
    index: int,
) -> _LcurveCandidate:
    """Refine a grid peak in log lambda, never returning less than the peak."""

    def negative_curvature(log_lam: float) -> float:
        lam = np.exp(np.array([log_lam]))
        return -float(_lcurve_curvature(path, energy, residual_zero, lam)[0])

    result = sp.optimize.minimize_scalar(
        negative_curvature,
        bounds=(float(np.log(lambdas[index - 1])), float(np.log(lambdas[index + 1]))),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 200, "disp": 0},
    )
    lam, value = float(np.exp(result.x)), -float(result.fun)
    if not value >= curvature[index]:
        lam, value = float(lambdas[index]), float(curvature[index])
    return _LcurveCandidate(
        index, lam, value, bool(result.success), int(result.nit), int(result.nfev)
    )


def _lcurve_corner(
    path: _PenaltyPath, energy: NDArray, residual_zero: float, lambdas: NDArray
) -> _LcurveCorner:
    """Refine every positive interior curvature peak and keep the largest."""
    curvature = _lcurve_curvature(path, energy, residual_zero, lambdas)
    eligible = len(lambdas) - 10
    nonfinite = int(np.count_nonzero(~np.isfinite(curvature[5:-5])))
    if nonfinite == eligible:
        raise ValueError(
            f"L-curve has no finite curvature among {eligible} eligible grid points"
        )
    peaks = _lcurve_peaks(curvature)
    if len(peaks) == 0:
        raise ValueError(
            f"L-curve has no positive interior curvature peak among {eligible} "
            f"eligible grid points in [{lambdas[0]:.3g}, {lambdas[-1]:.3g}]; "
            "no corner is selected"
        )
    candidates = [
        _refine_lcurve_peak(path, energy, residual_zero, lambdas, curvature, int(index))
        for index in peaks
    ]
    selected = max(candidates, key=lambda candidate: candidate.curvature)
    return _LcurveCorner(lambdas, curvature, candidates, selected, nonfinite)


def _lcurve_stability(
    path: _PenaltyPath,
    energy: NDArray,
    residual_zero: float,
    reference: _LcurveCandidate,
    lambdas: NDArray,
) -> dict[str, object]:
    """Compare the corner found on a check grid with the reference corner."""
    try:
        check = _lcurve_corner(path, energy, residual_zero, lambdas).selected
    except ValueError as error:
        return {"passed": False, "error": str(error)}
    template_change = path.template_change(check.lam, reference.lam)
    curvature_change = _relative_change(
        abs(check.curvature - reference.curvature), reference.curvature
    )
    return {
        "lambda": check.lam,
        "curvature": check.curvature,
        "template_change_over_ols_norm": template_change,
        "relative_curvature_change": curvature_change,
        "passed": template_change <= 1e-5 and curvature_change <= 1e-5,
    }


def _warn_lcurve(
    corner: _LcurveCorner, stability: dict[str, dict[str, object]]
) -> None:
    if corner.nonfinite:
        warnings.warn(
            f"L-curve excluded {corner.nonfinite} nonfinite eligible curvature values.",
            RuntimeWarning,
            stacklevel=3,
        )
    unconverged = [row.grid_index for row in corner.candidates if not row.success]
    if unconverged:
        warnings.warn(
            "L-curve corner refinement did not converge for grid peaks "
            f"{unconverged}; their best evaluated points are retained.",
            RuntimeWarning,
            stacklevel=3,
        )
    failed = {name: row for name, row in stability.items() if not row["passed"]}
    if failed:
        warnings.warn(
            f"L-curve corner is not stable under the {', '.join(failed)} check: "
            "allowed changes are 1e-5 of the OLS template norm and 1e-5 relative "
            f"curvature; got {failed}. Review this fit.",
            RuntimeWarning,
            stacklevel=3,
        )


def _select_lcurve(
    path: _PenaltyPath,
    X: NDArray,
    D: NDArray | scipy.sparse.spmatrix,
    n_lambdas: int = 80,
) -> tuple[float, dict[str, object]]:
    if n_lambdas < 11:
        raise ValueError(f"L-curve requires at least 11 grid points, got {n_lambdas}")
    energy = np.sum(path.rotated**2, axis=1)
    variation = path.variation_bound()
    # Within the stability tolerance every lambda gives the same estimate, and
    # roundoff in the penalized modes would draw a plausible-looking corner.
    if variation <= 1e-5:
        raise ValueError(
            "L-curve is undefined: templates along the whole regularization "
            f"path differ by at most {variation:.3e} of the OLS norm, within "
            "the 1e-5 stability tolerance"
        )
    fitted = np.asarray(D @ path.ols)
    residual_zero = float(np.sum((np.asarray(X, dtype=float) - fitted) ** 2))
    lower, upper = path.log10_envelope()
    corner = _lcurve_corner(
        path, energy, residual_zero, _lcurve_grid(lower, upper, n_lambdas, 1)
    )
    checks = {"denser": (lower, upper, 2), "outer": (lower - 2, upper + 2, 1)}
    stability = {
        name: _lcurve_stability(
            path,
            energy,
            residual_zero,
            corner.selected,
            _lcurve_grid(low, high, n_lambdas, density),
        )
        for name, (low, high, density) in checks.items()
    }
    _warn_lcurve(corner, stability)
    selected, index = corner.selected, corner.selected.grid_index
    return selected.lam, {
        "opt_val": selected.lam,
        "method": "lcurve",
        "grid_index": index,
        "grid_size": len(corner.lambdas),
        "grid_min": float(corner.lambdas[0]),
        "grid_max": float(corner.lambdas[-1]),
        "grid_log10_envelope": (lower, upper),
        "excluded_edge_points": 5,
        "at_first_eligible": index == 5,
        "at_last_eligible": index == len(corner.lambdas) - 6,
        "selected_grid_lambda": float(corner.lambdas[index]),
        "selected_grid_curvature": float(corner.curvature[index]),
        "selected_curvature": selected.curvature,
        "positive_corner": True,
        "corner_candidates": [row.as_dict() for row in corner.candidates],
        "nonfinite_eligible_curvatures": corner.nonfinite,
        "stability": stability,
        "stability_passed": all(bool(row["passed"]) for row in stability.values()),
    }


def select_lambda_lcurve(
    DtD: NDArray | scipy.sparse.spmatrix,
    DtX: NDArray | scipy.sparse.spmatrix,
    S: NDArray | scipy.sparse.spmatrix,
    X: NDArray,
    D: NDArray | scipy.sparse.spmatrix,
    n_lambdas: int = 80,
) -> tuple[float, dict[str, object]]:
    """Select regularization lambda via L-curve (maximum convex curvature).

    The log grid keeps the legacy [1e-3, 1e5] anchor and spacing of
    ``8 / (n_lambdas - 1)`` decades and extends it to six decades beyond the
    penalty's first and last transition scales. Residual and penalty norms
    and their log-lambda derivatives are evaluated analytically on a stable
    path. Five grid points at each end are excluded, every positive interior
    curvature peak is refined in log lambda, and the largest refined peak is
    the corner. A grid with doubled density and one with two extra decades at
    each end must reproduce it within 1e-5 of the OLS template norm and 1e-5
    relative curvature; otherwise the fit is flagged with a warning. Without
    a positive interior peak, or when no two lambdas can change the template
    by more than 1e-5 of the OLS norm (every lambda gives the same estimate),
    no corner is invented and ValueError is raised.

    Parameters
    ----------
    DtD : NDArray | scipy.sparse.spmatrix
        Gram matrix ``D^T D`` of shape ``(p, p)``; its condition number must
        stay below ``1e-7 / (p * eps)``.
    DtX : NDArray | scipy.sparse.spmatrix
        Cross-product ``D^T X`` of shape ``(p, C)``.
    S : NDArray | scipy.sparse.spmatrix
        Positive semidefinite penalty matrix of shape ``(p, p)``. Eigenvalues
        at most ``p * eps`` times the largest are treated as exact zeros.
    X : NDArray
        Data matrix of shape ``(n, C)``.
    D : NDArray | scipy.sparse.spmatrix
        Design matrix of shape ``(n, p)``.
    n_lambdas : int
        Points on the legacy [1e-3, 1e5] segment, which sets the grid
        spacing; at least 11.

    Returns
    -------
    tuple[float, dict[str, object]]
        Corner lambda and diagnostics for the grid, refined candidates, and
        stability checks.
    """
    path = _PenaltyPath(_dense(DtD), _dense(DtX), _penalty_factor(_dense(S)))
    return _select_lcurve(path, X, D, n_lambdas)


class _EmpBayesSystem:
    """Shared white-noise EB statistics with flat unpenalized components."""

    def __init__(
        self,
        DtD: NDArray | scipy.sparse.spmatrix,
        DtX: NDArray | scipy.sparse.spmatrix,
        S: NDArray | scipy.sparse.spmatrix,
        X: NDArray,
        D: NDArray | scipy.sparse.spmatrix,
    ) -> None:
        gram = _EmpBayesSystem._dense(DtD, "DtD")
        cross = _EmpBayesSystem._dense(DtX, "DtX")
        penalty = _EmpBayesSystem._dense(S, "S")
        data = _EmpBayesSystem._dense(X, "X")
        if data.ndim == 1:
            data = data[:, None]
        if cross.ndim == 1:
            cross = cross[:, None]
        if data.ndim != 2 or min(data.shape) == 0:
            raise ValueError("EB requires a nonempty data matrix")
        n_obs, n_cols = data.shape
        if gram.ndim != 2 or gram.shape[0] == 0:
            raise ValueError("EB requires a nonempty Gram matrix")
        n_parameters = gram.shape[0]
        if gram.shape != (n_parameters, n_parameters):
            raise ValueError("EB Gram matrix must be square")
        if penalty.shape != gram.shape:
            raise ValueError("EB penalty and Gram matrices must have equal shapes")
        if cross.shape != (n_parameters, n_cols):
            raise ValueError("EB cross-product shape does not match the model")
        if sp.sparse.issparse(D):
            sparse_design = sp.sparse.csr_matrix(D)
            _EmpBayesSystem._dense(sparse_design.data, "D entries")
            design = sp.sparse.csr_matrix(sparse_design, dtype=float)
        else:
            design = _EmpBayesSystem._dense(D, "D")
        if design.shape != (n_obs, n_parameters):
            raise ValueError("EB design shape does not match the data and Gram matrix")
        _EmpBayesSystem._psd_eigenvalues(gram, "DtD")
        values = _EmpBayesSystem._psd_eigenvalues(penalty, "S")
        # This is a numerical rank test, not a ridge or a change to S.
        rank_tolerance = float(
            n_parameters * np.finfo(float).eps * np.max(np.abs(values))
        )
        rank = int(np.count_nonzero(values > rank_tolerance))
        nullity = n_parameters - rank
        if rank == 0:
            raise ValueError("EB smoothing precision requires a nonzero penalty rank")
        if n_obs <= nullity:
            raise ValueError(
                "EB requires residual degrees of freedom beyond the nullspace"
            )
        self.gram = gram
        self.cross = cross
        self.penalty = penalty
        self.data = data
        self.design = design
        self.n_obs = n_obs
        self.n_cols = n_cols
        self.rank_tolerance = rank_tolerance
        self.rank = rank
        self.nullity = nullity

    @staticmethod
    def _dense(value: NDArray | scipy.sparse.spmatrix, name: str) -> NDArray:
        array = np.asarray(value.todense() if sp.sparse.issparse(value) else value)
        if not np.isrealobj(array):
            raise ValueError(f"EB requires real-valued {name}")
        array = np.asarray(array, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"EB requires finite {name}")
        return array

    @staticmethod
    def _psd_eigenvalues(matrix: NDArray, name: str) -> NDArray:
        tolerance = len(matrix) * np.finfo(float).eps * np.max(np.abs(matrix))
        if not np.allclose(matrix, matrix.T, rtol=0, atol=tolerance):
            raise ValueError(f"EB requires symmetric {name}")
        values = sp.linalg.eigvalsh(matrix)
        tolerance = len(matrix) * np.finfo(float).eps * np.max(np.abs(values))
        if np.min(values) < -tolerance:
            raise ValueError(f"EB requires positive-semidefinite {name}")
        return values

    def statistics(self, lam: float) -> tuple[NDArray, float, float, float]:
        factor = sp.linalg.cho_factor(self.gram + lam * self.penalty, lower=True)
        theta = sp.linalg.cho_solve(factor, self.cross)
        gamma = float(np.trace(sp.linalg.cho_solve(factor, self.gram)))
        # Direct residuals avoid subtraction of nearly equal quadratic forms.
        residual_ss = float(np.sum((self.data - self.design @ theta) ** 2))
        penalty_ss = float(np.sum(theta * (self.penalty @ theta)))
        return theta, gamma, residual_ss, penalty_ss

    def update(
        self, gamma: float, residual_ss: float, penalty_ss: float
    ) -> tuple[float, float]:
        quantities = {
            "penalized effective parameter count": gamma - self.nullity,
            "residual degrees of freedom": self.n_obs - gamma,
            "residual sum of squares": residual_ss,
            "penalty sum of squares": penalty_ss,
        }
        for name, value in quantities.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid EB {name}: {value}")
        alpha = self.n_cols * (gamma - self.nullity) / penalty_ss
        beta = self.n_cols * (self.n_obs - gamma) / residual_ss
        if not np.isfinite(alpha) or not np.isfinite(beta) or alpha <= 0 or beta <= 0:
            raise ValueError("EB precision update is not finite and positive")
        updated = alpha / beta
        if not np.isfinite(updated) or updated <= 0:
            raise ValueError("EB lambda update is not finite and positive")
        noise_var = 1.0 / beta
        if not np.isfinite(noise_var) or noise_var <= 0:
            raise ValueError("EB noise variance is not finite and positive")
        return float(updated), float(noise_var)


def _fit_empbayes(
    DtD: NDArray | scipy.sparse.spmatrix,
    DtX: NDArray | scipy.sparse.spmatrix,
    S: NDArray | scipy.sparse.spmatrix,
    X: NDArray,
    D: NDArray | scipy.sparse.spmatrix,
    max_iter: int,
    tol: float,
) -> tuple[float, NDArray, float, dict[str, object]]:
    if (
        isinstance(max_iter, bool)
        or not isinstance(max_iter, (int, np.integer))
        or max_iter < 1
    ):
        raise ValueError("EB max_iter must be a positive integer")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("EB tolerance must be finite and positive")
    system = _EmpBayesSystem(DtD, DtX, S, X, D)
    lam = 1.0
    converged = False
    theta, gamma, residual_ss, penalty_ss = system.statistics(lam)
    relative_update = 0.0
    for _iteration in range(1, max_iter + 1):
        updated, _ = system.update(gamma, residual_ss, penalty_ss)
        relative_update = abs(updated - lam) / lam
        if not np.isfinite(relative_update):
            raise ValueError("EB relative update is not finite")
        lam = updated
        # Return diagnostics and the fit at the returned lambda, not its predecessor.
        theta, gamma, residual_ss, penalty_ss = system.statistics(lam)
        if relative_update < tol:
            converged = True
            break
    next_lam, noise_var = system.update(gamma, residual_ss, penalty_ss)
    fixed_point_update = abs(next_lam - lam) / lam
    if not np.isfinite(fixed_point_update):
        raise ValueError("EB final fixed-point residual is not finite")
    info: dict[str, object] = {
        "opt_val": lam,
        "method": "empbayes",
        "gamma": gamma,
        "gamma_total": system.n_cols * gamma,
        "gamma_penalized": gamma - system.nullity,
        "gamma_penalized_total": system.n_cols * (gamma - system.nullity),
        "n_iter": _iteration,
        "converged": converged,
        "status": "converged" if converged else "iteration_limit",
        "penalty_rank": system.rank,
        "penalty_nullity": system.nullity,
        "penalty_rank_tolerance": system.rank_tolerance,
        "noise_var": noise_var,
        "residual_ss": residual_ss,
        "penalty_ss": penalty_ss,
        "last_relative_update": float(relative_update),
        "fixed_point_relative_update": float(fixed_point_update),
        "initial_lambda": 1.0,
        "max_iter": int(max_iter),
        "tolerance": float(tol),
    }
    return lam, theta, noise_var, info


def select_lambda_empbayes(
    DtD: NDArray | scipy.sparse.spmatrix,
    DtX: NDArray | scipy.sparse.spmatrix,
    S: NDArray | scipy.sparse.spmatrix,
    X: NDArray,
    D: NDArray | scipy.sparse.spmatrix,
    max_iter: int = 30,
    tol: float = 1e-4,
) -> tuple[float, dict[str, object]]:
    """Select regularization lambda via Empirical Bayes (Evidence Maximization).

    Uses shared precision parameters across channels. Unpenalized coefficients
    have flat priors independent of those precisions. The resulting restricted
    evidence is ordinary Gaussian evidence when the penalty is full rank.
    The penalty nullspace must be identifiable through the design matrix.

    Parameters
    ----------
    DtD : NDArray | scipy.sparse.spmatrix
        Gram matrix ``D^T D`` of shape ``(p, p)``.
    DtX : NDArray | scipy.sparse.spmatrix
        Cross-product ``D^T X`` of shape ``(p, C)``.
    S : NDArray | scipy.sparse.spmatrix
        Penalty matrix of shape ``(p, p)``.
    X : NDArray
        Data matrix of shape ``(n, C)``.
    D : NDArray | scipy.sparse.spmatrix
        Design matrix of shape ``(n, p)``.
    max_iter : int
        Maximum number of MacKay update iterations.
    tol : float
        Relative convergence tolerance for lambda.

    Returns
    -------
    tuple[float, dict[str, object]]
        Last finite lambda and diagnostics, including convergence status,
        penalty rank/nullity, effective parameter counts, noise variance,
        and the final fixed-point residual. ``gamma`` is per channel.
        An iteration limit is not reported as convergence. Invalid updates
        or a non-positive-definite posterior precision raise an exception.
    """
    lam, _, _, info = _fit_empbayes(DtD, DtX, S, X, D, max_iter, tol)
    return lam, info


# --- Helpers for get_events_templates SURE optimization ---


def _gen_h_diff(n: int) -> NDArray:
    """Generate an n-th order finite difference kernel.

    Parameters
    ----------
    n : int
        Difference order.

    Returns
    -------
    NDArray
        1-D kernel of length ``n + 1``.
    """
    h_diff: NDArray = np.array([1])
    for _ii in range(n):
        h_diff = np.convolve(h_diff, [1, -1])
    return h_diff


class _AR1Noise(NamedTuple):
    """Explicit AR(1) covariance inputs; *None* fields use the estimates."""

    phi: float | None = None
    noise_var: float | None = None
    noise_var_scale: float = 1.0


class _ColoredNoise(NamedTuple):
    covariance: NDArray | None
    noise_var: float
    record: dict[str, object]


def _colored_covariance(
    data: NDArray,
    ix: NDArray,
    lb: NDArray,
    events: NDArray,
    half_win: int,
    noise_est: str,
    sigsq: float,
    event_blocks: NDArray | None,
    ar1: _AR1Noise | None,
    verbose: bool = False,
) -> _ColoredNoise:
    """Build D^T Sigma_W D for the colored SURE paths (*None* for white noise).

    *ar1* None means no explicit inputs. AR(1) uses ``ar1.phi`` unclipped
    when given, else :func:`estimate_phi`, and ``ar1.noise_var`` when given,
    else ``sigsq`` times a non-unit ``ar1.noise_var_scale``. The record
    holds the estimated and used values, or the AR(3) autocovariance.
    """
    if noise_est == "ar1_colored":
        ar1 = _AR1Noise() if ar1 is None else ar1
        phi_estimated = estimate_phi(data, ix, half_win)
        phi = phi_estimated if ar1.phi is None else ar1.phi
        if ar1.noise_var is not None:
            noise_var = ar1.noise_var
        elif ar1.noise_var_scale != 1:
            noise_var = sigsq * ar1.noise_var_scale
        else:
            noise_var = sigsq
        covariance = build_DtSigmaD_ar1(
            ix,
            lb,
            events,
            half_win,
            phi,
            noise_var,
            data_length=len(data),
            event_blocks=event_blocks,
        )
        if verbose:
            print(f"phi: {phi:.3f}, correction vs white: {(1 + phi) / (1 - phi):.1f}x")
        record: dict[str, object] = {
            "phi_estimated": float(phi_estimated),
            "phi_used": float(phi),
            "noise_var_estimated": float(sigsq),
            "noise_var_scale": float(ar1.noise_var_scale),
        }
        return _ColoredNoise(covariance, noise_var, record)
    if noise_est == "ar3_colored":
        ar_order = 3
        acf = estimate_acf_ar(
            data,
            ix,
            half_win,
            ar_order=ar_order,
            max_lag=_required_covariance_lag(ix, lb, events, half_win, len(data)),
        )
        covariance = build_DtSigmaD_acf(
            ix, lb, events, half_win, acf, len(data), event_blocks
        )
        if verbose:
            print(f"AR({ar_order}) ACF[0:5]: {acf[:5]}")
        return _ColoredNoise(covariance, sigsq, {"ar3_acf": acf})
    return _ColoredNoise(None, sigsq, {})


def _optimize_kernel(
    data: NDArray,
    a: scipy.sparse.spmatrix,
    b: scipy.sparse.spmatrix,
    ix: NDArray,
    lb: NDArray,
    events: NDArray,
    half_win: int,
    M: int,
    T: int,
    noise_est: str,
    omega: float,
    vlen: int,
    symm: bool,
    verbose: bool,
    event_blocks: NDArray | None = None,
    ar1: _AR1Noise | None = None,
) -> tuple[float, scipy.sparse.spmatrix, dict[str, object]]:
    """Optimize the convolution kernel via SURE (vec_mode).

    Returns the regularization strength (always 1.0), the updated
    temporal penalty matrix, and a diagnostics dict. *event_blocks* labels
    each event's recording block for the colored covariance; *ar1* holds
    explicit AR(1) inputs.
    """

    if event_blocks is not None:
        _check_block_noise(noise_est, lb, half_win)
    theta_ml = np.linalg.inv(a.todense()) @ b
    sigsq = est_noise_var(
        data,
        method=noise_est
        if noise_est not in {"ar1_colored", "ar3_colored"}
        else "baseline",
        ix=ix,
        lb=lb,
        half_win=half_win,
    )

    colored = _colored_covariance(
        data, ix, lb, events, half_win, noise_est, sigsq, event_blocks, ar1, verbose
    )
    sigsq = colored.noise_var
    DtSigmaD_vec = colored.covariance

    x0 = _gen_h_diff(vlen)
    x0 = x0[: int(vlen - ((vlen - 1) / 2) * int(symm))]

    objective = _KernelSUREObjective(
        a.todense(),
        theta_ml,
        data,
        sigsq,
        M,
        symm,
        omega,
        DtSigmaD=DtSigmaD_vec,
    )
    cost_at_x0, _ = objective.value_gradient(x0)
    cost_scale = max(abs(cost_at_x0), 1e-30)
    objective.cost_scale = cost_scale

    opt_res = sp.optimize.minimize(
        objective.value_gradient,
        x0=x0,
        jac=True,
        method=None,
        options={"maxiter": 50, "disp": 0},
    )

    opt_vec = gen_corefilt(opt_res.x, symm)
    R = sp.linalg.convolution_matrix(opt_vec, T - len(opt_vec) + 1)
    RR = R @ R.T
    temporal_mat = sp.sparse.block_diag([RR] * M, format="csr")
    reg_temporal = 1.0

    if verbose:
        print(f"opt_vec: {opt_vec}")
        print(f"noise var: {sigsq}")
        print(f"cost func: {opt_res.fun}")

    # With no bounds or constraints, method=None selects BFGS. Its returned
    # results include jac and njev even when it stops without converging.
    inner: dict[str, object] = {
        "opt_val": opt_vec,
        "noise_var": sigsq,
        "cost": opt_res.fun,
        "optimizer_success": bool(opt_res.success),
        "optimizer_status": int(opt_res.status),
        "optimizer_message": str(opt_res.message),
        "optimizer_iterations": int(opt_res.nit),
        "optimizer_function_evaluations": int(opt_res.nfev),
        "optimizer_gradient_evaluations": int(opt_res.njev),
        "optimizer_gradient_norm": float(np.linalg.norm(opt_res.jac)),
        "optimizer_gradient_max_abs": float(np.max(np.abs(opt_res.jac))),
        "optimizer_cost_scale": float(cost_scale),
    }
    inner.update(colored.record)
    return reg_temporal, temporal_mat, inner


def _local_minimum_brackets(
    grid: NDArray, scores: NDArray
) -> list[tuple[float, float]]:
    """Bracket each sampled local minimum, skipping exactly flat triples."""
    local = (
        (scores[1:-1] <= scores[:-2])
        & (scores[1:-1] <= scores[2:])
        & ((scores[1:-1] < scores[:-2]) | (scores[1:-1] < scores[2:]))
    )
    indices = np.flatnonzero(local) + 1
    return [(float(grid[index - 1]), float(grid[index + 1])) for index in indices]


_ScalarCandidate = tuple[float, float, str, int]


class _ScalarScan(NamedTuple):
    grid: NDArray
    scores: NDArray
    selected: _ScalarCandidate | None


class _SolveRecord(NamedTuple):
    search: str
    bounds: tuple[float, float]
    lam: float
    cost: float
    success: bool
    status: int
    message: str
    iterations: int
    function_evaluations: int

    def as_dict(self) -> dict[str, object]:
        return {
            "search": self.search,
            "bounds": self.bounds,
            "lambda": self.lam,
            "cost": self.cost,
            "success": self.success,
            "status": self.status,
            "message": self.message,
            "iterations": self.iterations,
            "function_evaluations": self.function_evaluations,
        }


class _ScalarSureScore:
    """The existing scalar SURE objective, evaluated on a stable path.

    With shrinkage weights ``w``, SURE_cost_fast and SURE_cost_colored (with
    precomputed trace coefficients) equal ``w^T Q w + l^T w``. Their
    omega != 0.5 term is reproduced as implemented: it scales
    ``tr(G (D^T D + S) G)`` with ``G = (D^T D + lam S)^{-1}``.
    """

    def __init__(
        self,
        path: _PenaltyPath,
        trace_coefficients: NDArray,
        n_cols: int,
        omega: float,
        correction_variance: float,
    ) -> None:
        basis_gram = path.basis.T @ path.basis
        quadratic = basis_gram * (path.rotated @ path.rotated.T)
        self.path: _PenaltyPath = path
        self.quadratic: NDArray = omega * quadratic
        if abs(omega - 0.5) > 1e-5:
            correction = (2 * omega - 1) * correction_variance * n_cols
            self.quadratic -= np.diag(
                correction * (1 + path.values) * np.diag(basis_gram)
            )
        self.linear: NDArray = (
            2 * omega * (n_cols * trace_coefficients - quadratic.sum(axis=1))
        )

    def __call__(self, lam: float) -> float:
        weights = self.path.weights(lam)
        return float(weights @ self.quadratic @ weights + self.linear @ weights)

    def log_derivative(self, log_lam: float) -> float:
        weights, shrink = _shrinkage(self.path.values, float(np.exp(log_lam)))
        gradient = 2.0 * self.quadratic @ weights + self.linear
        return float(-(weights * shrink) @ gradient)


class _ScalarSureSearch:
    """Score scalar SURE candidates on log grids, recording every solve."""

    def __init__(self, score: _ScalarSureScore) -> None:
        self.score = score
        self.evaluations = 0
        self.nonfinite: list[dict[str, float | str]] = []
        self.refinements: list[_SolveRecord] = []
        self.roots: list[_SolveRecord] = []
        self.limit_scores: dict[str, float] = {
            "zero": self.evaluate(0.0),
            "infinite": self.evaluate(np.inf),
        }

    def evaluate(self, lam: float) -> float:
        self.evaluations += 1
        value = self.score(lam)
        if not np.isfinite(value):
            self.nonfinite.append({"lambda": float(lam), "score": str(value)})
            return np.inf
        return value

    def scan(self, name: str, lower: int, upper: int, per_decade: int) -> _ScalarScan:
        grid = np.power(
            10.0, np.linspace(lower, upper, (upper - lower) * per_decade + 1)
        )
        scores = np.array([self.evaluate(float(lam)) for lam in grid])
        candidates: list[_ScalarCandidate] = [
            (self.limit_scores["zero"], 0.0, "zero", 0),
            (self.limit_scores["infinite"], np.inf, "infinite", 0),
        ]
        candidates += [
            (float(score), float(lam), "grid", index)
            for index, (lam, score) in enumerate(zip(grid, scores))
        ]
        candidates += self._refine(name, np.log(grid), scores)
        candidates += self._stationary(name, np.log(grid))
        finite = [row for row in candidates if np.isfinite(row[0])]
        return _ScalarScan(grid, scores, min(finite) if finite else None)

    def _refine(
        self, name: str, log_grid: NDArray, scores: NDArray
    ) -> list[_ScalarCandidate]:
        found: list[_ScalarCandidate] = []
        for bounds in _local_minimum_brackets(log_grid, scores):
            result = sp.optimize.minimize_scalar(
                lambda log_lam: self.evaluate(float(np.exp(log_lam))),
                bounds=bounds,
                method="bounded",
                options={"xatol": 1e-10, "maxiter": 100, "disp": 0},
            )
            record = _SolveRecord(
                name,
                bounds,
                float(np.exp(result.x)),
                float(result.fun),
                bool(result.success),
                int(result.status),
                str(result.message),
                int(result.nit),
                int(result.nfev),
            )
            self.refinements.append(record)
            if np.isfinite(record.cost):
                found.append(
                    (record.cost, record.lam, "refinement", len(self.refinements) - 1)
                )
        return found

    def _stationary(self, name: str, log_grid: NDArray) -> list[_ScalarCandidate]:
        slopes = np.array([self.score.log_derivative(float(t)) for t in log_grid])
        found: list[_ScalarCandidate] = []
        for index in np.flatnonzero((slopes[:-1] < 0) & (slopes[1:] > 0)):
            bounds = (float(log_grid[index]), float(log_grid[index + 1]))
            root, report = sp.optimize.brentq(
                self.score.log_derivative,
                *bounds,
                xtol=1e-12,
                full_output=True,
                disp=False,
            )
            lam = float(np.exp(root))
            record = _SolveRecord(
                name,
                bounds,
                lam,
                self.evaluate(lam),
                bool(report.converged),
                0 if report.converged else 1,
                str(report.flag),
                int(report.iterations),
                int(report.function_calls),
            )
            self.roots.append(record)
            if np.isfinite(record.cost):
                found.append((record.cost, lam, "stationary", len(self.roots) - 1))
        return found


def _scalar_stability(
    path: _PenaltyPath, reference: _ScalarCandidate, check: _ScalarCandidate | None
) -> dict[str, object]:
    """Compare the selection from a check grid with the base selection."""
    if check is None:
        return {"passed": False, "error": "no finite candidate"}
    energy = float(np.sum(path.ols**2))
    score_change = _relative_change(abs(check[0] - reference[0]), energy)
    template_change = path.template_change(check[1], reference[1])
    return {
        "lambda": check[1],
        "selected_from": check[2],
        "score_change_over_ols_energy": score_change,
        "template_change_over_ols_norm": template_change,
        "passed": score_change <= 1e-8 and template_change <= 1e-5,
    }


def _warn_scalar_search(
    search: _ScalarSureSearch, stability: dict[str, dict[str, object]]
) -> None:
    if search.nonfinite:
        first = search.nonfinite[0]
        warnings.warn(
            f"Scalar SURE excluded {len(search.nonfinite)} nonfinite objective "
            f"evaluations; first at lambda={first['lambda']} "
            f"(score={first['score']}). Returning the best finite candidate; "
            "inspect numerical stability before using this fit.",
            RuntimeWarning,
            stacklevel=3,
        )
    failed = {name: row for name, row in stability.items() if not row["passed"]}
    if failed:
        warnings.warn(
            f"Scalar SURE selection is not stable under the {', '.join(failed)} "
            "check: allowed changes are 1e-8 of the OLS template energy in score "
            f"and 1e-5 of its norm in templates; got {failed}. Review this fit.",
            RuntimeWarning,
            stacklevel=3,
        )


def _scalar_search_diagnostics(
    search: _ScalarSureSearch,
    base: _ScalarScan,
    selected: _ScalarCandidate,
    stability: dict[str, dict[str, object]],
    exponents: tuple[int, int],
) -> dict[str, object]:
    cost, lam, kind, index = selected
    solves = search.refinements + search.roots
    failures = [row for row in solves if not row.success]
    message = (
        f"All {len(solves)} refinements and root searches converged; "
        "global optimum not certified."
    )
    if failures:
        message = (
            f"{len(failures)} of {len(solves)} refinements and root searches "
            f"did not converge; first in the {failures[0].search} scan: "
            f"{failures[0].message}"
        )
    bounds = (10.0 ** exponents[0], 10.0 ** exponents[1])
    return {
        "opt_val": lam,
        "cost": cost,
        "optimizer_success": not failures,
        "optimizer_status": failures[0].status if failures else 0,
        "optimizer_message": message,
        "optimizer_iterations": sum(row.iterations for row in solves),
        "optimizer_function_evaluations": search.evaluations,
        "search_method": "grid_and_bounded_refinement",
        "search_policy": "design_scaled_grid_with_zero_and_infinite_limits",
        "search_bounds": bounds,
        "search_log10_bounds": exponents,
        "search_limit_scores": search.limit_scores,
        "search_grid_lambdas": base.grid,
        "search_grid_scores": base.scores,
        "search_all_evaluations_finite": not search.nonfinite,
        "search_nonfinite_evaluations": search.nonfinite,
        "search_selected_from": kind,
        "search_selected_index": index,
        "search_near_lower_bound": lam <= bounds[0],
        "search_near_upper_bound": lam >= bounds[1],
        "search_refinements": [row.as_dict() for row in search.refinements],
        "search_stationary_points": [row.as_dict() for row in search.roots],
        "search_stability": stability,
        "search_stability_passed": all(
            bool(row["passed"]) for row in stability.values()
        ),
    }


def _search_scalar_sure_path(
    score: _ScalarSureScore,
) -> tuple[float, dict[str, object]]:
    """Search scalar SURE over [0, inf] on a design-scaled log grid.

    The finite grid spans six decades beyond the penalty's first and last
    transition scales at ten points per decade, and the exact lambda=0 and
    infinite-penalty limits are candidates. Sampled local minima are refined
    and derivative sign changes are root-bracketed in log lambda. Candidates
    are compared by the SURE objective only, and exact ties select the smaller
    lambda. A grid with 20 points per decade and one with two extra decades at
    each end must reproduce the selection within 1e-8 of the OLS template
    energy in score and 1e-5 of its norm in templates; otherwise the fit is
    flagged with a warning. Nonfinite scores are excluded and reported; no
    finite candidate is an explicit failure. A finite scan does not certify a
    global optimum.
    """
    search = _ScalarSureSearch(score)
    lower, upper = score.path.log10_envelope()
    base = search.scan("base", lower, upper, 10)
    selected = base.selected
    if selected is None:
        raise ValueError(
            "Scalar SURE search found no finite candidate after "
            f"{search.evaluations} objective evaluations on [0, inf]"
        )
    checks = {
        "denser": search.scan("denser", lower, upper, 20),
        "outer": search.scan("outer", lower - 2, upper + 2, 10),
    }
    stability = {
        name: _scalar_stability(score.path, selected, check.selected)
        for name, check in checks.items()
    }
    _warn_scalar_search(search, stability)
    diagnostics = _scalar_search_diagnostics(
        search, base, selected, stability, (lower, upper)
    )
    return selected[1], diagnostics


def _scalar_sure_score(
    path: _PenaltyPath,
    n_cols: int,
    sigsq: float,
    omega: float,
    DtSigmaD: NDArray | None,
) -> _ScalarSureScore:
    """Build the white (*DtSigmaD* None) or colored scalar SURE objective."""
    if DtSigmaD is None:
        trace = sigsq * np.sum(path.basis**2, axis=0)
        return _ScalarSureScore(path, trace, n_cols, omega, sigsq)
    inverse_basis = sp.linalg.cho_solve((path.chol, True), path.basis)
    trace = np.sum(path.basis * (DtSigmaD @ inverse_basis), axis=0)
    variance = float(np.trace(DtSigmaD)) / len(trace)
    return _ScalarSureScore(path, trace, n_cols, omega, variance)


def _optimize_lambda(
    data: NDArray,
    a: scipy.sparse.spmatrix,
    b: scipy.sparse.spmatrix,
    temporal_factor: NDArray,
    ix: NDArray,
    lb: NDArray,
    events: NDArray,
    half_win: int,
    noise_est: str,
    omega: float,
    verbose: bool,
    event_blocks: NDArray | None = None,
    ar1: _AR1Noise | None = None,
) -> tuple[float, dict[str, object], _PenaltyPath]:
    """Optimize the scalar lambda via SURE.

    Search [0, inf] on a design-scaled grid of a stable path built from
    ``temporal_factor`` (``F^T F`` is the temporal penalty), which is also
    returned for computing the templates. Return the best evaluated
    candidate and all stopping reports, including unsuccessful ones.
    *event_blocks* labels each event's recording block for the colored
    covariance; *ar1* holds explicit AR(1) inputs.
    """
    if event_blocks is not None:
        _check_block_noise(noise_est, lb, half_win)
    sigsq = est_noise_var(
        data,
        method=noise_est
        if noise_est not in {"ar1_colored", "ar3_colored"}
        else "baseline",
        ix=ix,
        lb=lb,
        half_win=half_win,
    )

    path = _PenaltyPath(_dense(a), _dense(b), temporal_factor)
    colored = _colored_covariance(
        data, ix, lb, events, half_win, noise_est, sigsq, event_blocks, ar1
    )
    sigsq, record = colored.noise_var, colored.record
    n_cols = data.shape[1] if data.ndim > 1 else 1
    score = _scalar_sure_score(path, n_cols, sigsq, omega, colored.covariance)
    reg_temporal, inner = _search_scalar_sure_path(score)

    inner["noise_var"] = sigsq
    inner.update(record)
    if not inner["optimizer_success"]:
        warnings.warn(
            "SURE(lambda) stopped without converging: "
            f"{inner['optimizer_message']} (status={inner['optimizer_status']}, "
            f"function evaluations={inner['optimizer_function_evaluations']}, "
            f"returned lambda={reg_temporal}).",
            RuntimeWarning,
            stacklevel=2,
        )

    if verbose:
        print(f"tmp_reg: {reg_temporal}")
        print(f"noise var: {sigsq}")

    return reg_temporal, inner, path


# Estimators that filter, or pair adjacent samples, across a block join.
_JOINED_NOISE_ESTIMATORS = frozenset({"hpf"})


def _event_blocks_from_starts(
    block_starts: NDArray, ix: NDArray, half_win: int, n_samples: int
) -> NDArray:
    """Validate concatenated-block starts and label each event by its block."""
    starts = np.asarray(block_starts)
    if starts.ndim != 1 or starts.size == 0:
        raise ValueError(
            f"block_starts must be a nonempty 1-D array, got shape {starts.shape}"
        )
    if not np.issubdtype(starts.dtype, np.integer):
        raise ValueError(
            f"block_starts must contain integers, got dtype {starts.dtype}"
        )
    if starts[0] != 0:
        raise ValueError(f"block_starts must begin at 0, got {starts[0]}")
    # Compare without subtraction, so unsigned and narrow integers cannot wrap.
    if np.any(starts[1:] <= starts[:-1]):
        raise ValueError(f"block_starts must be strictly increasing, got {starts}")
    if starts[-1] >= n_samples:
        raise ValueError(
            f"The last block start {starts[-1]} leaves no samples before the "
            f"data length {n_samples}"
        )
    onsets = np.asarray(ix)
    if not np.issubdtype(onsets.dtype, np.integer):
        raise ValueError(
            f"ix must contain integers with block_starts, got dtype {onsets.dtype}"
        )
    for name, outside in (
        ("starts before sample 0", onsets < half_win),
        (f"ends after the data length {n_samples}", onsets > n_samples - half_win),
    ):
        if np.any(outside):
            event = int(np.flatnonzero(outside)[0])
            raise ValueError(
                f"Event {event} at sample {onsets[event]}: its window {name}"
            )
    # Both are now bounded by n_samples.
    starts = starts.astype(np.int64)
    first = onsets.astype(np.int64) - half_win
    last = first + 2 * half_win - 1
    blocks = np.searchsorted(starts, first, side="right") - 1
    crossing = blocks != np.searchsorted(starts, last, side="right") - 1
    if np.any(crossing):
        event = int(np.flatnonzero(crossing)[0])
        raise ValueError(
            f"Event {event} at sample {ix[event]}: its window "
            f"[{first[event]}, {last[event] + 1}) crosses the block start "
            f"{starts[blocks[event] + 1]}"
        )
    return blocks


def _check_block_noise(noise_est: str, lb: NDArray, half_win: int) -> None:
    """Reject noise estimates that would mix samples across block joins."""
    if noise_est in _JOINED_NOISE_ESTIMATORS:
        raise ValueError(
            f"noise_est={noise_est!r} filters or pairs samples across block "
            "joins and is not supported with block_starts"
        )
    _, counts = np.unique(lb, return_counts=True)
    if noise_est == "pairwise" and not np.any(counts >= 2):
        raise ValueError(
            "Pairwise noise estimation needs two events of one label; its HPF "
            "fallback crosses block joins and is not supported with block_starts"
        )
    # Shortest baseline segments of estimate_phi and estimate_acf_ar.
    baseline = {"ar1_colored": 4, "ar3_colored": 7}.get(noise_est, 0)
    if baseline > 2 * half_win:
        raise ValueError(
            f"noise_est={noise_est!r} uses baseline segments of at least "
            f"{baseline} samples, longer than the {2 * half_win}-sample window, "
            "so they can cross block joins"
        )


def _real_value(argument: str, value: object) -> float:
    """Require a real scalar and convert it to float before any check or use."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"{argument} must be a real number, got {type(value).__name__} {value!r}"
        )
    try:
        return float(value)
    except OverflowError as error:
        raise ValueError(
            f"{argument} must be finite, got an integer beyond the float range"
        ) from error


def _real_ar1(ar1: _AR1Noise) -> _AR1Noise:
    """Explicit AR(1) inputs as floats; *None* stays for phi and variance."""
    return _AR1Noise(
        None if ar1.phi is None else _real_value("ar1_phi", ar1.phi),
        None if ar1.noise_var is None else _real_value("ar1_noise_var", ar1.noise_var),
        _real_value("noise_var_scale", ar1.noise_var_scale),
    )


def _check_ar1_values(ar1: _AR1Noise) -> None:
    """Reject explicit AR(1) inputs outside a stationary, positive model."""
    if ar1.phi is not None:
        if not np.isfinite(ar1.phi):
            raise ValueError(f"ar1_phi must be finite, got {ar1.phi!r}")
        if ar1.phi < 0:
            raise ValueError(f"ar1_phi must be >= 0, got {ar1.phi!r}")
        if ar1.phi >= 1:
            raise ValueError(
                f"ar1_phi must be < 1 for a stationary AR(1) covariance, got {ar1.phi!r}"
            )
    if ar1.noise_var is not None:
        if not np.isfinite(ar1.noise_var):
            raise ValueError(f"ar1_noise_var must be finite, got {ar1.noise_var!r}")
        if ar1.noise_var <= 0:
            raise ValueError(f"ar1_noise_var must be > 0, got {ar1.noise_var!r}")
    if not np.isfinite(ar1.noise_var_scale):
        raise ValueError(f"noise_var_scale must be finite, got {ar1.noise_var_scale!r}")
    if ar1.noise_var_scale <= 0:
        raise ValueError(f"noise_var_scale must be > 0, got {ar1.noise_var_scale!r}")
    if ar1.noise_var is not None and ar1.noise_var_scale != 1:
        raise ValueError(
            f"ar1_noise_var={ar1.noise_var!r} and "
            f"noise_var_scale={ar1.noise_var_scale!r} are mutually exclusive: "
            "a non-unit scale applies only to the estimated variance"
        )


def _check_ar1_context(
    method: str,
    opt_reg_temporal: bool,
    reg_method: str | None,
    noise_est: str,
) -> None:
    """Reject explicit AR(1) inputs where no AR(1)-colored SURE uses them."""
    where = (
        "ar1_phi, ar1_noise_var and noise_var_scale apply only to AR(1)-colored SURE"
    )
    if method != "ls":
        raise ValueError(f"{where}, which needs method='ls'; got method={method!r}")
    if not opt_reg_temporal:
        raise ValueError(
            f"{where}, which needs opt_reg_temporal=True; "
            f"got opt_reg_temporal={opt_reg_temporal!r}"
        )
    if reg_method is not None:
        raise ValueError(
            f"{where}, which needs reg_method=None; got reg_method={reg_method!r}"
        )
    if noise_est != "ar1_colored":
        raise ValueError(
            f"{where}, which needs noise_est='ar1_colored'; got noise_est={noise_est!r}"
        )


def _ar1_inputs(
    ar1: _AR1Noise,
    method: str,
    opt_reg_temporal: bool,
    reg_method: str | None,
    noise_est: str,
) -> _AR1Noise | None:
    """Validate explicit AR(1) inputs as floats; *None* when all are defaults."""
    ar1 = _real_ar1(ar1)
    if ar1.phi is None and ar1.noise_var is None and ar1.noise_var_scale == 1:
        return None
    _check_ar1_values(ar1)
    _check_ar1_context(method, opt_reg_temporal, reg_method, noise_est)
    return ar1


# --- Main template estimation function ---


def get_events_templates(
    data: NDArray,
    ix: NDArray,
    lb: NDArray,
    half_win: int = 100,
    method: str = "average",
    reg_temporal: float = 0,
    batch_size: int | None = None,
    opt_reg_temporal: bool = False,
    omega: float = 0.5,
    vec_mode: bool = False,
    vlen: int = 5,
    symm: bool = False,
    verbose: bool = False,
    reg_method: str | None = None,
    noise_est: str = "pairwise",
    block_starts: NDArray | None = None,
    *,
    ar1_phi: float | None = None,
    ar1_noise_var: float | None = None,
    noise_var_scale: float = 1.0,
) -> tuple[dict[object, NDArray], dict[str, object] | None]:
    """Estimate event templates from continuous data.

    Supports simple averaging (``method='average'``) and regularized
    least-squares (``method='ls'``) with optional SURE-optimized lambda
    or convolution kernel, GCV, L-curve, or Empirical Bayes selection.

    Parameters
    ----------
    data : NDArray
        Continuous data of shape ``(T, C)``.
    ix : NDArray
        1-D integer array of event-onset indices.
    lb : NDArray
        1-D array of event labels corresponding to *ix*.
    half_win : int
        Half-width of the template window (full window is ``2 * half_win``).
    method : str
        ``'average'`` for simple epoch averaging, ``'ls'`` for regularized
        least-squares.
    reg_temporal : float
        Fixed temporal regularization strength (used when *opt_reg_temporal*
        is *False* and *reg_method* is *None*).
    batch_size : int | None
        Number of samples per batch for design-matrix accumulation.  If
        *None*, processes the full signal in one pass.
    opt_reg_temporal : bool
        If *True*, optimize temporal regularization via SURE.
    omega : float
        SURE weighting parameter (0.5 gives Stein's unbiased estimate).
    vec_mode : bool
        If *True*, optimize the convolution kernel jointly via SURE.
    vlen : int
        Length of the convolution kernel seed.
    symm : bool
        If *True*, enforce a symmetric convolution kernel.
    verbose : bool
        If *True*, print optimization diagnostics.
    reg_method : str | None
        Lambda selection method (``'gcv'``, ``'lcurve'``, ``'empbayes'``).
        Takes precedence over *opt_reg_temporal*.
    noise_est : str
        Noise estimation method passed to :func:`est_noise_var`.
    block_starts : NDArray | None
        For separately recorded blocks concatenated in *data*: the integer
        sample index where each block begins, strictly increasing from 0; the
        last block ends at ``len(data)``. Every event window must lie inside
        one block. Colored-noise SURE then omits covariance between events of
        different blocks, and noise estimators that filter or pair samples
        across joins (``'hpf'`` and the HPF
        fallback of ``'pairwise'``) raise, as do colored estimates whose
        baseline segments are longer than the window. *None* treats *data* as
        one continuous recording.
    ar1_phi : float | None
        AR(1) coefficient for the colored covariance, in ``[0, 1)``, used
        as given (no 0.99 cap) instead of :func:`estimate_phi`.
    ar1_noise_var : float | None
        Marginal AR(1) noise variance (> 0) used instead of the baseline
        estimate.
    noise_var_scale : float
        Factor (> 0) on the baseline variance estimate; a value other than
        1 excludes *ar1_noise_var*. The three AR(1) inputs must be real
        scalars (bools and other types raise TypeError) and are converted to
        float before they are checked and used. Nondefault values
        apply only to AR(1)-colored SURE: ``method='ls'``,
        ``opt_reg_temporal=True``, ``reg_method=None`` and
        ``noise_est='ar1_colored'`` (scalar or kernel); other settings
        raise. AR(1) SURE records ``phi_estimated``, ``phi_used``,
        ``noise_var_estimated`` and ``noise_var_scale`` in *inner*, with
        ``noise_var`` the variance used. AR(3) SURE records the fitted
        autocovariance as ``ar3_acf``: through the largest event-pair lag,
        so up to the recording length (only lag 0 without baselines).

    Returns
    -------
    tuple[dict[object, NDArray], dict[str, object] | None]
        ``(templates, inner)`` where *templates* maps event labels to arrays
        of shape ``(2 * half_win, C)``, and *inner* holds optimization
        diagnostics (or *None* for simple averaging / fixed regularization).
    """

    ar1 = _ar1_inputs(
        _AR1Noise(ar1_phi, ar1_noise_var, noise_var_scale),
        method,
        opt_reg_temporal,
        reg_method,
        noise_est,
    )
    inner = None
    event_blocks = (
        None
        if block_starts is None
        else _event_blocks_from_starts(block_starts, ix, half_win, len(data))
    )

    if method == "average":
        X = epoch_data(data, ix, tmax=half_win, tmin=-half_win)
        return {str(label): X[lb == label].mean(0) for label in np.unique(lb)}, inner

    elif method == "ls":
        events = np.unique(lb)
        M = len(events)
        T = 2 * half_win
        C = data.shape[1]

        if batch_size is None:
            batch_size = len(data)

        kk = 0
        a = sp.sparse.csr_matrix((M * T, M * T))
        b = sp.sparse.csr_matrix((M * T, C))
        d = sp.sparse.csr_matrix((data.shape[0], M * T))

        while kk < len(data):
            sl = slice(kk, min(kk + batch_size, len(data)))
            sel = (ix >= sl.start) & (ix < sl.stop)

            design = get_design_matrix(
                data[sl], ix[sel] - kk, lb[sel], half_win=half_win, events=events
            )
            design = sp.sparse.csr_matrix(design)

            d += design
            a += design.T @ design
            b += design.T @ data[sl]

            kk += batch_size

        R = sp.linalg.convolution_matrix([1, -2, 1], T - 2)
        RR = R @ R.T

        temporal_mat = sp.sparse.block_diag([RR] * M, format="csr")
        # Exact factor: its null space is the structural constant/linear trend.
        temporal_factor = sp.linalg.block_diag(*([R.T] * M))

        path: _PenaltyPath | None = None
        if reg_method in {"gcv", "lcurve", "empbayes"}:
            if reg_method == "gcv":
                reg_temporal, inner = select_lambda_gcv(a, b, temporal_mat, data, d)
            elif reg_method == "lcurve":
                path = _PenaltyPath(_dense(a), _dense(b), temporal_factor)
                reg_temporal, inner = _select_lcurve(path, data, d)
            elif reg_method == "empbayes":
                reg_temporal, inner = select_lambda_empbayes(
                    a, b, temporal_mat, data, d
                )

        elif opt_reg_temporal:
            if vec_mode:
                reg_temporal, temporal_mat, inner = _optimize_kernel(
                    data=data,
                    a=a,
                    b=b,
                    ix=ix,
                    lb=lb,
                    events=events,
                    half_win=half_win,
                    M=M,
                    T=T,
                    noise_est=noise_est,
                    omega=omega,
                    vlen=vlen,
                    symm=symm,
                    verbose=verbose,
                    event_blocks=event_blocks,
                    ar1=ar1,
                )
            else:
                reg_temporal, inner, path = _optimize_lambda(
                    data=data,
                    a=a,
                    b=b,
                    temporal_factor=temporal_factor,
                    ix=ix,
                    lb=lb,
                    events=events,
                    half_win=half_win,
                    noise_est=noise_est,
                    omega=omega,
                    verbose=verbose,
                    event_blocks=event_blocks,
                    ar1=ar1,
                )

        if path is not None:
            templates = path.templates(reg_temporal).T
        else:
            templates = sp.sparse.linalg.spsolve(a + reg_temporal * temporal_mat, b).T

        templates = templates.reshape(data.shape[1], M, T).transpose(1, 2, 0)

        return {str(ev): templates[ii] for ii, ev in enumerate(events)}, inner
    else:
        raise ValueError(f"Unknown averaging method {method}")
