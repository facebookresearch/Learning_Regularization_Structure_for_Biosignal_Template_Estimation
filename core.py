# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

"""
Core template estimation methods.

Extracted from the research notebook:
"Optimizing L2-Norm Regularization with Application to Estimating Event Templates"
(Yonathan Morin, Oct. 2022)

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

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import scipy.sparse


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
    *sparse_out* flag.

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
    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    import scipy as sp

    if events is None:
        events = np.unique(lb)

    T = len(data)
    n_cols = len(events) * 2 * half_win

    if sparse_out or T * n_cols > 50_000_000:
        rows, cols = [], []
        for ii, ev in enumerate(events):
            sel = (lb == ev) & (ix < (T - half_win))
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
    for ii, ev in enumerate(events):
        sel = (lb == ev) & (ix < (T - half_win))
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


def calc_G(DtD: NDArray, regval: float, S: NDArray, D: NDArray) -> NDArray:
    """Compute the regularized gain matrix G = (D^T D + lam * S)^{-1} D^T.

    Parameters
    ----------
    DtD : NDArray
        Gram matrix ``D^T D`` of shape ``(p, p)``.
    regval : float
        Regularization parameter (lambda).
    S : NDArray
        Penalty matrix of shape ``(p, p)``.
    D : NDArray
        Design matrix of shape ``(n, p)``.

    Returns
    -------
    NDArray
        Gain matrix of shape ``(p, n)``.
    """
    return np.linalg.inv((DtD) + regval * S) @ D.T


def calc_Gx(G: NDArray, X: NDArray) -> NDArray:
    """Compute the regularized estimate G @ X.

    Parameters
    ----------
    G : NDArray
        Gain matrix of shape ``(p, n)``.
    X : NDArray
        Data matrix of shape ``(n, C)``.

    Returns
    -------
    NDArray
        Product of shape ``(p, C)``.
    """
    return G @ X


def calc_Gtilde(G: NDArray, Dpinv: NDArray) -> NDArray:
    """Compute G-tilde = G^T @ D^+.

    Parameters
    ----------
    G : NDArray
        Gain matrix of shape ``(p, n)``.
    Dpinv : NDArray
        Pseudoinverse of the design matrix of shape ``(p, n)``.

    Returns
    -------
    NDArray
        Product of shape ``(n, n)``.
    """
    return G.T @ Dpinv


def calc_trace_term(G: NDArray, Dpinv: NDArray, sigsq: float) -> float:
    """Compute the SURE trace term ``2 * sigma^2 * tr(G^T D^+)``.

    Parameters
    ----------
    G : NDArray
        Gain matrix of shape ``(p, n)``.
    Dpinv : NDArray
        Pseudoinverse of the design matrix of shape ``(p, n)``.
    sigsq : float
        Noise variance estimate.

    Returns
    -------
    float
        Scalar trace term.
    """
    return 2 * sigsq * np.multiply(G, Dpinv).sum()


def calc_normvec(
    Gx: NDArray,
    trace_term: float,
    omega: float,
    G: NDArray,
    sigsq: float,
) -> NDArray:
    """Compute the per-channel SURE norm vector.

    Parameters
    ----------
    Gx : NDArray
        Regularized estimate ``G @ X`` of shape ``(p, C)``.
    trace_term : float
        Precomputed trace term from :func:`calc_trace_term`.
    omega : float
        SURE weighting parameter (0.5 gives Stein's unbiased estimate).
    G : NDArray
        Gain matrix of shape ``(p, n)``.
    sigsq : float
        Noise variance estimate.

    Returns
    -------
    NDArray
        1-D array of length *C* with the per-channel norm contribution.
    """
    G_correction = 0
    if np.abs(omega - 0.5) > 1e-5:
        G_correction = (2 * omega - 1) * sigsq * np.multiply(G, G).sum()
    return omega * (np.linalg.norm(Gx, axis=0) ** 2 + trace_term) - G_correction


def calc_cost(
    norm_vec: NDArray,
    X: NDArray,
    G: NDArray,
    DpinvX: NDArray,
    omega: float,
) -> float:
    """Compute the total SURE cost from the norm vector and cross-term.

    Parameters
    ----------
    norm_vec : NDArray
        Per-channel norm vector of length *C*.
    X : NDArray
        Data matrix of shape ``(n, C)``.
    G : NDArray
        Gain matrix of shape ``(p, n)``.
    DpinvX : NDArray
        ``D^+ @ X`` of shape ``(p, C)``.
    omega : float
        SURE weighting parameter.

    Returns
    -------
    float
        Scalar SURE cost.
    """
    return np.sum(norm_vec - omega * 2 * np.diagonal(X.T @ G.T @ DpinvX))


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


def estimate_acf_ar(
    data: NDArray,
    ix: NDArray,
    half_win: int,
    ar_order: int = 3,
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

    Returns
    -------
    NDArray
        Autocorrelation function ``R(k)`` for ``k = 0, ..., 4 * half_win``.
    """
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

    # Pool autocorrelation estimates across segments and channels
    max_lag = 4 * half_win
    r = np.zeros(max_lag + 1)
    count = 0
    for seg in segments:
        for ch in range(seg.shape[1]):
            x = seg[:, ch] - seg[:, ch].mean()
            n = len(x)
            for k in range(min(n, max_lag + 1)):
                r[k] += np.sum(x[: n - k] * x[k:n])
            count += n

    if count == 0:
        return np.array([np.mean(np.var(data, axis=0))])

    r /= count

    # Fit AR(p) via Yule-Walker
    p = min(ar_order, len(r) - 1)
    if p < 1 or r[0] < 1e-30:
        acf = np.zeros(max_lag + 1)
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
        acf = np.zeros(max_lag + 1)
        acf[0] = r[0]
        return acf

    # Innovation variance
    sigma2_e = r[0] - np.dot(ar_coeffs, r[1 : p + 1])
    sigma2_e = max(sigma2_e, 1e-30)

    # Generate ACF from AR model: R(k) = sum_i phi_i R(k-i) for k > p
    acf = np.zeros(max_lag + 1)
    acf[: min(p + 1, max_lag + 1)] = r[: min(p + 1, max_lag + 1)]
    for k in range(p + 1, max_lag + 1):
        acf[k] = sum(ar_coeffs[i] * acf[k - 1 - i] for i in range(p))

    return acf


def build_DtSigmaD_acf(
    ix: NDArray,
    lb: NDArray,
    events: NDArray,
    half_win: int,
    acf: NDArray,
    data_length: int | None = None,
) -> NDArray:
    """Build D^T Sigma_W D from a general autocorrelation function.

    ``acf[k] = R(k)``, the noise autocovariance at lag *k*.

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
        Total length of the data array.  Used to exclude events too close to
        the boundary.

    Returns
    -------
    NDArray
        Symmetric matrix of shape ``(p, p)`` where ``p = len(events) * 2 * half_win``.
    """
    M = len(events)
    T = 2 * half_win
    p = M * T

    max_lag = len(acf) - 1
    offsets = np.arange(T)[:, None] - np.arange(T)[None, :]

    DtSigmaD = np.zeros((p, p))

    for ii, ev_i in enumerate(events):
        sel_i = lb == ev_i
        if data_length is not None:
            sel_i = sel_i & (ix < data_length - half_win)
        ix_i = ix[sel_i]
        for jj, ev_j in enumerate(events):
            sel_j = lb == ev_j
            if data_length is not None:
                sel_j = sel_j & (ix < data_length - half_win)
            ix_j = ix[sel_j]
            block = np.zeros((T, T))
            for ti in ix_i:
                for tj in ix_j:
                    dt_base = ti - tj
                    abs_lags = np.abs(dt_base + offsets)
                    valid = abs_lags <= max_lag
                    contrib = np.where(valid, acf[np.minimum(abs_lags, max_lag)], 0.0)
                    block += contrib
            DtSigmaD[ii * T : (ii + 1) * T, jj * T : (jj + 1) * T] = block

    return DtSigmaD


def build_DtSigmaD_ar1(
    ix: NDArray,
    lb: NDArray,
    events: NDArray,
    half_win: int,
    phi: float,
    sigsq: float,
    data_length: int | None = None,
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
        AR(1) coefficient.
    sigsq : float
        Noise variance (innovation variance scaled by AR process).
    data_length : int | None
        Total data length; used to exclude boundary events.

    Returns
    -------
    NDArray
        Symmetric matrix of shape ``(p, p)`` where ``p = len(events) * 2 * half_win``.
    """
    M = len(events)
    T = 2 * half_win
    p = M * T

    # Precompute phi^k lookup table
    max_lag = 4 * T
    phi_table = sigsq * np.power(abs(phi), np.arange(max_lag + 1))

    # Template offset grid: a - b for all (a, b) pairs
    offsets = np.arange(T)[:, None] - np.arange(T)[None, :]  # T x T

    DtSigmaD = np.zeros((p, p))

    for ii, ev_i in enumerate(events):
        sel_i = lb == ev_i
        if data_length is not None:
            sel_i = sel_i & (ix < data_length - half_win)
        ix_i = ix[sel_i]
        for jj, ev_j in enumerate(events):
            sel_j = lb == ev_j
            if data_length is not None:
                sel_j = sel_j & (ix < data_length - half_win)
            ix_j = ix[sel_j]
            block = np.zeros((T, T))
            for ti in ix_i:
                for tj in ix_j:
                    dt_base = ti - tj
                    abs_lags = np.abs(dt_base + offsets)
                    valid = abs_lags <= max_lag
                    contrib = np.where(
                        valid, phi_table[np.minimum(abs_lags, max_lag)], 0.0
                    )
                    block += contrib
            DtSigmaD[ii * T : (ii + 1) * T, jj * T : (jj + 1) * T] = block

    return DtSigmaD


def SURE_cost(
    regval: float,
    DtD: NDArray,
    D: NDArray,
    Dpinv: NDArray,
    DpinvX: NDArray,
    S: NDArray,
    theta_ml: NDArray,
    X: NDArray,
    sigsq: float,
    reg_l2: float | None,
    l2_mat: NDArray,
    omega: float = 0.5,
) -> float:
    """Compute the full (non-fast) SURE cost for a given regularization value.

    Parameters
    ----------
    regval : float
        Regularization parameter (lambda).
    DtD : NDArray
        Gram matrix ``D^T D`` of shape ``(p, p)``.
    D : NDArray
        Design matrix of shape ``(n, p)``.
    Dpinv : NDArray
        Pseudoinverse of the design matrix of shape ``(p, n)``.
    DpinvX : NDArray
        ``D^+ X`` of shape ``(p, C)``.
    S : NDArray
        Penalty matrix of shape ``(p, p)``.
    theta_ml : NDArray
        ML (OLS) estimate of shape ``(p, C)``.
    X : NDArray
        Data matrix of shape ``(n, C)``.
    sigsq : float
        Noise variance estimate.
    reg_l2 : float | None
        L2-mixing coefficient.  When *None*, pure Tikhonov is used.
    l2_mat : NDArray
        L2 penalty matrix of shape ``(p, p)``.
    omega : float
        SURE weighting parameter (default 0.5).

    Returns
    -------
    float
        Scalar SURE cost.
    """
    if reg_l2 is None:
        G = calc_G(DtD, regval, S, D)
    else:
        G = calc_G((1 - reg_l2) * (DtD) + reg_l2 * l2_mat, regval, S, D)

    Gx = calc_Gx(G, X)
    trace_term = calc_trace_term(G, Dpinv, sigsq)
    norm_vec = calc_normvec(Gx, trace_term, omega, G, sigsq)
    cost = calc_cost(norm_vec, X, G, DpinvX, omega)

    return cost


def gen_tempmat(vec: NDArray, TT: int, M: int) -> NDArray:
    """Generate the block-diagonal penalty matrix from a convolution kernel.

    Builds ``R R^T`` from the convolution matrix of *vec* and replicates it
    *M* times in a block-diagonal structure.

    Parameters
    ----------
    vec : NDArray
        1-D convolution kernel.
    TT : int
        Total number of template parameters (``M * T``).
    M : int
        Number of event types.

    Returns
    -------
    NDArray
        Block-diagonal penalty matrix of shape ``(TT, TT)``.
    """
    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    import scipy as sp

    L = len(vec)
    T = int(TT / M)
    R = sp.linalg.convolution_matrix(vec, T - (L - 1))
    RR = R @ R.T
    return sp.linalg.block_diag(*([RR] * M))


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


def SURE_cost_vec(
    vec: NDArray,
    DtD: NDArray,
    theta_ml: NDArray,
    X: NDArray,
    sigsq: float,
    M: int,
    symm: bool = False,
    omega: float = 0.5,
    _cost_scale: float | None = None,
    DtSigmaD: NDArray | None = None,
) -> float:
    """SURE cost as a function of the convolution-kernel vector.

    Builds the penalty matrix from *vec* via :func:`gen_tempmat` and evaluates
    the SURE cost at ``lambda = 1``.  Supports both white-noise and
    colored-noise variants.

    Parameters
    ----------
    vec : NDArray
        1-D convolution kernel parameters to optimise.
    DtD : NDArray
        Gram matrix ``D^T D`` of shape ``(p, p)``.
    theta_ml : NDArray
        ML (OLS) estimate of shape ``(p, C)``.
    X : NDArray
        Data matrix of shape ``(n, C)``.
    sigsq : float
        Noise variance estimate.
    M : int
        Number of event types.
    symm : bool
        Whether to symmetrise the kernel (default *False*).
    omega : float
        SURE weighting parameter (default 0.5).
    _cost_scale : float | None
        When provided, the cost is divided by this value (for numerical
        scaling during optimisation).
    DtSigmaD : NDArray | None
        ``D^T Sigma_W D`` for the colored-noise SURE variant.  When *None*,
        the white-noise variant is used.

    Returns
    -------
    float
        Scalar SURE cost (optionally rescaled).
    """
    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    import scipy as sp

    S = gen_tempmat(gen_corefilt(vec, symm), DtD.shape[0], M)

    DtD_arr = np.asarray(DtD)
    S_arr = np.asarray(S)
    theta_ml_arr = np.asarray(theta_ml)

    mu, V = sp.linalg.eigh(DtD_arr, DtD_arr + S_arr)
    mu = np.clip(mu, 0, 1)

    DtX = DtD_arr @ theta_ml_arr
    DtX_rot = V.T @ DtX

    n_obs = X.shape[0]
    n_cols = X.shape[1] if X.ndim > 1 else 1

    if DtSigmaD is not None:
        precomp = precompute_colored_trace(DtD_arr, DtSigmaD, V, mu)
        cost = SURE_cost_colored(
            1.0,
            mu,
            DtX_rot,
            V,
            theta_ml_arr,
            DtSigmaD,
            n_obs,
            n_cols,
            omega,
            _precomp=precomp,
        )
    else:
        cost = SURE_cost_fast(
            1.0, mu, DtX_rot, V, theta_ml_arr, sigsq, n_obs, n_cols, omega
        )

    if _cost_scale is not None:
        return cost / _cost_scale
    return cost


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
    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    import scipy.signal as sig

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
    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    import scipy.signal as sig

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


def est_noise_var_gcv(
    DtD: NDArray | scipy.sparse.spmatrix,
    DtX: NDArray | scipy.sparse.spmatrix,
    S: NDArray | scipy.sparse.spmatrix,
    X: NDArray,
    D: NDArray | scipy.sparse.spmatrix,
) -> float:
    """Estimate noise variance from residuals of a GCV-regularized fit.

    Uses corrected eigendecomposition (V is B-orthonormal, not orthonormal).

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

    Returns
    -------
    float
        Estimated noise variance.
    """
    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    import scipy as sp

    DtD_dense = np.asarray(
        DtD.todense() if sp.sparse.issparse(DtD) else DtD, dtype=float
    )
    S_dense = np.asarray(S.todense() if sp.sparse.issparse(S) else S, dtype=float)
    DtX_dense = np.asarray(
        DtX.todense() if sp.sparse.issparse(DtX) else DtX, dtype=float
    )

    n_obs = X.shape[0]
    n_cols = X.shape[1] if X.ndim > 1 else 1
    XtX_sum = np.sum(X**2)

    mu, V = sp.linalg.eigh(DtD_dense, DtD_dense + S_dense)
    mu = np.clip(mu, 0, 1)
    DtX_rot = V.T @ DtX_dense

    lambdas = np.logspace(-3, 5, 80)
    best_lam, best_score = lambdas[0], np.inf
    for lam in lambdas:
        g = 1.0 / (mu + lam * (1 - mu) + 1e-30)
        trace_H = np.sum(mu * g)
        denom = n_obs - trace_H
        if denom <= 0:
            continue
        Phi_hat = V @ (g[:, None] * DtX_rot)
        cross = np.sum(DtX_dense * Phi_hat)
        quad = np.sum(Phi_hat * (DtD_dense @ Phi_hat))
        residual_ss = XtX_sum - 2 * cross + quad
        score = residual_ss / (denom**2) * n_obs
        if score < best_score:
            best_score = score
            best_lam = lam

    g = 1.0 / (mu + best_lam * (1 - mu) + 1e-30)
    gamma = np.sum(mu * g)
    Phi_hat = V @ (g[:, None] * DtX_rot)
    cross = np.sum(DtX_dense * Phi_hat)
    quad = np.sum(Phi_hat * (DtD_dense @ Phi_hat))
    residual_ss = XtX_sum - 2 * cross + quad
    dof = n_obs * n_cols - gamma
    sigsq = residual_ss / max(dof, 1.0)

    return sigsq


def est_noise_var_mad(data: NDArray) -> float:
    """Estimate noise variance via MAD of finest Haar wavelet coefficients.

    Computes ``sigma = median(|d_finest|) / 0.6745`` where *d_finest* are the
    detail coefficients at the finest scale of a Haar wavelet decomposition.

    Parameters
    ----------
    data : NDArray
        Input signal of shape ``(T,)`` or ``(T, C)``.

    Returns
    -------
    float
        Scalar noise variance estimate averaged across channels.
    """
    n_samples = data.shape[0]
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Haar wavelet: finest detail coefficients = (x[2k] - x[2k+1]) / sqrt(2)
    n_pairs = n_samples // 2
    details = (data[: 2 * n_pairs : 2] - data[1 : 2 * n_pairs : 2]) / np.sqrt(2)

    mad_per_ch = np.median(np.abs(details), axis=0) / 0.6745
    return np.mean(mad_per_ch**2)


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


def est_noise_var_outofband(
    data_raw: NDArray,
    bandpass: tuple[float, float],
    fs: float,
    noise_band: tuple[float, float] | None = None,
) -> float:
    """Estimate noise variance from a frequency band with minimal signal content.

    Assumes the noise has a flat PSD (broadband).  Estimates the noise PSD from
    a band where the signal is negligible, then scales to the full working
    bandwidth.

    Parameters
    ----------
    data_raw : NDArray
        Data of shape ``(T, C)`` containing the noise band (raw or bandpassed).
    bandpass : tuple[float, float]
        Working signal band ``(low_hz, high_hz)``.
    fs : float
        Sampling frequency in Hz.
    noise_band : tuple[float, float] | None
        Frequency band ``(lo_hz, hi_hz)`` to estimate noise from.  Should be a
        region with minimal signal content.  If *None*, defaults to
        ``(high_hz * 1.2, Nyquist)``.

    Returns
    -------
    float
        Estimated in-band noise variance.
    """
    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    import scipy.signal as sig

    nyq = fs / 2.0
    low_hz, high_hz = bandpass

    if noise_band is None:
        noise_lo = high_hz * 1.2
        noise_hi = nyq * 0.95
        if noise_lo >= noise_hi:
            noise_lo = high_hz * 0.9
            noise_hi = min(high_hz, nyq * 0.95)
    else:
        noise_lo, noise_hi = noise_band

    if data_raw.ndim == 1:
        data_raw = data_raw.reshape(-1, 1)

    # Bandpass filter to isolate the noise estimation band
    bp_lo = noise_lo / nyq
    bp_hi = min(noise_hi / nyq, 0.99)

    sos = sig.butter(4, [bp_lo, bp_hi], btype="band", output="sos")
    filtered = sig.sosfiltfilt(sos, data_raw, axis=0)

    # Variance of filtered signal = noise PSD * noise_bandwidth
    var_filtered = np.mean(np.var(filtered, axis=0))

    # Noise PSD estimate (assuming flat)
    noise_bw = noise_hi - noise_lo
    sigma2_per_hz = var_filtered / noise_bw

    # Scale to the working signal band
    signal_bw = high_hz - low_hz
    sigma2_inband = sigma2_per_hz * signal_bw

    return sigma2_inband


def est_noise_var_ar1_corrected(
    data: NDArray,
    ix: NDArray,
    half_win: int,
    base_method: str = "baseline",
    lb: NDArray | None = None,
    **kwargs: object,
) -> float:
    """Compute AR(1)-corrected noise variance for temporally correlated noise.

    Estimates the lag-1 autocorrelation *phi* from pre-stimulus baselines, then
    scales the base noise estimate by ``(1 + phi) / (1 - phi)`` to account for
    the effective degrees of freedom with correlated noise.

    Parameters
    ----------
    data : NDArray
        Continuous data of shape ``(T,)`` or ``(T, C)``.
    ix : NDArray
        1-D integer array of event-onset indices.
    half_win : int
        Half-width of the template window.
    base_method : str
        Base noise estimation method (``'baseline'``, ``'pairwise'``,
        ``'mad'``, or ``'hpf'``).
    lb : NDArray | None
        Event labels, required when *base_method* is ``'pairwise'``.
    **kwargs : object
        Additional keyword arguments (unused, kept for API compatibility).

    Returns
    -------
    float
        Corrected noise variance estimate.
    """
    if base_method == "baseline":
        sigma2_base = est_noise_var_baseline(data, ix, half_win)
    elif base_method == "pairwise":
        if lb is None:
            raise ValueError("pairwise noise estimation requires lb")
        sigma2_base = est_noise_var_pairwise(data, ix, lb, half_win)
    elif base_method == "mad":
        sigma2_base = est_noise_var_mad(data)
    else:
        sigma2_base = est_noise_var_hpf(data)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_baseline = max(1, int(2 * half_win * 0.3))
    phi_estimates = []
    for idx in ix:
        start = idx - half_win
        end = start + n_baseline
        if start < 0 or end > len(data):
            continue
        segment = data[start:end]
        for ch in range(segment.shape[1]):
            x = segment[:, ch]
            if len(x) < 4:
                continue
            x = x - x.mean()
            r0 = np.mean(x**2)
            if r0 < 1e-30:
                continue
            r1 = np.mean(x[:-1] * x[1:])
            phi_estimates.append(r1 / r0)

    if not phi_estimates:
        return sigma2_base

    phi = np.median(phi_estimates)
    phi = np.clip(phi, -0.99, 0.99)

    correction = (1 + phi) / (1 - phi)
    correction = max(correction, 1.0)

    return sigma2_base * correction


def _est_noise_ar1(
    data: NDArray,
    ix: NDArray | None,
    lb: NDArray | None,
    half_win: int | None,
    **_kwargs: object,
) -> float:
    if ix is None:
        raise ValueError("AR1 noise estimation requires ix (got None)")
    if half_win is None:
        raise ValueError("AR1 noise estimation requires half_win (got None)")
    return est_noise_var_ar1_corrected(data, ix, half_win, lb=lb)


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


def _est_noise_outofband(
    data: NDArray,
    data_raw: NDArray | None,
    bandpass: tuple[float, float] | None,
    fs: float | None,
    noise_band: tuple[float, float] | None,
    **_kwargs: object,
) -> float:
    if data_raw is None:
        raise ValueError("Out-of-band noise estimation requires data_raw (got None)")
    if bandpass is None:
        raise ValueError("Out-of-band noise estimation requires bandpass (got None)")
    if fs is None:
        raise ValueError("Out-of-band noise estimation requires fs (got None)")
    return est_noise_var_outofband(data_raw, bandpass, fs, noise_band=noise_band)


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


def _est_noise_gcv(
    data: NDArray,
    DtD: NDArray | scipy.sparse.spmatrix | None,
    DtX: NDArray | scipy.sparse.spmatrix | None,
    S: NDArray | scipy.sparse.spmatrix | None,
    X: NDArray | None,
    D: NDArray | scipy.sparse.spmatrix | None,
    **_kwargs: object,
) -> float:
    if DtD is None:
        raise ValueError("GCV noise estimation requires DtD (got None)")
    if DtX is None:
        raise ValueError("GCV noise estimation requires DtX (got None)")
    if S is None:
        raise ValueError("GCV noise estimation requires S (got None)")
    if X is None:
        raise ValueError("GCV noise estimation requires X (got None)")
    if D is None:
        raise ValueError("GCV noise estimation requires D (got None)")
    return est_noise_var_gcv(DtD, DtX, S, X, D)


def _est_noise_empbayes_residual(
    data: NDArray,
    DtD: NDArray | scipy.sparse.spmatrix | None,
    DtX: NDArray | scipy.sparse.spmatrix | None,
    S: NDArray | scipy.sparse.spmatrix | None,
    X: NDArray | None,
    D: NDArray | scipy.sparse.spmatrix | None,
    ix: NDArray | None,
    lb: NDArray | None,
    half_win: int | None,
    **_kwargs: object,
) -> float:
    if DtD is None:
        raise ValueError("empbayes_residual requires DtD (got None)")
    if DtX is None:
        raise ValueError("empbayes_residual requires DtX (got None)")
    if S is None:
        raise ValueError("empbayes_residual requires S (got None)")
    if X is None:
        raise ValueError("empbayes_residual requires X (got None)")
    if D is None:
        raise ValueError("empbayes_residual requires D (got None)")
    return est_noise_var_empbayes_residual(
        DtD, DtX, S, X, D, ix=ix, lb=lb, half_win=half_win
    )


def _est_noise_mad(data: NDArray, **_kwargs: object) -> float:
    return est_noise_var_mad(data)


def _est_noise_hpf(data: NDArray, **_kwargs: object) -> float:
    return est_noise_var_hpf(data)


_NOISE_ESTIMATORS: dict[str, object] = {
    "ar1": _est_noise_ar1,
    "ar1_colored": _est_noise_colored,
    "ar3_colored": _est_noise_colored,
    "outofband": _est_noise_outofband,
    "pairwise": _est_noise_pairwise,
    "baseline": _est_noise_baseline,
    "gcv": _est_noise_gcv,
    "empbayes_residual": _est_noise_empbayes_residual,
    "mad": _est_noise_mad,
    "hpf": _est_noise_hpf,
}


def est_noise_var(
    data: NDArray,
    method: str = "pairwise",
    DtD: NDArray | scipy.sparse.spmatrix | None = None,
    DtX: NDArray | scipy.sparse.spmatrix | None = None,
    S: NDArray | scipy.sparse.spmatrix | None = None,
    X: NDArray | None = None,
    D: NDArray | scipy.sparse.spmatrix | None = None,
    ix: NDArray | None = None,
    lb: NDArray | None = None,
    half_win: int | None = None,
    data_raw: NDArray | None = None,
    bandpass: tuple[float, float] | None = None,
    fs: float | None = None,
    noise_band: tuple[float, float] | None = None,
) -> float:
    """Estimate noise variance with a configurable method.

    Parameters
    ----------
    data : NDArray
        Continuous data of shape ``(T, C)``.
    method : str
        Noise estimation method.  One of ``'outofband'``, ``'pairwise'``,
        ``'baseline'``, ``'ar1'``, ``'ar1_colored'``, ``'ar3_colored'``,
        ``'gcv'``, ``'empbayes_residual'``, ``'hpf'``, or ``'mad'``.
    DtD : NDArray | scipy.sparse.spmatrix | None
        Gram matrix ``D^T D``.  Required for ``'gcv'`` and
        ``'empbayes_residual'``.
    DtX : NDArray | scipy.sparse.spmatrix | None
        Cross-product ``D^T X``.  Required for ``'gcv'`` and
        ``'empbayes_residual'``.
    S : NDArray | scipy.sparse.spmatrix | None
        Penalty matrix.  Required for ``'gcv'`` and ``'empbayes_residual'``.
    X : NDArray | None
        Data matrix.  Required for ``'gcv'`` and ``'empbayes_residual'``.
    D : NDArray | scipy.sparse.spmatrix | None
        Design matrix.  Required for ``'gcv'`` and ``'empbayes_residual'``.
    ix : NDArray | None
        Event-onset indices.  Required for ``'pairwise'``, ``'baseline'``,
        ``'ar1'``, and colored variants.
    lb : NDArray | None
        Event labels.  Required for ``'pairwise'``.
    half_win : int | None
        Half-width of the template window.  Required for ``'pairwise'``,
        ``'baseline'``, ``'ar1'``, and colored variants.
    data_raw : NDArray | None
        Raw data for out-of-band estimation.
    bandpass : tuple[float, float] | None
        Working signal band ``(low_hz, high_hz)`` for ``'outofband'``.
    fs : float | None
        Sampling frequency in Hz for ``'outofband'``.
    noise_band : tuple[float, float] | None
        Noise estimation band for ``'outofband'``.

    Returns
    -------
    float
        Estimated noise variance.
    """
    estimator = _NOISE_ESTIMATORS.get(method)
    if estimator is None:
        raise ValueError(f"Unknown noise estimation method: {method}")
    return estimator(
        data=data,
        method=method,
        DtD=DtD,
        DtX=DtX,
        S=S,
        X=X,
        D=D,
        ix=ix,
        lb=lb,
        half_win=half_win,
        data_raw=data_raw,
        bandpass=bandpass,
        fs=fs,
        noise_band=noise_band,
    )


def est_noise_var_empbayes_residual(
    DtD: NDArray | scipy.sparse.spmatrix,
    DtX: NDArray | scipy.sparse.spmatrix,
    S: NDArray | scipy.sparse.spmatrix,
    X: NDArray,
    D: NDArray | scipy.sparse.spmatrix,
    ix: NDArray | None = None,
    lb: NDArray | None = None,
    half_win: int | None = None,
) -> float:
    """Estimate effective noise variance via EB-regularized epoch residuals.

    Two-stage approach:

    1. Run MacKay's EB to get optimal lambda (no sigma-squared needed).
    2. Compute epoch-level residuals and estimate sigma-squared from those.

    Using epoch-level residuals (not full-signal) avoids dilution by the
    vast majority of samples where no events occur.

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
    ix : NDArray | None
        Event-onset indices for epoch-level residual computation.
    lb : NDArray | None
        Event labels corresponding to *ix*.
    half_win : int | None
        Half-width of the template window.

    Returns
    -------
    float
        Estimated noise variance.
    """
    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    import scipy as sp

    DtD_dense = np.asarray(
        DtD.todense() if sp.sparse.issparse(DtD) else DtD, dtype=float
    )
    S_dense = np.asarray(S.todense() if sp.sparse.issparse(S) else S, dtype=float)
    DtX_dense = np.asarray(
        DtX.todense() if sp.sparse.issparse(DtX) else DtX, dtype=float
    )

    n_cols = X.shape[1] if X.ndim > 1 else 1

    mu, V = sp.linalg.eigh(DtD_dense, DtD_dense + S_dense)
    mu = np.clip(mu, 0, 1)
    DtX_rot = V.T @ DtX_dense
    XtX_sum = np.sum(X**2)
    n_obs = X.shape[0]

    lam = 1.0
    for _ in range(30):
        g = 1.0 / (mu + lam * (1 - mu) + 1e-30)
        gamma = np.sum(mu * g)
        Phi_hat = V @ (g[:, None] * DtX_rot)
        phi_norm_sq = np.sum(Phi_hat * (S_dense @ Phi_hat))
        cross = np.sum(DtX_dense * Phi_hat)
        quad = np.sum(Phi_hat * (DtD_dense @ Phi_hat))
        residual_ss = XtX_sum - 2 * cross + quad
        alpha = gamma / (phi_norm_sq / n_cols + 1e-30)
        beta = (n_obs * n_cols - gamma) / (residual_ss + 1e-30)
        lam_new = alpha / (beta + 1e-30)
        if abs(lam_new - lam) / (lam + 1e-30) < 1e-4:
            lam = lam_new
            break
        lam = lam_new

    templates_flat = sp.sparse.linalg.spsolve(DtD + lam * S, DtX)
    if templates_flat.ndim == 1:
        templates_flat = templates_flat.reshape(-1, 1)

    if ix is not None and lb is not None and half_win is not None:
        events = np.unique(lb)
        M = len(events)
        T = 2 * half_win
        templates = templates_flat.T.reshape(n_cols, M, T).transpose(1, 2, 0)

        epoch_residuals = []
        for _i, (idx, label) in enumerate(zip(ix, lb)):
            start = idx - half_win
            end = idx + half_win
            if start < 0 or end > len(X):
                continue
            epoch = X[start:end]
            ev_idx = list(events).index(label)
            predicted = templates[ev_idx]
            residual = epoch - predicted
            epoch_residuals.append(residual)

        if epoch_residuals:
            all_residuals = np.concatenate(epoch_residuals, axis=0)
            return np.mean(np.var(all_residuals, axis=0))

    sigsq = residual_ss / max(n_obs * n_cols - gamma, 1.0)
    return sigsq


# --- GCV, L-curve, Empirical Bayes lambda selection ---
# All operate on the SVD of the regularized normal equations.
# Given D^T D and Gamma^T Gamma, we compute a joint eigendecomposition
# so that the regularized solution and all criteria can be evaluated
# efficiently for any lambda.


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
    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    import scipy as sp

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


def select_lambda_lcurve(
    DtD: NDArray | scipy.sparse.spmatrix,
    DtX: NDArray | scipy.sparse.spmatrix,
    S: NDArray | scipy.sparse.spmatrix,
    X: NDArray,
    D: NDArray | scipy.sparse.spmatrix,
    n_lambdas: int = 80,
) -> tuple[float, dict[str, object]]:
    """Select regularization lambda via L-curve (maximum curvature).

    Uses generalized eigendecomposition with correct V norms.

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
    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    import scipy as sp

    DtD_dense = np.asarray(
        DtD.todense() if sp.sparse.issparse(DtD) else DtD, dtype=float
    )
    S_dense = np.asarray(S.todense() if sp.sparse.issparse(S) else S, dtype=float)
    DtX_dense = np.asarray(
        DtX.todense() if sp.sparse.issparse(DtX) else DtX, dtype=float
    )

    XtX_sum = np.sum(X**2)

    mu, V = sp.linalg.eigh(DtD_dense, DtD_dense + S_dense)
    mu = np.clip(mu, 0, 1)
    DtX_rot = V.T @ DtX_dense

    lambdas = np.logspace(-3, 5, n_lambdas)
    residual_ss = np.zeros(n_lambdas)
    solution_ss = np.zeros(n_lambdas)

    for i, lam in enumerate(lambdas):
        g = 1.0 / (mu + lam * (1 - mu) + 1e-30)
        Phi_hat = V @ (g[:, None] * DtX_rot)
        # ||X - D Phi||^2
        cross = np.sum(DtX_dense * Phi_hat)
        quad = np.sum(Phi_hat * (DtD_dense @ Phi_hat))
        residual_ss[i] = XtX_sum - 2 * cross + quad
        # ||Gamma Phi||^2 = Phi^T S Phi
        solution_ss[i] = np.sum(Phi_hat * (S_dense @ Phi_hat))

    residual_ss = np.maximum(residual_ss, 1e-30)
    solution_ss = np.maximum(solution_ss, 1e-30)

    xi = np.log(residual_ss)
    eta = np.log(solution_ss)
    log_lam = np.log(lambdas)

    dxi = np.gradient(xi, log_lam)
    deta = np.gradient(eta, log_lam)
    d2xi = np.gradient(dxi, log_lam)
    d2eta = np.gradient(deta, log_lam)

    curvature = (dxi * d2eta - deta * d2xi) / (dxi**2 + deta**2) ** 1.5
    curvature[:5] = -np.inf
    curvature[-5:] = -np.inf

    best_idx = np.argmax(curvature)
    return lambdas[best_idx], {"opt_val": lambdas[best_idx], "method": "lcurve"}


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

    Uses MacKay's iterative update equations with correct V norms.
    V from ``eigh(DtD, DtD+S)`` is B-orthonormal, **not** orthonormal.

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
        Converged lambda and a diagnostics dict with keys ``'opt_val'``,
        ``'gamma'``, ``'n_iter'``, and ``'method'``.
    """
    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    import scipy as sp

    DtD_dense = np.asarray(
        DtD.todense() if sp.sparse.issparse(DtD) else DtD, dtype=float
    )
    S_dense = np.asarray(S.todense() if sp.sparse.issparse(S) else S, dtype=float)
    DtX_dense = np.asarray(
        DtX.todense() if sp.sparse.issparse(DtX) else DtX, dtype=float
    )

    n_obs = X.shape[0]
    n_cols = X.shape[1] if X.ndim > 1 else 1
    XtX_sum = np.sum(X**2)

    mu, V = sp.linalg.eigh(DtD_dense, DtD_dense + S_dense)
    mu = np.clip(mu, 0, 1)
    DtX_rot = V.T @ DtX_dense

    lam = 1.0
    n_iters_run = 0

    for _iteration in range(max_iter):
        n_iters_run = _iteration + 1
        g = 1.0 / (mu + lam * (1 - mu) + 1e-30)

        # Effective number of parameters: tr(DtD (DtD+lam*S)^{-1}) = sum(mu_i * g_i)
        gamma = np.sum(mu * g)

        # Phi_hat in original basis
        Phi_hat = V @ (g[:, None] * DtX_rot)

        # ||Gamma Phi||^2 = Phi^T S Phi
        phi_norm_sq = np.sum(Phi_hat * (S_dense @ Phi_hat))

        # ||X - D Phi||^2
        cross = np.sum(DtX_dense * Phi_hat)
        quad = np.sum(Phi_hat * (DtD_dense @ Phi_hat))
        residual_ss = XtX_sum - 2 * cross + quad

        # MacKay updates
        alpha = gamma / (phi_norm_sq / n_cols + 1e-30)
        beta = (n_obs * n_cols - gamma) / (residual_ss + 1e-30)

        lam_new = alpha / (beta + 1e-30)

        if abs(lam_new - lam) / (lam + 1e-30) < tol:
            lam = lam_new
            break
        lam = lam_new

    return lam, {
        "opt_val": lam,
        "gamma": gamma,
        "n_iter": n_iters_run,
        "method": "empbayes",
    }


# --- xDAWN spatial filtering ---


def xdawn_filter(
    data: NDArray,
    ix: NDArray,
    lb: NDArray,
    half_win: int,
    n_components: int | None = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute xDAWN spatial filters (Rivet et al., 2009).

    Finds spatial filters W that maximize the signal-to-signal-plus-noise
    ratio of the evoked response via the generalized eigenvalue problem
    ``Sigma_signal w = lambda Sigma_x w``.

    Parameters
    ----------
    data : NDArray
        Continuous data of shape ``(T, C)``.
    ix : NDArray
        1-D integer array of event-onset indices of shape ``(N,)``.
    lb : NDArray
        1-D array of event labels of shape ``(N,)``.
    half_win : int
        Half-width of the template window.
    n_components : int | None
        Number of spatial components to keep.  If *None*, all *C* components
        are retained.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        ``(data_filtered, W, A)`` where *data_filtered* has shape
        ``(T, n_components)``, *W* has shape ``(C, n_components)`` (spatial
        filters), and *A* has shape ``(n_components, C)`` (spatial patterns
        for interpretation).
    """
    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    import scipy as sp

    events = np.unique(lb)
    C = data.shape[1]

    if n_components is None:
        n_components = C

    D = get_design_matrix(data, ix, lb, half_win=half_win, events=events)

    # Signal covariance estimate: D (D^T D)^{-1} D^T approximates the signal subspace
    # For xDAWN: maximize w^T Sigma_signal w / w^T Sigma_x w
    XtX = data.T @ data
    DtD = D.T @ D

    # Evoked signal estimate (OLS)
    Phi_ols = np.linalg.solve(DtD + 1e-6 * np.eye(DtD.shape[0]), D.T @ data)
    signal_estimate = D @ Phi_ols
    Sigma_signal = signal_estimate.T @ signal_estimate

    # Solve generalized eigenvalue problem
    eigvals, eigvecs = sp.linalg.eigh(Sigma_signal, XtX + 1e-6 * np.eye(C))

    # Sort by descending eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    W = eigvecs[:, idx[:n_components]]

    # Normalize filters
    for j in range(n_components):
        W[:, j] /= np.linalg.norm(W[:, j])

    # Spatial patterns (for interpretation): A = Sigma_x W (W^T Sigma_x W)^{-1}
    A = XtX @ W @ np.linalg.inv(W.T @ XtX @ W)

    data_filtered = data @ W
    return data_filtered, W, A


def get_events_templates_xdawn(
    data: NDArray,
    ix: NDArray,
    lb: NDArray,
    half_win: int = 100,
    n_components: int | None = None,
    reg_temporal: float = 0,
    opt_reg_temporal: bool = False,
    omega: float = 0.5,
    vec_mode: bool = False,
    vlen: int = 3,
    symm: bool = True,
    reg_method: str | None = None,
) -> tuple[dict[object, NDArray], dict[str, object] | None]:
    """Estimate templates with xDAWN spatial filtering.

    First applies xDAWN to reduce dimensionality and enhance evoked SNR,
    then estimates templates in the filtered space.

    Parameters
    ----------
    data : NDArray
        Continuous data of shape ``(T, C)``.
    ix : NDArray
        1-D integer array of event-onset indices.
    lb : NDArray
        1-D array of event labels.
    half_win : int
        Half-width of the template window.
    n_components : int | None
        Number of xDAWN spatial components to keep.
    reg_temporal : float
        Temporal regularization strength.
    opt_reg_temporal : bool
        If *True*, optimize regularization via SURE.
    omega : float
        SURE weighting parameter.
    vec_mode : bool
        If *True*, optimize the convolution kernel via SURE.
    vlen : int
        Length of the convolution kernel seed.
    symm : bool
        If *True*, enforce symmetric kernel.
    reg_method : str | None
        Lambda selection method (``'gcv'``, ``'lcurve'``, ``'empbayes'``).

    Returns
    -------
    tuple[dict[object, NDArray], dict[str, object] | None]
        ``(templates, inner)`` where *templates* maps event labels to arrays
        of shape ``(2 * half_win, C)`` in the original channel space, and
        *inner* holds optimization diagnostics or *None*.
    """
    data_f, W, A = xdawn_filter(data, ix, lb, half_win, n_components=n_components)

    templates_f, inner = get_events_templates(
        data_f,
        ix,
        lb,
        half_win=half_win,
        method="ls",
        reg_temporal=reg_temporal,
        opt_reg_temporal=opt_reg_temporal,
        omega=omega,
        vec_mode=vec_mode,
        vlen=vlen,
        symm=symm,
        reg_method=reg_method,
    )

    # Project templates back to original channel space for SNR comparison
    templates = {k: v @ W.T for k, v in templates_f.items()}
    return templates, inner


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


def _optimize_kernel(
    data: NDArray,
    a: scipy.sparse.spmatrix,
    b: scipy.sparse.spmatrix,
    d: scipy.sparse.spmatrix,
    temporal_mat: scipy.sparse.spmatrix,
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
    data_raw: NDArray | None,
    bandpass: tuple[float, float] | None,
    fs: float | None,
    noise_band: tuple[float, float] | None,
) -> tuple[float, scipy.sparse.spmatrix, dict[str, object]]:
    """Optimize the convolution kernel via SURE (vec_mode).

    Returns the regularization strength (always 1.0), the updated
    temporal penalty matrix, and a diagnostics dict.
    """
    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    import scipy as sp

    theta_ml = np.linalg.inv(a.todense()) @ b
    sigsq = est_noise_var(
        data,
        method=noise_est
        if noise_est not in {"ar1_colored", "ar3_colored"}
        else "baseline",
        DtD=a,
        DtX=b,
        S=temporal_mat,
        X=data,
        D=d,
        ix=ix,
        lb=lb,
        half_win=half_win,
        data_raw=data_raw,
        bandpass=bandpass,
        fs=fs,
        noise_band=noise_band,
    )

    DtSigmaD_vec = None
    if noise_est == "ar1_colored":
        phi = estimate_phi(data, ix, half_win)
        DtSigmaD_vec = build_DtSigmaD_ar1(
            ix, lb, events, half_win, phi, sigsq, data_length=len(data)
        )
        if verbose:
            print(f"phi: {phi:.3f}, correction vs white: {(1 + phi) / (1 - phi):.1f}x")
    elif noise_est == "ar3_colored":
        ar_order = 3
        acf = estimate_acf_ar(data, ix, half_win, ar_order=ar_order)
        DtSigmaD_vec = build_DtSigmaD_acf(
            ix, lb, events, half_win, acf, data_length=len(data)
        )
        if verbose:
            print(f"AR({ar_order}) ACF[0:5]: {acf[:5]}")

    x0 = _gen_h_diff(vlen)
    x0 = x0[: int(vlen - ((vlen - 1) / 2) * int(symm))]

    cost_at_x0 = SURE_cost_vec(
        x0,
        a.todense(),
        theta_ml,
        data,
        sigsq,
        M,
        symm,
        omega,
        DtSigmaD=DtSigmaD_vec,
    )
    cost_scale = max(abs(cost_at_x0), 1e-30)

    opt_res = sp.optimize.minimize(
        SURE_cost_vec,
        x0=x0,
        args=(
            a.todense(),
            theta_ml,
            data,
            sigsq,
            M,
            symm,
            omega,
            cost_scale,
            DtSigmaD_vec,
        ),
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

    inner: dict[str, object] = {
        "opt_val": opt_vec,
        "noise_var": sigsq,
        "cost": opt_res.fun,
    }
    return reg_temporal, temporal_mat, inner


def _optimize_lambda(
    data: NDArray,
    a: scipy.sparse.spmatrix,
    b: scipy.sparse.spmatrix,
    d: scipy.sparse.spmatrix,
    temporal_mat: scipy.sparse.spmatrix,
    l2_mat: scipy.sparse.spmatrix,
    ix: NDArray,
    lb: NDArray,
    events: NDArray,
    half_win: int,
    reg_l2: float,
    noise_est: str,
    omega: float,
    verbose: bool,
    data_raw: NDArray | None,
    bandpass: tuple[float, float] | None,
    fs: float | None,
    noise_band: tuple[float, float] | None,
) -> tuple[float, dict[str, object]]:
    """Optimize the scalar lambda via SURE.

    Returns the optimal regularization strength and a diagnostics dict.
    """
    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    import scipy as sp

    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    from scipy.optimize import minimize_scalar

    sigsq = est_noise_var(
        data,
        method=noise_est
        if noise_est not in {"ar1_colored", "ar3_colored"}
        else "baseline",
        DtD=a,
        DtX=b,
        S=temporal_mat,
        X=data,
        D=d,
        ix=ix,
        lb=lb,
        half_win=half_win,
        data_raw=data_raw,
        bandpass=bandpass,
        fs=fs,
        noise_band=noise_band,
    )

    # Use fast eigendecomposition-based SURE when no L2 mixing
    if reg_l2 == 0:
        DtD_dense = np.asarray(a.todense() if sp.sparse.issparse(a) else a, dtype=float)
        S_dense = np.asarray(
            temporal_mat.todense()
            if sp.sparse.issparse(temporal_mat)
            else temporal_mat,
            dtype=float,
        )
        DtX_dense = np.asarray(b.todense() if sp.sparse.issparse(b) else b, dtype=float)

        mu, V = sp.linalg.eigh(DtD_dense, DtD_dense + S_dense)
        mu = np.clip(mu, 0, 1)
        DtX_rot = V.T @ DtX_dense
        theta_ml_dense = np.linalg.solve(DtD_dense, DtX_dense)
        n_obs = data.shape[0]
        n_cols = data.shape[1] if data.ndim > 1 else 1

        if noise_est in {"ar1_colored", "ar3_colored"}:
            if noise_est == "ar1_colored":
                phi = estimate_phi(data, ix, half_win)
                DtSigmaD = build_DtSigmaD_ar1(
                    ix,
                    lb,
                    events,
                    half_win,
                    phi,
                    sigsq,
                    data_length=len(data),
                )
            else:
                acf = estimate_acf_ar(data, ix, half_win, ar_order=3)
                DtSigmaD = build_DtSigmaD_acf(
                    ix, lb, events, half_win, acf, data_length=len(data)
                )
            precomp = precompute_colored_trace(DtD_dense, DtSigmaD, V, mu)
            opt_res = minimize_scalar(
                SURE_cost_colored,
                bounds=(0, 1e5),
                method="bounded",
                args=(
                    mu,
                    DtX_rot,
                    V,
                    theta_ml_dense,
                    DtSigmaD,
                    n_obs,
                    n_cols,
                    omega,
                    None,
                    None,
                    precomp,
                ),
                options={"xatol": 1e-02, "maxiter": 25, "disp": 0},
            )
        else:
            opt_res = minimize_scalar(
                SURE_cost_fast,
                bounds=(0, 1e5),
                method="bounded",
                args=(
                    mu,
                    DtX_rot,
                    V,
                    theta_ml_dense,
                    sigsq,
                    n_obs,
                    n_cols,
                    omega,
                ),
                options={"xatol": 1e-02, "maxiter": 25, "disp": 0},
            )
    else:
        # Fallback to original SURE with L2 mixing
        theta_ml = np.linalg.inv(a.todense()) @ b
        Dpinv = np.linalg.inv(a.todense()) @ d.todense().T
        DpinvX = Dpinv @ data
        opt_res = minimize_scalar(
            SURE_cost,
            bounds=(0, 1e5),
            method="bounded",
            args=(
                a.todense(),
                d.todense(),
                Dpinv,
                DpinvX,
                temporal_mat.todense(),
                theta_ml,
                data,
                sigsq,
                reg_l2,
                l2_mat.todense(),
                omega,
            ),
            options={"xatol": 1e-02, "maxiter": 25, "disp": 0},
        )

    reg_temporal = opt_res.x

    if verbose:
        print(f"tmp_reg: {reg_temporal}")
        print(f"noise var: {sigsq}")

    inner: dict[str, object] = {
        "opt_val": reg_temporal,
        "noise_var": sigsq,
        "cost": opt_res.fun,
    }
    return reg_temporal, inner


# --- Main template estimation function ---


def get_events_templates(
    data: NDArray,
    ix: NDArray,
    lb: NDArray,
    half_win: int = 100,
    method: str = "average",
    reg_temporal: float = 0,
    reg_l2: float = 0,
    batch_size: int | None = None,
    opt_reg_temporal: bool = False,
    omega: float = 0.5,
    vec_mode: bool = False,
    vlen: int = 5,
    symm: bool = False,
    verbose: bool = False,
    reg_method: str | None = None,
    noise_est: str = "pairwise",
    data_raw: NDArray | None = None,
    bandpass: tuple[float, float] | None = None,
    fs: float | None = None,
    noise_band: tuple[float, float] | None = None,
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
    reg_l2 : float
        L2 (count-weighted identity) regularization mixing weight in [0, 1].
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
    data_raw : NDArray | None
        Raw data for out-of-band noise estimation.
    bandpass : tuple[float, float] | None
        Working signal band for out-of-band noise estimation.
    fs : float | None
        Sampling frequency for out-of-band noise estimation.
    noise_band : tuple[float, float] | None
        Noise estimation band for out-of-band estimation.

    Returns
    -------
    tuple[dict[object, NDArray], dict[str, object] | None]
        ``(templates, inner)`` where *templates* maps event labels to arrays
        of shape ``(2 * half_win, C)``, and *inner* holds optimization
        diagnostics (or *None* for simple averaging / fixed regularization).
    """
    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    import scipy as sp

    inner = None

    if method == "average":
        X = epoch_data(data, ix, tmax=half_win, tmin=-half_win)
        return {str(label): X[lb == label].mean(0) for label in np.unique(lb)}, inner

    elif method == "ls":
        assert (reg_l2 >= 0) & (reg_l2 <= 1)

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

        l2_mat = sp.sparse.block_diag(
            [np.eye(T) * c for _, c in zip(*np.unique(lb, return_counts=True))],
            format="csr",
        )
        temporal_mat = sp.sparse.block_diag([RR] * M, format="csr")

        if reg_method in {"gcv", "lcurve", "empbayes"}:
            if reg_method == "gcv":
                reg_temporal, inner = select_lambda_gcv(a, b, temporal_mat, data, d)
            elif reg_method == "lcurve":
                reg_temporal, inner = select_lambda_lcurve(a, b, temporal_mat, data, d)
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
                    d=d,
                    temporal_mat=temporal_mat,
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
                    data_raw=data_raw,
                    bandpass=bandpass,
                    fs=fs,
                    noise_band=noise_band,
                )
            else:
                reg_temporal, inner = _optimize_lambda(
                    data=data,
                    a=a,
                    b=b,
                    d=d,
                    temporal_mat=temporal_mat,
                    l2_mat=l2_mat,
                    ix=ix,
                    lb=lb,
                    events=events,
                    half_win=half_win,
                    reg_l2=reg_l2,
                    noise_est=noise_est,
                    omega=omega,
                    verbose=verbose,
                    data_raw=data_raw,
                    bandpass=bandpass,
                    fs=fs,
                    noise_band=noise_band,
                )

        templates = sp.sparse.linalg.spsolve(
            (1 - reg_l2) * (a) + reg_l2 * l2_mat + reg_temporal * temporal_mat, b
        ).T

        templates = templates.reshape(data.shape[1], M, T).transpose(1, 2, 0)

        return {str(ev): templates[ii] for ii, ev in enumerate(events)}, inner
    else:
        raise ValueError(f"Unknown averaging method {method}")
