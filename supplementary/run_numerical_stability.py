# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import concurrent.futures
import math
import multiprocessing
import os
import sys
import time
import traceback
import warnings
from collections.abc import Callable
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, cast

import mne
import numpy as np
import scipy
import threadpoolctl
from numpy.typing import NDArray

# Make sibling modules (core.py, data_loaders*.py in the parent folder)
# importable when this script is run directly from supplementary/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analysis_tools  # noqa: E402
import core  # noqa: E402
from data_loaders import load_eeg_erp  # noqa: E402
from data_loaders_ecg import load_ecg_mitbih  # noqa: E402
from data_loaders_p300 import load_bnci2014_009_blocks  # noqa: E402

DESCRIPTION = """Numerical stability of D^T Sigma_W D for AR(1) noise at high phi.

Per configuration: condition number and eigenvalues over a phi sweep, SURE cost on a
lambda grid, end-to-end SURE(kernel) at forced phi (EEG, EEG-2cls, P300, ECG), float64
vs exact long double (EEG), and the historical lag-4T truncated builder vs the exact one
on the historical inputs (BRIDGE).

    python3 -B supplementary/run_numerical_stability.py [--configs EEG] [--workers 4]

Writes <out>/numerical_stability/results/<config>.json, summary.json and run_info.json.
"""

# A configuration, a result and a summary are heterogeneous JSON-like records.
Record = dict[str, Any]
Data = tuple[NDArray, NDArray, NDArray, float, NDArray | None]

EEG4 = {"aud_l": 1, "aud_r": 2, "vis_l": 3, "vis_r": 4}
EEG2 = {"auditory_left": 1, "auditory_right": 2}
# exclude_bads drops EEG 053, the channel MNE marks bad.
EEG_SETUP: Record = {
    "dataset": "MNE sample",
    "bandpass": (1.0, 40.0),
    "exclude_bads": True,
}
P300_SETUP: Record = {"dataset": "BNCI2014-009", "subject": 1, "bandpass": (1.0, 20.0)}
P300_SETUP["resample"] = 128.0
ECG_SETUP: Record = {
    "dataset": "MIT-BIH",
    "record": "119",
    "lead": 0,
    "bandpass": (0.5, 40.0),
}
ECG_SETUP.update({"beat_types": ["N", "V"], "labels": {0: "Normal", 1: "PVC"}})
# BRIDGE: the submitted figure's inputs (60 EEG channels, absolute onsets, T = 360).
CONFIGS: dict[str, Record] = {
    "EEG": {"kind": "eeg", "event_ids": EEG4, "half_win": 120, **EEG_SETUP},
    "EEG-2cls": {"kind": "eeg", "event_ids": EEG2, "half_win": 120, **EEG_SETUP},
    "P300": {"kind": "p300", "half_win": 51, **P300_SETUP},
    "ECG": {"kind": "ecg", "half_win": 126, **ECG_SETUP},
    "BRIDGE": {"kind": "eeg_hist", "half_win": 180, **EEG_SETUP, "exclude_bads": False},
}
PHIS = [0.0, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
SURE_PHIS = [0.0, 0.9, 0.98, 0.99, 0.999]
LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0]
SK_PHIS = [0.98, 0.99, 0.995]
LD_PHIS = [0.9, 0.98, 0.999]
LD_CLASSES, LD_EVENTS, LD_HALF_WIN = 2, 6, 20  # the long-double subset
TRUNCATION_PHIS = (0.995, 0.999)
K, SEED = 5, 42
APPROVED_SK: Record = {
    "method": "ls",
    "reg_temporal": 500,
    "opt_reg_temporal": True,
    "vec_mode": True,
    "symm": True,
    "vlen": 3,
    "noise_est": "ar1_colored",
}  # the SURE(kernel) of the paper analyses
ORIGINAL_SK: Record = {
    "method": "ls",
    "opt_reg_temporal": True,
    "omega": 0.5,
    "vec_mode": True,
    "vlen": 5,
    "symm": True,
    "noise_est": "ar1_colored",
}  # the submitted script's call
EPS = 2.220446049250313e-16
# Raised, never recorded as a scientific failure.
INFRASTRUCTURE_ERRORS: tuple[type[BaseException], ...] = (
    *analysis_tools.INFRASTRUCTURE_ERRORS,
    concurrent.futures.BrokenExecutor,  # a worker process died
)


def _load_eeg(cfg: Record, data_dir: Path) -> Data:
    loaded = load_eeg_erp(
        data_dir=data_dir,
        event_ids=cfg.get("event_ids"),
        bandpass=cfg["bandpass"],
        exclude_bads=cfg["exclude_bads"],
        return_info=True,
    )
    if not isinstance(loaded, tuple):
        raise TypeError(f"load_eeg_erp returned {type(loaded).__name__}, not a tuple")
    if len(loaded) != 4:
        raise ValueError(
            f"load_eeg_erp(return_info=True) returned {len(loaded)} values, "
            "not 4 (data, ix, lb, info)"
        )
    data, ix, lb, info = cast(tuple[NDArray, NDArray, NDArray, Record], loaded)
    if cfg["kind"] == "eeg_hist":  # the historical loader kept absolute onsets
        fif = data_dir / "MNE-sample-data/MEG/sample/sample_audvis_raw.fif"
        ix = ix + mne.io.read_raw_fif(fif, preload=False, verbose=False).first_samp
    sfreq = float(info["sfreq"])
    return np.asarray(data), np.asarray(ix), np.asarray(lb), sfreq, None


def _load_p300(cfg: Record, data_dir: Path) -> Data:
    # MOABB finds BNCI under this MNE config key; the environment overrides it
    os.environ["MNE_DATASETS_BNCI_PATH"] = str(data_dir)
    blocks = load_bnci2014_009_blocks(
        cfg["subject"], bandpass=cfg["bandpass"], resample=cfg["resample"]
    )
    starts = np.cumsum([0] + [len(b.data) for b in blocks[:-1]])
    ix = np.concatenate([b.onsets + starts[i] for i, b in enumerate(blocks)])
    lb = np.concatenate([b.labels for b in blocks])
    data = np.concatenate([b.data for b in blocks])
    return data, ix, lb, cfg["resample"], starts.astype(np.int64)


def _load_ecg(cfg: Record, data_dir: Path) -> Data:
    data, ix, lb, sfreq = load_ecg_mitbih(
        data_dir=data_dir / "mitdb",
        records=[cfg["record"]],
        lead=cfg["lead"],
        bandpass=cfg["bandpass"],
        beat_types=cfg["beat_types"],
    )
    lb = np.array([cfg["labels"][int(v)] for v in lb])
    return np.asarray(data), ix, lb, float(sfreq), None


def load_config_data(name: str, data_dir: Path) -> Data:
    """(data, event_indices, event_labels, sfreq, block_starts) of one configuration."""
    cfg = CONFIGS[name]
    loader = {"eeg": _load_eeg, "eeg_hist": _load_eeg, "p300": _load_p300}
    return loader.get(cfg["kind"], _load_ecg)(cfg, data_dir)


def select_events(
    n_samples: int,
    event_indices: NDArray,
    event_labels: NDArray,
    half_win: int,
    block_starts: NDArray | None,
) -> tuple[NDArray, NDArray]:
    """K events per class with complete windows (inside their own block), sorted."""
    # the stream of the script that made the submitted figure
    rs = np.random.RandomState(SEED)
    ix_sub, lb_sub = [], []
    for ev in np.unique(event_labels):
        mask = event_labels == ev
        candidates = event_indices[mask]
        valid = (candidates - half_win >= 0) & (candidates + half_win <= n_samples)
        if block_starts is not None:
            ends = np.r_[block_starts[1:], n_samples]
            blk = np.searchsorted(block_starts, candidates, side="right") - 1
            valid &= (candidates - half_win >= block_starts[blk]) & (
                candidates + half_win <= ends[blk]
            )
        candidates = candidates[valid]
        sel = rs.choice(len(candidates), min(K, len(candidates)), replace=False)
        ix_sub.extend(candidates[sel])
        lb_sub.extend([ev] * len(sel))
    ix_sub = np.array(ix_sub)
    lb_sub = np.array(lb_sub)
    sort_idx = np.argsort(ix_sub)
    return ix_sub[sort_idx], lb_sub[sort_idx]


def load_and_prepare(config: str, data_dir: Path) -> Record:
    """The selection, D^T D, D^T X, the penalty S and sigma^2 of one configuration."""
    data, event_indices, event_labels, sfreq, block_starts = load_config_data(
        config, data_dir
    )
    half_win = CONFIGS[config]["half_win"]
    events = np.unique(event_labels)
    M, T = len(events), 2 * half_win
    ix, lb = select_events(
        len(data), event_indices, event_labels, half_win, block_starts
    )
    D = core.get_design_matrix(data, ix, lb, half_win=half_win, events=events)
    D_dense = np.asarray(D.toarray()) if isinstance(D, scipy.sparse.csr_matrix) else D
    R = scipy.linalg.convolution_matrix([1, -2, 1], T - 2)
    RR = R @ R.T
    blocks = None
    if block_starts is not None:
        blocks = np.searchsorted(block_starts, ix, side="right") - 1
    return {
        "data": data,
        "ix": ix,
        "lb": lb,
        "half_win": half_win,
        "events": events,
        "M": M,
        "T": T,
        "DtD": D_dense.T @ D_dense,
        "DtX": D_dense.T @ data,
        "S": scipy.linalg.block_diag(*([RR] * M)),
        "sigsq": core.est_noise_var_baseline(data, ix, half_win),
        "sfreq": sfreq,
        "block_starts": block_starts,
        "event_blocks": blocks,
        "phi_estimated": float(core.estimate_phi(data, ix, half_win)),
    }


def build_DtSigmaD_truncated(
    ix: NDArray,
    lb: NDArray,
    events: NDArray,
    half_win: int,
    phi: float,
    sigsq: float,
    data_length: int | None = None,
    event_blocks: NDArray | None = None,
) -> NDArray:
    """The builder behind the submitted figure (core.py e57802c1cba4), body verbatim:
    sigma^2 phi^|lag| is zero beyond lag 4T; events with ix >= data_length - half_win
    are dropped."""
    if event_blocks is not None:
        raise ValueError(
            "build_DtSigmaD_truncated: the historical builder has no event_blocks "
            f"(got {len(event_blocks)} block labels)"
        )
    M = len(events)
    T = 2 * half_win
    p = M * T
    max_lag = 4 * T
    phi_table = sigsq * np.power(abs(phi), np.arange(max_lag + 1))
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


def build(
    info: Record, phi: float, builder: Callable[..., NDArray] | None = None
) -> NDArray:
    """D^T Sigma_W D at *phi*: the exact core builder, or *builder* (no blocks)."""
    args = (
        info["ix"],
        info["lb"],
        info["events"],
        info["half_win"],
        phi,
        info["sigsq"],
    )
    if builder is None:
        return core.build_DtSigmaD_ar1(
            *args, data_length=len(info["data"]), event_blocks=info["event_blocks"]
        )
    return builder(*args, data_length=len(info["data"]))


def condition_numbers(info: Record, cache: dict[float, NDArray]) -> Record:
    eigvals_DtD = np.linalg.eigvalsh(info["DtD"])
    cond_DtD = np.max(np.abs(eigvals_DtD)) / max(np.min(np.abs(eigvals_DtD)), 1e-300)
    out: Record = {
        "DtD": {
            "cond": float(cond_DtD),
            "eigmin": float(np.min(eigvals_DtD)),
            "eigmax": float(np.max(eigvals_DtD)),
        }
    }
    sweep = {}
    for phi in sorted(cache):
        eigvals = np.linalg.eigvalsh(cache[phi])
        eigmin, eigmax = np.min(eigvals), np.max(eigvals)
        absv = np.abs(eigvals)
        sweep[str(phi)] = {
            # legacy ratio lambda_max / |lambda_min| (the figure); cond2: 2-norm
            "cond": float(eigmax / max(abs(eigmin), 1e-300)),
            "cond2": float(absv.max() / max(absv.min(), 1e-300)),
            "n_nonpositive": int(np.sum(eigvals <= 0)),
            "eigmin": float(eigmin),
            "eigmax": float(eigmax),
            "neg_eigenvalues": int(np.sum(eigvals < -1e-10 * eigmax)),
            "eigenvalues": eigvals.tolist(),
        }
    out["phi_sweep"] = sweep
    return out


def sure_inputs(info: Record) -> tuple[NDArray, ...]:
    """(D^T D, S, mu, V, V^T D^T X, theta_ml) for SURE_cost_colored / SURE_cost_fast."""
    DtD_d, S_d, DtX_d = (np.asarray(info[k], dtype=float) for k in ("DtD", "S", "DtX"))
    mu, V = cast(tuple[NDArray, NDArray], scipy.linalg.eigh(DtD_d, DtD_d + S_d))
    mu = np.clip(mu, 0, 1)
    return DtD_d, S_d, mu, V, V.T @ DtX_d, np.linalg.solve(DtD_d, DtX_d)


def colored_cost(
    info: Record, inputs: tuple[NDArray, ...], A: NDArray, lam: float
) -> float:
    DtD_d, S_d, mu, V, DtX_rot, theta_ml = inputs
    n_obs, n_cols = info["data"].shape
    return core.SURE_cost_colored(
        lam, mu, DtX_rot, V, theta_ml, A, n_obs, n_cols, 0.5, _DtD=DtD_d, _S=S_d
    )


def sure_costs(
    info: Record, cache: dict[float, NDArray]
) -> tuple[bool, list[dict[str, float]]]:
    inputs = sure_inputs(info)
    _, _, mu, V, DtX_rot, theta_ml = inputs
    n_obs, n_cols = info["data"].shape
    rows, all_finite = [], True
    for phi in SURE_PHIS:
        for lam in LAMBDAS:
            c_col = colored_cost(info, inputs, cache[phi], lam)
            c_wht = core.SURE_cost_fast(
                lam, mu, DtX_rot, V, theta_ml, info["sigsq"], n_obs, n_cols, 0.5
            )
            all_finite &= bool(np.isfinite(c_col) and np.isfinite(c_wht))
            rows.append(
                {"phi": phi, "lam": lam, "colored": float(c_col), "white": float(c_wht)}
            )
    return all_finite, rows


def fit_record(
    info: Record, templates: dict[object, NDArray], inner: Record, t0: float
) -> Record:
    """One end-to-end fit: kernel, cost, finiteness, SNR vs OLS, optimizer fields."""
    data, lb, half_win = info["data"], info["lb"], info["half_win"]
    opt_vec, cost = inner.get("opt_val"), inner.get("cost")
    noise_var = inner.get("noise_var")
    theta_ml = np.linalg.solve(info["DtD"], info["DtX"])
    events = np.unique(lb)
    C, T, M = data.shape[1], 2 * half_win, len(events)
    ols = theta_ml.T.reshape(C, M, T).transpose(1, 2, 0)
    snr = core.calculate_snr(
        {str(k): v for k, v in templates.items()},
        {str(ev): ols[ii] for ii, ev in enumerate(events)},
    )
    optimizer = {
        k: (v if isinstance(v, (bool, int, str)) else float(v))
        for k, v in inner.items()
        if k.startswith("optimizer_")
        or k in {"phi_used", "phi_estimated", "noise_var_estimated"}
    }
    scale = float(inner.get("optimizer_cost_scale", float("nan")))
    return {
        # the call returned; optimizer_success is the BFGS outcome
        "converged": True,
        "kernel": None if opt_vec is None else opt_vec.tolist(),
        "cost": None if cost is None else float(cost),
        "noise_var": None if noise_var is None else float(noise_var),
        "templates_finite": all(np.all(np.isfinite(t)) for t in templates.values()),
        "templates_max": float(max(np.max(np.abs(t)) for t in templates.values())),
        "snr_vs_ols": float(snr),
        "time": time.time() - t0,
        "kernel_finite": bool(opt_vec is not None and np.all(np.isfinite(opt_vec))),
        **optimizer,
        "cost_unscaled": None if cost is None else float(cost) * scale,
    }


def end_to_end(info: Record, sk: Record) -> dict[str, Record]:
    """SURE(kernel) at forced phi; a failed fit is recorded, not raised."""
    results: dict[str, Record] = {}
    for phi in SK_PHIS:
        t0 = time.time()
        try:
            templates, inner = core.get_events_templates(
                info["data"],
                info["ix"],
                info["lb"],
                half_win=info["half_win"],
                block_starts=info["block_starts"],
                ar1_phi=phi,
                **sk,
            )
            if inner is None:
                results[str(phi)] = {"converged": False, "time": time.time() - t0}
                continue
            results[str(phi)] = fit_record(info, templates, inner, t0)
        except INFRASTRUCTURE_ERRORS:
            raise
        except Exception as e:  # a failed fit is evidence: record it, keep going
            traceback.print_exc()
            results[str(phi)] = {
                "converged": False,
                "error": str(e),
                "time": time.time() - t0,
            }
    return results


def long_double_matrix(
    ix_s: NDArray,
    lb_s: NDArray,
    events: NDArray,
    half_win: int,
    phi: float,
    sigsq: float,
) -> NDArray:
    """Exact AR(1) D^T Sigma_W D in long double (every lag)."""
    M, T = len(events), 2 * half_win
    pp = M * T
    max_lag = int(ix_s.max() - ix_s.min()) + T
    phi_tab = np.longdouble(sigsq) * np.power(
        np.longdouble(phi), np.arange(max_lag + 1, dtype=np.longdouble)
    )
    offsets = np.arange(T)[:, None] - np.arange(T)[None, :]
    A_ld = np.zeros((pp, pp), dtype=np.longdouble)
    for ii, ev_i in enumerate(events):
        ix_i = ix_s[lb_s == ev_i]
        for jj, ev_j in enumerate(events):
            ix_j = ix_s[lb_s == ev_j]
            block = np.zeros((T, T), dtype=np.longdouble)
            for ti in ix_i:
                for tj in ix_j:
                    al = np.abs(int(ti) - int(tj) + offsets)
                    block += np.where(
                        al <= max_lag,
                        phi_tab[np.minimum(al, max_lag)],
                        np.longdouble(0),
                    )
            A_ld[ii * T : (ii + 1) * T, jj * T : (jj + 1) * T] = block
    return A_ld


def long_double(info: Record) -> dict[str, dict[str, float]]:
    """float64 builder vs exact long-double AR(1) on a small subset; empty when long
    double has no extra precision on this platform."""
    ld_eps = np.finfo(np.longdouble).eps
    if not ld_eps < np.finfo(np.float64).eps * 0.5:
        return {}
    events = info["events"][:LD_CLASSES]
    half_win = min(info["half_win"], LD_HALF_WIN)
    sigsq = info["sigsq"]
    mask = np.isin(info["lb"], events)
    ix_s = info["ix"][mask][:LD_EVENTS]
    lb_s = info["lb"][mask][:LD_EVENTS]
    out = {}
    for phi in LD_PHIS:
        A64 = core.build_DtSigmaD_ar1(
            ix_s, lb_s, events, half_win, phi, sigsq, data_length=len(info["data"])
        )
        A_ld = long_double_matrix(ix_s, lb_s, events, half_win, phi, sigsq)
        pp = A_ld.shape[0]
        diff = np.abs(A64[:pp, :pp].astype(np.longdouble) - A_ld)  # in long double
        rel_err = diff / np.maximum(np.abs(A_ld), np.longdouble(1e-300))
        out[str(phi)] = {
            "longdouble_eps": float(ld_eps),
            "max_abs_err": float(np.max(diff)),
            "max_rel_err": float(np.max(rel_err)),
            "mean_rel_err": float(np.mean(rel_err)),
        }
    return out


def truncated_sweep(info: Record) -> dict[str, Record]:
    """The truncated builder's eigenvalue statistics at every phi of the sweep."""
    tsweep = {}
    for phi in PHIS:
        e = np.linalg.eigvalsh(build(info, phi, build_DtSigmaD_truncated))
        a = np.abs(e)
        tsweep[str(phi)] = {
            "cond": float(e[-1] / max(abs(e[0]), 1e-300)),
            "cond2": float(a.max() / max(a.min(), 1e-300)),
            "eigmin": float(e[0]),
            "eigmax": float(e[-1]),
            "neg": int(np.sum(e < -1e-10 * e[-1])),
        }
    return tsweep


def builder_fit(info: Record, phi: float, builder: Callable[..., NDArray]) -> Record:
    """SURE(kernel) at forced phi with *builder* as core's covariance builder."""
    exact_builder = core.build_DtSigmaD_ar1
    # get_events_templates builds its covariance through this name
    core.build_DtSigmaD_ar1 = builder
    t0 = time.time()
    try:
        tpl, returned = core.get_events_templates(
            info["data"],
            info["ix"],
            info["lb"],
            half_win=info["half_win"],
            ar1_phi=phi,
            **APPROVED_SK,
        )
        f = kernel_fit_diagnostics(tpl, returned, phi, t0)
    except INFRASTRUCTURE_ERRORS:
        raise
    except Exception as e:  # a failed fit is evidence: record it, keep going
        return {"error": f"{type(e).__name__}: {e}", "time": time.time() - t0}
    finally:
        core.build_DtSigmaD_ar1 = exact_builder
    return f


def kernel_fit_diagnostics(
    tpl: dict[object, NDArray],
    returned: dict[str, object] | None,
    phi: float,
    t0: float,
) -> Record:
    """The diagnostics of one SURE(kernel) fit; raises if the fit returned none."""
    if returned is None:
        raise ValueError(f"SURE(kernel) at phi = {phi} returned no diagnostics")
    inner: Record = dict(returned)
    f = {
        "templates": tpl,
        "kernel": np.asarray(inner["opt_val"], dtype=float),
        "cost": float(inner["cost"]),
        "cost_scale": float(inner["optimizer_cost_scale"]),
        "success": bool(inner["optimizer_success"]),
        "message": str(inner["optimizer_message"]),
        "nit": int(inner["optimizer_iterations"]),
        "time": time.time() - t0,
    }
    f["finite"] = bool(
        np.all(np.isfinite(f["kernel"]))
        and np.isfinite(f["cost"])
        and all(np.all(np.isfinite(t)) for t in tpl.values())
    )
    return f


def fit_differences(fits: dict[str, Record]) -> Record:
    """Truncated vs exact fit: kernel (up to sign), templates and unscaled cost."""
    kt, ke = fits["truncated"]["kernel"], fits["exact"]["kernel"]
    nk = max(float(np.linalg.norm(ke)), 1e-300)
    exact_tpl, trunc_tpl = fits["exact"]["templates"], fits["truncated"]["templates"]
    tmax = max(*(float(np.max(np.abs(t))) for t in exact_tpl.values()), 1e-300)
    worst = max(np.max(np.abs(trunc_tpl[k] - exact_tpl[k])) for k in exact_tpl)
    ce = fits["exact"]["cost"] * fits["exact"]["cost_scale"]
    ct = fits["truncated"]["cost"] * fits["truncated"]["cost_scale"]
    return {
        # a kernel and its negative give the same penalty: compare up to sign
        "kernel_rel_diff": float(
            min(np.linalg.norm(kt - ke), np.linalg.norm(kt + ke)) / nk
        ),
        "template_max_abs_diff_rel": float(worst / tmax),
        "cost_unscaled_rel_diff": abs(ct - ce) / max(abs(ce), 1e-300),
    }


def truncation_at(info: Record, inputs: tuple[NDArray, ...], phi: float) -> Record:
    builders = {"truncated": build_DtSigmaD_truncated, "exact": core.build_DtSigmaD_ar1}
    mats = {name: build(info, phi, b) for name, b in builders.items()}
    r: Record = {}
    for name, A in mats.items():
        e = np.linalg.eigvalsh(A)
        r[f"{name}_eigmin"], r[f"{name}_eigmax"] = float(e[0]), float(e[-1])
        r[f"{name}_neg"] = int(np.sum(e < -1e-10 * e[-1]))
    r["matrix_max_abs_diff_rel"] = float(
        np.max(np.abs(mats["truncated"] - mats["exact"]))
        / max(float(np.max(np.abs(mats["exact"]))), 1e-300)
    )
    costs = []
    for lam in LAMBDAS:
        c = {n: float(colored_cost(info, inputs, A, lam)) for n, A in mats.items()}
        rel = abs(c["truncated"] - c["exact"]) / max(abs(c["exact"]), 1e-300)
        costs.append({"lam": lam, **c, "rel_diff": rel})
    r["sure_costs"] = costs
    fits = {name: builder_fit(info, phi, b) for name, b in builders.items()}
    r["sk"] = {
        name: {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in f.items()
            if k != "templates"
        }
        for name, f in fits.items()
    }
    if all("error" not in f and f["finite"] for f in fits.values()):
        r["sk"].update(fit_differences(fits))
    return r


def truncation_effect(info: Record) -> Record:
    """Truncated (historical) vs exact builder: sweep, SURE costs, SURE(kernel) fits."""
    out: Record = {"truncated_sweep": truncated_sweep(info)}
    inputs = sure_inputs(info)
    for phi in TRUNCATION_PHIS:
        out[str(phi)] = truncation_at(info, inputs, phi)
    return out


def run_config(config: str, data_dir: Path) -> Record:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    info = load_and_prepare(config, data_dir)
    cache = {phi: build(info, phi) for phi in PHIS}
    cond = condition_numbers(info, cache)
    sure_ok, sure_rows = sure_costs(info, cache)
    hist = config == "BRIDGE"
    task3 = {}
    if not hist:
        task3 = {
            "approved": end_to_end(info, APPROVED_SK),
            "original": end_to_end(info, ORIGINAL_SK),
        }
    starts = info["block_starts"]
    res = {
        "config": config,
        "n_channels": int(info["data"].shape[1]),
        "M": int(info["M"]),
        "T": int(info["T"]),
        "half_win": int(info["half_win"]),
        "events": [str(e) for e in info["events"]],
        "ix": [int(i) for i in info["ix"]],
        "lb": [str(v) for v in info["lb"]],
        "block_starts": None if starts is None else [int(b) for b in starts],
        "sigsq": float(info["sigsq"]),
        "sfreq": float(info["sfreq"]),
        "phi_estimated": info["phi_estimated"],
        "DtD": cond["DtD"],
        "phi_sweep": cond["phi_sweep"],
        "sure_all_finite": sure_ok,
        "sure_costs": sure_rows,
        "task3": task3,
        "task4": long_double(info) if config == "EEG" else {},
        "task5": truncation_effect(info) if hist else {},
    }
    print(f"{config}: done in {time.time() - t0:.0f}s", flush=True)
    return res


def slope2(sweep: Record, lo: str, hi: str, key: str = "cond") -> float | None:
    if not (sweep[lo][key] > 0 and sweep[hi][key] > 0):
        return None  # a zero matrix (sigma^2 = 0) has no log-log slope
    return math.log(sweep[hi][key] / sweep[lo][key]) / math.log(
        (1 - float(hi)) / (1 - float(lo))
    )


def slope_ls(sweep: Record, phis: tuple[str, ...], key: str = "cond") -> float | None:
    if not all(sweep[p][key] > 0 for p in phis):
        return None
    pts = [(math.log(1 - float(p)), math.log(sweep[p][key])) for p in phis]
    mx = sum(a for a, _ in pts) / len(pts)
    my = sum(b for _, b in pts) / len(pts)
    return sum((a - mx) * (b - my) for a, b in pts) / sum((a - mx) ** 2 for a, _ in pts)


SUBMITTED_RANGE = ("0.9", "0.95", "0.98", "0.99", "0.995")


def setup_facts(configs: list[str]) -> Record:
    return {
        "K": K,
        "seed": SEED,
        "phis": {
            "sweep": PHIS,
            "sure_cost": SURE_PHIS,
            "lambdas": LAMBDAS,
            "end_to_end": SK_PHIS,
            "long_double": LD_PHIS,
        },
        "configs": {c: CONFIGS[c] for c in configs},
        "long_double_subset": {
            "classes": LD_CLASSES,
            "events": LD_EVENTS,
            "T": 2 * LD_HALF_WIN,
        },
    }


def old_facts(osw: dict[str, Record]) -> Record:
    """The submitted figure's matrices: historical inputs, truncated builder."""
    keys = [str(p) for p in PHIS]
    top = osw["0.999"]
    return {
        "sweep": {
            p: {k: osw[p][k] for k in ("cond", "eigmin", "eigmax", "neg")} for p in keys
        },
        "cond2_0999": top["cond2"],
        "slope2_09_0995": slope2(osw, "0.9", "0.995"),
        "slope_ls_09_0995": slope_ls(osw, SUBMITTED_RANGE),
        # None for a zero matrix (sigma^2 = 0)
        "ratio_0999": top["eigmin"] / top["eigmax"] if top["eigmax"] != 0 else None,
    }


def decode_nonfinite(value: Any) -> Any:  # noqa: ANN401 - any JSON value
    """Undo analysis_tools.plain's "nan"/"inf"/"-inf" strings in reloaded results."""
    if isinstance(value, dict):
        return {k: decode_nonfinite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_nonfinite(v) for v in value]
    if value in {"nan", "inf", "-inf"}:
        return float(value)
    return value


def task3_facts(runs: dict[str, Record]) -> Record:
    maxima = [
        x["templates_max"]
        for x in runs.values()
        if x.get("templates_max") is not None and math.isfinite(x["templates_max"])
    ]
    return {
        "n": len(runs),
        "completed": sum(bool(x.get("converged")) for x in runs.values()),
        "finite": sum(
            bool(x.get("kernel_finite"))
            and bool(x.get("templates_finite"))
            and x.get("cost") is not None
            and math.isfinite(x["cost"])
            for x in runs.values()
        ),
        "bfgs_success": sum(bool(x.get("optimizer_success")) for x in runs.values()),
        "not_converged": sorted(
            p for p, x in runs.items() if not x.get("optimizer_success")
        ),
        "templates_max_max": max(maxima, default=None),
    }


def task5_facts(t: Record) -> Record:
    sk = t["sk"]
    grid = [x["rel_diff"] for x in t["sure_costs"]]
    finite_grid = [g for g in grid if math.isfinite(g)]
    # None: a fit failed or was not finite
    diffs = [
        sk.get(k)
        for k in (
            "kernel_rel_diff",
            "template_max_abs_diff_rel",
            "cost_unscaled_rel_diff",
        )
    ]
    ok = {name: bool(sk[name].get("success")) for name in ("truncated", "exact")}
    comparable = len(finite_grid) == len(grid) and all(
        d is not None and math.isfinite(d) for d in diffs
    )
    return {
        "trunc_neg": t["truncated_neg"],
        "exact_neg": t["exact_neg"],
        "trunc_eigmin": t["truncated_eigmin"],
        "exact_eigmin": t["exact_eigmin"],
        "matrix_rel": t["matrix_max_abs_diff_rel"],
        # over the finite comparisons; None when there is none
        "cost_grid_max_rel": max(finite_grid, default=None),
        "cost_grid_min_rel": min(finite_grid, default=None),
        "kernel_rel": diffs[0],
        "template_rel": diffs[1],
        "cost_rel": diffs[2],
        "both_success": ok["truncated"] and ok["exact"],
        "no_effect": comparable
        and ok["truncated"] == ok["exact"]
        and max(*diffs, *grid) <= 1e-6,
    }


def config_facts(r: Record) -> Record:
    keys = [str(p) for p in PHIS]
    sw = r["phi_sweep"]
    d: Record = {
        k: r[k] for k in ("n_channels", "M", "T", "half_win", "sigsq", "phi_estimated")
    }
    d["p"] = r["M"] * r["T"]
    fields = ("cond", "cond2", "eigmin", "eigmax")
    d["sweep"] = {
        p: {**{f: sw[p][f] for f in fields}, "neg": sw[p]["neg_eigenvalues"]}
        for p in keys
    }
    d["all_psd"] = all(
        sw[p]["neg_eigenvalues"] == 0 and sw[p]["eigmin"] > 0 for p in keys
    )
    d["slope2_09_0995"] = slope2(sw, "0.9", "0.995")
    d["slope_ls_09_0995"] = slope_ls(sw, SUBMITTED_RANGE)
    # the submitted rule: 0.9 to the largest PSD phi
    d["slope2_09_0999"] = slope2(sw, "0.9", "0.999")
    d["largest_psd_phi"] = max(
        (p for p in keys if float(p) >= 0.9 and sw[p]["neg_eigenvalues"] == 0),
        default=None,
    )
    d["cond_eps_0999"] = sw["0.999"]["cond2"] * EPS
    d["sure_all_finite"] = r["sure_all_finite"]
    d["task3"] = {v: task3_facts(runs) for v, runs in r["task3"].items()}
    if r["task4"]:
        d["task4_max_rel"] = max(x["max_rel_err"] for x in r["task4"].values())
        d["task4_min_rel"] = min(x["max_rel_err"] for x in r["task4"].values())
    if r["task5"]:
        d["task5"] = {str(p): task5_facts(r["task5"][str(p)]) for p in TRUNCATION_PHIS}
    return d


def summarize(R: dict[str, Record]) -> Record:
    """Every number the supplement's Numerical Stability section cites."""
    R = decode_nonfinite(R)  # results reloaded from JSON
    F: Record = {"setup": setup_facts(list(R))}
    if "BRIDGE" in R:
        F["old"] = old_facts(R["BRIDGE"]["task5"]["truncated_sweep"])
    for c, r in R.items():
        F[c] = config_facts(r)
    t3 = {
        k: [F[c]["task3"][k] for c in R if F[c]["task3"]]
        for k in ("approved", "original")
    }
    counts = ("n", "completed", "finite", "bfgs_success")
    F["t3_totals"] = {
        k: {m: sum(x[m] for x in t) for m in counts} for k, t in t3.items()
    }
    # over the configurations that completed; None when none did
    F["all_configs_psd"] = all(F[c]["all_psd"] for c in R) if R else None
    F["max_cond_eps"] = max((F[c]["cond_eps_0999"] for c in R), default=None)
    return F


def run_configs(args: argparse.Namespace) -> tuple[dict[str, Record], Record]:
    """Every configuration in a spawned process; scientific failures are collected."""
    R: dict[str, Record] = {}
    failed: Record = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=threadpoolctl.threadpool_limits,
        initargs=(args.blas_threads,),
    ) as pool:
        futures = {c: pool.submit(run_config, c, args.data_dir) for c in args.configs}
        for c, f in futures.items():
            try:
                R[c] = f.result()
            except INFRASTRUCTURE_ERRORS:
                raise
            except Exception as e:  # a failed configuration is recorded; others go on
                failed[c] = {
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": "".join(traceback.format_exception(e)),
                }
                print(f"{c}: FAILED {type(e).__name__}: {e}", flush=True)
    return R, failed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("MNE_DATA", Path.home() / "mne_data")),
    )
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--workers", type=int, default=1)
    # eigvalsh is not bitwise invariant to the BLAS thread count (the paper's run: 4,
    # BRIDGE 16)
    ap.add_argument("--blas-threads", type=int, default=1)
    ap.add_argument(
        "--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS)
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    started = datetime.now(UTC)
    out = args.out / "numerical_stability"
    (out / "results").mkdir(parents=True, exist_ok=True)
    # a rerun into the same --out must not leave this run's earlier outputs behind
    for stale in [out / "summary.json", out / "run_info.json"] + [
        out / "results" / f"{c}.json" for c in args.configs
    ]:
        stale.unlink(missing_ok=True)
    R, failed = run_configs(args)
    for c, r in R.items():
        analysis_tools.write_json(out / "results" / f"{c}.json", r)
    summary = summarize(R)
    summary["failed_configs"] = failed
    analysis_tools.write_json(out / "summary.json", summary)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    # under the workers' limit, so threadpool_info() shows their BLAS thread count
    with threadpoolctl.threadpool_limits(args.blas_threads):
        analysis_tools.write_run_info(out, settings, started)
    print(f"wrote {out}")
    if failed:
        raise SystemExit(f"failed configurations: {sorted(failed)}")


if __name__ == "__main__":
    main()
