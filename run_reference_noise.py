# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Noise of the full-data P300 references and the noise-corrected distance.

Reads a ``run_p300.py`` output folder (cache, references, Population, per-fit
records, selections and saved templates) and writes ``<out>/reference_noise/``:
- the OLS reference preflight per subject: rank and conditioning of D^T D, and
  the model-based noise of the OLS, EB and Average references (Q D^T Sigma D Q^T,
  Bartlett-tapered OLS residual autocovariance up to lag 204) with its
  shifted-residual Monte Carlo check;
- the noise-corrected distance NC = mean over fits and classes of
  sqrt(max(d^2 - sigma2, 0)) to the OLS reference at K = 5 (the central batch),
  10, 20, 50, with the cross terms X_lin and X_tune. Gates G1-G6 run first; a K
  whose gate fails is unavailable, with its failed gates, and the run continues.
  A failed fit is excluded and counted (decision OQ6);
- NC of the estimators and of the zero, lambda = inf and Population baselines for
  every K with the same formula (descriptive, also where the gated run is
  unavailable; response-letter material), and the squared-domain decomposition at
  K = 5.
Files: ``summary.json``, ``per_fit.csv``, ``references_uV.npz``, ``run_info.json``.
Usage: python3 -B run_reference_noise.py [--p300 DIR] [--out DIR] [--workers N]
           [--subjects 1 2 ...] [--reps N]
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import functools
import json
import math
import statistics
import warnings
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, cast

import analysis_tools
import core
import heldout_p300
import numpy as np
import protocol
import scipy
from numpy.typing import NDArray

UV = 1e6
HALF, GUARD = heldout_p300.HALF_WIN, heldout_p300.GUARD
T, C = 2 * HALF, protocol.N_CHANNELS
P = 2 * T
LAG = 4 * HALF  # 204
KS: tuple[int, ...] = (5, 10, 20, 50)
N_SHIFTS, SHIFT_SEED = 20, 20261005
REFERENCES: dict[str, dict[str, Any]] = {
    "OLS": {"method": "ls"},
    "EB": {"method": "ls", "reg_method": "empbayes"},
    "Average": {"method": "average"},
}
CORE: list[str] = list(protocol.CORE_METHODS)
EST: list[str] = CORE + list(protocol.COMPARATORS)
TUNED: list[str] = ["L-curve", "EmpBayes", "SURE(lambda)", "SURE(kernel)"]
ALL: list[str] = EST + [protocol.POPULATION]
KAPPAS: tuple[float, ...] = (1e-3, 1e-2)
G1_TOL, G1_REF_TOL, G2_TOL, G3_TOL, G4_Z = 1e-6, 1e-12, 1e-9, 1e-6, 4.0
G4_DRAWS, G4_SEED = 2000, 20261004
GATE_KEYS: tuple[str, ...] = tuple(f"G{i}_pass" for i in range(1, 7))
CLS: tuple[str, ...] = ("NonTarget", "Target")
Pre = Mapping[int, Mapping[str, Any]]
Units = Sequence[Mapping[str, Any]]


def fval(x: object) -> float:
    return float(cast(float, x))


@functools.cache
def operators() -> tuple[NDArray, NDArray, NDArray]:
    """R2 (102 x 100 second difference), its block penalty S2 and temporal factor."""
    r2 = scipy.linalg.convolution_matrix([1, -2, 1], T - 2)
    return (
        r2,
        scipy.linalg.block_diag(r2 @ r2.T, r2 @ r2.T),
        scipy.linalg.block_diag(r2.T, r2.T),
    )


def blocks_of(run: Path, s: int) -> list[heldout_p300.RecordingBlock]:
    return analysis_tools.load_blocks(run / "cache" / f"subject_{s:02d}.npz", s)


def class_trace(a: NDArray, k: int) -> float:
    return float(np.trace(a[k * T : (k + 1) * T, k * T : (k + 1) * T]))


def sparse_design(y: NDArray, ix: NDArray, lb: NDArray) -> scipy.sparse.csc_matrix:
    """The two-class design of the onsets *ix* with labels *lb* (CSC)."""
    design = core.get_design_matrix(
        y, ix, lb, HALF, events=np.array([0, 1]), sparse_out=True
    )
    csr = cast(scipy.sparse.csr_matrix, design)  # sparse_out=True returns CSR
    return cast(scipy.sparse.csc_matrix, csr.tocsc())


def rel_err(diff: float, scale: float) -> float:
    """diff / scale for a gate; 0 when both are 0, inf when undefined or nonfinite."""
    if not (math.isfinite(diff) and math.isfinite(scale)):
        return math.inf
    if scale == 0:
        return 0.0 if diff == 0 else math.inf
    return diff / scale


# ---------------------------------------------------------------------------
# OLS reference preflight
# ---------------------------------------------------------------------------


class Recording:
    """The complete events of a set of blocks, concatenated with block_starts (uV)."""

    def __init__(self, blocks: Sequence[heldout_p300.RecordingBlock]) -> None:
        lengths = [len(b.data) for b in blocks]
        self.starts: NDArray = np.cumsum([0] + lengths[:-1]).astype(np.int64)
        self.edges: list[int] = list(self.starts) + [int(sum(lengths))]
        self.data: NDArray = np.vstack([b.data for b in blocks]) * UV
        ix, lb = [], []
        for start, b in zip(self.starts, blocks):
            complete = (b.onsets >= HALF) & (b.onsets <= len(b.data) - HALF)
            events = np.flatnonzero(complete)
            ix.append(start + b.onsets[events].astype(np.int64))
            lb.append(b.labels[events].astype(np.int64))
        self.ix: NDArray = np.concatenate(ix)
        self.lb: NDArray = np.concatenate(lb)

    def design(self) -> scipy.sparse.csr_matrix:
        design = core.get_design_matrix(
            self.data, self.ix, self.lb, HALF, events=np.array([0, 1]), sparse_out=True
        )
        return cast(scipy.sparse.csr_matrix, design)  # sparse_out=True returns CSR


def fit_reference(
    name: str,
    rec: Recording,
    data: NDArray | None = None,
    settings: Mapping[str, Any] | None = None,
) -> tuple[NDArray, dict[str, Any]]:
    """One reference fit through ``core.get_events_templates``: templates and record."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        templates, inner = core.get_events_templates(
            rec.data if data is None else data,
            rec.ix,
            rec.lb,
            half_win=HALF,
            block_starts=rec.starts,
            **(REFERENCES[name] if settings is None else settings),
        )
    out = np.stack([np.asarray(templates[k]) for k in ("0", "1")])
    if out.shape[1:] != (T, rec.data.shape[1]):
        raise RuntimeError(
            f"{name} template has shape {out.shape}, not (2, {T}, {rec.data.shape[1]})"
        )
    if not np.all(np.isfinite(out)):
        raise RuntimeError(
            f"{name} template has {int(np.sum(~np.isfinite(out)))} nonfinite values"
        )
    record: dict[str, Any] = {"warnings": sorted({str(w.message) for w in caught})}
    if inner is not None and inner.get("method") == "empbayes":
        record |= {
            "eb_lambda": fval(inner["opt_val"]),
            "eb_status": inner["status"],
            "eb_converged": bool(inner["converged"]),
        }
    return out, record


def rank_conditioning(rec: Recording) -> tuple[dict[str, Any], NDArray]:
    """Rank and spectrum of the full-data D^T D, computed before any fit."""
    design = rec.design()
    a = (design.T @ design).toarray()
    eig = np.linalg.eigvalsh(a)
    return {
        "n_parameters": int(a.shape[0]),
        "rank": int(np.linalg.matrix_rank(a)),
        "eig_min": float(eig.min()),
        "eig_max": float(eig.max()),
        "eig_median": float(np.median(eig)),
        "cond_eig": float(eig.max() / eig.min()) if eig.min() > 0 else math.inf,
    }, a


def model_noise(
    rec: Recording, a: NDArray, ols: NDArray, eb_lambda: float
) -> dict[str, Any]:
    """Model-based noise of the full-data references, which are linear in the data
    (EB conditional on its lambda): theta = Q D^T y, Q = A^-1 (OLS), (A + lambda S)^-1
    (EB), diag(1 / N_class) (Average). Per class: sqrt of the expected squared error."""
    n_ch = rec.data.shape[1]
    design = cast(scipy.sparse.csc_matrix, rec.design().tocsc())
    residual = rec.data - design @ ols.reshape(2 * T, n_ch)
    counts = np.array([np.sum(rec.lb == 0), np.sum(rec.lb == 1)])
    projectors = {
        "OLS": np.linalg.inv(a),
        "EB": np.linalg.inv(a + eb_lambda * operators()[1]),
        "Average": np.diag(np.repeat(1.0 / counts, T)),
    }
    acov = analysis_tools.residual_autocovariance(residual, rec.edges, LAG)
    g = analysis_tools.noise_gram(
        design, rec.edges, analysis_tools.bartlett_kernel(acov)
    )
    eig = [np.linalg.eigvalsh(g[c]) for c in range(n_ch)]
    entry: dict[str, Any] = {
        "G_min_eig_over_max_eig": min(float(e[0]) / float(e[-1]) for e in eig)
    }
    for name, q in projectors.items():
        colored = np.sqrt(analysis_tools.class_noise_power(q, g, T))
        entry[name] = {
            "colored_uv_per_class": colored,
            "colored_uv": float(np.mean(colored)),
        }
    entry["residual_sd_uv_per_channel"] = np.sqrt(acov[0])
    return {f"max_lag_{LAG}": entry}


def shift_noise(
    rec: Recording, ols: NDArray, eb_lambda: float, records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Monte Carlo check of model_noise: the OLS residual, circularly shifted within
    each block by random offsets, refitted as data whose true template is 0 (EB at
    its full-data lambda)."""
    blocks = list(zip(rec.edges[:-1], rec.edges[1:]))
    residual = rec.data - rec.design() @ ols.reshape(2 * T, rec.data.shape[1])
    rng = np.random.RandomState(SHIFT_SEED)
    settings = {
        "OLS": {"method": "ls"},
        "EB": {"method": "ls", "reg_temporal": eb_lambda},
        "Average": {"method": "average"},
    }
    norms: dict[str, list[list[float]]] = {name: [] for name in settings}
    offsets = []
    for _ in range(N_SHIFTS):
        offset = [int(rng.randint(1, e - s)) for s, e in blocks]
        offsets.append(offset)
        shifted = np.vstack(
            [np.roll(residual[s:e], o, axis=0) for (s, e), o in zip(blocks, offset)]
        )
        for name, kwargs in settings.items():
            t, record = fit_reference(name, rec, data=shifted, settings=kwargs)
            records.append(record)
            norms[name].append([float(np.linalg.norm(t[k])) for k in range(2)])
    out: dict[str, Any] = {"offsets": offsets}
    for name, v in norms.items():
        rms = np.sqrt((np.asarray(v) ** 2).mean(0))  # over shifts, per class
        out[name] = {"norms_per_shift": v, "uv_per_class": rms, "uv": float(rms.mean())}
    return out


def preflight_subject(
    run: Path, s: int
) -> tuple[dict[str, Any], dict[str, NDArray] | None]:
    """Preflight of one subject: diagnostics and the full-data references (uV)."""
    blocks = blocks_of(run, s)
    rec = Recording(blocks)
    spacing = np.concatenate([np.diff(b.onsets) for b in blocks])
    result: dict[str, Any] = {
        "subject": s,
        "data": {
            "block_lengths": [len(b.data) for b in blocks],
            "complete_events": len(rec.ix),
            "guard_eligible_events": int(sum(b.eligible().sum() for b in blocks)),
            "complete_per_class": [int(np.sum(rec.lb == c)) for c in (0, 1)],
            "onset_spacing_samples": {
                "min": int(spacing.min()),
                "median": float(np.median(spacing)),
                "max": int(spacing.max()),
            },
        },
    }
    result["rank"], a = rank_conditioning(rec)
    if result["rank"]["rank"] < result["rank"]["n_parameters"]:
        # Rank deficiency is itself the finding; OLS and everything built on it is
        # undefined.
        return result | {"status": "rank_deficient: no fits"}, None
    records: list[dict[str, Any]] = []
    full = {}
    try:
        for name in REFERENCES:
            full[name], record = fit_reference(name, rec)
            records.append(record)
        eb_lambda = records[1]["eb_lambda"]
        result["full_eb"] = records[1]
        result["template_norm_uv"] = {
            n: [float(np.linalg.norm(t[c])) for c in range(2)] for n, t in full.items()
        }
        result["model_noise"] = model_noise(rec, a, full["OLS"], eb_lambda)
        result["shift_noise"] = shift_noise(rec, full["OLS"], eb_lambda, records)
    except analysis_tools.INFRASTRUCTURE_ERRORS:
        raise
    except Exception as error:  # a failed reference fit excludes the subject (OQ6)
        return result | {"status": f"failed: {type(error).__name__}: {error}"}, None
    result["n_fits"] = len(records)
    result["warnings"] = sorted({w for r in records for w in r["warnings"]})
    return result | {"status": "complete"}, full


def preflight_summary(results: Pre) -> dict[str, Any]:
    """Per-subject scalars (uV) and their median, min and max over subjects."""
    rows = []
    complete = {s: r for s, r in results.items() if r["status"] == "complete"}
    for r in complete.values():
        mn, sn = r["model_noise"][f"max_lag_{LAG}"], r["shift_noise"]
        row: dict[str, Any] = {
            "subject": r["subject"],
            "complete_events": r["data"]["complete_events"],
            "guard_eligible_events": r["data"]["guard_eligible_events"],
            "rank": r["rank"]["rank"],
            "n_parameters": r["rank"]["n_parameters"],
            "cond": r["rank"]["cond_eig"],
            "eb_lambda": r["full_eb"]["eb_lambda"],
            "G_min_eig_ratio": mn["G_min_eig_over_max_eig"],
        }
        for n in REFERENCES:
            row[f"{n}_template_norm_uv"] = sum(r["template_norm_uv"][n]) / 2
            row[f"{n}_model_noise_uv"] = mn[n]["colored_uv"]
            row[f"{n}_shift_noise_uv"] = sn[n]["uv"]
            norm = row[f"{n}_template_norm_uv"]
            row[f"{n}_noise_to_norm"] = row[f"{n}_model_noise_uv"] / norm
        rows.append(row)
    out: dict[str, Any] = {"per_subject": rows}
    for key in rows[0] if rows else ():
        if key != "subject":
            v = [row[key] for row in rows]
            out[key] = {"median": float(statistics.median(v)), "min": min(v)}
            out[key]["max"] = max(v)
    out["all_rank_full"] = all(
        r["rank"]["rank"] == r["rank"]["n_parameters"] for r in results.values()
    )
    out["all_eb_converged"] = all(
        r["full_eb"]["eb_converged"] for r in complete.values()
    )
    out["any_warnings"] = sorted({w for r in complete.values() for w in r["warnings"]})
    out["all_G_psd"] = all(row["G_min_eig_ratio"] > -1e-12 for row in rows)
    return out


# ---------------------------------------------------------------------------
# Noise-corrected distance: inputs, linear operators and per-fit terms
# ---------------------------------------------------------------------------


@functools.cache
def fit_records(run: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    """run_p300's analysis-A records by (K, subject, repetition, estimator) (estimators
    and lambda = inf) and its Population records by ("P", subject)."""
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    with (Path(run) / "per_fit.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            name, s = row["estimator"], int(row["subject"])
            if row["analysis"] != "A" or name not in {*ALL, "lambda=inf"}:
                continue
            rec: dict[str, Any] = {
                "status": row["status"],
                "failure_type": row["failure_type"] or None,
                "template_file": row["template"],
                "distances": {
                    r: float(row[f"distance_{r}"]) if row[f"distance_{r}"] else None
                    for r in ("OLS", "Average")
                },
            }
            for col in ("flags", "kernel", "inner"):
                rec[col] = json.loads(row[col]) if row[col] else None
            if row["lambda"]:
                rec["lambda"] = float(row["lambda"])
            if name == protocol.POPULATION:
                out["P", s] = rec
                continue
            k, r = int(row["K"]), int(row["repetition"])
            rec["key"] = f"A|K{k:02d}|s{s:02d}|r{r:02d}|{name}"
            out[k, s, r, name] = rec
    return out


def fit_record(run: str, k: int, s: int, r: int, name: str) -> dict[str, Any]:
    key = (k, s, r, name)
    if key not in fit_records(run):
        raise KeyError(f"{run}/per_fit.csv has no analysis-A record {key}")
    return fit_records(run)[key]


@functools.cache
def selections(run: str) -> dict[str, Any]:
    return json.loads((Path(run) / "selections.json").read_text())


def selection(run: str, k: int, s: int, r: int) -> NDArray:
    return np.asarray(selections(run)["A"][str(k)][str(s)][r], dtype=np.int64)


class Subject:
    """One subject's joined recording (uV), full-data design, references and kernel."""

    def __init__(self, run: Path, s: int) -> None:
        blocks = blocks_of(run, s)
        lengths = [len(b.data) for b in blocks]
        starts = np.cumsum([0] + lengths[:-1]).astype(np.int64)
        edges = list(starts) + [int(sum(lengths))]
        data_v = np.vstack([b.data for b in blocks])
        y = data_v * UV
        self.start = {b.block: int(t) for b, t in zip(blocks, starts)}
        self.by_id = {b.block: b for b in blocks}
        rows = [
            (b.block, int(e))
            for b in blocks
            for e in np.flatnonzero(b.eligible(HALF, GUARD))
        ]
        ix, lb = self.events(rows)
        design = sparse_design(y, ix, lb)
        counts = np.array([np.sum(lb == 0), np.sum(lb == 1)])
        with np.load(run / f"references/subject_{s:02d}.npz") as z:
            refs = {"OLS": z["OLS"] * UV, "Average": z["Average"] * UV}
        self.pop: NDArray | None = None  # only with a successful record of this run
        rec = fit_records(str(run)).get(("P", s))
        if rec is not None and rec["status"] == "success":
            with np.load(run / f"population/subject_{s:02d}.npz") as z:
                self.pop = z["templates"] * UV
        self.resid: NDArray = y - design @ refs["OLS"].reshape(P, C)
        self.s = s
        self.blocks: list[heldout_p300.RecordingBlock] = blocks
        self.edges: list[int] = edges
        self.data_v: NDArray = data_v
        self.y: NDArray = y
        self.D: scipy.sparse.csc_matrix = design
        self.QR: NDArray = np.linalg.inv((design.T @ design).toarray())
        self.QA: NDArray = np.diag(np.repeat(1.0 / counts, T))
        self.R: dict[str, NDArray] = refs
        self.kernel: NDArray = analysis_tools.bartlett_kernel(
            analysis_tools.residual_autocovariance(self.resid, edges, LAG)
        )
        self.joint_fft: tuple[NDArray, int] | None = None

    def events(
        self, rows: Sequence[tuple[int, int]] | NDArray
    ) -> tuple[NDArray, NDArray]:
        """Joined-recording onsets and labels of (block id, event) rows."""
        ix = [self.start[b] + self.by_id[b].onsets[e] for b, e in rows]
        lb = [self.by_id[b].labels[e] for b, e in rows]
        return np.array(ix, dtype=np.int64), np.array(lb, dtype=np.int64)

    def joint_kernel(self) -> tuple[NDArray, float]:
        """gamma_ij(k), k = -L..L (2L+1, C, C): Bartlett-tapered biased cross-covariance
        pooled over blocks; the diagonal is replaced by the marginal kernel (returned:
        its max relative difference before)."""
        g = np.zeros((2 * LAG + 1, C, C))
        for a_, b_ in zip(self.edges[:-1], self.edges[1:]):
            r = self.resid[a_:b_]
            nfft = 1 << int(np.ceil(np.log2(len(r) + LAG + 1)))
            f = np.fft.rfft(r, nfft, axis=0)
            # c_ij(k) = sum_t r_i(t) r_j(t+k), k >= 0
            cc = np.fft.irfft(np.conj(f)[:, :, None] * f[:, None, :], nfft, axis=0)
            cc = cc[: LAG + 1]
            g[LAG:] += cc
            g[:LAG] += np.transpose(cc[1:][::-1], (0, 2, 1))  # c_ij(-k) = c_ji(k)
        g /= len(self.y)
        g *= (1 - np.abs(np.arange(-LAG, LAG + 1)) / (LAG + 1))[:, None, None]
        diag = np.einsum("kii->ki", g)
        rel = float(np.max(np.abs(diag - self.kernel) / np.max(np.abs(self.kernel))))
        for c in range(C):
            g[:, c, c] = self.kernel[:, c]
        return g, rel

    def fft_of(self, g: NDArray) -> tuple[NDArray, int]:
        n = max(b - a for a, b in zip(self.edges[:-1], self.edges[1:]))
        nfft = 1 << int(np.ceil(np.log2(n + 2 * LAG + 1)))
        return np.fft.rfft(g[::-1], nfft, axis=0), nfft

    def joint_apply(self, u: NDArray) -> NDArray:
        """Sigma_joint u (u: N x C), (Sigma u)_i(t) = sum_j sum_k gamma_ij(k) u_j(t+k),
        within blocks."""
        if self.joint_fft is None:
            self.joint_fft = self.fft_of(self.joint_kernel()[0])
        h, nfft = self.joint_fft
        out = np.zeros_like(u)
        for a_, b_ in zip(self.edges[:-1], self.edges[1:]):
            v = np.einsum("fij,fj->fi", h, np.fft.rfft(u[a_:b_], nfft, axis=0))
            out[a_:b_] = np.fft.irfft(v, nfft, axis=0)[LAG : LAG + (b_ - a_)]
        return out


_SUBJECTS: dict[tuple[str, int], Subject] = {}


def get_subject(run: str, s: int) -> Subject:
    """Per-process cache of the two most recent subjects."""
    if (run, s) not in _SUBJECTS:
        while len(_SUBJECTS) >= 2:
            _SUBJECTS.pop(next(iter(_SUBJECTS)))
        _SUBJECTS[run, s] = Subject(Path(run), s)
    return _SUBJECTS[run, s]


class Fit:
    """Linear operator Q_m (P x P) with T_m = Q_m D_sel^T y, and tuning derivatives."""

    def __init__(
        self, subj: Subject, rows: NDArray, name: str, rec: Mapping[str, Any], k: int
    ) -> None:
        ix, lb = subj.events(rows)
        if list(np.unique(lb)) != [0, 1]:
            raise ValueError(
                f"{name} operator: the selection has labels {np.unique(lb)}, not [0, 1]"
            )
        dsel = sparse_design(subj.y, ix, lb)
        self.ix: NDArray = ix
        self.Dsel: scipy.sparse.csc_matrix = dsel
        self.asel: NDArray = (dsel.T @ dsel).toarray()
        self.bsel: NDArray = np.asarray(dsel.T @ subj.y)
        self.derivs: list[NDArray] = []  # dT/du_k (P x C)
        self.theta: list[float] = []
        self.Sk: NDArray | None = None
        s2 = operators()[1]
        if name == "Average":
            self.Q: NDArray = np.eye(P) / float(k)
        elif name == "OLS":
            self.Q = np.linalg.inv(self.asel)
        elif name in {"GCV", "EmpBayes"}:
            lam = fval(rec["lambda"])
            self.Q = np.linalg.inv(self.asel + lam * s2)
            if name == "EmpBayes":  # u = log(lambda): dT/du = lambda dT/dlambda
                self.theta = [lam]
                self.derivs = [-lam * (self.Q @ s2 @ (self.Q @ self.bsel))]
        elif name in {"L-curve", "SURE(lambda)"}:
            # The eigenbasis core.get_events_templates uses for L-curve and
            # SURE(lambda), so the operator reproduces the saved template (gate G3).
            lam = fval(rec["lambda"])
            factor = operators()[2]
            path = core._PenaltyPath(self.asel, self.bsel, factor)  # noqa: SLF001
            self.Q = path.basis @ (path.weights(lam)[:, None] * path.basis.T)
            self.theta = [lam]
            if np.isinf(lam):
                self.derivs = [np.zeros((P, C))]
            else:
                dw = -path.values / (1 + lam * path.values) ** 2
                self.derivs = [lam * (path.basis @ (dw[:, None] * path.rotated))]
        elif name == "SURE(kernel)":
            self.kernel_penalty(np.asarray(rec["kernel"], dtype=float))
        else:
            raise ValueError(f"no linear operator for estimator {name!r}")

    def kernel_penalty(self, k: NDArray) -> None:
        if len(k) != 3 or k[0] != k[2]:
            raise ValueError(f"SURE(kernel) kernel {k} is not symmetric with 3 taps")
        rk = scipy.linalg.convolution_matrix(k, T - len(k) + 1)
        self.Sk = scipy.linalg.block_diag(rk @ rk.T, rk @ rk.T)
        self.Q = np.linalg.inv(self.asel + self.Sk)
        tm = self.Q @ self.bsel
        self.theta = [k[0], k[1]]
        for basis in ([1.0, 0.0, 1.0], [0.0, 1.0, 0.0]):
            dr = scipy.linalg.convolution_matrix(basis, T - 2)
            ds = scipy.linalg.block_diag(dr @ rk.T + rk @ dr.T, dr @ rk.T + rk @ dr.T)
            self.derivs.append(-self.Q @ ds @ tm)

    def template(self) -> NDArray:
        return (self.Q @ self.bsel).reshape(2, T, C)


def theta_of(name: str, inner: Mapping[str, Any]) -> list[float]:
    """Raw tuning values as stored: [lambda] or [x0, x1] (kernel = x0, x1, x0)."""
    if name == "SURE(kernel)":
        v = np.asarray(inner["opt_val"], dtype=float)
        return [float(v[0]), float(v[1])]
    return [fval(inner["opt_val"])]


def coord(name: str, theta: Sequence[float]) -> list[float]:
    """Tuning coordinates u: log(lambda) for scalar methods, (x0, x1) for the kernel."""
    if name == "SURE(kernel)":
        return list(theta)
    lam = theta[0]
    if math.isfinite(lam) and lam > 0:
        return [math.log(lam)]
    return [math.inf if lam == math.inf else math.nan]


def refit_theta(
    subj: Subject, rows: NDArray, name: str, data_v: NDArray
) -> list[float]:
    """The tuning values of *name* refitted on *data_v* (volts), same selection."""
    edges = subj.edges
    blocks = [
        dataclasses.replace(b, data=data_v[edges[i] : edges[i + 1]])
        for i, b in enumerate(subj.blocks)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        templates, inner = heldout_p300.fit_templates(
            blocks, rows, heldout_p300.reference_methods()[name], HALF
        )
    bad = sum(int(np.sum(~np.isfinite(t))) for t in templates.values())
    if bad:  # handled by the callers' failure paths (G5 fails, X_tune unquantified)
        raise ValueError(f"{name} refit on {len(rows)} events: {bad} nonfinite values")
    if inner is None:
        raise RuntimeError(
            f"{name} refit on {len(rows)} events returned no diagnostics"
        )
    return theta_of(name, inner)


def failed_entry(name: str, rec: Mapping[str, Any]) -> dict[str, Any]:
    """A failed fit (OQ6): NaN everywhere, so the metric excludes and counts it."""
    nan2 = [math.nan, math.nan]
    e: dict[str, Any] = {
        "failed_fit": True,
        "key": rec["key"],
        "failure_type": rec["failure_type"],
        "d2_OLS": nan2,
        "d2_Average": nan2,
        "journal_distance_OLS_uV": math.nan,
        "journal_distance_Average_uV": math.nan,
        "flags": rec["flags"],
    }
    if name in CORE:
        e["X_lin_OLS"] = nan2
    if name in TUNED:
        part = {"X_tune": None, "parts": [{"status": "failed_fit_excluded"}]}
        e["theta"] = None
        e["X_tune_OLS"] = {
            "per_class": {c: {"0.001": part, "0.01": part} for c in ("0", "1")}
        }
    return e


def unit(run: str, k: int, s: int, r: int, full: bool) -> dict[str, Any]:
    """One (subject, repetition): d2, operators (G3); with *full*, X_lin and X_tune."""
    subj = get_subject(run, s)
    rows = selection(run, k, s, r)
    out: dict[str, Any] = {"subject": s, "repetition": r, "est": {}}
    for name in EST:
        rec = fit_record(run, k, s, r, name)
        if rec["status"] != "success":
            out["est"][name] = failed_entry(name, rec)
            continue
        with np.load(Path(run) / rec["template_file"]) as z:
            tm = z["templates"] * UV
        e: dict[str, Any] = {}
        for ref in ("OLS", "Average"):
            e[f"d2_{ref}"] = [
                float(np.sum((tm[c] - subj.R[ref][c]) ** 2)) for c in range(2)
            ]
        e["journal_distance_OLS_uV"] = rec["distances"]["OLS"] * UV
        e["journal_distance_Average_uV"] = rec["distances"]["Average"] * UV
        e["flags"] = rec["flags"]
        if name in CORE:
            try:
                fit = Fit(subj, rows, name, rec, k)
            except (ValueError, np.linalg.LinAlgError) as error:  # fails G3
                out["est"][name] = e | {"G3_rel_err": math.inf, "error": repr(error)}
                continue
            diff = float(np.linalg.norm(fit.template() - tm))
            e["G3_rel_err"] = rel_err(diff, float(np.linalg.norm(tm)))
            if full:
                e["X_lin_OLS"] = xlin(subj, fit)
                if name in TUNED:
                    e["theta"] = fit.theta
                    e["X_tune_OLS"] = xtune(subj, fit, rows, name, rec)
        out["est"][name] = e
    return out


def window_mask(subj: Subject, fit: Fit) -> NDArray:
    rows = np.zeros(len(subj.y), dtype=bool)
    for i in fit.ix:
        rows[i - HALF : i + HALF] = True
    return rows


def xlin(subj: Subject, fit: Fit) -> list[float]:
    """X_lin[c] = sum_ch trace_c(Q_m D_sel^T Sigma_ch D Q_R), from the rows of
    Sigma_ch D on the selected windows (within blocks)."""
    sel = np.flatnonzero(window_mask(subj, fit))
    dsel_s = analysis_tools.dense_rows(fit.Dsel, sel)  # |S| x P
    x = np.zeros(2)
    for a_, b_ in zip(subj.edges[:-1], subj.edges[1:]):
        sb = sel[(sel >= a_) & (sel < b_)]
        if sb.size == 0:
            continue
        lo, hi = max(a_, sb.min() - LAG), min(b_, sb.max() + LAG + 1)
        dreg = analysis_tools.dense_rows(subj.D, slice(lo, hi))
        for c in range(C):
            # 'same'-aligned convolution over the region; rows of sb are at least L from
            # a region edge unless that edge is the block edge (covariance truncated).
            sd = scipy.signal.fftconvolve(
                dreg, subj.kernel[:, c][:, None], mode="same", axes=0
            )
            prod = fit.Q @ (dsel_s[np.isin(sel, sb)].T @ sd[sb - lo]) @ subj.QR
            for k in range(2):
                x[k] += class_trace(prod, k)
    return x.tolist()


def tune_entry(
    name: str,
    k: int,
    kappa: float,
    tp: list[float],
    tm: list[float],
    h: float,
    ctx: Mapping[str, Any],
) -> dict[str, Any]:
    """One central difference with refits *tp*, *tm*: 0 at a lambda = inf base whose
    refits both stay inf, unquantified for nonfinite refits and branch switches."""
    up, um = coord(name, tp), coord(name, tm)
    entry: dict[str, Any] = {"h": h, "theta_plus": tp, "theta_minus": tm}
    if ctx["base_inf"]:
        if tp[0] == math.inf and tm[0] == math.inf:
            return entry | {"term": 0.0, "status": "boundary_locally_constant"}
        return entry | {"term": None, "status": "branch_switch_at_boundary"}
    if not (math.isfinite(up[k]) and math.isfinite(um[k])):
        return entry | {"term": None, "status": "nonfinite_refit"}
    jump = False
    if kappa == KAPPAS[0]:
        kern, base_u = ctx["kern"], ctx["base_u"]
        if kern is not None:
            moved = max(
                np.linalg.norm(np.array([t[0], t[1], t[0]]) - kern) for t in (tp, tm)
            )
            jump = moved > 0.1 * np.linalg.norm(kern)
        else:
            jump = max(abs(up[k] - base_u[k]), abs(um[k] - base_u[k])) > 1.0
    if jump:
        return entry | {"term": None, "status": "branch_switch"}
    return entry | {"term": (up[k] - um[k]) / (2 * h), "status": "ok"}


def tune_part(
    subj: Subject,
    fit: Fit,
    rows: NDArray,
    name: str,
    cls: int,
    k: int,
    kappa: float,
    ctx: Mapping[str, Any],
) -> dict[str, Any]:
    """One tuning coordinate's term for one class and step size."""
    g = np.zeros((P, C))
    g[cls * T : (cls + 1) * T] = fit.derivs[k][cls * T : (cls + 1) * T]
    if not ctx["base_inf"] and not np.any(g):
        return {"term": 0.0, "status": "zero_derivative"}
    if ctx["base_inf"]:
        # The direction along dT/dlambda is zero at the boundary; probe local constancy
        # along Sigma_joint D Q_R applied to the class block of the template itself.
        g[cls * T : (cls + 1) * T] = fit.template().reshape(P, C)[
            cls * T : (cls + 1) * T
        ]
    v = subj.joint_apply(np.asarray(subj.D @ (subj.QR @ g)))
    vrms = float(np.sqrt(np.mean(v[ctx["win"]] ** 2)))
    if not vrms > 0:
        return {"term": None, "status": "zero_probe_rms"}
    h = kappa * ctx["yrms"] / vrms
    try:
        tp = refit_theta(subj, rows, name, subj.data_v + (h * v) / UV)
        tm = refit_theta(subj, rows, name, subj.data_v - (h * v) / UV)
    except analysis_tools.INFRASTRUCTURE_ERRORS:
        raise
    except Exception as err:  # a numerical refit failure leaves the term unquantified
        return {"term": None, "status": "refit_exception", "error": repr(err)[:300]}
    return tune_entry(name, k, kappa, tp, tm, h, ctx)


def xtune(
    subj: Subject, fit: Fit, rows: NDArray, name: str, rec: Mapping[str, Any]
) -> dict[str, Any]:
    """X_tune: central differences of the tuning coordinates along
    Sigma_joint D Q_R dT_c/du_k (zero derivative 0; lambda = inf base 0 if both
    refits stay inf; nonfinite refits and branch switches unquantified)."""
    base_theta = theta_of(name, rec["inner"])
    win = window_mask(subj, fit)
    ctx = {
        "base_u": coord(name, base_theta),
        "kern": (
            np.asarray(rec["kernel"], dtype=float) if name == "SURE(kernel)" else None
        ),
        "win": win,
        "yrms": float(np.sqrt(np.mean(subj.y[win] ** 2))),
        "base_inf": name != "SURE(kernel)" and base_theta[0] == math.inf,
    }
    res: dict[str, Any] = {"base_theta": base_theta, "per_class": {}}
    for cls in range(2):
        res["per_class"][str(cls)] = {}
        for kappa in KAPPAS:
            parts = [
                tune_part(subj, fit, rows, name, cls, k, kappa, ctx)
                for k in range(len(fit.derivs))
            ]
            terms = [p["term"] for p in parts]
            total = None if None in terms else sum(terms, 0.0)
            res["per_class"][str(cls)][str(kappa)] = {"X_tune": total, "parts": parts}
    return res


# ---------------------------------------------------------------------------
# Noise-corrected distance: gates
# ---------------------------------------------------------------------------


def g6_checks(subj: Subject, s: int) -> dict[str, float | str]:
    """G6: the joint covariance is Hermitian PSD with the marginal diagonal, and its
    apply agrees with the per-channel apply and with a brute-force sum."""
    g, diag_rel = subj.joint_kernel()
    lags = np.arange(-LAG, LAG + 1)
    freqs = np.arange(1024) / 1024
    spec = np.einsum("kij,fk->fij", g, np.exp(-2j * np.pi * np.outer(freqs, lags)))
    herm = np.max(np.abs(spec - np.conj(np.transpose(spec, (0, 2, 1)))))
    ev = np.linalg.eigvalsh(spec)
    out: dict[str, float | str] = {
        "G6_hermitian_rel": float(herm / np.max(np.abs(spec))),
        "G6_min_over_max_eig": float(ev.min() / ev.max()),
        "G6_diag_rel_before_copy": diag_rel,
    }
    u = np.random.default_rng(s).standard_normal(subj.y.shape)
    saved = subj.joint_fft
    gd = np.zeros_like(g)
    for c in range(C):
        gd[:, c, c] = g[:, c, c]
    subj.joint_fft = subj.fft_of(gd)
    try:
        a1 = subj.joint_apply(u)
    finally:
        subj.joint_fft = saved
    a2 = np.zeros_like(u)
    for a_, b_ in zip(subj.edges[:-1], subj.edges[1:]):
        for c in range(C):
            a2[a_:b_, c] = scipy.signal.fftconvolve(
                u[a_:b_, c], subj.kernel[:, c], mode="same"
            )
    out["joint_apply_diag_vs_marginal_rel"] = float(
        np.linalg.norm(a1 - a2) / np.linalg.norm(a2)
    )
    # The brute-force sum needs LAG samples on both sides in block 0; a block too
    # short to probe fails G6 instead of stopping the run.
    lo, hi = subj.edges[0] + LAG, subj.edges[1] - LAG
    if lo >= hi:
        out["joint_apply_bruteforce_rel"] = math.inf
        out["G6_probe_unavailable"] = (
            f"subject {s}: first block [{subj.edges[0]}, {subj.edges[1]}) is shorter "
            f"than 2 * {LAG} + 1 samples"
        )
        return out
    t0, i = min(max(5000, lo), hi - 1), 3
    jt = subj.joint_apply(u)
    bf = sum(g[k + LAG, i, j] * u[t0 + k, j] for j in range(C) for k in lags)
    out["joint_apply_bruteforce_rel"] = float(abs(jt[t0, i] - bf) / abs(bf))
    return out


def gate_sigma2(
    run: str, s: int, pre: Mapping[str, Any], ols: NDArray
) -> dict[str, Any]:
    """G1: sigma2 recomputed from the run's cache equals the preflight's; G6."""
    subj = get_subject(run, s)
    g = analysis_tools.noise_gram(subj.D, subj.edges, subj.kernel)
    out: dict[str, Any] = {}
    for name, q in (("OLS", subj.QR), ("Average", subj.QA)):
        val = analysis_tools.class_noise_power(q, g, T)
        ref = np.asarray(pre[f"max_lag_{LAG}"][name]["colored_uv_per_class"]) ** 2
        out[f"{name}_lag{LAG}"] = {
            "recomputed": val.tolist(),
            "preflight": ref.tolist(),
            "max_rel": float(np.max(np.abs(val - ref) / ref)),
        }
    out["ref_norm_rel"] = float(
        np.linalg.norm(subj.R["OLS"] - ols) / np.linalg.norm(ols)
    )
    return out | g6_checks(subj, s)


def gate_population(run: str, s: int) -> list[float]:
    """G2, Population part: recomputed Population distances vs run_p300's records."""
    subj, rec = get_subject(run, s), fit_records(run).get(("P", s))
    if rec is None or subj.pop is None or rec["status"] != "success":
        return []
    rel = []
    for ref in ("OLS", "Average"):
        dist = np.mean([np.linalg.norm(subj.pop[c] - subj.R[ref][c]) for c in range(2)])
        recorded = rec["distances"][ref] * UV
        rel.append(rel_err(abs(float(dist) - recorded), recorded))
    return rel


def gate5(run: str, k: int, s: int, r: int, name: str) -> dict[str, Any]:
    """G5: refitting the unperturbed data reproduces the recorded tuning exactly."""
    rec, subj = fit_record(run, k, s, r, name), get_subject(run, s)
    if rec["status"] != "success":
        return {"s": s, "r": r, "name": name, "equal": None, "skipped": "failed fit"}
    try:
        th = refit_theta(subj, selection(run, k, s, r), name, subj.data_v)
    except analysis_tools.INFRASTRUCTURE_ERRORS:
        raise
    except Exception as error:  # a refit failure fails G5 (the K becomes unavailable)
        return {"s": s, "r": r, "name": name, "equal": False, "error": repr(error)}
    base = theta_of(name, rec["inner"])
    return {"s": s, "r": r, "name": name, "equal": th == base, "refit": th} | {
        "journal": base
    }


def circulant_noise(subj: Subject, rng: np.random.Generator, check: bool) -> NDArray:
    """Gaussian noise with the tapered Toeplitz covariance per channel and block
    (circulant embedding; exact for the within-block Toeplitz covariance when the
    embedding is PSD)."""
    noise = np.zeros_like(subj.y)
    for a_, b_ in zip(subj.edges[:-1], subj.edges[1:]):
        n = b_ - a_
        m = 1 << int(np.ceil(np.log2(2 * n)))
        for c in range(C):
            k = subj.kernel[LAG:, c]  # lags 0..L
            col = np.zeros(m)
            col[: LAG + 1] = k
            col[m - LAG :] = k[1:][::-1]
            lam = np.fft.fft(col).real
            if check and lam.min() < -1e-8 * lam.max():
                raise ValueError(
                    f"subject {subj.s} channel {c}: circulant embedding not PSD "
                    f"(min eigenvalue {lam.min():.3g}, max {lam.max():.3g})"
                )
            lam = np.clip(lam, 0, None)
            z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
            noise[a_:b_, c] = np.fft.fft(np.sqrt(lam / m) * z).real[:n]
    return noise


def gate4(run: str, k: int) -> dict[str, Any]:
    """G4: Monte Carlo check of X_lin (subject 1, repetition 0, two anchors)."""
    subj = get_subject(run, 1)
    recs = {n: fit_record(run, k, 1, 0, n) for n in ("Average", "SURE(kernel)")}
    failed = [n for n, rec in recs.items() if rec["status"] != "success"]
    if failed:
        return {
            n: {"pass": False, "error": "anchor fit failed (excluded)"} for n in failed
        }
    rng = np.random.default_rng(G4_SEED)
    try:
        fits = {n: Fit(subj, selection(run, k, 1, 0), n, r, k) for n, r in recs.items()}
    except (ValueError, np.linalg.LinAlgError) as error:  # no operator: G4 fails
        return {n: {"pass": False, "error": repr(error)} for n in recs}
    samples = {n: np.zeros((G4_DRAWS, 2)) for n in fits}
    for i in range(G4_DRAWS):
        try:
            noise = circulant_noise(subj, rng, check=i == 0)
        except ValueError as error:  # the noise model cannot be simulated: G4 fails
            return {n: {"pass": False, "error": str(error)} for n in fits}
        e_r = (subj.QR @ np.asarray(subj.D.T @ noise)).reshape(2, T, C)
        for name, f in fits.items():
            em = (f.Q @ np.asarray(f.Dsel.T @ noise)).reshape(2, T, C)
            samples[name][i] = [float(np.sum(em[c] * e_r[c])) for c in range(2)]
    out = {}
    for name, f in fits.items():
        analytic = xlin(subj, f)
        mc = samples[name].mean(0)
        se = samples[name].std(0, ddof=1) / np.sqrt(G4_DRAWS)
        z = (np.array(analytic) - mc) / se
        out[name] = {"analytic": analytic, "mc_mean": mc.tolist(), "mc_se": se.tolist()}
        out[name] |= {"z": z.tolist(), "pass": bool(np.all(np.abs(z) <= G4_Z))}
    return out


# ---------------------------------------------------------------------------
# Noise-corrected distance: summary of one K and the runs
# ---------------------------------------------------------------------------


def ranking(means: Mapping[str, float | None]) -> list[list[str]]:
    """Ascending; names whose means differ by <= 1e-9 are ties (None is left out)."""
    present = {m: v for m, v in means.items() if v is not None}
    out: list[list[str]] = []
    for m in sorted(present, key=lambda m: present[m]):
        if out and abs(present[m] - present[out[-1][0]]) <= 1e-9:
            out[-1].append(m)
        else:
            out.append([m])
    return out


def sigma2_arrays(pre: Pre, subjects: Sequence[int]) -> tuple[NDArray, NDArray]:
    """sigma2 per subject and class (model noise) and the SD of the realized squared
    norms of the shifted-residual refits."""
    mn = [pre[s]["model_noise"][f"max_lag_{LAG}"]["OLS"] for s in subjects]
    shift = [pre[s]["shift_noise"]["OLS"]["norms_per_shift"] for s in subjects]
    return np.array([np.square(x["colored_uv_per_class"]) for x in mn]), np.array(
        [np.std(np.square(x), axis=0, ddof=1) for x in shift]
    )


def cross_terms(
    by: Mapping[tuple[int, int], Mapping[str, Any]],
    subjects: Sequence[int],
    m: str,
    d2: NDArray,
    xl: NDArray,
    sig: NDArray,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """X_lin and X_tune summaries of estimator *m*, and its NC adjusted by both terms
    per step size (unavailable when a non-failed cell is unquantified)."""
    reps = d2.shape[1]
    ratio = 2 * xl / sig[:, None, :]
    some = bool((~np.isnan(xl)).any())  # else every fit failed: None
    q: dict[str, Any] = {
        "Xlin_mean": float(np.nanmean(xl)) if some else None,
        "2Xlin_over_sigma2_median": float(np.nanmedian(ratio)) if some else None,
        "Xlin_mean_by_class": np.nanmean(xl, axis=(0, 1)).tolist() if some else None,
    }
    smooth = {}
    for kappa in ("0.001", "0.01"):
        xt, status = np.zeros((len(subjects), reps, 2)), Counter[str]()
        for i, s in enumerate(subjects):
            for r in range(reps):
                for c in range(2):
                    if m not in TUNED:
                        status["no_smooth_tuning_term"] += 1
                        continue
                    pk = by[s, r]["est"][m]["X_tune_OLS"]["per_class"][str(c)][kappa]
                    for p in pk["parts"]:
                        status[p["status"]] += 1
                    xt[i, r, c] = np.nan if pk["X_tune"] is None else pk["X_tune"]
        ok = ~np.isnan(xt)
        q[f"Xtune_{kappa}_n_quantified"] = int(ok.sum())
        q[f"Xtune_{kappa}_mean_quantified"] = float(xt[ok].mean()) if ok.any() else None
        q[f"status_{kappa}"] = dict(status)
        failed = np.isnan(d2)  # an excluded fit, not an unquantified cell
        if (np.isnan(xt) & ~failed).any():
            n = int((np.isnan(xt) & ~failed).sum())
            smooth[kappa] = {"available": False, "n_unquantified_cells": n}
        else:
            extra = 2 * (xl + np.where(failed, 0.0, xt))
            smooth[kappa] = {"available": True} | analysis_tools.nc_metric(
                d2, sig, extra
            )
    return q, smooth


def population_d2(run: str, subjects: Sequence[int]) -> NDArray | None:
    """Squared class distances of the Population templates to OLS, (S, 1, 2); None
    when a subject has no Population template."""
    subj = [get_subject(run, s) for s in subjects]
    if any(x.pop is None for x in subj):
        return None
    d2 = [
        [
            float(np.sum((cast(NDArray, x.pop)[c] - x.R["OLS"][c]) ** 2))
            for c in range(2)
        ]
        for x in subj
    ]
    return np.array(d2)[:, None, :]


def rankings(res: dict[str, Any], unc: Mapping[str, NDArray], names: list[str]) -> None:
    """Uncorrected cohort means (of subject means over included fits) and rankings."""
    res["uncorrected_mean"] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an all-NaN row is NaN
        for m in names:
            per = np.nanmean(unc[m], axis=1)
            res["uncorrected_mean"][m] = (
                None if np.isnan(per).all() else float(np.nanmean(per))
            )
    res["ranking_NC_all"] = ranking({m: res["NC"][m]["mean"] for m in names})
    res["ranking_NC_core"] = ranking({m: res["NC"][m]["mean"] for m in CORE})
    res["ranking_uncorrected_all"] = ranking(res["uncorrected_mean"])
    res["ranking_NCadj_lin_core"] = ranking(
        {m: res["NCadj_lin"][m]["mean"] for m in CORE}
    )


def nc_results(
    run: str, units: Units, pre: Pre, subjects: Sequence[int], reps: int
) -> dict[str, Any]:
    """NC, its cross terms and the cross-term-adjusted NC of one K from its units."""
    metric = analysis_tools.nc_metric
    sig, shift_sd = sigma2_arrays(pre, subjects)
    by = {(u["subject"], u["repetition"]): u for u in units}

    def grid(m: str, field: str) -> NDArray:
        return np.array(
            [[by[s, r]["est"][m][field] for r in range(reps)] for s in subjects]
        )

    d2 = {m: grid(m, "d2_OLS") for m in EST}
    unc = {m: grid(m, "journal_distance_OLS_uV") for m in EST}
    pop = population_d2(run, subjects)
    names = EST if pop is None else ALL
    if pop is not None:
        d2[protocol.POPULATION] = pop
        unc[protocol.POPULATION] = np.sqrt(pop).mean(axis=2)
    xl = {m: grid(m, "X_lin_OLS") for m in CORE}
    nc = {m: metric(d2[m], sig) for m in names}
    res: dict[str, Any] = {
        "sigma2": sig.tolist(),
        "shift_sd_sq_norm": shift_sd.tolist(),
        "NC": nc,
        "NCadj_lin": {m: metric(d2[m], sig, 2 * xl[m]) for m in CORE},
        "cross_terms": {},
        "NCadj_smooth": {},
    }
    rankings(res, unc, names)
    for m in CORE:
        res["cross_terms"][m], smooth = cross_terms(by, subjects, m, d2[m], xl[m], sig)
        for kappa, adj in smooth.items():
            res["NCadj_smooth"].setdefault(kappa, {})[m] = adj
    keys: dict[str, list[str]] = defaultdict(list)  # the excluded fits
    for m, e in ((m, e) for u in units for m, e in u["est"].items()):
        if e.get("failed_fit"):
            keys[m].append(e["key"])
    for m in names:
        res["NC"][m]["failed_fits"] = keys[m]
    for m in CORE:
        res["NCadj_lin"][m]["failed_fits"] = keys[m]
    return res


def work(task: tuple[str, tuple[Any, ...]]) -> Any:  # noqa: ANN401
    kind, args = task
    calls = {
        "preflight": preflight_subject,
        "sigma2": gate_sigma2,
        "population": gate_population,
        "gate5": gate5,
        "gate4": gate4,
        "unit": unit,
        "inf": inf_unit,
    }
    return calls[kind](*args)


def shared_gates(
    run: str, subjects: Sequence[int], pre: Pre, refs: Pre, workers: int
) -> dict[str, Any]:
    """G1 per subject, G6 and the Population part of G2."""
    tasks = [
        ("sigma2", (run, s, pre[s]["model_noise"], refs[s]["OLS"])) for s in subjects
    ]
    done = analysis_tools.run_pool(workers, work, tasks)
    pop = [x for s in subjects for x in work(("population", (run, s)))]
    g1 = dict(zip(subjects, done))
    out: dict[str, Any] = {
        "G1": g1,
        "G6_min_over_max_eig": min(v["G6_min_over_max_eig"] for v in done),
        "G6_diag_rel_before_copy_max": max(v["G6_diag_rel_before_copy"] for v in done),
        "joint_apply_checks_max_rel": max(
            max(v["joint_apply_diag_vs_marginal_rel"], v["joint_apply_bruteforce_rel"])
            for v in done
        ),
        "G6_hermitian_max_rel": max(v["G6_hermitian_rel"] for v in done),
        "G2_population_max_rel": max(pop, default=0.0),
    }
    out["G6_pass"] = (
        out["G6_min_over_max_eig"] >= -1e-10
        and out["joint_apply_checks_max_rel"] <= 1e-8
        and out["G6_hermitian_max_rel"] <= 1e-10
        and out["G6_diag_rel_before_copy_max"] <= 1e-10
    )
    return out


def k_gates(
    k: int,
    units: Units,
    g5: list[dict[str, Any]],
    g4: dict[str, Any],
    shared: Mapping[str, Any],
) -> dict[str, Any]:
    """The gates of one K from its gate units, G5 refits, G4 and the shared gates."""
    g3: dict[str, float] = {}  # G3: the operators reproduce the saved templates
    for u in units:
        for n, e in u["est"].items():
            if "G3_rel_err" in e:
                g3[n] = max(g3.get(n, 0.0), e["G3_rel_err"])
    g1 = shared["G1"].values()
    g1_max = max(v[key]["max_rel"] for v in g1 for key in v if isinstance(v[key], dict))
    g1_ref = max(v["ref_norm_rel"] for v in g1)
    g2 = max(  # G2: the recorded distances recomputed
        (
            rel_err(
                abs(
                    float(np.mean(np.sqrt(e[f"d2_{r}"])))
                    - e[f"journal_distance_{r}_uV"]
                ),
                e[f"journal_distance_{r}_uV"],
            )
            for u in units
            for e in u["est"].values()
            if not e.get("failed_fit")
            for r in ("OLS", "Average")
        ),
        default=0.0,
    )
    gates: dict[str, Any] = {
        "K": k,
        "G1_max_rel": g1_max,
        "G1_ref_norm_max_rel": g1_ref,
        "G1_pass": g1_max <= G1_TOL and g1_ref <= G1_REF_TOL,
        **{key: shared[key] for key in shared if key.startswith(("G6", "joint"))},
        "G2_population_max_rel": shared["G2_population_max_rel"],
        "G2_max_rel": g2,
        "G3_max_rel_by_estimator": g3,
        "G5": g5,
        "G5_pass": all(x["equal"] for x in g5 if not x.get("skipped")),
        "G2_pass": g2 <= G2_TOL and shared["G2_population_max_rel"] <= G2_TOL,
        "G3_pass": all(x <= G3_TOL for x in g3.values()),
        "G4": g4,
        "G4_pass": all(v["pass"] for v in g4.values()),
    }
    return gates


def unavailable(reason: str, failed: list[str]) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason, "failed_gates": failed}


def run_k(
    run: str,
    k: int,
    subjects: Sequence[int],
    reps: int,
    pre: Pre,
    shared: Mapping[str, Any],
    workers: int,
) -> dict[str, Any]:
    """Gates of one K, then (when all pass) its units and summary; for an unavailable K
    the gate units (squared distances, operator errors) are kept."""
    cells = [(s, r) for s in subjects for r in range(reps)]
    tasks = [("unit", (run, k, s, r, False)) for s, r in cells]
    out: dict[str, Any] = {"units": analysis_tools.run_pool(workers, work, tasks)}
    failed_shared = [] if shared["G6_pass"] else ["G6_pass"]
    if shared["G2_population_max_rel"] > G2_TOL:
        failed_shared.append("G2_population")
    if failed_shared:
        out["status"] = unavailable("shared gate failed", failed_shared)
        return out
    g5_reps = sorted({0, min(10, reps - 1)})
    tasks = [
        ("gate5", (run, k, s, r, n)) for n in TUNED for s in subjects for r in g5_reps
    ]
    g4_task = [("gate4", (run, k))] if 1 in subjects else []  # anchor: subject 1
    g5 = analysis_tools.run_pool(workers, work, [*g4_task, *tasks])
    g4 = g5.pop(0) if g4_task else {"subject_1": {"pass": False, "error": "absent"}}
    gates = k_gates(k, out["units"], g5, g4, shared)
    out["gates"] = gates
    failed = [key for key in GATE_KEYS if not gates[key]]
    if failed:
        out["status"] = unavailable("gate failed", failed)
        return out
    tasks = [("unit", (run, k, s, r, True)) for s, r in cells]
    out["units"] = analysis_tools.run_pool(workers, work, tasks)
    out["status"] = {"status": "complete"}
    out["results"] = nc_results(run, out["units"], pre, subjects, reps)
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


PER_FIT = (
    *("K", "subject", "repetition", "estimator", "failed_fit", "key", "failure_type"),
    *("d2_OLS", "d2_Average", "journal_distance_OLS_uV", "journal_distance_Average_uV"),
    *("G3_rel_err", "X_lin_OLS", "theta", "X_tune_OLS", "flags"),
)


def csv_cell(v: object) -> object:
    if v is None:
        return ""
    v = analysis_tools.plain(v)
    return json.dumps(v) if isinstance(v, (list, dict)) else v


def preflight_stage(
    run: str, subjects: Sequence[int], workers: int, out: Path
) -> tuple[dict[int, Any], dict[int, dict[str, NDArray]], dict[int, str]]:
    """The preflight of every subject (references saved to ``references_uV.npz``) and
    the subjects excluded from the NC with the reason (failed references)."""
    tasks = [("preflight", (Path(run), s)) for s in subjects]
    done = analysis_tools.run_pool(min(workers, len(subjects)), work, tasks)
    pre = {s: r for s, (r, _) in zip(subjects, done)}
    refs = {s: t for s, (_, t) in zip(subjects, done) if t is not None}
    excluded = {s: pre[s]["status"] for s in subjects if s not in refs}
    for s in refs:  # NC needs run_p300's OLS and Average references (failure policy)
        with np.load(Path(run) / f"references/subject_{s:02d}.npz") as z:
            if not {"OLS", "Average"} <= set(z.files):
                excluded[s] = f"run_p300 references of subject {s}: {z.files}"
    arrays = {f"{n}_s{s:02d}": t[n] for s, t in refs.items() for n in t}
    np.savez(out / "references_uV.npz", **arrays)
    return pre, refs, excluded


def write_per_fit(
    path: Path, ks: Mapping[int, Mapping[str, Any]], extra: Units = ()
) -> None:
    """One row per fit: the units of every K (gated, or gate units), then *extra*."""
    rows = [
        {"K": k, "subject": u["subject"], "repetition": u["repetition"], "estimator": n}
        | e
        for k, res in ks.items()
        for u in res["units"]
        for n, e in u["est"].items()
    ]
    rows += extra
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PER_FIT, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({c: csv_cell(row.get(c)) for c in PER_FIT} for row in rows)


# ---------------------------------------------------------------------------
# Baselines and noise-corrected distances (response letter), squared domain
# (supplement: why a trivial template can be closer) and facts
# ---------------------------------------------------------------------------


def inf_unit(run: str, k: int, s: int, r: int) -> dict[str, Any]:
    """The saved lambda = inf template: squared distances to OLS and X_lin."""
    subj, rec = get_subject(run, s), fit_record(run, k, s, r, "lambda=inf")
    out = {"K": k, "subject": s, "repetition": r, "estimator": "lambda=inf"}
    if rec["status"] != "success":
        return out | {"failed_fit": True, "key": rec["key"]}
    with np.load(Path(run) / rec["template_file"]) as z:
        arr = z["templates"] * UV
    fit = Fit(subj, selection(run, k, s, r), "L-curve", {"lambda": math.inf}, k)
    d2 = [float(np.sum((arr[c] - subj.R["OLS"][c]) ** 2)) for c in range(2)]
    return out | {"d2_OLS": d2, "X_lin_OLS": xlin(subj, fit)}


def nc_of(
    d2: Sequence[float], sig: Sequence[float] | NDArray, x: Sequence[float] = (0, 0)
) -> tuple[float, int]:
    """NC of one fit (mean over classes) and its number of truncated classes."""
    d2c = [d2[c] - sig[c] + 2 * x[c] for c in range(2)]
    return float(np.mean([math.sqrt(max(v, 0.0)) for v in d2c])), sum(
        v < 0 for v in d2c
    )


Cell = dict[tuple[int, str, int], dict[str, list[float]]]


def baseline_cells(
    run: str,
    ks: Mapping[int, Mapping[str, Any]],
    inf: Units,
    sig: Mapping[int, NDArray],
) -> tuple[Cell, list[str], dict[int, list[float]]]:
    """Per (K, series, subject), the NC, truncation counts and (lambda = inf, zero,
    Population) NCadj_lin of every included fit; the series; the zero d2 per subject."""
    cell: Cell = defaultdict(lambda: defaultdict(list))
    for k, res in ks.items():
        for u in sorted(res["units"], key=lambda u: (u["subject"], u["repetition"])):
            for name, e in u["est"].items():
                if not e.get("failed_fit"):
                    value, tr = nc_of(e["d2_OLS"], sig[u["subject"]])
                    cell[k, name, u["subject"]]["NC"].append(value)
                    cell[k, name, u["subject"]]["trunc"].append(tr)
    for u in sorted(inf, key=lambda u: (u["K"], u["subject"], u["repetition"])):
        if u.get("failed_fit"):
            continue
        c, sg = cell[u["K"], "lambda=inf", u["subject"]], sig[u["subject"]]
        value, tr = nc_of(u["d2_OLS"], sg)
        c["NC"].append(value)
        c["trunc"].append(tr)
        c["NCadj_lin"].append(nc_of(u["d2_OLS"], sg, u["X_lin_OLS"])[0])
    names = ["zero", "lambda=inf", *EST]
    zero_d2: dict[int, list[float]] = {}
    for s in sig:
        subj = get_subject(run, s)
        zero_d2[s] = [float(np.sum(subj.R["OLS"][c] ** 2)) for c in range(2)]
        fixed = {"zero": zero_d2[s]}
        if subj.pop is not None:
            fixed[protocol.POPULATION] = [
                float(np.sum((subj.pop[c] - subj.R["OLS"][c]) ** 2)) for c in range(2)
            ]
        for name, d2 in fixed.items():
            value, tr = nc_of(d2, sig[s])
            for k in ks:
                cell[k, name, s] |= {"NC": [value], "trunc": [tr], "NCadj_lin": [value]}
    if all((KS[0], protocol.POPULATION, s) in cell for s in sig):
        names.append(protocol.POPULATION)
    return cell, names, zero_d2


def baselines(
    run: str,
    ks: Mapping[int, Mapping[str, Any]],
    inf: Units,
    sig: Mapping[int, NDArray],
) -> dict[str, Any]:
    """Per K, cohort means of the subject-mean NC of the estimators, zero,
    lambda = inf and Population with one formula, also for a K whose gated run is
    unavailable (descriptive); NCadj_lin of lambda = inf from its own X_lin, of the
    estimators from the gated run."""
    cell, names, zero_d2 = baseline_cells(run, ks, inf, sig)
    out: dict[str, Any] = {"K": {}, "first_K_below_zero": {}}
    for k, res in ks.items():
        out["K"][str(k)] = {}
        zero = {s: float(np.mean(cell[k, "zero", s]["NC"])) for s in sig}
        for n in names:
            cells = {s: cell[k, n, s] for s in sig if cell[k, n, s]["NC"]}
            means = {
                s: {key: float(np.mean(v)) for key, v in c.items() if key != "trunc"}
                for s, c in cells.items()
            }
            row: dict[str, Any] = {
                "NC": (
                    float(np.mean([m["NC"] for m in means.values()])) if means else None
                ),
                "subject": means,
                "trunc_cells": int(sum(sum(c["trunc"]) for c in cells.values())),
                "n_cells": int(2 * sum(len(c["trunc"]) for c in cells.values())),
                "n_subjects_below_zero_NC": sum(
                    m["NC"] < zero[s] for s, m in means.items()
                ),
            }
            if means and all("NCadj_lin" in m for m in means.values()):
                row["NCadj_lin"] = float(
                    np.mean([m["NCadj_lin"] for m in means.values()])
                )
            elif "results" in res and n in res["results"]["NCadj_lin"]:
                row["NCadj_lin"] = res["results"]["NCadj_lin"][n]["mean"]
            out["K"][str(k)][n] = row
    for n in names[1:]:
        nc = {k: out["K"][str(k)][n]["NC"] for k in ks}
        below = [
            k
            for k in ks
            if nc[k] is not None and nc[k] < out["K"][str(k)]["zero"]["NC"]
        ]
        out["first_K_below_zero"][n] = {"NC": below[0] if below else None}
        out["first_K_below_zero"][n]["NC_all_K_below"] = below
    return out | {"zero_d2_OLS": zero_d2}


def squared_domain(
    units: Units,
    inf: Units,
    zero_d2: Mapping[int, Sequence[float]],
    sig: Mapping[int, NDArray],
) -> dict[str, Any]:
    """K = 5 without truncation: ||R||^2 - sigma2 estimates ||T||^2 (zero template), and
    d2 - sigma2 + 2 X_lin estimates E||T_hat - T||^2 (linear part of X only)."""
    index = {s: i for i, s in enumerate(sig)}
    t2 = np.array([[zero_d2[s][c] - sig[s][c] for c in range(2)] for s in sig])
    out: dict[str, Any] = {"T_norm2": {CLS[c]: float(t2[:, c].mean()) for c in (0, 1)}}
    out["T_norm2"]["pooled"] = float(t2.mean())
    out["zero_truncated_cells"] = [
        [s, CLS[c]] for s in sig for c in range(2) if t2[index[s], c] < 0
    ]
    rows: dict[str, list[tuple[int, Sequence[float], Sequence[float]]]] = {
        n: [
            (u["subject"], u["est"][n]["d2_OLS"], u["est"][n]["X_lin_OLS"])
            for u in units
        ]
        for n in ("SURE(kernel)", "EmpBayes", "Average")
    }
    rows["lambda=inf"] = [
        (u["subject"], u["d2_OLS"], u["X_lin_OLS"])
        for u in inf
        if u["K"] == 5 and not u.get("failed_fit")
    ]
    out["err2"] = {}
    for n, entries in rows.items():
        acc, count = np.zeros((len(sig), 2)), np.zeros(len(sig))
        for s, d2, x in entries:
            if not np.isnan(d2).any():  # a failed fit (any class) is excluded
                acc[index[s]] += [d2[c] - sig[s][c] + 2 * x[c] for c in range(2)]
                count[index[s]] += 1
        if not count.any():
            out["err2"][n] = None
            continue
        e = acc[count > 0] / count[count > 0, None]
        out["err2"][n] = {CLS[c]: float(e[:, c].mean()) for c in range(2)}
        out["err2"][n]["pooled"] = float(e.mean())
        out["err2"][n]["n_subjects_err2_gt_T2_pooled"] = int(
            (e.mean(1) > t2[count > 0].mean(1)).sum()
        )
    pooled = {"T": out["T_norm2"]["pooled"]}
    pooled |= {n: v["pooled"] for n, v in out["err2"].items() if v}
    # A negative untruncated estimate has no root (signed value kept above).
    out["rms"] = {n: float(np.sqrt(v)) if v >= 0 else None for n, v in pooled.items()}
    return out


def shared_data_gap(units: Units) -> dict[str, Any]:
    """R2.9 (not in the final paper or supplement), K = 5: per subject, the mean over
    repetitions of the class sums of 2 (X_lin[Average] - X_lin[SURE(kernel)]), the
    linear shared-data correction of the gap, and of d2[Average] - d2[SURE(kernel)]
    to OLS, the total gap (uV^2)."""
    terms: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for u in units:
        a, k = u["est"]["Average"], u["est"]["SURE(kernel)"]
        if not (a.get("failed_fit") or k.get("failed_fit")):
            x = sum(2 * (a["X_lin_OLS"][c] - k["X_lin_OLS"][c]) for c in range(2))
            terms["shared_data_correction_avg_minus_sk_uv2"][u["subject"]].append(x)
            d = sum(a["d2_OLS"][c] - k["d2_OLS"][c] for c in range(2))
            terms["total_gap_avg_minus_sk_uv2"][u["subject"]].append(d)
    out = {}
    for key, per_subject in terms.items():
        per = {s: float(np.mean(v)) for s, v in per_subject.items()}
        out[key] = {"subject": per, "mean": float(np.mean(list(per.values())))}
        out[key] |= {"min": min(per.values()), "max": max(per.values())}
    return out


def facts(k5: Mapping[str, Any], sq: Mapping[str, Any]) -> dict[str, Any]:
    """Derived numbers of the trivial-template explanation (P300 results and the
    supplement) and of the noise-corrected distance (response letter)."""
    sig, sd = np.array(k5["sigma2"]), np.array(k5["shift_sd_sq_norm"])
    change = {
        m: k5["NCadj_lin"][m]["mean"] / k5["NC"][m]["mean"] - 1
        for m in CORE
        if k5["NC"][m]["mean"] and k5["NCadj_lin"][m]["mean"] is not None
    }
    return {
        "sigma2_min_max": [float(sig.min()), float(sig.max())],
        "shift_sd_sq_norm_min_max": [float(sd.min()), float(sd.max())],
        "NCadj_lin_over_NC_minus_1_core": change,
        "NCadj_lin_over_NC_minus_1_core_min_max": [
            min(change.values(), default=None),
            max(change.values(), default=None),
        ],
        "average_err2_over_T_norm2": (
            sq["err2"]["Average"]["pooled"] / sq["T_norm2"]["pooled"]
            if sq["err2"]["Average"] and sq["T_norm2"]["pooled"] > 0
            else None
        ),
        "zero_truncated_cells_by_class": {
            c: sum(x[1] == c for x in sq["zero_truncated_cells"]) for c in CLS
        },
    }


def baseline_summary(
    run: str,
    ks: Mapping[int, Mapping[str, Any]],
    pre: Pre,
    subjects: Sequence[int],
    reps: int,
    workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The baseline, squared-domain and facts parts of the summary, and the lambda = inf
    per-fit rows."""
    tasks = [("inf", (run, k, s, r)) for k in KS for s in subjects for r in range(reps)]
    inf = analysis_tools.run_pool(workers, work, tasks)
    sig = dict(zip(subjects, sigma2_arrays(pre, subjects)[0]))
    base = baselines(run, ks, inf, sig)
    zero_d2 = base.pop("zero_d2_OLS")
    k5 = ks[5].get("results")  # None when a gate failed at K = 5
    sq = squared_domain(ks[5]["units"], inf, zero_d2, sig) if k5 else None
    fact = None
    if k5 and sq:
        fact = facts(k5, sq) | shared_data_gap(ks[5]["units"])
    return {"baselines": base, "squared_domain": sq, "facts": fact}, inf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--p300", default="./runs/p300", help="run_p300.py output folder"
    )
    parser.add_argument("--out", default="./runs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--subjects", type=int, nargs="+", default=list(protocol.SUBJECTS)
    )
    parser.add_argument("--reps", type=int, default=heldout_p300.REPETITIONS)
    args = parser.parse_args()
    started = datetime.now(UTC)
    run, subjects, reps = str(Path(args.p300).resolve()), args.subjects, args.reps
    out = Path(args.out).resolve() / "reference_noise"
    out.mkdir(parents=True, exist_ok=True)
    pre, refs, excluded = preflight_stage(run, subjects, args.workers, out)
    subjects = [s for s in subjects if s not in excluded]
    if not subjects:
        raise RuntimeError(
            f"no subject left for the noise-corrected distance: {excluded}"
        )
    shared = shared_gates(run, subjects, pre, refs, args.workers)
    ks = {k: run_k(run, k, subjects, reps, pre, shared, args.workers) for k in KS}
    extra, inf = baseline_summary(run, ks, pre, subjects, reps, args.workers)
    write_per_fit(out / "per_fit.csv", ks, inf)
    nc = {
        str(k): {x: v for x, v in res.items() if x != "units"} for k, res in ks.items()
    }
    summary = {
        "settings": {"p300": run, "subjects": args.subjects, "repetitions": reps}
        | {"half_win": HALF, "guard": GUARD, "max_lag": LAG, "Ks": list(KS)},
        "preflight": {"subjects": pre, **preflight_summary(pre)},
        "excluded_subjects": excluded,
        "nc": {"shared_gates": shared, "K": nc},
    } | extra
    analysis_tools.write_json(out / "summary.json", summary)
    analysis_tools.write_run_info(out, vars(args), started)


if __name__ == "__main__":
    main()
