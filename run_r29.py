# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Post hoc, descriptive analyses for reviewer comment R2.9 on the saved P300 fits.

Reads a ``run_p300.py`` output and the raw BNCI2014-009 files (speller codes).
Amplitude value: the LLR L = s / (2 v) of each held-out fit's nearest-template
score s, v the variance of the unselected training events along the difference
template: information gain, recalibration, decisions, speller, shrinkage control.
Direction check: common/difference and direction/scale splits of the error to the
full-data references; held-out difference directions against the AUC.
Writes ``<out>/r29/``: ``per_fit.csv``, ``summary.json``, ``run_info.json``.
Usage: python3 -B run_r29.py [--p300 DIR] [--data-dir DIR] [--out DIR]
           [--workers N] [--subjects 1 2 ...] [--reps N]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, cast

import analysis_tools as at
import core
import heldout_p300
import numpy as np
import protocol
import scipy
from numpy.typing import ArrayLike, NDArray

UV = 1e6
HALF, GUARD = heldout_p300.HALF_WIN, heldout_p300.GUARD
FOLDS, N_CH, T_LEN = protocol.N_BLOCKS, protocol.N_CHANNELS, 2 * HALF
EST: tuple[str, ...] = protocol.ESTIMATORS
REFS: tuple[str, ...] = protocol.REFERENCES
LC, SHRINK, POP, BASE = at.R29_LC, at.R29_SHRINK, protocol.POPULATION, protocol.ANCHOR
MI: dict[str, int] = {m: i for i, m in enumerate(EST)}
N_EVENTS, N_TARGETS, NSEQ = 576, 96, at.R29_NSEQ
NAN = math.nan
NO_DIAG: dict[str, float] = dict.fromkeys(
    ["delta_ho", "log_rho_var", "abs_mid_err", "mid_err", "log_rho_amp", "log_scale"],
    NAN,
)
B_FIELDS: tuple[str, ...] = (
    "auc",
    "cos_allAverage",
    "cos_allOLS",
    "cos_fullOLS",
    "cos_fullAverage",
)
NOISE_LAG = 204
SENS_KS: tuple[int, ...] = (5, 10)  # overlap-excluded variance sensitivity
# The metrics a fit can leave undefined (counted per estimator for the caption).
UNDEFINED_KEYS: tuple[str, ...] = (
    *("log_rho_amp", "log_scale", "b_scale", "b_joint", "gain_offset_bits"),
    *("gain_scale_bits", "gain_joint_bits", "IG_recal_joint_bits", "delta_ho"),
)
Fits = dict[str, dict[str, dict[str, Any]]]  # unit -> estimator -> record


def unit_name(k: int | str, s: int, fold: int | None, rep: int) -> str:
    kn = "Kall" if k == "all" else f"K{int(k):02d}"
    tail = f"r{rep:02d}" if fold is None else f"f{fold}_r{rep:02d}"
    return f"{'A' if fold is None else 'B'}_{kn}_s{s:02d}_{tail}"


def read_fits(p300: Path) -> tuple[dict[int, Fits], dict[int, dict[int, float]]]:
    """Per subject, the estimator records of every unit; Population's AUC per fold."""
    fits: dict[int, Fits] = {}
    pop: dict[int, dict[int, float]] = {}
    with (p300 / "per_fit.csv").open() as stream:
        for row in csv.DictReader(stream):
            an, est, s = row["analysis"], row["estimator"], int(row["subject"])
            if an == "B" and est == POP and row["auc"]:  # a failed block has no AUC
                pop.setdefault(s, {})[int(row["fold"])] = float(row["auc"])
            if an not in {"A", "B"} or est not in EST:
                continue
            k = row["K"] if row["K"] == "all" else int(row["K"])
            fold = int(row["fold"]) if an == "B" else None
            unit = unit_name(k, s, fold, int(row["repetition"]))
            rec = {"status": row["status"], "key": unit.replace("_", "|") + f"|{est}"}
            rec["auc"] = float(row["auc"]) if row["auc"] else None
            rec["template_file"] = row["template"]
            fits.setdefault(s, {}).setdefault(unit, {})[est] = rec
    return fits, pop


def unit_records(fits: Fits, unit: str) -> dict[str, dict[str, Any]]:
    if sorted(fits.get(unit, {})) != sorted(EST):
        got = sorted(fits.get(unit, {}))
        raise RuntimeError(f"{unit}: run_p300 fits {got}, expected {sorted(EST)}")
    at.check_r29_failures(fits[unit], unit)
    return fits[unit]


def load_t(run: Path, rel: str, name: str = "templates") -> NDArray:
    with np.load(run / rel) as z:
        return z[name] * UV


def load_blocks(p300: Path, s: int) -> list[dict[str, NDArray]]:
    """Event windows (uV) and labels per block (G2: 576 events, 96 targets)."""
    blocks = []
    with np.load(p300 / f"cache/subject_{s:02d}.npz") as z:
        for b in range(FOLDS):
            data, on = z[f"data_{b}"], z[f"onsets_{b}"]
            elig = heldout_p300.eligible_events(on, len(data), HALF, GUARD)
            if not elig.all():
                raise RuntimeError(f"subject {s} block {b}: {(~elig).sum()} ineligible")
            wv = np.stack([data[o - HALF : o + HALF] for o in on])
            lab = z[f"labels_{b}"].astype(int)
            if len(lab) != N_EVENTS or int(lab.sum()) != N_TARGETS:
                raise RuntimeError(
                    f"G2 failed: subject {s} block {b}: {lab.sum()} targets of {len(lab)}"
                )
            nat = z[f"native_onsets_{b}"]
            blocks.append({"W": wv * UV, "on": on, "nat": nat, "lab": lab})
    return blocks


def speller_map(
    mat: Path, s: int, blocks: Sequence[Mapping[str, NDArray]]
) -> list[dict[str, NDArray]]:
    """Flash codes, character and sequence of every event, target row/column codes
    (G5: the raw file's codes and labels agree with the cached events)."""
    m = scipy.io.loadmat(mat, squeeze_me=True, struct_as_record=False)
    out = []
    for b, block in enumerate(blocks):
        where, el, lab = f"G5 failed: subject {s} block {b}", m["data"][b], block["lab"]
        ys, y = np.asarray(el.y_stim).astype(int), np.asarray(el.y).astype(int)
        on = np.flatnonzero((ys != 0) & (ys != np.r_[0, ys[:-1]]))
        if not np.array_equal(on, block["nat"]):
            raise RuntimeError(f"{where}: {len(on)} raw onsets differ from the cache")
        if not np.array_equal(y[on] - 1, lab):
            raise RuntimeError(f"{where}: raw labels differ from the cached labels")
        codes, tcs = ys[on], []
        for c in range(6):
            sl = slice(c * 96, (c + 1) * 96)
            for q in range(NSEQ):
                cs = sorted(codes[sl][q * 12 : (q + 1) * 12].tolist())
                if cs != list(range(1, 13)):
                    raise RuntimeError(f"{where}: character {c} sequence {q}: {cs}")
            tc = np.unique(codes[sl][lab[sl] == 1])
            if len(tc) != 2 or not (1 <= tc[0] <= 6 and 7 <= tc[1] <= 12):
                raise RuntimeError(f"{where}: character {c} target codes {tc}")
            if not np.array_equal(lab[sl] == 1, np.isin(codes[sl], tc)):
                raise RuntimeError(f"{where}: character {c} labels are not its codes")
            tcs.append(tc)
        idx, (trow, tcol) = np.arange(N_EVENTS), np.array(tcs).T
        mp = {"codes": codes, "char": idx // 96, "seq": (idx % 96) // 12}
        out.append(mp | {"trow": trow, "tcol": tcol})
    return out


def recalibrate(
    L: NDArray, y: NDArray, halves: tuple[NDArray, NDArray]
) -> dict[str, float | int]:
    """Cross-fitted offset, scale and joint recalibration: gain in bits per event."""
    raw = float(at.ll_terms(at.R29_KAPPA + L, y).mean())
    out: dict[str, float | int] = {}
    for variant in ("offset", "scale", "joint"):
        eta_cf, ok, rev, bs = np.empty_like(L), True, 0, []
        for fit, ev in ((halves[0], halves[1]), (halves[1], halves[0])):
            a, b, conv, rv, bdiag = at.fit_variant(variant, L[fit], y[fit])
            ok, rev = ok and conv, rev + rv
            bs.append(bdiag)
            eta_cf[ev] = at.R29_KAPPA + a + b * L[ev]
        gain = (raw - float(at.ll_terms(eta_cf, y).mean())) / at.R29_LN2
        out[f"gain_{variant}_bits"] = gain if ok else NAN
        out[f"fail_{variant}"] = int(not ok)
        out[f"rev_{variant}"] = rev
        out[f"b_{variant}"] = (
            float(np.mean(bs)) if ok and np.all(np.isfinite(bs)) else NAN
        )
    return out


def score_set(
    L: NDArray, y: NDArray, halves: tuple[NDArray, NDArray]
) -> dict[str, Any]:
    out: dict[str, Any] = dict(at.event_metrics(L, y))
    out.update(recalibrate(L, y, halves))
    out.update(at.decisions(L, y))
    out["IG_recal_joint_bits"] = out["IG_bits"] + out["gain_joint_bits"]
    return out


def scores(L: NDArray, fc: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """score_set and speller of LLRs L; all NaN when L is undefined (zero variance)."""
    if np.isfinite(L).all():
        return score_set(L, fc["y"], fc["halves"]), at.speller(L, fc["map"])
    keys = score_set(np.zeros_like(L), fc["y"], fc["halves"])
    nan8, stop = [NAN] * NSEQ, {str(t): [NAN] * 7 for t in at.R29_TAUS}
    sp = {"ll": nan8, "pmax": nan8, "acc": nan8, "stop": stop}
    return dict.fromkeys(keys, NAN), sp


def log_or_nan(q: float) -> float:
    """log(q), NaN where q is not positive (e.g. a ratio that underflowed to 0)."""
    return math.log(q) if q > 0 else NAN


def diagnostics(
    x: NDArray, y: NDArray, length: float, c_proj: float, v: float
) -> dict[str, float]:
    xt, xn = x[y == 1], x[y == 0]
    mt, mn = float(xt.mean()), float(xn.mean())
    delta, mid = mt - mn, 0.5 * (mt + mn)
    v_ho = float((np.sum((xt - mt) ** 2) + np.sum((xn - mn) ** 2)) / (len(x) - 2))
    sd = math.sqrt(v_ho) if v_ho > 0 else NAN  # zero variances: undefined (NaN)
    out = {
        "delta_ho": delta,
        "log_rho_var": log_or_nan(v / v_ho) if v_ho > 0 else NAN,
        "abs_mid_err": abs(c_proj - mid) / sd,
        "mid_err": (c_proj - mid) / sd,
        "log_rho_amp": NAN,
        "log_scale": NAN,
    }
    if delta > 0 and length > 0:
        out["log_rho_amp"] = log_or_nan(length / delta)
        out["log_scale"] = out["log_rho_amp"] - out["log_rho_var"]
    return out


def evaluate(
    T: NDArray, fc: Mapping[str, Any], pool: NDArray, pool_ov: NDArray | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The fit's record, and what the shrinkage control reuses."""
    W, y = fc["W_ho"], fc["y"]
    s = np.sum((W - T[0]) ** 2, axis=(1, 2)) - np.sum((W - T[1]) ** 2, axis=(1, 2))
    length = float(np.linalg.norm(T[1] - T[0]))
    rec: dict[str, Any] = {"auc": float(heldout_p300.auc(s, y)), "normD": length}
    rec["n_pool"] = int(pool.sum())
    ext: dict[str, Any] = {"length": length, "v": NAN, "L": np.zeros_like(s)}
    ext |= {"diag": dict(NO_DIAG), "xtr": None}
    if length > 0:
        D, Cm = T[1] - T[0], 0.5 * (T[0] + T[1])
        u = (D / length).ravel()
        xtr = fc["Wtr"] @ u
        v = float(np.var(xtr[pool], ddof=1))
        c_proj = float(Cm.ravel() @ u)
        rec.update(diagnostics(fc["W_ho_flat"] @ u, y, length, c_proj, v))
        L = s / (2 * v) if v > 0 else np.full_like(s, NAN)
        ext |= {"v": v, "L": L, "xtr": xtr}
        ext["diag"] = {k: rec[k] for k in NO_DIAG}
    else:
        rec.update(NO_DIAG)
    rec["v_hat"] = ext["v"]
    sc, sp = scores(ext["L"], fc)
    rec.update(sc)
    rec["sp"] = sp
    if pool_ov is not None:
        v_ov = float(np.var(ext["xtr"][pool_ov], ddof=1)) if length > 0 else NAN
        rec |= {"v_hat_ov": v_ov, "n_pool_ov": int(pool_ov.sum())}
        rec["v_ratio"] = v_ov / ext["v"] if ext["v"] > 0 else NAN
        prior = length == 0  # identical class templates predict the prior here too
        ext["L_ov"] = s / (2 * v_ov) if v_ov > 0 else (ext["L"] if prior else s * NAN)
        rec["ov"] = scores(ext["L_ov"], fc)[0]
    return rec, ext


def shrink(
    avg_rec: Mapping[str, Any],
    avg: Mapping[str, Any],
    fc: Mapping[str, Any],
    pools: tuple[NDArray, NDArray | None],
    K: int,
) -> dict[str, Any]:
    """Average's difference scaled by b = max(0, 1 - E_hat / ||D||^2)."""
    y, (pool, pool_ov) = fc["y"], pools
    rec: dict[str, Any] = {"n_pool": avg_rec["n_pool"], "v_hat": avg["v"]}

    def factor(mask: NDArray) -> float:
        e_hat = (2.0 / K) * float(np.sum(np.var(fc["Wtr"][mask], axis=0, ddof=1)))
        return max(0.0, 1.0 - e_hat / avg["length"] ** 2) if avg["length"] > 0 else 0.0

    b = factor(pool)
    L = b * avg["L"]
    rec |= {"b_shrink": b, "b_zero": int(b == 0.0)}  # noqa: RUF069 - exact by the clip
    rec |= {
        "auc": float(heldout_p300.auc(L, y)) if np.isfinite(L).all() else NAN,
        "normD": b * avg["length"],
    }
    d = dict(avg["diag"])
    if b > 0 and np.isfinite(d["log_rho_amp"]):
        d["log_rho_amp"] += math.log(b)
        d["log_scale"] = d["log_rho_amp"] - d["log_rho_var"]
    elif b == 0:  # noqa: RUF069 - exactly zero by the max(0, .) clip
        d = dict(NO_DIAG)
    rec.update(d)
    sc, sp = scores(L, fc)
    rec.update(sc)
    rec["sp"] = sp
    if pool_ov is not None and "L_ov" in avg:
        b_ov = factor(pool_ov)
        rec["b_shrink_ov"] = b_ov
        rec["ov"] = scores(b_ov * avg["L_ov"], fc)[0]
    return rec


def fold_context(
    blocks: Sequence[Mapping[str, NDArray]],
    maps: Sequence[Mapping[str, NDArray]],
    f: int,
) -> dict[str, Any]:
    trb = [b for b in range(FOLDS) if b != f]
    return {
        "W_ho": blocks[f]["W"],
        "W_ho_flat": blocks[f]["W"].reshape(N_EVENTS, -1),
        "y": blocks[f]["lab"],
        "Wtr": np.concatenate([blocks[b]["W"].reshape(N_EVENTS, -1) for b in trb]),
        "halves": tuple(np.isin(maps[f]["char"], h) for h in ((0, 2, 4), (1, 3, 5))),
        "map": maps[f],
        "trb": trb,
    }


def selection_pools(
    blocks: Sequence[Mapping[str, NDArray]],
    trb: list[int],
    rows: Sequence[Sequence[int]],
    K: int,
) -> tuple[NDArray, NDArray | None]:
    """Training events outside the selection, and (K = 5, 10) also outside its windows."""
    if len(rows) != 2 * K:
        raise RuntimeError(f"{len(rows)} selected events, expected {2 * K}")
    pool = np.ones(len(trb) * N_EVENTS, bool)
    pool_ov = pool.copy() if K in SENS_KS else None
    for b, e in rows:
        if b not in trb:
            raise RuntimeError(f"selected event {e} lies in held-out block {b}")
        i = trb.index(b)
        pool[i * N_EVENTS + e] = False
        if pool_ov is not None:
            near = np.abs(blocks[b]["on"] - blocks[b]["on"][e]) < 2 * HALF
            pool_ov[i * N_EVENTS : (i + 1) * N_EVENTS][near] = False
    return pool, pool_ov


def amp_records(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One subject: per fold, Population, then K, repetition, estimator."""
    p300, s, fits, sel = task["p300"], task["subject"], task["fits"], task["sel"]
    blocks = load_blocks(p300, s)
    maps = speller_map(task["mat_dir"] / f"A{s:02d}S.mat", s, blocks)
    pop_t = load_t(p300, f"population/subject_{s:02d}.npz")
    records: list[dict[str, Any]] = []
    for f in range(FOLDS):
        fc = fold_context(blocks, maps, f)
        rec, _ = evaluate(pop_t, fc, np.ones(len(fc["trb"]) * N_EVENTS, bool), None)
        records.append(rec | {"K": "pop", "subject": s, "fold": f, "rep": 0})
        records[-1]["method"] = POP
        for K in task["ks"]:
            for r in range(task["reps"]):
                jr = unit_records(fits, unit_name(K, s, f, r))
                rows = sel[str(K)][str(s)][str(f)][r]
                pools = selection_pools(blocks, fc["trb"], rows, K)
                for m in EST:
                    head = {"K": K, "subject": s, "fold": f, "rep": r, "method": m}
                    if jr[m]["status"] != "success":  # R1: excluded and counted
                        records.append(head | {"failed": jr[m]["key"]})
                        continue
                    T = load_t(p300, jr[m]["template_file"])
                    rec, ext = evaluate(T, fc, *pools)
                    records.append(rec | head)
                    if m == "Average":
                        sh = shrink(rec, ext, fc, pools, K)
                        records.append(sh | head | {"method": SHRINK})
    return records


def v_ratios(records: Sequence[Mapping[str, Any]], K: int) -> dict[str, dict[str, Any]]:
    """Overlap-excluded over primary variance proxy, per estimator."""
    vr = {}
    for m in EST:
        ok = [r for r in records if r["K"] == K and r["method"] == m]
        v = np.array([r["v_ratio"] for r in ok if "failed" not in r], dtype=float)
        v = v[np.isfinite(v)]
        stats = (np.median(v), v.min(), v.max()) if len(v) else (NAN,) * 3
        vr[m] = dict(zip(("median", "min", "max"), map(float, stats)))
        vr[m]["n"] = len(v)
    return vr


def amp_summary(
    records: Sequence[dict[str, Any]], subj: Sequence[int], ks: Sequence[int]
) -> dict[str, Any]:
    """Per K, metric and method: subject values, bootstrap mean, paired differences."""
    per, per_lc = at.r29_per_subject(records)
    out: dict[str, Any] = {}
    for K in ks:
        key = f"K{K:02d}"
        tab = {m: per.get((K, m), {}) for m in at.R29_METHODS if m != POP}
        tab[POP] = per.get(("pop", POP), {})
        lc_refs = {ref: per_lc.get((K, ref), {}) for ref, _ in at.R29_REFS_PAIRED}
        first = [next((v for v in t.values() if v), {}) for t in tab.values()]
        metrics = sorted(set().union(*[set(v) for v in first]))
        out[key] = {
            metric: at.r29_metric_entry(tab, lc_refs, metric, subj)
            for metric in metrics
        }
        if K in SENS_KS:
            out[key]["v_ratio_overlap_vs_primary"] = v_ratios(records, K)
    return out


def metric_fields(T: NDArray, refs: Mapping[str, NDArray]) -> dict[str, float]:
    """The six metrics against each reference, as `<metric>_<reference>`."""
    out: dict[str, float] = {}
    for ref in REFS:
        values = at.fit_metrics(T, refs[ref])
        out.update({f"{k}_{ref}": v for k, v in zip(at.R29_METRICS, values)})
    return out


def ols_residual(
    p300: Path, s: int
) -> tuple[scipy.sparse.csc_matrix, list[int], NDArray, NDArray]:
    """Full-data design of the joined recording (uV), block edges, the residual of
    the OLS reference, and the class labels of the eligible events."""
    blocks = at.load_blocks(p300 / f"cache/subject_{s:02d}.npz", s)
    lengths = [len(b.data) for b in blocks]
    starts = np.cumsum([0] + lengths[:-1]).astype(np.int64)
    edges = list(starts) + [int(sum(lengths))]
    y = np.vstack([b.data for b in blocks]) * UV
    rows = [
        (b.block, e) for b in blocks for e in np.flatnonzero(b.eligible(HALF, GUARD))
    ]
    ix = np.array([starts[b] + blocks[b].onsets[e] for b, e in rows], dtype=np.int64)
    lb = np.array([blocks[b].labels[e] for b, e in rows], dtype=np.int64)
    dm = core.get_design_matrix(
        y, ix, lb, HALF, events=np.array([0, 1]), sparse_out=True
    )
    design = scipy.sparse.csc_matrix(dm)
    ols = load_t(p300, f"references/subject_{s:02d}.npz", "OLS")
    return design, edges, y - design @ ols.reshape(2 * T_LEN, N_CH), lb


def noise_gram(
    design: scipy.sparse.csc_matrix, edges: list[int], resid: NDArray
) -> NDArray:
    """D^T Sigma D per channel, Sigma the lag-204 Bartlett-tapered autocovariance of
    the residual within blocks."""
    acov = np.zeros((NOISE_LAG + 1, N_CH))
    for a_, b_ in zip(edges[:-1], edges[1:]):
        r = resid[a_:b_]
        for k in range(NOISE_LAG + 1):
            acov[k] += np.sum(r[: len(r) - k] * r[k:], axis=0)
    acov /= len(resid)
    tapered = acov * (1 - np.arange(NOISE_LAG + 1) / (NOISE_LAG + 1))[:, None]
    kernel = np.concatenate([tapered[:0:-1], tapered])
    G = np.zeros((N_CH, 2 * T_LEN, 2 * T_LEN))
    for a_, b_ in zip(edges[:-1], edges[1:]):
        d = cast(scipy.sparse.csc_matrix, design[a_:b_]).toarray()
        for c in range(N_CH):
            G[c] += d.T @ scipy.signal.fftconvolve(
                d, kernel[:, c][:, None], mode="same", axes=0
            )
    return 0.5 * (G + G.transpose(0, 2, 1))


def noise_floor(p300: Path, s: int) -> dict[str, dict[str, float]]:
    """Expected squared norms of the OLS and Average reference noise in C and D:
    per channel Q D^T Sigma D Q^T ([0:102] NonTarget, [102:204] Target)."""
    design, edges, resid, lb = ols_residual(p300, s)
    G = noise_gram(design, edges, resid)
    counts = np.array([np.sum(lb == 0), np.sum(lb == 1)])
    out = {}
    for name, Q in (
        ("OLS", np.linalg.inv((design.T @ design).toarray())),
        ("Average", np.diag(np.repeat(1.0 / counts, T_LEN))),
    ):
        tnn = ttt = ttn = 0.0
        for c in range(N_CH):
            cov = Q @ G[c] @ Q.T
            tnn += float(np.trace(cov[:T_LEN, :T_LEN]))
            ttt += float(np.trace(cov[T_LEN:, T_LEN:]))
            ttn += float(np.trace(cov[T_LEN:, :T_LEN]))
        out[name] = {"sigma2_nontarget": tnn, "sigma2_target": ttt}
        out[name] |= {"cross_target_nontarget": ttn, "nu_D": ttt + tnn - 2 * ttn}
        out[name]["nu_C"] = 0.25 * (ttt + tnn + 2 * ttn)
    return out


def kall_record(
    task: Mapping[str, Any], f: int, d_pop: NDArray
) -> tuple[dict[str, Any], NDArray, NDArray]:
    """The all-training-events fits of one fold, and their Average and OLS differences."""
    s = task["subject"]
    krec = unit_records(task["fits"], unit_name("all", s, f, 0))
    if failed := {m: v["status"] for m, v in krec.items() if v["status"] != "success"}:
        raise RuntimeError(f"subject {s} fold {f}: all-events fits failed: {failed}")
    d_all = {
        m: at.template_parts(load_t(task["p300"], krec[m]["template_file"]))[1]
        for m in EST
    }
    d_avg, d_ols = d_all["Average"], d_all["OLS"]
    kall = {"kind": "Kall", "subject": s, "fold": f}
    kall |= {"auc": {m: krec[m]["auc"] for m in EST}}
    kall["cos_allOLS_allAverage"] = at.cosine(d_ols, d_avg)
    kall["cos_to_allAverage"] = {m: at.cosine(d_all[m], d_avg) for m in EST}
    kall["pop_auc"] = task["pop_auc"].get(f, NAN)  # NaN for a failed block
    kall["pop_cos_allAverage"] = at.cosine(d_pop, d_avg)
    kall["pop_cos_allOLS"] = at.cosine(d_pop, d_ols)
    return kall, d_avg, d_ols


def dir_records(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Noise floor, Population, A (whole recording), Kall and B (held out), full data."""
    p300, s, fits = task["p300"], task["subject"], task["fits"]
    refs = {r: load_t(p300, f"references/subject_{s:02d}.npz", r) for r in REFS}
    pop = load_t(p300, f"population/subject_{s:02d}.npz")
    out: list[dict[str, Any]] = [
        {"kind": "noise", "subject": s, **noise_floor(p300, s)},
        {"kind": "P", "subject": s, **metric_fields(pop, refs)},
    ]
    for K in task["ks"]:
        for r in range(task["reps"]):
            recs = unit_records(fits, unit_name(K, s, None, r))
            for m in EST:
                head = {"kind": "A", "K": K, "subject": s, "rep": r, "method": m}
                if recs[m]["status"] != "success":  # R1: excluded and counted
                    out.append(head | {"failed": recs[m]["key"]})
                    continue
                T = load_t(p300, recs[m]["template_file"])
                out.append(head | metric_fields(T, refs))
    d_pop = at.template_parts(pop)[1]
    d_full = {r: at.template_parts(refs[r])[1] for r in REFS}
    for f in range(FOLDS):
        kall, d_avg, d_ols = kall_record(task, f, d_pop)
        out.append(kall)
        directions = (d_avg, d_ols, d_full["OLS"], d_full["Average"])
        for K in task["ks"]:
            for r in range(task["reps"]):
                recs = unit_records(fits, unit_name(K, s, f, r))
                for m in EST:
                    head = {"kind": "B", "K": K, "subject": s, "fold": f, "rep": r}
                    head["method"] = m
                    if recs[m]["status"] != "success":  # R1: excluded and counted
                        out.append(head | {"failed": recs[m]["key"]})
                        continue
                    d = at.template_parts(load_t(p300, recs[m]["template_file"]))[1]
                    vals = [recs[m]["auc"]] + [at.cosine(d, x) for x in directions]
                    out.append(head | dict(zip(B_FIELDS, vals)))
    full = {"kind": "full", "subject": s}
    full["cos_fullOLS_fullAverage"] = at.cosine(d_full["OLS"], d_full["Average"])
    full["cos_fullEB_fullAverage"] = at.cosine(d_full["EmpBayes"], d_full["Average"])
    full["dref2"] = float(np.sum((refs["OLS"][1] - refs["OLS"][0]) ** 2))
    return out + [full]


def subject_task(
    task: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return amp_records(task), dir_records(task)


def per_fit_arrays(
    records: Sequence[Mapping[str, Any]], ks: Sequence[int], subj: list[int], reps: int
) -> tuple[NDArray, NDArray, NDArray]:
    """A (k, s, r, m, ref, metric), P (s, ref, metric), B (k, s, f, r, m, field);
    NaN at the excluded fits."""
    nm, nk, ns = len(at.R29_METRICS), len(ks), len(subj)
    A = np.full((nk, ns, reps, len(EST), len(REFS), nm), np.nan)
    P = np.full((ns, len(REFS), nm), np.nan)
    B = np.full((nk, ns, FOLDS, reps, len(EST), len(B_FIELDS)), np.nan)
    for r in records:
        if r["kind"] not in {"A", "B", "P"} or "failed" in r:
            continue
        si = subj.index(r["subject"])
        if r["kind"] == "B":
            ki, mi = ks.index(r["K"]), MI[r["method"]]
            B[ki, si, r["fold"], r["rep"], mi] = [r[k] for k in B_FIELDS]
            continue
        cells = [[r[f"{k}_{ref}"] for k in at.R29_METRICS] for ref in REFS]
        if r["kind"] == "P":
            P[si] = cells
        else:
            A[ks.index(r["K"]), si, r["rep"], MI[r["method"]]] = cells
    return A, P, B


def boot_list(values: ArrayLike) -> list[float]:
    """[mean, 2.5%, 97.5%, n]."""
    e = at.boot_dict(np.asarray(values, dtype=float))
    return [e["mean"], e["ci"][0], e["ci"][1], e["n"]]


def _gaps(msq: NDArray, msq_base_lc: NDArray) -> dict[str, Any]:
    gaps: dict[str, dict[str, list[float] | int]] = {}
    for m in EST:
        if m == BASE:
            continue
        mb = msq_base_lc if m == LC else msq[:, MI[BASE]]  # R3 for the L-curve
        gC = 2 * (msq[:, MI[m], 1] - mb[:, 1])
        gD = 0.5 * (msq[:, MI[m], 2] - mb[:, 2])
        gT = msq[:, MI[m], 0] - mb[:, 0]
        share = np.where(gT > 0, gD / np.where(gT > 0, gT, 1), np.nan)
        gaps[m] = {"gap_total_uV2": boot_list(gT), "gap_common_uV2": boot_list(gC)}
        gaps[m] |= {"gap_difference_uV2": boot_list(gD)}
        gaps[m] |= {"share_difference": boot_list(share)}
        gaps[m]["n_subjects_gap_total_positive"] = int(np.sum(gT > 0))
        gaps[m]["n_subjects_gap_difference_positive"] = int(np.sum(gD > 0))
    return gaps


def part_a(A: NDArray, P: NDArray, ki: int) -> dict[str, Any]:
    out = {}
    for ri_, ref in enumerate(REFS):
        sub = A[ki, :, :, :, ri_, :]  # (s, r, m, metric)
        msq = sub[:, :, :, :3].mean(axis=1)  # per subject tot2, ec2, ed2: (s, m, 3)
        # R2: the L-curve over its valid repetitions; R3: SURE(kernel) on the same ones
        lc = MI[LC]
        msq[:, lc] = np.nanmean(sub[:, :, lc, :3], axis=1)
        valid = ~np.isnan(sub[:, :, lc, 0])  # (s, r)
        msq_base_lc = np.array(
            [
                (
                    sub[si][valid[si], MI[BASE], :3].mean(axis=0)
                    if not valid[si].all()
                    else msq[si, MI[BASE]]
                )
                for si in range(sub.shape[0])
            ]
        )
        res: dict[str, Any] = {"rms_C": {}, "rms_D": {}, "mean_cos_D": {}}
        res["mean_amp_D"] = {}
        for m in EST:
            j = MI[m]
            res["rms_C"][m] = boot_list(np.sqrt(msq[:, j, 1]))
            res["rms_D"][m] = boot_list(np.sqrt(msq[:, j, 2]))
            for name, col in (("mean_cos_D", 3), ("mean_amp_D", 4)):
                x = sub[:, :, j, col]
                res[name][m] = boot_list(
                    x.mean(axis=1) if m != LC else np.nanmean(x, axis=1)
                )
        res["gaps_vs_SURE_kernel"] = _gaps(msq, msq_base_lc)
        res["population"] = {
            "rms_C": boot_list(np.sqrt(P[:, ri_, 1])),
            "rms_D": boot_list(np.sqrt(P[:, ri_, 2])),
            "cos_D": boot_list(P[:, ri_, 3]),
            "amp_D": boot_list(P[:, ri_, 4]),
        }
        out[ref] = res
    return out


def _summary3(v: list[float]) -> dict[str, Any]:
    """Median and count of positive values over the subjects where v is defined."""
    ok = [x for x in v if math.isfinite(x)]
    median = float(np.median(ok)) if ok else NAN
    n_pos = int(np.sum(np.array(v) > 0))
    return {"per_subject": v, "median": median, "n_positive": n_pos, "n": len(ok)}


def part_b(b: NDArray, n_obs: dict[str, list[int]]) -> dict[str, Any]:
    """b: (s, f, r, m, field). Means, and the R5(b) Spearman / R4(a) Pearson rules."""
    subj_mean = np.nanmean(b, axis=(1, 2))  # (s, m, 5)
    resb: dict[str, Any] = {
        f"mean_{k}": {m: boot_list(subj_mean[:, MI[m], col]) for m in EST}
        for col, k in enumerate(B_FIELDS)
    }
    idx = [MI[m] for m in at.R29_RANK_EST]
    for name, col in (("allAverage", 1), ("allOLS", 2)):
        resb[f"spearman_across_estimators_cos_{name}_vs_auc"] = _summary3(
            at.spearman_r5b(b, idx, col, subj_mean)
        )
        within, n_obs[name] = at.pearson_r4a(b, idx, col)
        resb[f"pearson_within_cell_cos_{name}_vs_auc"] = _summary3(within)
    return resb


def sensitivity(b: NDArray, col: int) -> dict[str, Any]:
    """The other variants of the two rules: Spearman over each estimator's own
    valid cells (R5(a)); within-cell Pearson on the cells all estimators fitted (R4(b)).
    """
    rho, r_drop = [], []
    for x in b[:, :, :, [MI[m] for m in at.R29_RANK_EST], :]:  # (f, r, m, 5)
        pair = np.isfinite(x[..., col]) & np.isfinite(x[..., 0])  # (f, r, m)
        sm = np.nanmean(np.where(pair[..., None], x, np.nan), axis=(0, 1))
        rho.append(float(scipy.stats.spearmanr(sm[:, col], sm[:, 0])[0]))
        ok = pair.all(axis=2)
        xs, ys = x[..., col][ok], x[..., 0][ok]
        xc = (xs - xs.mean(axis=-1, keepdims=True)).ravel()
        yc = (ys - ys.mean(axis=-1, keepdims=True)).ravel()
        r_drop.append(float(np.corrcoef(xc, yc)[0, 1]))
    return {
        "spearman_own_cells": _summary3(rho),
        "pearson_within_drop_cells": _summary3(r_drop),
    }


def part_c(kall: Mapping[Any, Mapping[str, Any]], subj: list[int]) -> dict[str, Any]:
    """The all-training-events fits, the candidate target directions, Population."""

    def fold_mean(field: str) -> list[float]:
        return [float(np.mean([kall[s, f][field] for f in range(FOLDS)])) for s in subj]

    def by_est(field: str) -> dict[str, list[float]]:
        return {
            m: boot_list(
                [np.mean([kall[s, f][field][m] for f in range(FOLDS)]) for s in subj]
            )
            for m in EST
        }

    c_fold = np.array(
        [[kall[s, f]["cos_allOLS_allAverage"] for f in range(FOLDS)] for s in subj]
    ).mean(axis=1)
    full = ("cos_fullOLS_fullAverage", "cos_fullEB_fullAverage")
    out: dict[str, list[float] | dict[str, list[float]]] = {}
    out["cos_allOLS_allAverage_fold_mean"] = boot_list(c_fold)
    out |= {k: boot_list([kall["full", s][k] for s in subj]) for k in full}
    out["Kall_auc"] = by_est("auc")
    out["Kall_cos_to_allAverage"] = by_est("cos_to_allAverage")
    for k in ("auc", "cos_allAverage", "cos_allOLS"):
        out[f"population_{k}"] = boot_list(fold_mean(f"pop_{k}"))
    return out


def noise_medians(nf: Mapping[str, Any], subj: list[int]) -> dict[str, Any]:
    """Median, min, max over subjects of sqrt(nu_C), sqrt(nu_D) (uV) and of the
    target/non-target noise correlation."""
    out = {}
    for ref in ("OLS", "Average"):
        x = [nf[str(s)][ref] for s in subj]
        vals = {f"sqrt_nu_{p}_uV": np.sqrt([v[f"nu_{p}"] for v in x]) for p in "CD"}
        prod = np.array([v["sigma2_target"] * v["sigma2_nontarget"] for v in x])
        vals["rho"] = np.array([v["cross_target_nontarget"] for v in x]) / np.sqrt(prod)
        stats = (np.median, np.min, np.max)
        out[ref] = {k: [float(f(a)) for f in stats] for k, a in vals.items()}
    return out


def dir_summary(
    records: Sequence[Mapping[str, Any]], ks: list[int], subj: list[int], reps: int
) -> dict[str, Any]:
    nf = {
        str(r["subject"]): {k: r[k] for k in ("OLS", "Average")}
        for r in records
        if r["kind"] == "noise"
    }
    A, P, B = per_fit_arrays(records, ks, subj, reps)
    out: dict[str, Any] = {"A": {}, "B": {}, "sensitivity": {}}
    out["noise_floor"] = {s: nf[s] for s in map(str, subj)}
    out["noise_floor_summary"] = noise_medians(nf, subj)
    n_obs: dict[str, dict[str, list[int]]] = {}
    for ki, K in enumerate(ks):
        key = f"K{K:02d}"
        out["A"][key] = part_a(A, P, ki)
        n_obs[key] = {}
        out["B"][key] = part_b(B[ki], n_obs[key])
        cols = (("allAverage", 1), ("allOLS", 2))
        out["sensitivity"][key] = {n: sensitivity(B[ki], c) for n, c in cols}
    kall = {(r["subject"], r["fold"]): r for r in records if r["kind"] == "Kall"}
    full = {r["subject"]: r for r in records if r["kind"] == "full"}
    out["C"] = part_c(kall | {("full", s): r for s, r in full.items()}, subj)
    lc = MI[LC]
    valid = {"A": A[:, :, :, lc, 0, 0], "B": B[:, :, :, :, lc, 0]}  # k, s, ...
    out["checks"] = {
        "failed_fits_excluded": sorted({r["failed"] for r in records if "failed" in r}),
        "within_cell_pearson_n_observations": n_obs,
        "valid_LC_cells_by_subject": {
            an: {
                f"K{K:02d}": [int(np.sum(~np.isnan(x))) for x in v[ki]]
                for ki, K in enumerate(ks)
            }
            for an, v in valid.items()
        },
    }
    dref2 = [full[s]["dref2"] for s in subj]
    out["amp_direction_split"] = at.amp_direction_split(A, dref2, ks)
    if 50 in ks:  # the association depends on the estimator set: seven core estimators
        b = B[ks.index(50)]
        idx = [MI[m] for m in protocol.CORE_METHODS]
        out["core7_K50_spearman_cos_allAverage_vs_auc"] = _summary3(
            at.spearman_r5b(b, idx, 1, np.nanmean(b, axis=(1, 2)))
        )
    return out


def amp_facts(records: Sequence[Mapping[str, Any]], ks: list[int]) -> dict[str, Any]:
    """Pool sizes, the excluded fits, and per K and estimator the successful fits
    and the undefined values among them."""
    ok = [r for r in records if "failed" not in r]
    base = [r for r in ok if r["method"] != SHRINK]

    def pool_range(K: int | str, field: str) -> list[int]:
        vals = [r[field] for r in base if r["K"] == K]
        return [int(min(vals)), int(max(vals))]

    undefined: dict[str, dict[str, dict[str, int]]] = {}
    n_fits: dict[str, dict[str, int]] = {}
    for r in records:  # every observed (K, method) group, failed fits included
        key = "pop" if r["K"] == "pop" else f"K{r['K']:02d}"
        m = r["method"]
        n_fits.setdefault(key, {}).setdefault(m, 0)
        und = undefined.setdefault(key, {})
        und = und.setdefault(m, dict.fromkeys(UNDEFINED_KEYS, 0))
        if "failed" not in r:
            n_fits[key][m] += 1
            for k in UNDEFINED_KEYS:
                und[k] += int(not math.isfinite(r.get(k, NAN)))
    sens = [K for K in ks if K in SENS_KS]
    return {
        "pool": {str(K): pool_range(K, "n_pool") for K in [*ks, "pop"]},
        "pool_overlap_excluded": {str(K): pool_range(K, "n_pool_ov") for K in sens},
        "failed_fits_excluded": sorted(r["failed"] for r in records if "failed" in r),
        "n_successful_fits": n_fits,
        "undefined_counts": undefined,
    }


def write_per_fit(
    path: Path, amp: Sequence[Mapping[str, Any]], dirs: Sequence[Mapping[str, Any]]
) -> None:
    """One row per record; nested fields (speller, overlap scores) as JSON."""
    rows = [
        {
            k: json.dumps(at.plain(v)) if isinstance(v, (dict, list)) else at.plain(v)
            for k, v in r.items()
        }
        for r in [{"analysis": "amplitude", **r} for r in amp]
        + [{"analysis": "direction", **r} for r in dirs]
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r))
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--p300", default="./runs/p300", help="run_p300.py output")
    parser.add_argument(
        "--data-dir", default=os.environ.get("MNE_DATA", str(Path.home() / "mne_data"))
    )
    parser.add_argument("--out", default="./runs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--subjects", type=int, nargs="+")
    parser.add_argument("--reps", type=int)
    args = parser.parse_args()
    started = datetime.now(UTC)
    p300 = Path(args.p300).resolve()
    settings = json.loads((p300 / "summary.json").read_text())["settings"]
    subj = list(args.subjects or settings["subjects"])
    reps = int(args.reps or settings["repetitions"])
    ks = [int(k) for k in settings["Ks"]]
    out = Path(args.out).resolve() / "r29"
    out.mkdir(parents=True, exist_ok=True)
    fits, pop_auc = read_fits(p300)
    if missing := [s for s in subj if s not in pop_auc]:
        raise RuntimeError(f"subjects {missing} have no Population fit (R2.9 rules)")
    sel = json.loads((p300 / "selections.json").read_text())["B"]  # K, subject, fold
    mat_dir = Path(args.data_dir) / "MNE-bnci-data" / "~bci" / "database" / "009-2014"
    tasks = [
        {"p300": p300, "subject": s, "fits": fits[s], "pop_auc": pop_auc[s]}
        | {"sel": {k: {str(s): v[str(s)]} for k, v in sel.items()}}
        | {"ks": ks, "reps": reps, "mat_dir": mat_dir}
        for s in subj
    ]
    done = at.run_pool(min(args.workers, len(subj)), subject_task, tasks)
    amp = [r for a, _ in done for r in a]
    dirs = [r for _, d in done for r in d]
    summary = {
        "settings": {"p300": str(p300), "subjects": subj}
        | {"repetitions": reps, "Ks": ks},
        "amplitude_value": {"summary": amp_summary(amp, subj, ks)},
        "direction_check": dir_summary(dirs, ks, subj, reps),
    }
    summary["amplitude_value"] |= amp_facts(amp, ks)
    write_per_fit(out / "per_fit.csv", amp, dirs)
    at.write_json(out / "summary.json", summary)
    at.write_run_info(out, vars(args), started)


if __name__ == "__main__":
    main()
