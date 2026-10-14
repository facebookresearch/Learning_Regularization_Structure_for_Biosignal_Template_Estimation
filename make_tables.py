# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""LaTeX tables of the paper and the supplement from the analysis outputs.

Reads ``<runs>/<analysis>/summary.json`` written by ``run_p300.py`` (``p300``),
``run_reference_noise.py`` (``reference_noise``), ``run_real_data.py``
(``real_data``), ``run_simulations.py`` (``calibration``, ``known_covariance``,
``q_symmetry``), ``run_noise_diagnostics.py`` (``noise_diagnostics``),
``run_p300_sensitivity.py`` (``p300_colored_factorial``, ``p300_phi_variance``,
``p300_q_symmetry``, also its ``per_fit.csv``), ``run_timing.py`` (``timing``)
and ``run_r29.py`` (``r29``), and writes ``<out>/<name>.tex`` for each table of
``TABLES``: the supplement's ``\\input`` files under their input names, and the
tabulars of the paper's tables ``tab_summary`` and ``tab_p300_k5`` and of the
supplement's squared-domain table (``tab_sup_squared``), which the manuscripts
hold inline. A value a run could not compute (failed fits, gated-out or omitted
data sets, small slices) prints ``n/a``.

Usage: python3 -B make_tables.py [--runs DIR] [--out DIR] [NAME ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import protocol

LABEL: dict[str, str] = {
    "SURE(lambda)": "SURE($\\lambda$)",
    "EmpBayes": "Emp.\\ Bayes",
    "Molina filtered DTW": "Molina (filtered)",
    "zero": "Zero template",
    "lambda=inf": "$\\lambda \\hsp{-2}{-2}{=} \\infty$ (line)",
    "Q1": "Q1 (ridge)",
    "Q3-sym": "Q3 symmetric",
    "Q3-free": "Q3 free",
    "Q5-sym": "Q5 symmetric",
    "Q5-free": "Q5 free",
    "phi_f0.5": "$\\hat\\varphi^{2}$",
    "phi_f0.75": "$\\hat\\varphi^{4/3}$",
    "baseline": "$\\hat\\varphi$ (estimated)",
    "phi_f1.5": "$\\hat\\varphi^{2/3}$",
    "phi_f2": "$\\hat\\varphi^{1/2}$",
    "var_x0.8": "variance $\\times 0.8$",
    "var_x1.25": "variance $\\times 1.25$",
}
# Distance order of the paper's P300 table at K = 5 (tab_p300_k5).
P300_ORDER: tuple[str, ...] = (
    *("SURE(kernel)", "EmpBayes", "SURE(lambda)", "GCV", "L-curve", "Average"),
    *("OLS", protocol.FILTERED),
)
# Method order of the EEG/ECG tables at K = 5 (tab_summary, b3_eeg_ecg_k5).
REAL_ORDER: tuple[str, ...] = protocol.CORE_METHODS
BASELINES: tuple[str, ...] = ("lambda=inf", "zero", protocol.POPULATION)


def label(name: str) -> str:
    return LABEL.get(name, name)


def summary(runs: Path, analysis: str) -> dict[str, Any]:
    return json.loads((runs / analysis / "summary.json").read_text())


def per_fit(runs: Path, analysis: str) -> list[dict[str, str]]:
    with (runs / analysis / "per_fit.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def at(tree: object, *path: str | int) -> Any:  # noqa: ANN401 - a JSON value
    """``tree[path[0]][path[1]]...``; None where a level is missing."""
    for key in path:
        if isinstance(tree, Mapping):
            tree = tree.get(key)
        elif isinstance(tree, list) and isinstance(key, int) and key < len(tree):
            tree = tree[key]
        else:
            return None
    return tree


def mapping(tree: object, *path: str | int) -> dict[str, Any]:
    """The mapping at `path`, empty where it is missing (None: an unavailable result)."""
    value = at(tree, *path)
    return value if isinstance(value, dict) else {}


def fnum(x: object) -> float | None:
    """A summary number: None for None, NaN, the "nan" string and non-numbers; the
    "inf"/"-inf" strings of ``analysis_tools.plain`` as floats."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        v = float(x)
    elif isinstance(x, str) and x in {"inf", "-inf", "nan"}:
        v = float(x)
    else:
        return None
    return None if math.isnan(v) else v


def num(x: object, spec: str) -> str:
    """`x` formatted, without a signed zero; an undefined value is "n/a"."""
    v = fnum(x)
    if v is None:
        return "n/a"
    s = format(v, spec)
    if re.fullmatch(r"[-+]0(\.0+)?", s):
        s = ("+" if spec.startswith("+") else "") + s[1:]
    return s


def tex_num(x: object, nd: int) -> str:
    s = num(x, f".{nd}f")
    return f"${s}$" if s.startswith("-") else s


def frac(k: object, n: object) -> str:
    """``k/n`` for counts; "n/a" when `k` is undefined."""
    return "n/a" if fnum(k) is None else f"{num(k, 'g')}/{num(n, 'g')}"


def minus(s: str) -> str:
    """A minus sign before a digit as a LaTeX minus."""
    return re.sub(r"(?<![\w.])-(?=\d)", "$-$", s)


def row(cells: Sequence[str]) -> str:
    return " & ".join(cells) + " \\\\"


def latex_tabular(
    spec: str, head: Sequence[str], rows: Sequence[Sequence[str] | str]
) -> str:
    """A booktabs tabular; a row given as a string (``\\midrule``) is copied as is."""
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", *head, "\\midrule"]
    lines += [r if isinstance(r, str) else row(r) for r in rows]
    return "\n".join([*lines, "\\bottomrule", "\\end{tabular}"]) + "\n"


def simple(head: str, rows: Sequence[Sequence[str] | str]) -> str:
    """A tabular ``l r ... r`` under the header row `head` (cells joined by " & "); every
    cell after the first gets LaTeX minus signs."""
    body = [r if isinstance(r, str) else [r[0], *map(minus, r[1:])] for r in rows]
    return latex_tabular("l" + "r" * head.count(" & "), [f"{head} \\\\"], body)


def subject_keys(cells: Mapping[str, Any], *path: str) -> list[str]:
    """The subjects present under `path` in any cell, in numeric order."""
    keys = {s for c in cells.values() for s in mapping(c, *path)}
    return sorted(keys, key=lambda s: int(s) if s.isdigit() else math.inf)


def real_k(real: Mapping[str, Any], ds: str) -> dict[str, Any]:
    """``baselines.K`` of one real-data set; empty when the run omitted the set or could
    not fit it."""
    return mapping(real, ds, "baselines", "K")


# ---------------------------------------------------------------------------
# Main paper
# ---------------------------------------------------------------------------
def tab_p300_k5(runs: Path) -> str:
    b = mapping(summary(runs, "p300"), "baselines")
    molina = {protocol.FILTERED: "Molina et al.~\\cite{molina2024enhanced}"}

    def cells(m: str) -> list[str]:
        refs = (num(at(b, "K", "5", m, r), ".1f") for r in protocol.REFERENCES)
        auc = num(at(b, "auc", "5", m, "mean"), ".3f")
        return [molina.get(m, label(m)), *refs, auc]

    rows = [*map(cells, P300_ORDER), "\\midrule", *map(cells, BASELINES)]
    head = [
        " & \\multicolumn{3}{c}{Distance to reference} & \\\\",
        "\\cmidrule(lr){2-4}",
        "Estimator & OLS & EB & Average & AUC \\\\",
    ]
    return latex_tabular("lcccc", head, rows)


def tab_summary(runs: Path) -> str:
    """Paper table: auditory EEG and ECG distances to the OLS reference at K = 5."""
    real = summary(runs, "real_data")
    cols = {
        "EEG": (real_k(real, "EEG").get("5"), ".1f"),
        "ECG": (real_k(real, "ECG").get("5"), ".3f"),
    }
    baselines = ("lambda=inf", "zero")
    names = (*REAL_ORDER, *baselines)
    ols = {c: {m: fnum(at(d, m, "OLS")) for m in names} for c, (d, _) in cols.items()}
    # Bold: every estimator whose rounded distance equals the rounded minimum.
    best = {
        c: num(
            min((x for m in REAL_ORDER if (x := ols[c][m]) is not None), default=None),
            f,
        )
        for c, (_, f) in cols.items()
    }

    def cells(m: str) -> list[str]:
        out = [label(m)]
        for c, (_, f) in cols.items():
            s = num(ols[c][m], f)
            bold = m in REAL_ORDER and s == best[c] != "n/a"
            out.append(f"\\textbf{{{s}}}" if bold else s)
        return out

    rows: list[Sequence[str] | str] = [cells(m) for m in REAL_ORDER]
    rows += ["\\midrule", *(cells(m) for m in baselines)]
    return latex_tabular("lcc", ["Method & Aud.\\ EEG & ECG \\\\"], rows)


# ---------------------------------------------------------------------------
# Supplement: P300 families, subjects, squared domain
# ---------------------------------------------------------------------------
ENDPOINTS: dict[str, str] = {
    "distance_OLS": "dist.\\ OLS",
    "distance_EmpBayes": "dist.\\ EB",
    "distance_Average": "dist.\\ Avg.",
    "heldout_auc": "AUC",
}


def family_row(fam: str, key: str, c: Mapping[str, Any]) -> list[str]:
    ep, _, other = key.partition("|")
    nd = 4 if ep == "heldout_auc" else 3
    effects = [fnum(at(v, "effect")) for v in mapping(c, "per_subject").values()]
    favour = sum(e is not None and e > 0 for e in effects)
    return [
        *("Primary" if fam == "primary" else fam, label(other)),
        *(ENDPOINTS.get(ep, ep), tex_num(at(c, "mean"), nd)),
        f"[{tex_num(at(c, 'ci95', 0), nd)}, {tex_num(at(c, 'ci95', 1), nd)}]",
        *(num(at(c, "test", "p"), ".3f"), num(at(c, "test", "p_holm"), ".3f")),
        tex_num(at(c, "test", "rank_biserial"), 2),
        f"{favour}/{num(at(c, 'n_subjects'), 'g')}",
    ]


def families(runs: Path) -> str:
    c0 = mapping(summary(runs, "p300"), "central")
    rows = [
        family_row(fam, key, mapping(c0, "comparisons", key))
        for fam, keys in mapping(c0, "families").items()
        for key in keys
    ]
    head = "Family & SURE(kernel) vs & Endpoint & Mean & 95\\% CI & $p$ & $p_{\\text{Holm}}$ & $r$ & Subj. \\\\"
    return latex_tabular("lllrrrrrr", [head], rows)


def subjects(runs: Path) -> str:
    S = summary(runs, "p300")
    sm = mapping(S, "central", "descriptive", "subject_means")
    pop = mapping(S, "central", "descriptive", "population")
    zero = mapping(S, "baselines", "K", "5", "zero", "subject")
    dist = ("SURE(kernel)", "EmpBayes", "SURE(lambda)", "Average", protocol.FILTERED)
    rows = []
    for s in sm:
        cells = [s, *(tex_num(at(sm, s, m, "distance_OLS_uV"), 1) for m in dist)]
        cells += [
            tex_num(at(pop, s, "distances_uV", "OLS"), 1),
            tex_num(at(zero, s, "OLS"), 1),
        ]
        auc = ("SURE(kernel)", "EmpBayes", "Average")
        cells += [tex_num(at(sm, s, m, "heldout_auc"), 3) for m in auc]
        rows.append([*cells, tex_num(at(pop, s, "heldout_auc"), 3)])
    head = [
        " & \\multicolumn{7}{c}{Distance to the full-data OLS reference ($\\mu$V)} & \\multicolumn{4}{c}{Held-out AUC} \\\\",
        "\\cmidrule(lr){2-8}\\cmidrule(lr){9-12}",
        "Subj. & SK & EB & S$\\lambda$ & Avg. & Mol. & Pop. & Zero & SK & EB & Avg. & Pop. \\\\",
    ]
    return latex_tabular("l rrrrrrr rrrr", head, rows)


def d2_defined(cell: str) -> bool:
    """A per-fit ``d2_OLS`` (a JSON list of class values) defined in every class."""
    try:
        values = json.loads(cell)
    except json.JSONDecodeError:
        return False
    return isinstance(values, list) and all(fnum(v) is not None for v in values)


def squared_subjects(runs: Path) -> dict[str, int]:
    """Per estimator, the subjects in the squared domain: those with a K = 5 fit whose
    ``d2_OLS`` is defined in both classes (``run_reference_noise.squared_domain``)."""
    subjects: dict[str, set[str]] = {}
    for r in per_fit(runs, "reference_noise"):
        if r["K"] == "5" and not r["failed_fit"] and d2_defined(r["d2_OLS"]):
            subjects.setdefault(r["estimator"], set()).add(r["subject"])
    return {m: len(s) for m, s in subjects.items()}


def tab_sup_squared(runs: Path) -> str:
    """``squared_domain`` is None when the K = 5 gate failed: every cell is n/a."""
    sq = mapping(summary(runs, "reference_noise"), "squared_domain")
    n = squared_subjects(runs)
    t2, t = num(at(sq, "T_norm2", "pooled"), ".0f"), num(at(sq, "rms", "T"), ".1f")
    rows = [["Zero template ($\\|\\mathbf{T}\\|^2$)", t2, t, "n/a"]]
    names = {"lambda=inf": "Straight line ($\\lambda \\hsp{-2}{-2}{=} \\infty$)"}
    for m in ("lambda=inf", "SURE(kernel)", "EmpBayes", "Average"):
        above = frac(at(sq, "err2", m, "n_subjects_err2_gt_T2_pooled"), n.get(m))
        rows.append(
            [
                *(names.get(m, label(m)), num(at(sq, "err2", m, "pooled"), ".0f")),
                *(num(at(sq, "rms", m), ".1f"), above),
            ]
        )
    return simple(
        "Template & Estimate ($\\mu$V$^2$) & Square root ($\\mu$V) & Subjects above $\\|\\mathbf{T}\\|^2$",
        rows,
    )


# ---------------------------------------------------------------------------
# Supplement: R2.9 tables
# ---------------------------------------------------------------------------
def r29_ci(e: object, d: int = 3) -> str:
    """mean [interval] with signs; a signed zero is kept."""
    values = [fnum(at(e, "mean")), fnum(at(e, "ci", 0)), fnum(at(e, "ci", 1))]
    x = [v for v in values if v is not None]
    if len(x) < 3:
        return "n/a"
    return f"{x[0]:+.{d}f} [{x[1]:+.{d}f}, {x[2]:+.{d}f}]"


def r29_gap(runs: Path) -> str:
    S = summary(runs, "r29")
    g = mapping(S, "direction_check", "A", "K05", "OLS", "gaps_vs_SURE_kernel")
    spl = mapping(S, "direction_check", "amp_direction_split")
    rows = []
    for m in (
        "Average",
        "EmpBayes",
        "GCV",
        "SURE(lambda)",
        "L-curve",
        protocol.FILTERED,
    ):
        s = mapping(spl, f"K05|{m}")
        parts = [at(g, m, f"gap_{k}_uV2", 0) for k in ("total", "common", "difference")]
        parts += [
            at(s, "gap_D_direction_part_mean"),
            at(s, "gap_D_amplitude_part_mean"),
        ]
        rows.append([label(m), *(num(x, ",.0f") for x in parts)])
    head = "Estimator & total & common part & difference part & of which direction & of which scale"
    return simple(head, rows)


# r29_prob names the Molina comparators by their low-pass filter.
R29_PROB_LABEL: dict[str, str] = {
    protocol.FILTERED: "Molina",
    protocol.PLAIN: "Molina, no low-pass filter",
}


def r29_prob(runs: Path) -> str:
    T = mapping(summary(runs, "r29"), "amplitude_value", "summary", "K05")
    rows = []
    for m in (*protocol.ESTIMATORS, "Average + shrinkage", protocol.POPULATION):
        log_rho = fnum(at(T, "log_rho_amp", m, "mean"))
        sep = None if log_rho is None else math.exp(log_rho)
        rows.append(
            [
                R29_PROB_LABEL.get(m, label(m)),
                num(at(T, "auc", m, "mean"), ".3f"),
                num(at(T, "IG_bits", m, "mean"), "+.3f"),
                num(sep, ".1f"),
            ]
        )
    head = "Estimator & AUC & information gain (bits/event) & separation ratio"
    return simple(head, rows)


def higher(a: object, b: object) -> str:
    """k/n, where n counts the subjects with both values defined (lists `a` and `b`)
    and k those whose value in `a` is the larger; "n/a" when n = 0."""
    pairs = zip(a if isinstance(a, list) else [], b if isinstance(b, list) else [])
    both = [
        (x, y)
        for x, y in ((fnum(p), fnum(q)) for p, q in pairs)
        if x is not None and y is not None
    ]
    return frac(sum(x > y for x, y in both), len(both)) if both else "n/a"


def r29_diff(runs: Path) -> str:
    S = summary(runs, "r29")
    rows = []
    for K, T in mapping(S, "amplitude_value", "summary").items():
        sk, av = (
            at(T, "IG_bits", m, "per_subject") for m in ("SURE(kernel)", "Average")
        )
        auc = r29_ci(at(T, "auc", "SURE(kernel)", "vs_Avg"), 4)
        ig = r29_ci(at(T, "IG_bits", "SURE(kernel)", "vs_Avg"), 3)
        rows.append([K.removeprefix("K").lstrip("0"), auc, ig, higher(sk, av)])
    head = "K & AUC & information gain (bits/event) & subjects higher"
    return simple(head, rows)


# ---------------------------------------------------------------------------
# Supplement: EEG/ECG at K = 5, simulations, sensitivity, noise model, timing
# ---------------------------------------------------------------------------
# b3_eeg_ecg_k5's columns: data set, reference, format.
EEG_ECG_K5_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("EEG", "OLS", ".1f"),
    ("EEG", "EmpBayes", ".1f"),
    ("ECG", "OLS", ".3f"),
    ("ECG", "EmpBayes", ".3f"),
    ("ECG", "Average", ".3f"),
)


def b3_eeg_ecg_k5(runs: Path) -> str:
    """EEG (its Average reference equals the OLS one) and ECG distances at K = 5."""
    real = summary(runs, "real_data")
    k5 = {ds: mapping(real_k(real, ds), "5") for ds in ("EEG", "ECG")}
    rows: list[Sequence[str] | str] = [
        (
            m
            if m == "\\midrule"
            else [
                label(m),
                *(num(at(k5[d], m, r), f) for d, r, f in EEG_ECG_K5_COLUMNS),
            ]
        )
        for m in (*REAL_ORDER, "\\midrule", "lambda=inf", "zero")
    ]
    short = {"EmpBayes": "EB"}
    head = [f"{d}, {short.get(r, r)}" for d, r, _ in EEG_ECG_K5_COLUMNS]
    return simple(" & ".join(["Method", *head]), rows)


def b3_calibration(runs: Path) -> str:
    per = mapping(summary(runs, "calibration"), "per_reference")

    def cell(r: str, m: str) -> str:
        mean = at(per, r, "intervals", m, "mean")
        rank = num(at(per, r, "ranks", m), "g")
        return "n/a" if fnum(mean) is None else f"{num(mean, '.4g')} ({rank})"

    order = ("SURE(kernel)", "SURE(lambda)", "EmpBayes", "GCV", "L-curve", "Average")
    rows = [
        [label(m), *(cell(r, m) for r in ("truth", *protocol.REFERENCES))]
        for m in (*order, "OLS")
    ]
    head = "Method & Truth (rank) & OLS ref.\\ (rank) & EB ref.\\ (rank) & Average ref.\\ (rank)"
    return simple(head, rows)


def b3_colored_p300(runs: Path) -> str:
    P = mapping(summary(runs, "p300_colored_factorial"), "p300", "OLS")
    effects = ("kernel_effect_white", "kernel_effect_colored", "noise_effect_fixed")
    effects += ("noise_effect_learned", "interaction")
    rows = [
        [s, *(num(at(v, e), ".4f") for e in effects)]
        for s, v in P.items()
        if isinstance(v, dict)
    ]
    head = "Subject & Kernel (white) & Kernel (colored) & Noise (fixed) & Noise (learned) & Interaction"
    return simple(head, rows)


def b3_known_cov(runs: Path) -> str:
    rows = []
    for key, x in mapping(summary(runs, "known_covariance"), "sim").items():
        phi, _, est = key.removeprefix("phi").partition("|")
        k = mapping(x, "known_minus_estimated")
        snr = (num(at(x, v, "mean"), ".3f") for v in ("estimated", "known"))
        diff = f"{num(at(k, 'mean'), '.3f')} [{num(at(k, 'lo'), '.3f')}, {num(at(k, 'hi'), '.3f')}]"
        phi_hat = num(at(x, "phi_estimated_mean"), ".4f")
        rows.append([phi, label(est), *snr, diff, phi_hat])
    head = "$\\varphi$ & Estimator & Estimated & Known & Known $-$ estimated & Mean $\\hat\\varphi$"
    return simple(head, rows)


def b3_q5_convergence(runs: Path) -> str:
    p300 = mapping(summary(runs, "p300_q_symmetry"), "q5_convergence", "convergence")
    sim = mapping(summary(runs, "q_symmetry"), "q5_convergence", "convergence")

    def conv(x: object) -> str:
        return f"{num(at(x, 'converged_50'), 'g')} / {num(at(x, 'converged_500'), 'g')}"

    rows = []
    for c in p300:
        both = (at(p300, c), at(sim, c))
        its = (num(at(x, "median_iterations_final_converged"), "g") for x in both)
        rows.append([label(c), *map(conv, both), " / ".join(its)])
    head = "Kernel & P300 conv.\\ at 50 / 500 & Simulation conv.\\ at 50 / 500 & Median iterations, P300 / sim."
    return simple(head, rows)


# b3_kernel_p300: Q1 from the per-fit distances, the other kernels from the summary.
KERNEL_P300_CELLS: tuple[str, ...] = ("Q3-free", "Q5-sym", "Q5-free")


def log_distance(rec: Mapping[str, str], ref: str) -> float | None:
    """log of a successful fit's distance to `ref`; None for a failed fit or a
    distance that is missing, nonfinite or not positive."""
    if rec["status"] != "success":
        return None
    try:
        d = float(rec.get(f"distance_{ref}", ""))
    except ValueError:
        return None
    return math.log(d) if math.isfinite(d) and d > 0 else None


def q1_minus_q3sym(rows: Sequence[Mapping[str, str]], ref: str) -> dict[str, Any]:
    """Per subject, the mean over units of log d[Q1] - log d[Q3-sym] at the default
    iteration limit (units where both fits succeeded), then the mean over subjects;
    sums in per-fit order, divided by the count."""
    by: dict[str, dict[str, dict[str, float | None]]] = {"Q1": {}, "Q3-sym": {}}
    for r in rows:
        if r["label"] in by:
            units = by[r["label"]].setdefault(r["subject"], {})
            units[r["unit"]] = log_distance(r, ref)
    per: dict[str, float | None] = {}
    for s in subject_keys(by):
        q1, q3 = by["Q1"].get(s, {}), by["Q3-sym"].get(s, {})
        d = [
            a - b
            for u, a in q1.items()
            if a is not None and (b := q3.get(u)) is not None
        ]
        per[s] = sum(d) / len(d) if d else None
    vals = [v for v in per.values() if v is not None]
    return {"per_subject": per, "mean": sum(vals) / len(vals) if vals else None}


def kernel_minus_q3sym(runs: Path) -> dict[str, dict[str, Any]]:
    """{kernel: {reference: {"per_subject": {s: mean}, "mean": mean}}}: P300 log
    distance of each kernel minus Q3 symmetric (Q3 free and Q5: converged final fits
    at the raised limit)."""
    Q = mapping(summary(runs, "p300_q_symmetry"), "q5_convergence", "q5_vs_q3sym")
    rows = per_fit(runs, "p300_q_symmetry")
    out = {"Q1": {r: q1_minus_q3sym(rows, r) for r in protocol.REFERENCES}}
    for c in KERNEL_P300_CELLS:
        out[c] = {}
        for r in protocol.REFERENCES:
            e = mapping(Q, r, c, "converged final only")
            per = {s: at(v, "mean") for s, v in mapping(e, "per_subject").items()}
            out[c][r] = {"per_subject": per, "mean": at(e, "mean_over_subjects")}
    return out


def b3_kernel_p300(runs: Path) -> str:
    cells = kernel_minus_q3sym(runs)
    subs = subject_keys(cells, "OLS", "per_subject")
    rows = []
    for c, v in cells.items():
        per = " / ".join(num(at(v, "OLS", "per_subject", s), ".4f") for s in subs)
        means = (num(at(v, r, "mean"), ".4f") for r in protocol.REFERENCES)
        rows.append([label(c), per, *means])
    names = " / ".join(f"S{s}" for s in subs)
    head = f"Kernel & OLS ref.\\ {names} & Mean (OLS) & Mean (EB) & Mean (Average)"
    return simple(head, rows)


def b3_phi_variance(runs: Path) -> str:
    P = mapping(summary(runs, "p300_phi_variance"), "p300")
    cols = [
        (bw, est)
        for est in ("SURE(kernel)", "SURE(lambda)")
        for bw in ("bw1-20", "bw1-30")
    ]
    settings = ("phi_f0.5", "phi_f0.75", "baseline", "phi_f1.5", "phi_f2")
    rows = [phi_variance_row(P, c, cols) for c in (*settings, "var_x0.8", "var_x1.25")]
    head = (
        "Setting & $\\varphi$ used (1--20\\,Hz) & SK 1--20\\,Hz & SK 1--30\\,Hz & "
        "S$\\lambda$ 1--20\\,Hz & S$\\lambda$ 1--30\\,Hz"
    )
    return simple(head, rows)


def phi_variance_row(
    P: Mapping[str, Any], c: str, cols: Sequence[tuple[str, str]]
) -> list[str]:
    """The baseline row is 0 by definition where its fits give a defined mean."""
    means = (at(P, bw, e, "OLS", c, "mean") for bw, e in cols)
    vals = (
        num(x, ".4f") if c != "baseline" else "0" if fnum(x) is not None else "n/a"
        for x in means
    )
    phi = num(at(P, "bw1-20", "SURE(kernel)", "OLS", c, "phi_used_mean"), ".4f")
    return [label(c), phi, *vals]


def b3_noise_model(runs: Path) -> str:
    M = mapping(
        summary(runs, "noise_diagnostics"),
        "manuscript",
        "noise_model_mean_log_distance_OLS",
    )
    inputs = ("pairwise", "ar1_colored", "ar3_colored")
    keys = [f"{e}|{n}" for e in ("SURE(kernel)", "SURE(lambda)") for n in inputs]
    # A dataset whose performance fits were gated out has no entry: n/a cells.
    rows = [
        [ds, *(num(at(M, k, ds), ".4f") for k in keys)] for ds in ("EEG", "ECG", "P300")
    ]
    head = (
        "Dataset & SK pairwise & SK AR(1) & SK AR(3) & "
        "S$\\lambda$ pairwise & S$\\lambda$ AR(1) & S$\\lambda$ AR(3)"
    )
    return simple(head, rows)


def b3_timing(runs: Path) -> str:
    methods = (
        "SURE(kernel)",
        "SURE(lambda)",
        "EmpBayes",
        "GCV",
        "L-curve",
        "OLS",
        protocol.FILTERED,
    )
    rows = []
    for name, v in mapping(summary(runs, "timing"), "configs").items():
        m = re.fullmatch(r"L(\d+)_M(\d+)_C(\d+)", name)
        # Three significant figures with trailing zeros (this table only).
        times = [num(at(v, "methods", x, "median_seconds"), "#.3g") for x in methods]
        config = " / ".join(m.groups()) if m else "Auditory EEG"
        rows.append([config, num(at(v, "P"), "g"), *times])
    head = (
        "$L$ / $M$ / $C$ & $P$ & SURE(kernel) & SURE($\\lambda$) & Emp.\\ Bayes & GCV"
        " & L-curve & OLS & Molina"
    )
    return simple(head, rows)


TABLES: dict[str, Callable[[Path], str]] = {
    "tab_summary": tab_summary,
    "tab_p300_k5": tab_p300_k5,
    "b3_q5_convergence": b3_q5_convergence,
    "b3_kernel_p300": b3_kernel_p300,
    "b3_colored_p300": b3_colored_p300,
    "b3_known_cov": b3_known_cov,
    "b3_calibration": b3_calibration,
    "b3_eeg_ecg_k5": b3_eeg_ecg_k5,
    "families": families,
    "subjects": subjects,
    "tab_sup_squared": tab_sup_squared,
    "r29_gap": r29_gap,
    "r29_prob": r29_prob,
    "r29_diff": r29_diff,
    "b3_noise_model": b3_noise_model,
    "b3_phi_variance": b3_phi_variance,
    "b3_timing": b3_timing,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--runs", default="./runs", help="folder holding the analysis outputs"
    )
    parser.add_argument("--out", default="./tables")
    parser.add_argument("names", nargs="*", help=f"default: all of {list(TABLES)}")
    args = parser.parse_args()
    unknown = sorted(set(args.names) - set(TABLES))
    if unknown:
        parser.error(f"unknown tables {unknown}; expected names from {list(TABLES)}")
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.names or TABLES:
        (out / f"{name}.tex").write_text(TABLES[name](runs))


if __name__ == "__main__":
    main()
