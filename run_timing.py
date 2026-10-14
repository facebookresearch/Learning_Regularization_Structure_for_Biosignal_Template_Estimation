# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Computation time of every estimator: L/M/C scaling and the auditory EEG case.

Run alone on an idle machine: the times are wall-clock measurements. The
configurations run serially. Each configuration runs in two fresh processes of
this interpreter with one BLAS thread. The first makes, per method, one warm-up
and five timed calls (``time.perf_counter``) on the same input. The second makes
one instrumented call of SURE(lambda) and SURE(kernel) for the stage split, so
the instrumentation never perturbs a timed call. Each process reports its own
peak resident memory (``VmHWM`` in ``/proc/self/status``). Writes
``<out>/timing/per_fit.csv`` (one row per call), ``summary.json`` and
``run_info.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import functools
import json
import math
import os
import platform
import socket
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import analysis_tools
import core
import dtw_average
import numpy as np
import protocol
import scipy
import threadpoolctl
from numpy.typing import NDArray

FAILURES = bool(protocol.DECISIONS["OQ6_exceptions_are_failures"])
REAL = "real_eeg"
KERNEL_OBJECTIVE = "kernel_objective_other"
BLAS_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
COLUMNS = (
    "config",
    "P",
    "method",
    "call",
    "warmup",
    "profile",
    "status",
    "failure_type",
    "failure_message",
    "seconds",
    "flags",
    "peak_rss_kb_so_far",
)


class Config(TypedDict):
    """A sweep configuration (L samples, M event types, C channels) or the EEG."""

    name: str
    L: NotRequired[int]
    M: NotRequired[int]
    C: NotRequired[int]


class StageOptions(TypedDict, total=False):
    top_level_only: bool
    inside: str


def all_configs() -> list[Config]:
    """The sweep configurations in sweep order (the base once), then the EEG."""
    seen, out = set(), []
    for factor, values in protocol.TIMING_SWEEPS.items():
        for v in values:
            lmc = {**protocol.TIMING_BASE, factor: v}
            key = (lmc["L"], lmc["M"], lmc["C"])
            if key not in seen:
                seen.add(key)
                name = f"L{key[0]}_M{key[1]}_C{key[2]}"
                out.append(Config(name=name, L=key[0], M=key[1], C=key[2]))
    return out + [Config(name=REAL)]


def methods() -> dict[str, Mapping[str, Any] | None]:
    """``core.get_events_templates`` settings per method; None is the Molina average."""
    return {**protocol.METHODS_AR1COL, protocol.FILTERED: None}


def config_input(
    cfg: Config, data_dir: Path
) -> tuple[dict[str, NDArray] | None, str | None]:
    """The configuration's recording, or None and the reason it is unavailable."""
    if "L" in cfg and "M" in cfg and "C" in cfg:
        data, ix, lb = analysis_tools.timing_case(cfg["L"], cfg["M"], cfg["C"])
        hw = np.array(cfg["L"] // 2)
        return {"data": data, "ix": ix, "lb": lb, "half_win": hw}, None
    # Deferred: the timing workers re-run this file, and run_real_data imports MNE,
    # which would add about 40 MB to every worker's VmHWM (the reported memory).
    import run_real_data  # @manual

    k = str(protocol.TIMING_EEG_K)
    if k not in map(str, run_real_data.KS["EEG"]):
        raise ValueError(
            f"TIMING_EEG_K={k} is not among run_real_data.KS['EEG']="
            f"{run_real_data.KS['EEG']}, whose first selection the EEG case times"
        )
    a, _ = run_real_data.load_input("EEG", data_dir)
    selections, unavailable = run_real_data.make_selections(a["lb"], "EEG", 1)
    if k in unavailable:
        return None, f"EEG K={k}: {unavailable[k]}"
    rows = np.array(selections[k][0])
    hw = np.array(run_real_data.HALF_WIN["EEG"])
    return {
        "data": a["data"],
        "ix": a["ix"][rows],
        "lb": a["lb"][rows],
        "half_win": hw,
    }, None


# ---------------------------------------------------------------------------
# Worker: one configuration in a fresh process
# ---------------------------------------------------------------------------


def molina_fit(a: Mapping[str, NDArray]) -> tuple[dict[str, NDArray], None]:
    hw = int(a["half_win"])
    epochs = core.epoch_data(a["data"], a["ix"], -hw, hw)
    return {
        str(c): np.asarray(dtw_average.molina_template(epochs[a["lb"] == c]))
        for c in np.unique(a["lb"])
    }, None


def timed_call(a: Mapping[str, NDArray], name: str) -> dict[str, Any]:
    """One fit under the failure policy; ``seconds`` is its ``perf_counter`` time."""
    kwargs = methods()[name]
    if kwargs is None:
        call = functools.partial(molina_fit, a)
    else:
        call = functools.partial(
            core.get_events_templates,
            a["data"],
            a["ix"],
            a["lb"],
            half_win=int(a["half_win"]),
            **kwargs,
        )
    keys = [str(c) for c in np.unique(a["lb"])]
    record, _ = analysis_tools.fit_call(call, keys, FAILURES)
    return record


def instrument_plan() -> list[tuple[object, str, str, StageOptions]]:
    """(owner, attribute, stage, options) of every instrumented function."""
    top: StageOptions = {"top_level_only": True}
    sparse = scipy.sparse
    dne = "design_and_normal_equations"
    return [
        # Sparse construction, products and sums done inline at the top level of
        # get_events_templates: D, D^T D, D^T X and their accumulation (also the
        # small top-level penalty assembly).
        (sparse._compressed._cs_matrix, "__init__", dne, top),  # noqa: SLF001
        (sparse._base._spbase, "__matmul__", dne, top),  # noqa: SLF001
        (sparse._base._spbase, "__rmatmul__", dne, top),  # noqa: SLF001
        (sparse._base._spbase, "__add__", dne, top),  # noqa: SLF001
        (sparse._base._spbase, "__iadd__", dne, top),  # noqa: SLF001
        (core, "get_design_matrix", "design_matrix", {}),
        (core, "est_noise_var", "noise_variance_and_covariance", {}),
        (core, "estimate_phi", "noise_variance_and_covariance", {}),
        (core, "estimate_acf_ar", "noise_variance_and_covariance", {}),
        (core, "build_DtSigmaD_ar1", "noise_variance_and_covariance", {}),
        (core, "build_DtSigmaD_acf", "noise_variance_and_covariance", {}),
        (core._KernelSUREObjective, "value_gradient", KERNEL_OBJECTIVE, {}),  # noqa: SLF001
        (
            scipy.linalg,
            "cho_factor",
            "kernel_objective_cholesky_solves",
            {"inside": KERNEL_OBJECTIVE},
        ),
        (
            scipy.linalg,
            "cho_solve",
            "kernel_objective_cholesky_solves",
            {"inside": KERNEL_OBJECTIVE},
        ),
        (scipy.optimize, "minimize", "kernel_optimizer_other", {}),
        (core._PenaltyPath, "__init__", "scalar_cholesky_whitening_svd", {}),  # noqa: SLF001
        (core, "_scalar_sure_score", "scalar_sure_setup", {}),
        (core, "_search_scalar_sure_path", "scalar_search", {}),
        (scipy.sparse.linalg, "spsolve", "final_solve", top),
        (core._PenaltyPath, "templates", "final_solve", top),  # noqa: SLF001
    ]


@contextlib.contextmanager
def instrumented(timer: analysis_tools.StageTimer) -> Iterator[None]:
    """Wrap the stage functions for the instrumented calls; restored on exit."""
    saved = []
    try:
        for owner, name, stage, opts in instrument_plan():
            original = getattr(owner, name)
            saved.append((owner, name, original))
            setattr(owner, name, timer.wrap(stage, original, **opts))
        yield
    finally:
        for owner, name, original in reversed(saved):
            setattr(owner, name, original)


def stage_split(timer: analysis_tools.StageTimer, total: float) -> dict[str, Any]:
    stages = dict(timer.exclusive)
    stages["other_unattributed"] = total - sum(timer.exclusive.values())
    return {
        "exclusive_seconds": stages,
        "calls": dict(timer.calls),
        "total_seconds": total,
    }


def peak_rss_kb() -> int:
    """This process's own resident-set high-water mark. ``getrusage`` is not used:
    RUSAGE_CHILDREN keeps the largest earlier child and ru_maxrss is inherited
    across fork and exec."""
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    raise RuntimeError("/proc/self/status has no VmHWM line")


def one_call(
    a: Mapping[str, NDArray], name: str, call: int, profile: bool
) -> dict[str, Any]:
    """One timed call, or one instrumented call with its stage split."""
    if profile:
        timer = analysis_tools.StageTimer()
        with instrumented(timer):
            record = timed_call(a, name)
        if record["status"] == "success":
            record["stage_split"] = stage_split(timer, record["seconds"])
    else:
        record = timed_call(a, name)
    record.update(
        method=name,
        call=call,
        warmup=not profile and call < protocol.TIMING_WARMUP,
        profile=profile,
        peak_rss_kb_so_far=peak_rss_kb(),
    )
    for key in ("inner", "kernel", "lambda", "noise_var", "traceback"):
        record.pop(key, None)
    return record


def worker(path: Path, profile: bool, reps: int) -> dict[str, Any]:
    """Every call of one configuration: timed (per method its warm-up and *reps*
    timed calls) or instrumented (one call per stage-split method)."""
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    names = protocol.TIMING_STAGE_SPLIT if profile else tuple(methods())
    n_calls = 1 if profile else protocol.TIMING_WARMUP + reps
    with threadpoolctl.threadpool_limits(1):
        records = [
            one_call(a, name, call, profile)
            for name in names
            for call in range(n_calls)
        ]
        threads = [p["num_threads"] for p in threadpoolctl.threadpool_info()]
    labels, counts = np.unique(a["lb"], return_counts=True)
    hw = int(a["half_win"])
    return {
        "records": records,
        "peak_rss_kb": peak_rss_kb(),
        "data_shape": list(a["data"].shape),
        "n_events": len(a["ix"]),
        "class_counts": {str(c): int(n) for c, n in zip(labels, counts)},
        "L": 2 * hw,
        "M": len(labels),
        "C": int(a["data"].shape[1]),
        "P": int(len(labels) * 2 * hw),
        "blas_threads": threads,
    }


def run_worker(path: Path, name: str, profile: bool, reps: int) -> dict[str, Any]:
    """One configuration's worker in a fresh process with one BLAS thread."""
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker"]
    command += [str(path), "--reps", str(reps)] + (["--profile"] if profile else [])
    env = {**os.environ, **dict.fromkeys(BLAS_VARIABLES, "1")}
    load_before = os.getloadavg()[0]
    done = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    if done.returncode != 0:
        raise RuntimeError(
            f"timing worker {name} (profile={profile}) exited with code "
            f"{done.returncode}:\n{done.stderr[-2000:]}"
        )
    lines = done.stdout.splitlines()
    if not lines:
        raise RuntimeError(
            f"timing worker {name} (profile={profile}) exited 0 without output; "
            f"stderr: {done.stderr[-2000:]}"
        )
    result = json.loads(lines[-1])
    result["loadavg_1min"] = [load_before, os.getloadavg()[0]]
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def method_summary(recs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Median and range over the successful timed calls; counts of each."""
    timed = [r for r in recs if not r["warmup"]]
    ok = [r["seconds"] for r in timed if r["status"] == "success"]
    return {
        "n_timed": len(timed),
        "n_timed_success": len(ok),
        "n_failed_calls": sum(r["status"] != "success" for r in recs),
        "median_seconds": float(np.median(ok)) if ok else None,
        "range_seconds": [min(ok), max(ok)] if ok else None,
        "warmup_seconds": [r["seconds"] for r in recs if r["warmup"]],
        "timed_seconds_all": [r["seconds"] for r in timed],
        "timed_status": [r["status"] for r in timed],
        "flags": sorted({f for r in recs for f in r.get("flags", [])}),
    }


def config_summary(
    cfg: Config, timed: Mapping[str, Any], profiled: Mapping[str, Any]
) -> dict[str, Any]:
    recs = timed["records"]
    return {
        "config": dict(cfg),
        "P": timed["P"],
        "data_shape": timed["data_shape"],
        "n_events": timed["n_events"],
        "class_counts": timed["class_counts"],
        "L": timed["L"],
        "M": timed["M"],
        "C": timed["C"],
        "methods": {
            n: method_summary([r for r in recs if r["method"] == n]) for n in methods()
        },
        "stage_split": {
            r["method"]: {
                "status": r["status"],
                "flags": r.get("flags", []),
                "failure_message": r.get("failure_message"),
                "split": r.get("stage_split"),
            }
            for r in profiled["records"]
        },
        "peak_rss_gib": max(timed["peak_rss_kb"], profiled["peak_rss_kb"]) / 2**20,
        "peak_rss_kb": {
            "timed": timed["peak_rss_kb"],
            "profile": profiled["peak_rss_kb"],
        },
        "loadavg_1min": {
            "timed": timed["loadavg_1min"],
            "profile": profiled["loadavg_1min"],
        },
        "blas_threads": sorted(set(timed["blas_threads"] + profiled["blas_threads"])),
    }


def scaling(configs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Per method and along L and M: the median time per doubling of P and the
    endpoint exponent log(t_last / t_first) / log(P_last / P_first)."""
    out: dict[str, Any] = {}
    for factor in ("L", "M"):
        names = [
            "L{L}_M{M}_C{C}".format(**{**protocol.TIMING_BASE, factor: v})
            for v in protocol.TIMING_SWEEPS[factor]
        ]
        if not all(n in configs for n in names):
            continue
        ps = [configs[n]["P"] for n in names]
        for m in methods():
            t = [configs[n]["methods"][m]["median_seconds"] for n in names]
            entry: dict[str, Any] = {"P": ps, "median_seconds": t}
            # Ratios need positive times and distinct P endpoints.
            if all(x is not None and x > 0 for x in t) and ps[-1] != ps[0]:
                entry["doubling_ratios"] = [b / a for a, b in zip(t, t[1:])]
                entry["endpoint_exponent"] = math.log(t[-1] / t[0]) / math.log(
                    ps[-1] / ps[0]
                )
            out.setdefault(m, {})[factor] = entry
    return out


def eeg_stage_split(configs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    """R2.10 stage split (not in the final paper or supplement): kernel-objective
    time and noise/covariance time of the instrumented EEG SURE(kernel) call,
    against its total."""
    split = configs.get(REAL, {}).get("stage_split", {}).get("SURE(kernel)", {})
    if split.get("split") is None:
        return None
    ex = split["split"]["exclusive_seconds"]  # a stage never entered took 0 s
    return {
        "kernel_objective_seconds": ex.get("kernel_objective_cholesky_solves", 0.0)
        + ex.get("kernel_objective_other", 0.0),
        "noise_variance_and_covariance_seconds": ex.get(
            "noise_variance_and_covariance", 0.0
        ),
        "total_seconds": split["split"]["total_seconds"],
    }


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def measure(
    configs: Sequence[Config], data_dir: Path, out: Path, reps: int
) -> tuple[dict[str, dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    """Per configuration its summary, or the reason it is unavailable; every call."""
    summaries: dict[str, dict[str, Any]] = {}
    unavailable: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="inputs_", dir=out) as tmp:
        paths: dict[str, Path] = {}
        for cfg in configs:
            arrays, reason = config_input(cfg, data_dir)
            if arrays is None:
                unavailable[cfg["name"]] = str(reason)
                continue
            paths[cfg["name"]] = Path(tmp) / f"{cfg['name']}.npz"
            np.savez(paths[cfg["name"]], **arrays)
        for cfg in configs:
            if cfg["name"] not in paths:
                continue
            timed = run_worker(paths[cfg["name"]], cfg["name"], False, reps)
            profiled = run_worker(paths[cfg["name"]], cfg["name"], True, reps)
            summaries[cfg["name"]] = config_summary(cfg, timed, profiled)
            rows += [
                {**r, "config": cfg["name"], "P": timed["P"]}
                for r in timed["records"] + profiled["records"]
            ]
    return summaries, unavailable, rows


def write_per_fit(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(COLUMNS)
        for r in rows:
            writer.writerow(
                ";".join(str(f) for f in r.get(c) or [])
                if c == "flags"
                else ("" if r.get(c) is None else r[c])
                for c in COLUMNS
            )


def build_summary(
    summaries: Mapping[str, Mapping[str, Any]],
    unavailable: Mapping[str, str],
    rows: Sequence[Mapping[str, Any]],
    reps: int,
) -> dict[str, Any]:
    peaks = [s["peak_rss_gib"] for s in summaries.values()]
    return {
        "configs": summaries,
        "unavailable": unavailable,
        "scaling": scaling(summaries),
        "eeg_stage_split": eeg_stage_split(summaries),
        "peak_rss_gib_range": [min(peaks), max(peaks)] if peaks else None,
        "n_calls": len(rows),
        "n_failed_calls": sum(r["status"] != "success" for r in rows),
        "host": socket.getfqdn(),
        "os": platform.platform(),
        "cpu": cpu_model(),
        "n_cpus": os.cpu_count(),
        "protocol": {
            "warmup": protocol.TIMING_WARMUP,
            "timed_calls": reps,
            "base": protocol.TIMING_BASE,
            "simulation": protocol.TIMING_SIM,
            "eeg_k": protocol.TIMING_EEG_K,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--data-dir", default=os.environ.get("MNE_DATA", str(Path.home() / "mne_data"))
    )
    parser.add_argument("--out", default="./runs")
    names = [c["name"] for c in all_configs()]
    parser.add_argument("--configs", nargs="+", choices=names, default=names)
    parser.add_argument("--reps", type=int, default=protocol.TIMING_REPEATS)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--profile", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.reps < 1:
        parser.error(f"--reps must be at least 1, got {args.reps}")
    if args.worker is not None:
        result = worker(args.worker, args.profile, args.reps)
        print(json.dumps(analysis_tools.plain(result)))
        return
    started = datetime.now(UTC)
    out = Path(args.out) / "timing"
    out.mkdir(parents=True, exist_ok=True)
    configs = [c for c in all_configs() if c["name"] in args.configs]
    summaries, unavailable, rows = measure(configs, Path(args.data_dir), out, args.reps)
    write_per_fit(out / "per_fit.csv", rows)
    summary = build_summary(summaries, unavailable, rows, args.reps)
    analysis_tools.write_json(out / "summary.json", summary)
    settings = {
        "configs": [c["name"] for c in configs],
        "reps": args.reps,
        "sweeps": protocol.TIMING_SWEEPS,
        "stage_split": protocol.TIMING_STAGE_SPLIT,
        "methods": methods(),
        "data_dir": args.data_dir,
    }
    analysis_tools.write_run_info(out, settings, started)


if __name__ == "__main__":
    main()
