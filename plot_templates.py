# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Estimated templates on the analyzed real data.

``fig_templates_real`` (main paper): (a, b) P300 target and non-target at Cz of the
lower-median subject (``select_subject``), (c, d) ECG record 119 Normal and PVC beats;
``figS_templates_eeg``: auditory EEG at the fronto-central channel with the largest
N100 (``select_eeg_channel``). Each estimator is its mean over the 20 K = 5 selections
of ``run_p300.py`` / ``run_real_data.py`` with a +/- 1 SD band (np.std, ddof 0); P300
and EEG in uV, ECG in mV. Also writes ``<out>/<name>.json``: the selection and its
result, and the panel numbers the text cites. A figure whose inputs are missing
(failed references, single-dataset runs) is reported and not drawn.

Usage: python3 -B plot_templates.py [--runs DIR] [--data-dir DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import analysis_tools
import data_loaders
import figure_style as fs
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray

METHODS: tuple[str, ...] = ("Average", "EmpBayes", "SURE(kernel)")
K = 5
CHANNEL = "Cz"
P300_SFREQ = 128.0
UV = 1e6
MEAN_LW = 1.1
BAND_ALPHA = 0.18
REFERENCE_KW: dict[str, Any] = {
    "color": fs.REFERENCE_COLOR,
    "lw": 1.6,
    "alpha": 0.7,
    "ls": "-",
}
P300_CLASS: Mapping[str, int] = {"Target": 1, "Non-target": 0}
ECG_CLASS: Mapping[str, int] = {"Normal": 0, "PVC": 1}
N100_WINDOW_MS: tuple[float, float] = (80.0, 120.0)
FRONTO_CENTRAL = r"FC(z|\d+h?)"  # FCz, FC1..FC6, FC1h..FC6h; not FFC/FCC
EEG_CLASS_TITLE: Mapping[str, str] = {
    "aud_L": "left-ear tone",
    "aud_R": "right-ear tone",
}


def time_ms(n: int, sfreq: float) -> NDArray:
    """Sample times of a template of n samples, centred at sample n // 2."""
    return (np.arange(n) - n // 2) / sfreq * 1000.0


class Unavailable(RuntimeError):
    """A figure's inputs are missing (failed references or fits, sliced runs)."""


def select_subject(subject_means: Mapping[str, Any]) -> dict[str, Any]:
    """Lower-median subject: rank ceil(n/2) (5 of 10) by SURE(kernel) mean distance to the
    OLS reference at K = 5, ascending; ties broken by subject number. The rule needs
    the distance of every subject."""
    dist = {
        s: v.get("SURE(kernel)", {}).get("distance_OLS_uV")
        for s, v in subject_means.items()
    }
    if not dist:
        raise Unavailable("subject rule: the p300 summary has no subject means")
    missing = sorted((s for s, d in dist.items() if d is None), key=int)
    if missing:
        raise Unavailable(
            f"subject rule: no SURE(kernel) distance_OLS_uV for subjects {missing}"
        )
    vals = sorted((d, int(s)) for s, d in dist.items())
    rank = (len(vals) + 1) // 2
    value, subject = vals[rank - 1]
    return {
        "rule": "lower median (rank ceil(n/2), ascending) of SURE(kernel) distance_OLS_uV at K=5",
        "rank": rank,
        "n_subjects": len(vals),
        "subject": subject,
        "value_uV": value,
        "ranking": [[s, v] for v, s in vals],
    }


def stacked_fits(run: Path, keep: Mapping[str, str]) -> dict[str, NDArray]:
    """Each method's saved templates over the per-fit rows matching `keep`, stacked in
    repetition order. The figures have no failure rule: a failed fit makes the figure
    unavailable."""
    with (run / "per_fit.csv").open(newline="") as stream:
        rows = [
            r
            for r in csv.DictReader(stream)
            if all(r[k] == v for k, v in keep.items()) and r["estimator"] in METHODS
        ]
    out: dict[str, list[NDArray]] = {m: [] for m in METHODS}
    for r in sorted(rows, key=lambda r: int(r["repetition"])):
        if r["status"] != "success":
            raise Unavailable(
                f"{run}: {r['estimator']} failed at {keep} repetition "
                f"{r['repetition']} (status {r['status']!r}); the template figures "
                "have no failure rule"
            )
        with np.load(run / r["template"], allow_pickle=False) as z:
            out[r["estimator"]].append(np.asarray(z["templates"]))
    empty = [m for m, v in out.items() if not v]
    if empty:
        raise Unavailable(f"{run}: no saved {keep} templates of {empty}")
    return {m: np.stack(v) for m, v in out.items()}


def ols_reference(path: Path) -> NDArray:
    """The full-data OLS reference; a failed reference fit leaves no OLS key."""
    if not path.exists():
        raise Unavailable(f"no reference file {path}")
    with np.load(path) as z:
        if "OLS" not in z.files:
            raise Unavailable(f"{path}: no OLS reference (its fit failed)")
        return np.asarray(z["OLS"])


def read_summary(run: Path) -> dict[str, Any]:
    path = run / "summary.json"
    if not path.exists():
        raise Unavailable(f"no analysis summary {path} (the analysis was not run)")
    return json.loads(path.read_text())


def dataset_input(real: Path, ds: str) -> Mapping[str, Any]:
    S = read_summary(real)
    if ds not in S:
        raise Unavailable(f"{real} has no {ds} dataset (a single-dataset run)")
    return S[ds]["input"]


def legend_entries(population: bool) -> list[Line2D]:
    def line(m: str, label: str) -> Line2D:
        st = fs.method_style(m)
        return Line2D([], [], color=st.color, ls=st.linestyle, lw=MEAN_LW, label=label)

    handles = [Line2D([], [], label="Full-data OLS reference", **REFERENCE_KW)]
    if population:
        handles.append(line("Population", "Population template (P300)"))
    for m in METHODS:
        label = f"{fs.METHOD_LABELS[m]}, $K$ = {K} (mean $\\pm$ 1 SD)"
        handles.append(line(m, label))
    handles.append(Line2D([], [], label="Zero template", **fs.ZERO_LINE_KW))
    return handles


def peak(y: NDArray) -> float:
    """The value of largest magnitude (the first one on ties)."""
    a = np.asarray(y, dtype=np.float64)
    return float(a[int(np.argmax(np.abs(a)))])


def _template_panel(
    ax: Axes,
    x: NDArray,
    ref: NDArray,
    pop: NDArray | None,
    fits: Mapping[str, NDArray],
    scale: float,
) -> dict[str, Any]:
    """Reference, Population, each method's mean +/- 1 SD over the selections (`fits[m]`
    is (n, T)). Returns the reference peak and per method the peak of the mean and the
    time-averaged SD (half the band width): the numbers the text reads off the panel."""
    fs.zero_line(ax)
    ax.plot(x, ref * scale, zorder=1, **REFERENCE_KW)
    if pop is not None:
        st = fs.method_style("Population")
        ax.plot(x, pop * scale, color=st.color, ls=st.linestyle, lw=MEAN_LW, zorder=3)
    out: dict[str, Any] = {"reference_peak": peak(ref * scale)}
    for m in METHODS:
        st = fs.method_style(m)
        f = fits[m] * scale
        mu, s = f.mean(axis=0), f.std(axis=0)
        lo, hi = mu - s, mu + s
        ax.fill_between(x, lo, hi, color=st.color, alpha=BAND_ALPHA, lw=0, zorder=2)
        z = 4 if m == "SURE(kernel)" else 3
        ax.plot(x, mu, color=st.color, ls=st.linestyle, lw=MEAN_LW, zorder=z)
        out[m] = {
            "mean_peak": peak(mu),
            "sd_time_mean": statistics.mean(((hi - lo) / 2).tolist()),
        }
    ax.set_xlim(x[0], x[-1])
    return out


def p300_inputs(p300: Path, s: int) -> tuple[NDArray, int, NDArray | None]:
    """Subject s's OLS reference, the index of CHANNEL and its Population template
    (None: undefined with one subject, a quick slice)."""
    ref = ols_reference(p300 / "references" / f"subject_{s:02d}.npz")
    with np.load(p300 / "cache" / f"subject_{s:02d}.npz", allow_pickle=False) as z:
        channels = list(z["channels"])
    if CHANNEL not in channels:
        raise Unavailable(f"subject {s}: no channel {CHANNEL} in {channels}")
    pop_file = p300 / "population" / f"subject_{s:02d}.npz"
    pop = None
    if pop_file.exists():
        with np.load(pop_file) as z:
            pop = np.asarray(z["templates"])
    return ref, channels.index(CHANNEL), pop


def templates_real(runs: Path) -> tuple[Figure, dict[str, Any]]:
    p300, real = runs / "p300", runs / "real_data"
    S = read_summary(p300)
    means = S.get("central", {}).get("descriptive", {}).get("subject_means", {})
    sel = select_subject(means)
    ecg_sfreq = dataset_input(real, "ECG")["sfreq"]
    ecg_ref = ols_reference(real / "references" / "ecg.npz")
    s = sel["subject"]
    ref, ch, pop = p300_inputs(p300, s)
    fits = stacked_fits(p300, {"analysis": "A", "K": str(K), "subject": str(s)})
    ecg = stacked_fits(real, {"dataset": "ECG", "K": str(K)})
    fig = fs.new_figure((fs.DOUBLE_COLUMN[0], 4.8))
    axes = fig.subplots(2, 2)
    numbers: dict[str, dict[str, Any]] = {}
    for ax, name in zip(axes[0], ("Target", "Non-target")):
        c = P300_CLASS[name]
        x = time_ms(ref.shape[1], P300_SFREQ)
        series = {m: f[:, c, :, ch] for m, f in fits.items()}
        p = None if pop is None else pop[c, :, ch]
        letter = "a" if name == "Target" else "b"
        numbers[letter] = _template_panel(ax, x, ref[c, :, ch], p, series, UV)
        ax.set_title(f"({letter}) P300 {name.lower()}, subject {s}, {CHANNEL}")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(r"Amplitude ($\mu$V)")
    for ax, name in zip(axes[1], ("Normal", "PVC")):
        c = ECG_CLASS[name]
        x = time_ms(ecg_ref.shape[1], ecg_sfreq)
        series = {m: f[:, c, :, 0] for m, f in ecg.items()}
        letter = "c" if name == "Normal" else "d"
        numbers[letter] = _template_panel(ax, x, ecg_ref[c, :, 0], None, series, 1.0)
        ax.set_title(f"({letter}) ECG record 119, {name} beat")
        ax.set_xlabel("Time relative to R peak (ms)")
        ax.set_ylabel("Amplitude (mV)")
    fs.figure_legend(fig, legend_entries(population=pop is not None), ncol=3)
    fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=1.0)
    record = {
        "subject_selection": sel,
        "channel": CHANNEL,
        "K": K,
        "n_selections": {m: int(v.shape[0]) for m, v in fits.items()},
        "n_ecg_repetitions": {m: int(v.shape[0]) for m, v in ecg.items()},
        "panels": numbers,
    }
    return fig, record


def nearest_1005_labels(
    positions: Mapping[str, Any], names: Sequence[str]
) -> dict[str, str]:
    """Nearest standard 10-05 position (head frame) of each channel; a tie goes to the
    alphabetically first label."""
    std = sorted(positions["standard_head"])
    pts = np.array([positions["standard_head"][n] for n in std], dtype=float)
    out: dict[str, str] = {}
    for c in names:
        d = np.linalg.norm(pts - np.asarray(positions["channels"][c], float), axis=1)
        out[c] = std[int(np.argmin(d))]
    return out


def select_eeg_channel(
    ref_uv: NDArray, names: Sequence[str], labels: Mapping[str, str], sfreq: float
) -> dict[str, Any]:
    """Largest N100: among fronto-central channels, the channel whose class-mean
    full-data OLS reference is most negative in 80-120 ms."""
    t = time_ms(ref_uv.shape[1], sfreq)
    w = (t >= N100_WINDOW_MS[0]) & (t <= N100_WINDOW_MS[1])
    if not w.any():
        raise Unavailable(
            f"EEG channel rule: no sample in {N100_WINDOW_MS} ms "
            f"(template from {t[0]:.1f} to {t[-1]:.1f} ms)"
        )
    m = ref_uv.mean(axis=0)
    names = list(names)
    cands = {
        c: float(m[w, names.index(c)].min())
        for c in names
        if re.fullmatch(FRONTO_CENTRAL, labels[c])
    }
    if not cands:
        raise Unavailable(f"EEG channel rule: no fronto-central channel in {labels}")
    chosen = min(cands, key=lambda c: (cands[c], c))
    i = names.index(chosen)
    j = int(np.argmin(np.where(w, m[:, i], np.inf)))
    return {
        "rule": "most negative class-mean OLS reference in 80-120 ms among fronto-central channels "
        "(nearest 10-05 label FCz, FC1..FC6, FC1h..FC6h)",
        "channel": chosen,
        "label_1005": labels[chosen],
        "n100_uV": cands[chosen],
        "n100_latency_ms": float(t[j]),
        "candidates": {
            c: {"label_1005": labels[c], "n100_uV": v}
            for c, v in sorted(cands.items(), key=lambda kv: (kv[1], kv[0]))
        },
    }


def templates_eeg(runs: Path, data_dir: Path) -> tuple[Figure, dict[str, Any]]:
    real = runs / "real_data"
    info = dataset_input(real, "EEG")
    ref = ols_reference(real / "references" / "eeg.npz")
    names, sfreq = list(info["channels"]), float(info["sfreq"])
    positions = data_loaders.load_eeg_positions(data_dir)
    if positions["coord_frame"] != "head":
        raise Unavailable(
            f"EEG positions in frame {positions['coord_frame']!r}, expected head"
        )
    labels = nearest_1005_labels(positions, names)
    sel = select_eeg_channel(ref * UV, names, labels, sfreq)
    fits = stacked_fits(real, {"dataset": "EEG", "K": str(K)})
    ch = names.index(sel["channel"])
    x = time_ms(ref.shape[1], sfreq)
    fig = fs.new_figure((fs.DOUBLE_COLUMN[0], 2.9))
    axes = fig.subplots(1, 2, sharey=True)
    numbers: dict[str, dict[str, Any]] = {}
    for k, (ax, (_cls, title)) in enumerate(zip(axes, EEG_CLASS_TITLE.items())):
        series = {m: f[:, k, :, ch] for m, f in fits.items()}
        numbers["ab"[k]] = _template_panel(ax, x, ref[k, :, ch], None, series, UV)
        ax.set_title(
            f"({'ab'[k]}) {title.capitalize()}, {sel['channel']} "
            f"(near {sel['label_1005']})"
        )
        ax.set_xlabel("Time (ms)")
    axes[0].set_ylabel(r"Amplitude ($\mu$V)")
    fs.figure_legend(fig, legend_entries(population=False), ncol=3)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    n = {m: int(v.shape[0]) for m, v in fits.items()}
    record = {"channel_selection": sel, "K": K, "n_repetitions": n, "panels": numbers}
    return fig, record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--runs", default="./runs", help="folder holding p300/ and real_data/"
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("MNE_DATA", str(Path.home() / "mne_data")),
        help="MNE data folder (EEG channel positions of the sample recording)",
    )
    parser.add_argument("--out", default="./figures")
    args = parser.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    makers = {
        "fig_templates_real": lambda: templates_real(runs),
        "figS_templates_eeg": lambda: templates_eeg(runs, Path(args.data_dir)),
    }
    with fs.paper_style():
        for name, make in makers.items():
            try:
                fig, record = make()
            except Unavailable as e:
                print(f"{name} not drawn: {e}", file=sys.stderr)
                for ext in (".pdf", ".png", ".json"):  # no stale earlier output
                    (out / f"{name}{ext}").unlink(missing_ok=True)
                continue
            fs.save_figure(fig, out, name)
            analysis_tools.write_json(out / f"{name}.json", record)


if __name__ == "__main__":
    main()
