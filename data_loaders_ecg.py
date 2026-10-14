# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ECG loader for the MIT-BIH Arrhythmia Database (PhysioNet).

Downloads and loads ECG recordings with their annotated beats (R-peak sample and
beat type) as (data, event_indices, event_labels). MIT-BIH contains 48 half-hour
two-lead ECG recordings from 47 subjects, sampled at 360 Hz, with
cardiologist-annotated beat labels.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import wfdb
from scipy.signal import butter, sosfiltfilt


# MIT-BIH records live under the user's home MNE data dir, not in the repo.
# Override the base with the MNE_DATA env var if set.
_DATA_DIR = Path(os.environ.get("MNE_DATA", Path.home() / "mne_data")) / "mitdb"


def download_ecg_mitbih(
    data_dir: str | Path | None = None,
    records: list[str] | None = None,
) -> Path:
    """Download MIT-BIH records from PhysioNet."""

    if data_dir is None:
        data_dir = _DATA_DIR
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if records is None:
        records = [
            "100",
            "101",
            "103",
            "105",
            "106",
            "108",
            "112",
            "113",
            "114",
            "115",
            "116",
            "117",
            "119",
            "121",
            "122",
            "123",
        ]

    for rec_id in records:
        if (data_dir / f"{rec_id}.dat").exists():
            print(f"  Record {rec_id}: already downloaded")
            continue
        print(f"  Downloading record {rec_id}...")
        wfdb.dl_database("mitdb", str(data_dir), records=[rec_id])

    print(f"MIT-BIH: {len(records)} records in {data_dir}")
    return data_dir


def load_ecg_mitbih(
    data_dir: str | Path | None = None,
    records: list[str] | None = None,
    lead: int = 0,
    bandpass: tuple[float, float] | None = (0.5, 40.0),
    beat_types: list[str] | None = None,
    max_beats_per_record: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load MIT-BIH ECG data as continuous signal + beat event annotations.

    Parameters
    ----------
    data_dir : path, optional
        Directory containing MIT-BIH .dat/.hea/.atr files.
    records : list of str, optional
        Which record IDs to load. Default: a curated set of clean records.
    lead : int
        Which lead to use (0 = MLII, 1 = V1/V5 depending on record).
    bandpass : tuple or None
        (low, high) Hz bandpass filter.
    beat_types : list of str, optional
        Which AAMI beat types to include. Default: ['N', 'L', 'R', 'V', 'A'].
        N=Normal, L=LBBB, R=RBBB, V=PVC, A=APB.
    max_beats_per_record : int or None
        Cap beats per record (useful for balancing).

    Returns
    -------
    data : ndarray (T, 1)
        Single-lead ECG (kept as 2D for consistency with multichannel API).
    event_indices : ndarray (N,)
        Sample indices of R-peaks.
    event_labels : ndarray (N,)
        Beat type labels (as integers: N=0, L=1, R=2, V=3, A=4).
    sfreq : float
        Sampling frequency (360 Hz for MIT-BIH).
    """

    if data_dir is None:
        data_dir = _DATA_DIR
    data_dir = Path(data_dir)

    if records is None:
        records = [
            "100",
            "101",
            "103",
            "105",
            "106",
            "108",
            "112",
            "113",
            "114",
            "115",
            "116",
            "117",
            "119",
            "121",
            "122",
            "123",
        ]

    if beat_types is None:
        beat_types = ["N", "L", "R", "V", "A"]

    label_map = {bt: i for i, bt in enumerate(beat_types)}

    all_signals = []
    event_indices_list = []
    event_labels_list = []
    offset = 0
    sfreq = None

    for rec_id in records:
        rec_path = str(data_dir / rec_id)
        try:
            rec = wfdb.rdrecord(rec_path)
            ann = wfdb.rdann(rec_path, "atr")
        except Exception as e:
            print(f"  Warning: skipping record {rec_id}: {e}")
            continue

        if sfreq is None:
            sfreq = float(rec.fs)

        sig = rec.p_signal[:, lead : lead + 1]

        beat_count = 0
        for sample, symbol in zip(ann.sample, ann.symbol):
            if symbol in label_map:
                event_indices_list.append(offset + sample)
                event_labels_list.append(label_map[symbol])
                beat_count += 1
                if max_beats_per_record and beat_count >= max_beats_per_record:
                    break

        all_signals.append(sig)
        offset += sig.shape[0]

    if not all_signals:
        raise FileNotFoundError(f"No ECG records found in {data_dir}")
    if sfreq is None:
        raise ValueError("No valid records loaded, sfreq not set")

    data = np.vstack(all_signals)

    if bandpass is not None:
        nyq = sfreq / 2.0
        low = bandpass[0] / nyq
        high = bandpass[1] / nyq
        sos = butter(4, [low, high], btype="band", output="sos")
        data = sosfiltfilt(sos, data, axis=0)

    event_indices = np.array(event_indices_list, dtype=int)
    event_labels = np.array(event_labels_list, dtype=int)

    return data, event_indices, event_labels, sfreq


if __name__ == "__main__":
    print("=" * 60)
    print("  ECG Data Loader Verification")
    print("=" * 60)

    print("\n--- Downloading MIT-BIH ---")
    download_ecg_mitbih(records=["119"])

    # The paper's ECG input: record 119, lead 0, 0.5-40 Hz, Normal and PVC beats.
    print("\n--- Loading record 119 ---")
    data, ix, lb, fs = load_ecg_mitbih(records=["119"], beat_types=["N", "V"])
    print(f"  Data shape: {data.shape}")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Num beats: {len(ix)}")
    print(f"  Labels: {np.unique(lb, return_counts=True)}")
