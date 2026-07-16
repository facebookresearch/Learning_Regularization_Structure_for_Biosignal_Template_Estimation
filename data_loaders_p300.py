# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

"""P300 BCI data loader using MOABB.

Loads P300/ERP BCI datasets for template estimation benchmarks
and downstream single-trial classification evaluation.

Datasets:
  - BNCI2014-009 (P300 speller, 10 subjects, 8 EEG channels)
  - EPFL P300 (if BNCI unavailable)
"""


import mne
import numpy as np

from moabb.datasets import BI2013a, BNCI2014_009


def load_p300_bnci2014_009(
    subjects: list[int] | None = None,
    bandpass: tuple[float, float] | None = (1.0, 20.0),
    resample: float | None = 128.0,
    pick_channels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
    """Load BNCI2014-009 P300 dataset via MOABB.

    This dataset contains EEG recordings from a P300 speller paradigm.
    Subjects view a matrix of characters; the target row/column elicits
    a P300 response while non-targets do not.

    Parameters
    ----------
    subjects : list of int, optional
        Which subjects to load (1-10). Default: [1].
    bandpass : tuple or None
        (low, high) Hz bandpass filter.
    resample : float or None
        Resample to this frequency (reduces computation).

    Returns
    -------
    data : ndarray (T, C)
        Continuous multichannel EEG.
    event_indices : ndarray (N,)
        Sample indices of stimulus onsets.
    event_labels : ndarray (N,)
        Labels: 1 = target (P300), 0 = non-target.
    sfreq : float
        Sampling frequency after resampling.
    info : dict
        Metadata (ch_names, n_target, n_nontarget, etc.)
    """

    if subjects is None:
        subjects = [1]

    dataset = BNCI2014_009()

    all_signals = []
    event_indices_list = []
    event_labels_list = []
    offset = 0
    sfreq = None
    ch_names = None

    for subj in subjects:
        subj_data = dataset.get_data(subjects=[subj])

        for _session_name, session_data in subj_data[subj].items():
            for _run_name, raw in session_data.items():
                if bandpass is not None:
                    raw.filter(bandpass[0], bandpass[1], verbose=False)

                if resample is not None:
                    raw.resample(resample, verbose=False)

                # Extract events BEFORE picking channels (annotations need stim channels)
                events, event_id = mne.events_from_annotations(raw, verbose=False)

                # Pick only EEG channels (exclude stim/misc channels)
                raw.pick(picks="eeg", verbose=False)

                # Optionally pick a subset of channels
                if pick_channels is not None:
                    raw.pick(pick_channels, verbose=False)

                sfreq = raw.info["sfreq"]
                if ch_names is None:
                    ch_names = raw.ch_names

                sig = raw.get_data().T  # (T, C)

                for sample, _, eid in events:
                    # Map event IDs to target/non-target
                    # BNCI2014-009: event_id mapping varies, find Target/NonTarget
                    event_indices_list.append(offset + sample)
                    # Convention: Target=1, NonTarget=0
                    label = 1 if eid == max(event_id.values()) else 0
                    event_labels_list.append(label)

                all_signals.append(sig)
                offset += sig.shape[0]

    if not all_signals:
        raise ValueError("No data loaded from BNCI2014-009")
    if sfreq is None:
        raise ValueError("No valid runs loaded, sfreq not set")

    data = np.vstack(all_signals)
    event_indices = np.array(event_indices_list, dtype=int)
    event_labels = np.array(event_labels_list, dtype=int)

    n_target = np.sum(event_labels == 1)
    n_nontarget = np.sum(event_labels == 0)

    info = {
        "ch_names": ch_names,
        "sfreq": sfreq,
        "n_target": int(n_target),
        "n_nontarget": int(n_nontarget),
        "n_subjects": len(subjects),
        "dataset": "BNCI2014-009",
    }

    return data, event_indices, event_labels, sfreq, info


def _detect_target_nontarget_ids(
    event_id: dict[str, int],
) -> tuple[int, int | None]:
    """Detect target and non-target event IDs from an event_id mapping.

    Searches for names containing 'target' (excluding 'non-target' variants)
    and 'nontarget'/'non_target'/'non-target'. Falls back to using the
    largest ID as target and smallest as non-target.

    Returns (target_id, nontarget_id) where nontarget_id may be None.
    """
    target_id: int | None = None
    nontarget_id: int | None = None

    for name, eid in event_id.items():
        name_lower = name.lower()
        if "target" in name_lower and "non" not in name_lower:
            target_id = eid
        elif (
            "nontarget" in name_lower
            or "non_target" in name_lower
            or "non-target" in name_lower
        ):
            nontarget_id = eid

    if target_id is None:
        ids = sorted(event_id.values())
        target_id = ids[-1]
        nontarget_id = ids[0] if len(ids) > 1 else None

    return target_id, nontarget_id


def load_p300_bi2013a(
    subjects: list[int] | None = None,
    bandpass: tuple[float, float] | None = (1.0, 20.0),
    resample: float | None = 128.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
    """Load BI2013a (Brain Invaders) P300 dataset via MOABB.

    Brain Invaders is a P300-based BCI game. 16 EEG channels, 512 Hz.
    24 subjects, many repetitions per target/non-target.

    Parameters
    ----------
    subjects : list of int, optional
        Which subjects to load. Default: [1].
    bandpass, resample : same as load_p300_bnci2014_009.

    Returns
    -------
    Same format as load_p300_bnci2014_009.
    """

    if subjects is None:
        subjects = [1]

    dataset = BI2013a()

    all_signals = []
    event_indices_list = []
    event_labels_list = []
    offset = 0
    sfreq = None
    ch_names = None

    for subj in subjects:
        subj_data = dataset.get_data(subjects=[subj])

        for _session_name, session_data in subj_data[subj].items():
            for _run_name, raw in session_data.items():
                if bandpass is not None:
                    raw.filter(bandpass[0], bandpass[1], verbose=False)

                if resample is not None:
                    raw.resample(resample, verbose=False)

                sfreq = raw.info["sfreq"]
                if ch_names is None:
                    ch_names = [ch for ch in raw.ch_names if ch != "stim"]

                # Pick only EEG channels
                raw.pick(picks="eeg", verbose=False)

                events, event_id = mne.events_from_annotations(raw, verbose=False)

                sig = raw.get_data().T  # (T, C)

                target_id, nontarget_id = _detect_target_nontarget_ids(event_id)

                for sample, _, eid in events:
                    if eid == target_id:
                        event_indices_list.append(offset + sample)
                        event_labels_list.append(1)
                    elif nontarget_id is not None and eid == nontarget_id:
                        event_indices_list.append(offset + sample)
                        event_labels_list.append(0)

                all_signals.append(sig)
                offset += sig.shape[0]

    if not all_signals:
        raise ValueError("No data loaded from BI2013a")
    if sfreq is None:
        raise ValueError("No valid runs loaded, sfreq not set")

    data = np.vstack(all_signals)
    event_indices = np.array(event_indices_list, dtype=int)
    event_labels = np.array(event_labels_list, dtype=int)

    info = {
        "ch_names": ch_names,
        "sfreq": sfreq,
        "n_target": int(np.sum(event_labels == 1)),
        "n_nontarget": int(np.sum(event_labels == 0)),
        "n_subjects": len(subjects),
        "dataset": "BI2013a",
    }

    return data, event_indices, event_labels, sfreq, info


if __name__ == "__main__":
    print("=" * 60)
    print("  P300 BCI Data Loader Verification")
    print("=" * 60)

    print("\n--- Loading BNCI2014-009 (subject 1) ---")
    try:
        data, ix, lb, fs, info = load_p300_bnci2014_009(subjects=[1])
        print(f"  Data shape: {data.shape}")
        print(f"  sfreq: {fs} Hz")
        print(
            f"  Events: {len(ix)} (target={info['n_target']}, non-target={info['n_nontarget']})"
        )
        print(f"  Channels: {len(info['ch_names'])}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n--- Loading BI2013a (subject 1) ---")
    try:
        data, ix, lb, fs, info = load_p300_bi2013a(subjects=[1])
        print(f"  Data shape: {data.shape}")
        print(f"  sfreq: {fs} Hz")
        print(
            f"  Events: {len(ix)} (target={info['n_target']}, non-target={info['n_nontarget']})"
        )
        print(f"  Channels: {len(info['ch_names'])}")
    except Exception as e:
        print(f"  Error: {e}")
