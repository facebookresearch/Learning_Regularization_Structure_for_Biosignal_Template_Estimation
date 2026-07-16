# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

"""
Data loaders for real biosignal datasets.

Loads publicly available EEG and EMG datasets and returns them in the format:
    (data, event_indices, event_labels)

where:
    data:          np.ndarray of shape (T, C) - multichannel time series
    event_indices: np.ndarray of shape (N,)   - sample indices of events
    event_labels:  np.ndarray of shape (N,)   - integer label for each event

This matches the problem structure X = D*Phi + W, where events define when
templates (columns of Phi) are placed into the signal via the dictionary D.

Datasets:
    1. EEG/ERP: MNE sample dataset (auditory/visual event-related potentials)
       - 59 EEG channels, ~600 Hz, 4 event types, ~72 reps each
    2. EMG: GrabMyo (PhysioNet) - surface EMG during hand gestures
       - 16 forearm EMG channels, 2048 Hz, 17 gesture types, 7 reps each
"""


import os
import re
import zipfile
from pathlib import Path

import mne
import numpy as np

import requests

import wfdb


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Datasets live in the user's home directory (the conventional MNE location),
# not inside the repo. Override with the MNE_DATA env var if set.
_DATA_DIR = Path(os.environ.get("MNE_DATA", Path.home() / "mne_data"))
EEG_DIR = _DATA_DIR
_EMG_DIR = _DATA_DIR / "emg"


# ===================================================================
# EEG / ERP Loader  (MNE Sample Dataset)
# ===================================================================


def load_eeg_erp(
    data_dir: str | Path | None = None,
    pick_channels: str = "eeg",
    event_ids: dict[str, int] | None = None,
    bandpass: tuple[float, float] | None = (1.0, 40.0),
    tmin: float = -0.2,
    tmax: float = 0.5,
    return_epochs: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | dict:
    """Load the MNE sample dataset as continuous multichannel EEG + events.

    The MNE sample dataset contains auditory and visual event-related potentials
    recorded with a 306-channel Neuromag system (204 gradiometers, 102 magnetometers,
    60 EEG channels). We extract EEG channels by default.

    Event types (stimulus channel STI 014):
        1 = auditory left,  2 = auditory right
        3 = visual left,    4 = visual right
        5 = smiley face,   32 = button press

    Parameters
    ----------
    data_dir : path, optional
        Root of the MNE-sample-data directory.  Defaults to data/eeg/.
    pick_channels : str
        "eeg" (59 ch), "mag" (102 ch), "grad" (204 ch), or "all".
    event_ids : dict, optional
        Mapping of event names to integer IDs to include.
        Default: {"aud_l": 1, "aud_r": 2, "vis_l": 3, "vis_r": 4}.
    bandpass : tuple or None
        (low, high) Hz bandpass filter applied to raw data.
    tmin, tmax : float
        Epoch window relative to event onset (seconds). Only used if
        return_epochs=True.
    return_epochs : bool
        If True, return a dict with 'epochs' (N x C x T_epoch), 'labels',
        'times', and 'sfreq' instead of the continuous format.

    Returns
    -------
    data : ndarray (T, C)
        Continuous multichannel data.
    event_indices : ndarray (N,)
        Sample indices where events occur.
    event_labels : ndarray (N,)
        Integer label for each event.
    """

    if data_dir is None:
        data_dir = EEG_DIR

    data_dir = Path(data_dir)

    # Find the raw .fif file
    fif_path = data_dir / "MNE-sample-data" / "MEG" / "sample" / "sample_audvis_raw.fif"
    if not fif_path.exists():
        raise FileNotFoundError(
            f"MNE sample data not found at {fif_path}. "
            "Download it by running: "
            "mne.datasets.sample.data_path(path='data/eeg')"
        )

    # Default event IDs: the 4 main stimulus types
    if event_ids is None:
        event_ids = {
            "auditory_left": 1,
            "auditory_right": 2,
            "visual_left": 3,
            "visual_right": 4,
        }

    # Load raw data
    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)

    # Pick channels
    if pick_channels == "eeg":
        raw.pick(picks="eeg", verbose=False)
    elif pick_channels == "mag":
        raw.pick(picks="mag", verbose=False)
    elif pick_channels == "grad":
        raw.pick(picks="grad", verbose=False)
    elif pick_channels == "all":
        raw.pick(picks=["eeg", "meg"], verbose=False)
    else:
        raise ValueError(f"Unknown pick_channels={pick_channels}")

    # Apply bandpass filter
    if bandpass is not None:
        raw.filter(bandpass[0], bandpass[1], verbose=False)

    # Extract events from stimulus channel
    # Note: events are read from the stim channel which is separate from picked data
    raw_with_stim = mne.io.read_raw_fif(str(fif_path), preload=False, verbose=False)
    events = mne.find_events(raw_with_stim, stim_channel="STI 014", verbose=False)

    # Filter to requested event types
    event_values = set(event_ids.values())
    mask = np.isin(events[:, 2], list(event_values))
    events = events[mask]

    # Get continuous data as (T, C)
    data = raw.get_data().T  # raw.get_data() is (C, T), transpose to (T, C)
    sfreq = raw.info["sfreq"]

    event_indices = events[:, 0]  # sample indices
    event_labels = events[:, 2]  # event codes

    if return_epochs:
        n_pre = int(abs(tmin) * sfreq)
        n_post = int(tmax * sfreq)
        epoch_len = n_pre + n_post
        n_channels = data.shape[1]

        valid_mask = (event_indices - n_pre >= 0) & (
            event_indices + n_post <= data.shape[0]
        )
        valid_indices = event_indices[valid_mask]
        valid_labels = event_labels[valid_mask]

        epochs = np.zeros((len(valid_indices), n_channels, epoch_len))
        for i, idx in enumerate(valid_indices):
            epochs[i] = data[idx - n_pre : idx + n_post, :].T

        times = np.linspace(tmin, tmax, epoch_len, endpoint=False)
        return {
            "epochs": epochs,
            "labels": valid_labels,
            "times": times,
            "sfreq": sfreq,
            "ch_names": raw.ch_names,
            "event_ids": event_ids,
        }

    return data, event_indices, event_labels


# ===================================================================
# EMG Loader  (GrabMyo - PhysioNet)
# ===================================================================


def load_emg_grabmyo(
    data_dir: str | Path | None = None,
    session: int = 1,
    participant: int = 1,
    gestures: list[int] | None = None,
    use_forearm_only: bool = True,
    bandpass: tuple[float, float] | None = (20.0, 500.0),
    rectify: bool = False,
    return_epochs: bool = False,
    epoch_duration: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | dict:
    """Load GrabMyo EMG dataset from PhysioNet.

    GrabMyo contains surface EMG recordings from 43 participants performing
    17 hand gestures (including rest), with 7 repetitions per gesture per session.
    Recordings use 16 forearm + 12 wrist channels (bipolar), sampled at 2048 Hz.

    Each trial is a separate file containing ~5 seconds of EMG during one gesture.
    We concatenate all trials into a continuous signal and mark gesture onsets.

    Parameters
    ----------
    data_dir : path, optional
        Directory containing the grabmyo/ folder.
    session : int
        Session number (1, 2, or 3).
    participant : int
        Participant number (1-43).
    gestures : list of int, optional
        Which gesture IDs to include (1-17). Default: all available.
    use_forearm_only : bool
        If True, use only the 16 forearm channels (drop wrist/unused).
    bandpass : tuple or None
        (low, high) Hz bandpass filter.
    rectify : bool
        If True, full-wave rectify after filtering (common for EMG analysis).
    return_epochs : bool
        If True, return dict with 'epochs' array instead of continuous data.
    epoch_duration : float
        Duration of each epoch in seconds (default 5.0, matching trial length).

    Returns
    -------
    data : ndarray (T, C)
        Continuous multichannel EMG data (all trials concatenated with rest gaps).
    event_indices : ndarray (N,)
        Sample indices where each gesture trial starts.
    event_labels : ndarray (N,)
        Gesture ID for each event.
    """

    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    from scipy.signal import butter, sosfiltfilt

    if data_dir is None:
        data_dir = _EMG_DIR

    data_dir = Path(data_dir)
    subj_dir = (
        data_dir
        / "grabmyo"
        / f"Session{session}"
        / f"session{session}_participant{participant}"
    )

    if not subj_dir.exists():
        raise FileNotFoundError(
            f"GrabMyo data not found at {subj_dir}. Run download_emg_grabmyo() first."
        )

    # Discover available gesture/trial files
    hea_files = sorted(subj_dir.glob("*.hea"))
    if not hea_files:
        raise FileNotFoundError(f"No .hea files found in {subj_dir}")

    # Parse filenames: session{s}_participant{p}_gesture{g}_trial{t}

    trial_info: list[tuple[int, int, str]] = []  # (gesture, trial, record_path)
    for hf in hea_files:
        m = re.match(
            r"session(\d+)_participant(\d+)_gesture(\d+)_trial(\d+)\.hea",
            hf.name,
        )
        if m:
            g, t = int(m.group(3)), int(m.group(4))
            if gestures is None or g in gestures:
                rec_path = str(hf.with_suffix(""))
                trial_info.append((g, t, rec_path))

    trial_info.sort()  # sort by gesture then trial

    if not trial_info:
        raise ValueError(f"No trials found for gestures={gestures} in {subj_dir}")

    # Read all trials and concatenate
    all_signals: list[np.ndarray] = []
    event_indices_list: list[int] = []
    event_labels_list: list[int] = []
    current_offset = 0

    # Read first record to get metadata
    rec0 = wfdb.rdrecord(trial_info[0][2])
    fs = rec0.fs
    _ = rec0.n_sig
    sig_names = rec0.sig_name

    # Determine channel selection
    if use_forearm_only:
        # Forearm channels are the first 16 (8 bipolar pairs x 2 rings)
        # Wrist channels are 16-27, unused are 28-31
        ch_indices = list(range(16))
    else:
        # All 28 used channels (drop 4 unused)
        ch_indices = list(range(28))

    # Add rest gaps between trials (0.5 seconds of zeros)
    rest_gap_samples = int(0.5 * fs)

    for gesture_id, _trial_num, rec_path in trial_info:
        rec = wfdb.rdrecord(rec_path)
        sig = rec.p_signal[:, ch_indices]  # (T_trial, C)

        # Record event onset
        event_indices_list.append(current_offset)
        event_labels_list.append(gesture_id)

        all_signals.append(sig)
        current_offset += sig.shape[0]

        # Add rest gap
        rest = np.zeros((rest_gap_samples, len(ch_indices)))
        all_signals.append(rest)
        current_offset += rest_gap_samples

    data = np.vstack(all_signals)

    # Apply bandpass filter
    if bandpass is not None:
        nyq = fs / 2.0
        low = bandpass[0] / nyq
        high = bandpass[1] / nyq
        sos = butter(4, [low, high], btype="band", output="sos")
        data = sosfiltfilt(sos, data, axis=0)

    # Optionally rectify
    if rectify:
        data = np.abs(data)

    event_indices = np.array(event_indices_list, dtype=int)
    event_labels = np.array(event_labels_list, dtype=int)

    if return_epochs:
        n_epoch = int(epoch_duration * fs)
        n_ch = data.shape[1]
        valid = event_indices + n_epoch <= data.shape[0]
        valid_idx = event_indices[valid]
        valid_lbl = event_labels[valid]

        epochs = np.zeros((len(valid_idx), n_ch, n_epoch))
        for i, idx in enumerate(valid_idx):
            epochs[i] = data[idx : idx + n_epoch, :].T

        times = np.arange(n_epoch) / fs
        return {
            "epochs": epochs,
            "labels": valid_lbl,
            "times": times,
            "sfreq": fs,
            "ch_names": [sig_names[i] for i in ch_indices],
            "gestures": sorted(set(event_labels_list)),
        }

    return data, event_indices, event_labels


# ===================================================================
# EMG Loader  (UCI EMG for Gestures)
# ===================================================================


def load_emg_uci(
    data_dir: str | Path | None = None,
    subjects: list[int] | None = None,
    gestures: list[int] | None = None,
    bandpass: tuple[float, float] | None = (20.0, 450.0),
    return_epochs: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | dict:
    """Load the UCI EMG for Gestures dataset.

    This dataset has 36 subjects performing 6 hand gestures (+ rest),
    with 8 EMG channels sampled at ~1000 Hz.

    Note: only ~4 repetitions per gesture per subject, so this dataset
    is best used by pooling across subjects.

    Parameters
    ----------
    data_dir : path, optional
        Directory containing EMG_data_for_gestures-master/.
    subjects : list of int, optional
        Which subjects to load (1-36). Default: [1, 2, 3, 4, 5].
    gestures : list of int, optional
        Which gesture IDs to include (1-6). Default: all.
    bandpass : tuple or None
        (low, high) Hz bandpass filter.
    return_epochs : bool
        If True, return segmented epochs.

    Returns
    -------
    data : ndarray (T, C)
        Continuous 8-channel EMG data.
    event_indices : ndarray (N,)
    event_labels : ndarray (N,)
    """

    # Deferred: not compatible with Cinder lazy imports (loaded eagerly otherwise).
    from scipy.signal import butter, sosfiltfilt

    if data_dir is None:
        data_dir = _EMG_DIR

    data_dir = Path(data_dir)
    base = data_dir / "EMG_data_for_gestures-master"

    if not base.exists():
        raise FileNotFoundError(
            f"UCI EMG dataset not found at {base}. "
            "Download and extract emg_gestures.zip first."
        )

    if subjects is None:
        subjects = [1, 2, 3, 4, 5]

    all_signals: list[np.ndarray] = []
    event_indices_list: list[int] = []
    event_labels_list: list[int] = []
    offset = 0

    for subj in subjects:
        subj_dir = base / f"{subj:02d}"
        if not subj_dir.exists():
            continue

        txt_files = sorted(subj_dir.glob("*.txt"))
        for fpath in txt_files:
            raw = np.loadtxt(fpath, delimiter="\t", skiprows=1)
            # Columns: time, ch1-ch8, class
            emg_data = raw[:, 1:9]  # 8 EMG channels
            labels = raw[:, -1].astype(int)

            # Detect gesture onsets: transition from 0 (rest) to non-zero
            for i in range(1, len(labels)):
                if labels[i] != 0 and labels[i - 1] == 0:
                    if gestures is None or labels[i] in gestures:
                        event_indices_list.append(offset + i)
                        event_labels_list.append(labels[i])

            all_signals.append(emg_data)
            offset += emg_data.shape[0]

    if not all_signals:
        raise ValueError("No data loaded. Check subject directories.")

    data = np.vstack(all_signals)

    # Estimate sampling rate from time column (approximately 1000 Hz)
    fs = 1000.0

    # Apply bandpass filter
    if bandpass is not None:
        nyq = fs / 2.0
        low = bandpass[0] / nyq
        high = bandpass[1] / nyq
        sos = butter(4, [low, high], btype="band", output="sos")
        data = sosfiltfilt(sos, data, axis=0)

    event_indices = np.array(event_indices_list, dtype=int)
    event_labels = np.array(event_labels_list, dtype=int)

    if return_epochs:
        epoch_len = int(3.0 * fs)  # 3 second epochs
        n_ch = data.shape[1]
        valid = event_indices + epoch_len <= data.shape[0]
        valid_idx = event_indices[valid]
        valid_lbl = event_labels[valid]

        epochs = np.zeros((len(valid_idx), n_ch, epoch_len))
        for i, idx in enumerate(valid_idx):
            epochs[i] = data[idx : idx + epoch_len, :].T

        return {
            "epochs": epochs,
            "labels": valid_lbl,
            "times": np.arange(epoch_len) / fs,
            "sfreq": fs,
        }

    return data, event_indices, event_labels


# ===================================================================
# Download helpers
# ===================================================================


def download_eeg_mne(data_dir: str | Path | None = None) -> Path:
    """Download the MNE sample dataset."""

    if data_dir is None:
        data_dir = EEG_DIR
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    path = mne.datasets.sample.data_path(path=str(data_dir))
    print(f"MNE sample data at: {path}")
    return Path(path)


def download_emg_grabmyo(
    data_dir: str | Path | None = None,
    session: int = 1,
    participant: int = 1,
    gestures: list[int] | None = None,
) -> Path:
    """Download GrabMyo EMG data from PhysioNet for one participant."""

    if data_dir is None:
        data_dir = _EMG_DIR
    data_dir = Path(data_dir)

    subj_dir = (
        data_dir
        / "grabmyo"
        / f"Session{session}"
        / f"session{session}_participant{participant}"
    )
    subj_dir.mkdir(parents=True, exist_ok=True)

    base_url = (
        f"https://physionet.org/files/grabmyo/1.0.2/"
        f"Session{session}/session{session}_participant{participant}"
    )

    if gestures is None:
        gestures = list(range(1, 18))

    downloaded = 0
    skipped = 0
    for g in gestures:
        for t in range(1, 8):
            fname_base = (
                f"session{session}_participant{participant}_gesture{g}_trial{t}"
            )
            for ext in [".dat", ".hea"]:
                fname = fname_base + ext
                filepath = subj_dir / fname
                if filepath.exists():
                    skipped += 1
                    continue
                url = f"{base_url}/{fname}"
                r = requests.get(url, timeout=60)
                if r.status_code == 200:
                    filepath.write_bytes(r.content)
                    downloaded += 1
                else:
                    print(
                        f"  Warning: failed to download {fname} (HTTP {r.status_code})"
                    )

    print(
        f"GrabMyo: downloaded {downloaded} files, "
        f"skipped {skipped} existing. Dir: {subj_dir}"
    )
    return subj_dir


def download_emg_uci(data_dir: str | Path | None = None) -> Path:
    """Download the UCI EMG for Gestures dataset."""

    if data_dir is None:
        data_dir = _EMG_DIR
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "emg_gestures.zip"
    extracted = data_dir / "EMG_data_for_gestures-master"

    if extracted.exists():
        print(f"UCI EMG already extracted at {extracted}")
        return extracted

    if not zip_path.exists():
        url = "https://archive.ics.uci.edu/static/public/481/emg+data+for+gestures.zip"
        print(f"Downloading UCI EMG dataset from {url} ...")
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            f.writelines(r.iter_content(chunk_size=8192))
        print(f"  Saved to {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)

    print(f"UCI EMG extracted to {extracted}")
    return extracted


# ===================================================================
# Summary / diagnostic
# ===================================================================


def summarize_dataset(
    data: np.ndarray,
    event_indices: np.ndarray,
    event_labels: np.ndarray,
    name: str = "Dataset",
) -> None:
    """Print a summary of a loaded dataset."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Data shape:         {data.shape}  (T x C)")
    print(f"  Data dtype:         {data.dtype}")
    print(f"  Data range:         [{data.min():.6g}, {data.max():.6g}]")
    print(f"  Data mean (abs):    {np.abs(data).mean():.6g}")
    print(f"  Num events:         {len(event_indices)}")
    print(f"  Event index range:  [{event_indices.min()}, {event_indices.max()}]")

    unique_labels, counts = np.unique(event_labels, return_counts=True)
    print(f"  Event types:        {len(unique_labels)}")
    for lbl, cnt in zip(unique_labels, counts):
        print(f"    Label {lbl:3d}: {cnt:4d} events")
    print(f"{'=' * 60}\n")


# ===================================================================
# Main: verify loaders
# ===================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Data Loader Verification")
    print("=" * 60)

    # --- EEG/ERP ---
    print("\n--- Loading EEG/ERP (MNE Sample) ---")
    try:
        eeg_data, eeg_events, eeg_labels = load_eeg_erp()
        summarize_dataset(eeg_data, eeg_events, eeg_labels, "EEG/ERP (MNE Sample)")
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")

    # --- EMG: GrabMyo ---
    print("\n--- Loading EMG (GrabMyo) ---")
    try:
        emg_data, emg_events, emg_labels = load_emg_grabmyo()
        summarize_dataset(emg_data, emg_events, emg_labels, "EMG (GrabMyo)")
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")

    # --- EMG: UCI ---
    print("\n--- Loading EMG (UCI Gestures) ---")
    try:
        uci_data, uci_events, uci_labels = load_emg_uci()
        summarize_dataset(uci_data, uci_events, uci_labels, "EMG (UCI Gestures)")
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")
