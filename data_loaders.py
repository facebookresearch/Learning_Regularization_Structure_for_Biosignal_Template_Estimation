# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Loader of the MNE sample auditory/visual EEG recording.

Returns the recording in the format:
    (data, event_indices, event_labels)

where:
    data:          np.ndarray of shape (T, C) - multichannel time series
    event_indices: np.ndarray of shape (N,)   - sample indices of events
    event_labels:  np.ndarray of shape (N,)   - integer label for each event

This matches the problem structure X = D*Phi + W, where events define when
templates (columns of Phi) are placed into the signal via the dictionary D.

Dataset: MNE sample dataset (auditory/visual event-related potentials),
59 EEG channels (EEG 053 marked bad), ~600 Hz, 4 event types, ~72 reps each.
"""

from __future__ import annotations

import os
from pathlib import Path

import mne
import mne.transforms
import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Datasets live in the user's home directory (the conventional MNE location),
# not inside the repo. Override with the MNE_DATA env var if set.
_DATA_DIR = Path(os.environ.get("MNE_DATA", Path.home() / "mne_data"))
EEG_DIR = _DATA_DIR


# ===================================================================
# EEG / ERP Loader  (MNE Sample Dataset)
# ===================================================================


def load_eeg_erp(  # noqa: C901
    data_dir: str | Path | None = None,
    pick_channels: str = "eeg",
    event_ids: dict[str, int] | None = None,
    bandpass: tuple[float, float] | None = (1.0, 40.0),
    tmin: float = -0.2,
    tmax: float = 0.5,
    return_epochs: bool = False,
    exclude_bads: bool = False,
    return_info: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]
    | dict
):
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
        Folder holding ``MNE-sample-data/``. Defaults to ``$MNE_DATA`` or
        ``~/mne_data``.
    pick_channels : str
        "eeg" (60 ch, including the marked bad EEG 053), "mag" (102 ch),
        "grad" (204 ch), or "all".
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
    exclude_bads : bool
        If True, drop the picked channels listed in ``raw.info["bads"]``
        (EEG 053 for "eeg", leaving 59 channels), in both output formats.
    return_info : bool
        If True, return ``{"ch_names": [...], "sfreq": float}`` as a fourth
        value of the continuous format. Not allowed with return_epochs=True,
        whose dict already holds 'ch_names' and 'sfreq'.

    Returns
    -------
    data : ndarray (T, C)
        Continuous multichannel data.
    event_indices : ndarray (N,)
        Zero-based sample indices relative to the returned data array.
    event_labels : ndarray (N,)
        Integer label for each event.
    info : dict
        Only with return_info=True: channel names and sampling rate of *data*.
    """

    if return_info and return_epochs:
        raise ValueError(
            "return_info=True needs the continuous format: the return_epochs=True "
            "dict already holds 'ch_names' and 'sfreq'"
        )

    if data_dir is None:
        data_dir = EEG_DIR

    data_dir = Path(data_dir)

    # Find the raw .fif file
    fif_path = data_dir / "MNE-sample-data" / "MEG" / "sample" / "sample_audvis_raw.fif"
    if not fif_path.exists():
        raise FileNotFoundError(
            f"MNE sample data not found at {fif_path}. "
            f"Download it with download_eeg_mne('{data_dir}')."
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

    if exclude_bads:
        raw.drop_channels(raw.info["bads"])

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

    # MNE events use absolute sample numbers; get_data() starts at first_samp.
    event_indices = events[:, 0] - raw.first_samp
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

    if return_info:
        info = {"ch_names": list(raw.ch_names), "sfreq": float(sfreq)}
        return data, event_indices, event_labels, info
    return data, event_indices, event_labels


def load_eeg_positions(data_dir: str | Path | None = None) -> dict[str, object]:
    """Channel positions of the MNE sample recording and the standard 10-05 montage.

    Reads only the header of ``sample_audvis_raw.fif``. Returns a JSON-able dict:
    ``recording`` (the file read), ``coord_frame`` ("head" when every EEG
    channel is digitized in the head frame), ``channels`` {EEG channel: [x, y, z]} (m, head
    frame), ``montage`` ("standard_1005") and ``standard_head`` (the montage placed in the head
    frame by its fiducials with ``mne.transforms.get_ras_to_neuromag_trans``).
    """
    fif_path = (
        Path(data_dir or EEG_DIR)
        / "MNE-sample-data"
        / "MEG"
        / "sample"
        / "sample_audvis_raw.fif"
    )
    if not fif_path.exists():
        raise FileNotFoundError(f"MNE sample data not found at {fif_path}")
    raw = mne.io.read_raw_fif(str(fif_path), preload=False, verbose=False)
    eeg = [
        c for c in raw.info["chs"] if c["kind"] == mne.io.constants.FIFF.FIFFV_EEG_CH
    ]
    head = all(c["coord_frame"] == mne.io.constants.FIFF.FIFFV_COORD_HEAD for c in eeg)
    pos = mne.channels.make_standard_montage("standard_1005").get_positions()
    trans = mne.transforms.get_ras_to_neuromag_trans(
        pos["nasion"], pos["lpa"], pos["rpa"]
    )
    return {
        "recording": fif_path.name,
        "coord_frame": "head" if head else "not head",
        "channels": {c["ch_name"]: [float(v) for v in c["loc"][:3]] for c in eeg},
        "montage": "standard_1005",
        "standard_head": {
            n: [float(v) for v in mne.transforms.apply_trans(trans, p)]
            for n, p in pos["ch_pos"].items()
        },
    }


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


# ===================================================================
# Main: verify the loader
# ===================================================================

if __name__ == "__main__":
    # The paper's EEG input: auditory left/right tones, 1-40 Hz, EEG 053 dropped.
    try:
        eeg_data, eeg_events, eeg_labels, eeg_info = load_eeg_erp(
            event_ids={"auditory_left": 1, "auditory_right": 2},
            exclude_bads=True,
            return_info=True,
        )
    except FileNotFoundError as e:
        print(f"Skipped: {e}")
    else:
        print(f"EEG data {eeg_data.shape} (T x C), sfreq {eeg_info['sfreq']} Hz")
        print(f"{len(eeg_events)} events")
        for lbl, cnt in zip(*np.unique(eeg_labels, return_counts=True)):
            print(f"  label {lbl}: {cnt} events")
        positions = load_eeg_positions()
        print(f"EEG positions in the {positions['coord_frame']} frame")
