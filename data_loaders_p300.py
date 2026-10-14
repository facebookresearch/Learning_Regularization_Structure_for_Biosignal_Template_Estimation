# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""BNCI2014-009 P300 speller data (10 subjects, 16 EEG channels) via MOABB."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import mne
import numpy as np

from moabb.datasets import BNCI2014_009

if TYPE_CHECKING:
    import heldout_p300
else:
    heldout_p300 = importlib.import_module(
        f"{__package__}.heldout_p300" if __package__ else "heldout_p300"
    )


def load_bnci2014_009_blocks(
    subject: int,
    bandpass: tuple[float, float] = (1.0, 20.0),
    resample: float = 128.0,
) -> list[heldout_p300.RecordingBlock]:
    """Load one BNCI2014-009 subject as three separately preprocessed blocks.

    Each stored block is filtered and resampled on its own, so no filter or
    resampler mixes samples across blocks. Data are in volts with the 16 EEG
    channels in stored order. Onsets are given at the native and the resampled
    rate, and labels come from the explicit event names (Target = 1,
    NonTarget = 0).
    """
    sessions = BNCI2014_009().get_data(subjects=[subject])[subject]
    raws = [raw for runs in sessions.values() for raw in runs.values()]
    if len(raws) != 3:
        raise ValueError(f"Subject {subject} has {len(raws)} stored blocks, expected 3")
    blocks = []
    for block, raw in enumerate(raws):
        native, native_id = mne.events_from_annotations(raw, verbose=False)
        native_first = raw.first_samp
        raw.filter(bandpass[0], bandpass[1], verbose=False)
        raw.resample(resample, verbose=False)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        if event_id != native_id:
            raise ValueError(
                f"Block {block}: event names changed in resampling, "
                f"{native_id} -> {event_id}"
            )
        if not np.array_equal(events[:, 2], native[:, 2]):
            raise ValueError(f"Block {block}: event codes changed in resampling")
        raw.pick(picks="eeg", verbose=False)
        if raw.info["sfreq"] != resample:
            raise ValueError(
                f"Block {block}: sampling rate {raw.info['sfreq']} after resampling, "
                f"expected {resample}"
            )
        if len(raw.ch_names) != 16:
            raise ValueError(
                f"Block {block}: {len(raw.ch_names)} EEG channels, expected 16"
            )
        if blocks and tuple(raw.ch_names) != blocks[0].channels:
            raise ValueError(
                f"Block {block}: channels {raw.ch_names} differ from block 0 "
                f"{list(blocks[0].channels)}"
            )
        blocks.append(
            heldout_p300.RecordingBlock(
                subject=subject,
                block=block,
                data=raw.get_data().T,
                onsets=events[:, 0] - raw.first_samp,
                native_onsets=native[:, 0] - native_first,
                labels=heldout_p300.event_labels(events[:, 2], event_id),
                channels=tuple(raw.ch_names),
            )
        )
    return blocks


if __name__ == "__main__":
    print("=" * 60)
    print("  P300 BCI Data Loader Verification")
    print("=" * 60)

    print("\n--- Loading BNCI2014-009 blocks (subject 1) ---")
    try:
        for b in load_bnci2014_009_blocks(1):
            print(
                f"  Block {b.block}: data {b.data.shape}, {len(b.onsets)} events "
                f"(target={int(np.sum(b.labels == 1))}), {len(b.channels)} channels"
            )
    except Exception as e:
        print(f"  Error: {e}")
