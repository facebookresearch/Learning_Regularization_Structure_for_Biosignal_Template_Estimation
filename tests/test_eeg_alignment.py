# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Execute the EEG loader against MNE's absolute-sample event convention."""

from __future__ import annotations

import ast
import importlib
import unittest
from pathlib import Path
from typing import Any, TYPE_CHECKING
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_array_equal
from numpy.typing import NDArray

if TYPE_CHECKING:
    import core
else:
    core = importlib.import_module("core")


class RawFixture:
    """Small interface used by the continuous branch of load_eeg_erp."""

    def __init__(self, first_sample: int, data: NDArray) -> None:
        self.first_samp = first_sample
        self.info: dict[str, Any] = {"sfreq": 10.0, "bads": ["EEG 002"]}
        self.ch_names = ["EEG 001", "EEG 002"]
        self.data = data

    def get_data(self) -> NDArray:
        return self.data.T

    def pick(self, *, picks: str, verbose: bool) -> None:
        pass

    def drop_channels(self, names: list[str]) -> None:
        keep = [name not in names for name in self.ch_names]
        self.data = self.data[:, keep]
        self.ch_names = [name for name in self.ch_names if name not in names]
        self.info["bads"] = [name for name in self.info["bads"] if name not in names]


class MneFixture:
    """Return a fixed recording and its absolute-sample event array."""

    def __init__(self, raw: RawFixture, events: NDArray) -> None:
        self.io = self
        self.raw = raw
        self.events = events

    def read_raw_fif(self, path: str, *, preload: bool, verbose: bool) -> RawFixture:
        return self.raw

    def find_events(
        self, raw: RawFixture, *, stim_channel: str, verbose: bool
    ) -> NDArray:
        return self.events


def execute_loader(
    first_sample: int, *, return_epochs: bool = False, **options: bool
) -> tuple[Any, NDArray, NDArray]:
    # Execute the maintained function without optional download dependencies.
    source = Path(core.__file__).with_name("data_loaders.py").read_text()
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "load_eeg_erp"
    )
    data = np.arange(100.0).reshape(50, 2)
    events = np.array(
        [
            [first_sample + center, 0, label]
            for center, label in ((2, 1), (15, 2), (38, 3), (48, 1))
        ]
    )
    mne_fixture = MneFixture(RawFixture(first_sample, data), events)
    namespace: dict[str, Any] = {
        "np": np,
        "Path": Path,
        "EEG_DIR": Path("unused_eeg_fixture"),
        "mne": mne_fixture,
    }
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]), "<EEG loader test>", "exec"
        ),
        namespace,
    )
    with patch.object(Path, Path.exists.__name__, return_value=True):
        result = namespace["load_eeg_erp"](
            event_ids={"auditory_left": 1, "auditory_right": 2},
            bandpass=None,
            tmin=-0.2,
            tmax=0.3,
            return_epochs=return_epochs,
            **options,
        )
    return result, data, events


class TestEegAlignment(unittest.TestCase):
    def test_continuous_events_are_relative_to_returned_data(self) -> None:
        for first_sample in (0, 25800):
            with self.subTest(first_sample=first_sample):
                result, data, events = execute_loader(first_sample)
                actual, indices, labels = result
                assert_array_equal(actual, data)
                assert_array_equal(indices, [2, 15, 48])
                assert_array_equal(labels, [1, 2, 1])
                # Conversion must not mutate MNE's absolute event array.
                assert_array_equal(
                    events[:, 0], np.array([2, 15, 38, 48]) + first_sample
                )

    def test_epoch_windows_use_the_same_relative_indices(self) -> None:
        for first_sample in (0, 25800):
            with self.subTest(first_sample=first_sample):
                result, data, _ = execute_loader(first_sample, return_epochs=True)
                assert_array_equal(
                    result["epochs"], np.stack([data[:5].T, data[13:18].T])
                )
                assert_array_equal(result["labels"], [1, 2])
                self.assertEqual(result["sfreq"], 10.0)


class BadChannelTest(unittest.TestCase):
    def test_exclude_bads_drops_marked_channels_from_both_formats(self) -> None:
        # EEG 053 is marked bad in the MNE sample file; the paper's EEG
        # regeneration uses the remaining 59 channels.
        result, data, _ = execute_loader(0, exclude_bads=True, return_info=True)
        actual, indices, labels, info = result
        assert_array_equal(actual, data[:, :1])
        assert_array_equal(indices, [2, 15, 48])
        assert_array_equal(labels, [1, 2, 1])
        self.assertEqual(info, {"ch_names": ["EEG 001"], "sfreq": 10.0})
        epochs, _, _ = execute_loader(0, return_epochs=True, exclude_bads=True)
        self.assertEqual(epochs["ch_names"], ["EEG 001"])
        assert_array_equal(
            epochs["epochs"], np.stack([data[:5, :1].T, data[13:18, :1].T])
        )

    def test_return_info_keeps_marked_channels_by_default(self) -> None:
        result, data, _ = execute_loader(0, return_info=True)
        actual, _, _, info = result
        assert_array_equal(actual, data)
        self.assertEqual(info, {"ch_names": ["EEG 001", "EEG 002"], "sfreq": 10.0})

    def test_return_info_is_rejected_with_epochs(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "return_info=True needs the continuous format"
        ):
            execute_loader(0, return_epochs=True, return_info=True)


if __name__ == "__main__":
    unittest.main()
