# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Separately recorded blocks as separate domains for colored-noise SURE."""

from __future__ import annotations

import importlib
import unittest
import warnings
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray

if TYPE_CHECKING:
    import core
else:
    core = importlib.import_module("core")

HALF_WIN = 20
BLOCK_LENGTH = 240
# Zero samples between blocks in the reference layout. Cross-block covariance
# at this lag is far below double precision for the fitted noise models.
GAP = 20000
LAYOUT = (
    ([60, 120, 170, 220], ["A", "B", "A", "B"]),
    ([20, 70, 130, 180], ["B", "A", "B", "A"]),
)
SURE_PATHS: dict[str, dict[str, Any]] = {
    "scalar": {"opt_reg_temporal": True},
    "kernel": {"opt_reg_temporal": True, "vec_mode": True, "symm": True, "vlen": 3},
}
COLORED = ("ar1_colored", "ar3_colored")

Layout = tuple[NDArray, NDArray, NDArray, NDArray]


def recorded_blocks() -> list[tuple[NDArray, NDArray, NDArray]]:
    """Two blocks of strongly correlated noise with responses near both ends."""
    rs = np.random.RandomState(3)
    phase = np.linspace(0, np.pi, 2 * HALF_WIN)[:, None]
    shapes = {
        "A": np.sin(phase) * np.array([[1.0, 0.5]]),
        "B": np.cos(phase) * np.array([[0.3, -1.0]]),
    }
    blocks = []
    for onsets, labels in LAYOUT:
        innovations = rs.normal(size=(BLOCK_LENGTH, 2))
        noise = np.zeros_like(innovations)
        for t in range(BLOCK_LENGTH):
            noise[t] = (0.99 * noise[t - 1] if t else 0) + innovations[t]
        data = 0.3 * noise
        for onset, label in zip(onsets, labels):
            data[onset - HALF_WIN : onset + HALF_WIN] += shapes[label]
        blocks.append((data, np.array(onsets), np.array(labels)))
    return blocks


def concatenate(blocks: list[tuple[NDArray, NDArray, NDArray]], gap: int = 0) -> Layout:
    """Join blocks in order, optionally separated by zero samples."""
    pieces, onsets, labels, starts = [], [], [], []
    offset = 0
    for index, (data, block_onsets, block_labels) in enumerate(blocks):
        if index and gap:
            pieces.append(np.zeros((gap, data.shape[1])))
            offset += gap
        starts.append(offset)
        pieces.append(data)
        onsets.append(block_onsets + offset)
        labels.append(block_labels)
        offset += len(data)
    return (
        np.vstack(pieces),
        np.concatenate(onsets),
        np.concatenate(labels),
        np.array(starts),
    )


def fit(
    layout: Layout, path: str, noise_est: str, with_blocks: bool
) -> tuple[NDArray, NDArray]:
    """Return stacked templates and the selected lambda or kernel."""
    data, ix, lb, starts = layout
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        templates, inner = core.get_events_templates(
            data,
            ix,
            lb,
            half_win=HALF_WIN,
            method="ls",
            noise_est=noise_est,
            block_starts=starts if with_blocks else None,
            **SURE_PATHS[path],
        )
    assert inner is not None, f"{path} SURE returned no diagnostics"
    stacked = np.stack([templates[key] for key in sorted(templates, key=str)])
    return stacked, np.asarray(inner["opt_val"], dtype=float)


def relative(actual: NDArray, expected: NDArray) -> float:
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


class BlockCovarianceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.blocks = recorded_blocks()
        self.joined = concatenate(self.blocks)
        self.events = np.array(["A", "B"])

    def per_block_sum(self, build: str, parameter: NDArray | float) -> NDArray:
        total = np.zeros((4 * HALF_WIN, 4 * HALF_WIN))
        for _data, onsets, labels in self.blocks:
            if build == "ar1":
                total += core.build_DtSigmaD_ar1(
                    onsets, labels, self.events, HALF_WIN, float(parameter), 0.7
                )
            else:
                total += core.build_DtSigmaD_acf(
                    onsets, labels, self.events, HALF_WIN, np.asarray(parameter)
                )
        return total

    def test_builders_sum_block_covariances_without_cross_pairs(self) -> None:
        # The joined builder must equal independently built per-block
        # covariances; joining without labels adds the artificial terms.
        data, ix, lb, starts = self.joined
        event_blocks = np.searchsorted(starts, ix, side="right") - 1
        acf = core.estimate_acf_ar(data, ix, HALF_WIN, max_lag=2 * BLOCK_LENGTH)
        for build, parameter in (("ar1", 0.99), ("acf", acf)):
            with self.subTest(build=build):
                expected = self.per_block_sum(build, parameter)
                if build == "ar1":
                    blockwise = core.build_DtSigmaD_ar1(
                        ix,
                        lb,
                        self.events,
                        HALF_WIN,
                        0.99,
                        0.7,
                        len(data),
                        event_blocks,
                    )
                    joined = core.build_DtSigmaD_ar1(
                        ix, lb, self.events, HALF_WIN, 0.99, 0.7, len(data)
                    )
                else:
                    blockwise = core.build_DtSigmaD_acf(
                        ix, lb, self.events, HALF_WIN, acf, len(data), event_blocks
                    )
                    joined = core.build_DtSigmaD_acf(
                        ix, lb, self.events, HALF_WIN, acf, len(data)
                    )
                scale = np.linalg.norm(expected)
                assert_allclose(blockwise, expected, rtol=1e-12, atol=1e-12 * scale)
                self.assertGreater(relative(joined, expected), 1e-3)

    def test_both_sure_paths_match_separate_block_covariance(self) -> None:
        # Forwarding block labels to either SURE path gives the fit with no
        # cross-block covariance; the ordinary joined fit measurably differs.
        reference = concatenate(self.blocks, GAP)
        for path in SURE_PATHS:
            for noise_est in COLORED:
                with self.subTest(path=path, noise_est=noise_est):
                    expected, expected_value = fit(reference, path, noise_est, False)
                    actual, value = fit(self.joined, path, noise_est, True)
                    joined, _ = fit(self.joined, path, noise_est, False)
                    assert_allclose(value, expected_value, rtol=1e-10)
                    self.assertLess(relative(actual, expected), 1e-10)
                    self.assertGreater(relative(joined, expected), 1e-6)

    def test_block_order_does_not_change_fits(self) -> None:
        # Same selected events, blocks joined in the opposite order. Bounded
        # refinement moves scalar lambda by about 1e-7 for rounding-level
        # changes, so the tolerances sit well below the joined differences.
        swapped = concatenate(self.blocks[::-1])
        for path in SURE_PATHS:
            for noise_est in COLORED:
                with self.subTest(path=path, noise_est=noise_est):
                    expected, expected_value = fit(self.joined, path, noise_est, True)
                    actual, value = fit(swapped, path, noise_est, True)
                    joined, _ = fit(self.joined, path, noise_est, False)
                    joined_swapped, _ = fit(swapped, path, noise_est, False)
                    assert_allclose(value, expected_value, rtol=1e-5)
                    self.assertLess(relative(actual, expected), 1e-7)
                    self.assertGreater(relative(joined_swapped, joined), 1e-6)

    def test_single_block_reproduces_default_bitwise(self) -> None:
        data, ix, lb, _ = self.joined
        single = (data, ix, lb, np.array([0]))
        for path in SURE_PATHS:
            for noise_est in COLORED:
                with self.subTest(path=path, noise_est=noise_est):
                    expected, expected_value = fit(single, path, noise_est, False)
                    actual, value = fit(single, path, noise_est, True)
                    assert_array_equal(actual, expected)
                    assert_array_equal(value, expected_value)
