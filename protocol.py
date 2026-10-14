# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Settings of the paper analyses: estimator tables and the P300 protocol.

The held-out window, guard, K, repetitions and selection seed are defined in
``heldout_p300`` (``HALF_WIN``, ``GUARD``, ``K``, ``REPETITIONS``, ``SEED``).
"""

from __future__ import annotations

from typing import Any, Final, NamedTuple, TypedDict

# ``core.get_events_templates`` settings. Simulation and ECG: METHODS_PAIRWISE;
# P300 and EEG: METHODS_AR1COL.
METHODS_PAIRWISE: Final[dict[str, dict[str, Any]]] = {
    "Average": {"method": "average"},
    "OLS": {"method": "ls"},
    "GCV": {"method": "ls", "reg_method": "gcv"},
    "L-curve": {"method": "ls", "reg_method": "lcurve"},
    "EmpBayes": {"method": "ls", "reg_method": "empbayes"},
    "SURE(lambda)": {
        "method": "ls",
        "reg_temporal": 500,
        "opt_reg_temporal": True,
        "noise_est": "pairwise",
    },
    "SURE(kernel)": {
        "method": "ls",
        "reg_temporal": 500,
        "opt_reg_temporal": True,
        "vec_mode": True,
        "symm": True,
        "vlen": 3,
        "noise_est": "pairwise",
    },
}

METHODS_AR1COL: Final[dict[str, dict[str, Any]]] = {
    "Average": {"method": "average"},
    "OLS": {"method": "ls"},
    "GCV": {"method": "ls", "reg_method": "gcv"},
    "L-curve": {"method": "ls", "reg_method": "lcurve"},
    "EmpBayes": {"method": "ls", "reg_method": "empbayes"},
    "SURE(lambda)": {
        "method": "ls",
        "reg_temporal": 500,
        "opt_reg_temporal": True,
        "noise_est": "ar1_colored",
    },
    "SURE(kernel)": {
        "method": "ls",
        "reg_temporal": 500,
        "opt_reg_temporal": True,
        "vec_mode": True,
        "symm": True,
        "vlen": 3,
        "noise_est": "ar1_colored",
    },
}

# Central P300 batch (BNCI2014-009).
SUBJECTS: Final[tuple[int, ...]] = tuple(range(1, 11))
BANDPASS: Final[tuple[float, float]] = (1.0, 20.0)
RESAMPLE: Final[float] = 128.0
N_BLOCKS: Final[int] = 3
N_CHANNELS: Final[int] = 16
BLOCK_CLASS_COUNTS: Final[dict[int, int]] = {0: 480, 1: 96}

CORE_METHODS: Final[tuple[str, ...]] = tuple(METHODS_AR1COL)
FILTERED: Final[str] = "Molina filtered DTW"
PLAIN: Final[str] = "Molina plain DTW"
COMPARATORS: Final[tuple[str, ...]] = (FILTERED, PLAIN)
ESTIMATORS: Final[tuple[str, ...]] = CORE_METHODS + COMPARATORS
REFERENCES: Final[tuple[str, ...]] = ("OLS", "EmpBayes", "Average")
POPULATION: Final[str] = "Population"
ANCHOR: Final[str] = "SURE(kernel)"
PRIMARY_COMPARATORS: Final[tuple[str, ...]] = ("EmpBayes", "SURE(lambda)", "Average")

# Protocol decisions, labelled by their OQ identifiers; the scripts read them by key.
DECISIONS: Final[dict[str, object]] = {
    # Reference-distance selections: a fresh RandomState(seed) per subject.
    "OQ2_selection_scheme": "per_subject_42",
    "OQ2_selection_seed": 42,
    "OQ3_bootstrap_resamples": 10000,
    "OQ3_bootstrap_seed": 42,
    "OQ3_alpha": 0.05,
    # Endpoints of secondary family S4 (Molina filtered DTW vs SURE(kernel)).
    "OQ4_s4_endpoints": ("distance_OLS", "heldout_auc"),
    # An exception raised by an estimator is a counted failure, not a stop.
    "OQ6_exceptions_are_failures": True,
    # Reference whose other-subject mean is the Population template.
    "OQ8_population_reference": "OLS",
}

# ---------------------------------------------------------------------------
# Simulations (run_simulations.py)
# ---------------------------------------------------------------------------

# True templates, the two white-noise sweeps (submitted benchmark settings).
SIM_T: Final[int] = 100
SIM_C: Final[int] = 8
SIM_HALF_WIN: Final[int] = 50
SIM_SEED: Final[int] = 42
SIM_REPETITIONS: Final[int] = 20
NOISE_SWEEP: Final[tuple[float, ...]] = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
NOISE_SWEEP_I: Final[int] = 20
NOISE_SWEEP_MEAN_INTERVAL: Final[int] = 40
NEVENTS_SWEEP: Final[tuple[int, ...]] = (4, 8, 15, 30, 50, 100)
NEVENTS_NOISE: Final[float] = 0.1
NEVENTS_MEAN_INTERVAL: Final[int] = 50

# AR(1) noise (PCN generator), its phi grid, and the 2x2 factorial: fixed
# [1, -2, 1] or learned kernel, white (pairwise) or colored (AR(1)) procedure.
PCN_I: Final[int] = 15
PCN_NOISE: Final[float] = 0.1
FACTORIAL_PHIS: Final[tuple[float, ...]] = (0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95, 0.98)
ORACLE_LAMBDAS: Final[tuple[float, float, int]] = (-1.0, 6.0, 100)  # np.logspace


class CoreSettings(TypedDict, total=False):
    """Keyword settings of ``core.get_events_templates`` used by the simulations."""

    method: str
    reg_temporal: float
    opt_reg_temporal: bool
    vec_mode: bool
    symm: bool
    vlen: int
    noise_est: str


SURE_LAMBDA: Final[CoreSettings] = {
    "method": "ls",
    "reg_temporal": 500,
    "opt_reg_temporal": True,
}
SURE_KERNEL: Final[CoreSettings] = {
    **SURE_LAMBDA,
    "vec_mode": True,
    "symm": True,
    "vlen": 3,
}
FACTORIAL_CELLS: Final[dict[str, CoreSettings]] = {
    "fixed_white": {**SURE_LAMBDA, "noise_est": "pairwise"},
    "fixed_colored": {**SURE_LAMBDA, "noise_est": "ar1_colored"},
    "learned_white": {**SURE_KERNEL, "noise_est": "pairwise"},
    "learned_colored": {**SURE_KERNEL, "noise_est": "ar1_colored"},
}

# Kernel length and symmetry (Q = 1 is ridge only); SURE(kernel)'s BFGS limits:
# core's own, raised, and raised again for fits still capped.
Q_CELLS: Final[dict[str, CoreSettings]] = {
    "Q1": {"vlen": 1, "symm": True},
    "Q3-sym": {"vlen": 3, "symm": True},
    "Q3-free": {"vlen": 3, "symm": False},
    "Q5-sym": {"vlen": 5, "symm": True},
    "Q5-free": {"vlen": 5, "symm": False},
}
BFGS_LIMITS: Final[tuple[int, int, int]] = (50, 500, 5000)

# Block decisions.
# OQ-B1: kernel magnitude on 256 frequencies; DC "near zero" below 1e-3 x max.
KERNEL_N_FREQS: Final[int] = 256
KERNEL_DC_REL_THRESHOLD: Final[float] = 1e-3
# OQ-B3 and OQ-B7 use the event-count generator (NEVENTS_*, RandomState(42 + r)).
# OQ-B3: 100 events per class; RandomState(42) picks K per class.
CAL_I: Final[int] = 100
CAL_K: Final[int] = 5
CAL_SELECTION_SEED: Final[int] = 42
# OQ-B5: RandomState(offset + 1000 * repetition + int(100 * phi)).
COLORED_MEAN_INTERVALS: Final[tuple[int, ...]] = (40, 200)
COLORED_SEED_OFFSET: Final[int] = 0
COLORED_STATIONARY_START: Final[bool] = False
# OQ-B7: 15 events per class, pairwise noise estimate.
Q_SIM_I: Final[int] = 15
Q_SIM_NOISE_EST: Final[str] = "pairwise"
# OQ-B13: known vs estimated covariance on new, stationary-start data.
COV_MEAN_INTERVAL: Final[int] = 40
COV_PHIS: Final[tuple[float, ...]] = (0.5, 0.85, 0.98)
COV_TRUE_VAR: Final[float] = 0.01
COV_STATIONARY_START: Final[bool] = True
COV_SEED_OFFSET: Final[int] = 100000


# PD02: bias of the baseline variance and phi estimates on unit-variance
# stationary AR(1) noise (the noise-bound check, condition p300_ar1).
class Pd02Control(NamedTuple):
    length: int = 102
    channels: int = 16
    phi: float = 0.88
    events: int = 10
    gap: int = 8
    replicates: int = 500
    seed: int = 73101


# ---------------------------------------------------------------------------
# Real-data inputs (run_real_data.py, run_noise_diagnostics.py)
# ---------------------------------------------------------------------------

# Auditory EEG (MNE sample data): left/right tones, EEG 053 (marked bad) dropped.
EEG_EVENT_IDS: Final[dict[str, int]] = {"auditory_left": 1, "auditory_right": 2}
EEG_LABELS: Final[dict[int, str]] = {1: "aud_L", 2: "aud_R"}
EEG_BANDPASS: Final[tuple[float, float]] = (1.0, 40.0)
EEG_HALF_WIN: Final[int] = 120
EEG_N_CHANNELS: Final[int] = 59
EEG_BAD: Final[str] = "EEG 053"
# MIT-BIH record 119, lead 0: Normal and PVC beats.
ECG_RECORD: Final[str] = "119"
ECG_LEAD: Final[int] = 0
ECG_BANDPASS: Final[tuple[float, float]] = (0.5, 40.0)
ECG_BEATS: Final[tuple[str, ...]] = ("N", "V")
ECG_LABELS: Final[dict[int, str]] = {0: "Normal", 1: "PVC"}
ECG_HALF_WIN: Final[int] = 126
PHI_CAP: Final[float] = 0.99  # core.estimate_phi's clip; a fit at it is flagged


# ---------------------------------------------------------------------------
# Noise-model diagnostics (run_noise_diagnostics.py)
# ---------------------------------------------------------------------------


class ArDiagnosticSettings(TypedDict):
    """Baseline segments of max(min_baseline, int(2 * half_win * baseline_frac))
    samples; CLS drops drop_first samples per segment; bootstrap sets after a burn-in,
    from RandomState(seed) per (dataset, channel) and RandomState([seed, 1]) for the
    bias-correction pilot sets. The OLS residual series reuse these values."""

    baseline_frac: float
    min_baseline: int
    drop_first: int
    acf_lags: int
    resid_lags: int
    bootstrap: int
    seed: int
    burn_in: int


class BiasCorrectionSettings(TypedDict):
    pilot: int
    iterations: int
    tol: float


class TriggerSettings(TypedDict):
    """OQ-B17: the AR order each modality's estimator assumes, the BIC margin, the
    residual-test level and the channel majority."""

    baseline_order: dict[str, int]
    bic_margin: float
    alpha: float
    channel_fraction: float


DIAG_SUBJECTS: Final[tuple[int, ...]] = (1, 2, 3)
AR_ORDERS: Final[tuple[int, ...]] = (0, 1, 2, 3)
NOISE_MODELS: Final[tuple[str, ...]] = ("pairwise", "ar1_colored", "ar3_colored")
NOISE_MODEL_METHODS: Final[tuple[str, ...]] = ("SURE(lambda)", "SURE(kernel)")
AR_DIAGNOSTICS: Final[ArDiagnosticSettings] = {
    "baseline_frac": 0.3,
    "min_baseline": 4,
    "drop_first": 3,
    "acf_lags": 10,
    "resid_lags": 5,
    "bootstrap": 200,
    "seed": 42,
    "burn_in": 2000,
}
BIAS_CORRECTION: Final[BiasCorrectionSettings] = {
    "pilot": 50,
    "iterations": 3,
    "tol": 0.005,
}
TRIGGER: Final[TriggerSettings] = {
    "baseline_order": {"EEG": 1, "ECG": 0},
    "bic_margin": 10.0,
    "alpha": 0.05,
    "channel_fraction": 0.5,
}
# Full-data OLS residual series: the bootstrap runs on
# the first two channels per modality.
RESIDUAL_BOOTSTRAP_CHANNELS: Final[dict[str, tuple[int, ...]]] = {
    "P300_s01": (0, 1),
    "EEG": (0, 1),
    "ECG": (0,),
}
RESIDUAL_ORTHOGONALITY_TOLERANCE: Final[float] = 1e-6
# P300 sensitivity analyses (run_p300_sensitivity.py)
# ---------------------------------------------------------------------------

# Subjects 1-3 at K = 5 with the central selections. Block decisions: the white
# P300 cells use the pairwise variance as in
# FACTORIAL_CELLS (OQ-B6); kernel length and symmetry at BANDPASS only
# (OQ-B8); SURE(lambda) is perturbed too (OQ-B11); the references are
# recomputed at SECOND_BANDPASS (OQ-B12).
SENSITIVITY_SUBJECTS: Final[tuple[int, ...]] = (1, 2, 3)
SECOND_BANDPASS: Final[tuple[float, float]] = (1.0, 30.0)
# ar1_phi = phi_hat ** (1 / f); the marginal variance is scaled at phi_hat.
PHI_FACTORS: Final[tuple[float, ...]] = (0.5, 0.75, 1.5, 2.0)
VARIANCE_SCALES: Final[tuple[float, ...]] = (0.8, 1.25)

# ---------------------------------------------------------------------------
# Timing (run_timing.py)
# ---------------------------------------------------------------------------

# Per method one warm-up and five timed calls on the same input. One factor is
# varied at a time from the base configuration (L samples, M event types, C
# channels).
TIMING_WARMUP: Final[int] = 1
TIMING_REPEATS: Final[int] = 5
TIMING_BASE: Final[dict[str, int]] = {"L": 100, "M": 2, "C": 16}
TIMING_SWEEPS: Final[dict[str, tuple[int, ...]]] = {
    "L": (50, 100, 200, 400),
    "M": (1, 2, 4, 8),
    "C": (1, 4, 16, 64, 128),
}


# Decision OQ-B15: simulated data with I = 5 events
# per class, RandomState(seed); METHODS_AR1COL (AR(1)-colored noise for SURE)
# plus the filtered Molina average; one instrumented call of each stage-split
# method. The auditory EEG case uses the first RB selection at K = 5.
class TimingSimulation(TypedDict):
    I: int
    noise: float
    mean_interval: int
    seed: int


TIMING_SIM: Final[TimingSimulation] = {
    "I": 5,
    "noise": 0.1,
    "mean_interval": 40,
    "seed": 42,
}
TIMING_STAGE_SPLIT: Final[tuple[str, ...]] = ("SURE(lambda)", "SURE(kernel)")
TIMING_EEG_K: Final[int] = 5
