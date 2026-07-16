# Learning Regularization Structure for Biosignal Template Estimation

Companion code for:

> **Learning Regularization Structure for Biosignal Template Estimation**
> Y. Morin, A. Gramfort (under review)

## Overview

Estimating event-locked templates from biosignal recordings via regularized
least-squares requires choosing both the regularization structure and its
magnitude. This code implements a SURE-based framework that jointly optimizes
both, learning the penalty kernel directly from the data.

## Methods

| Name | Description |
|------|-------------|
| Average | Epoch averaging (baseline) |
| OLS | Ordinary least-squares (no regularization) |
| GCV | Generalized cross-validation λ selection |
| L-curve | Maximum-curvature λ selection |
| Emp. Bayes | MacKay's evidence maximization |
| SURE(λ) | SURE-optimized λ with fixed [1,−2,1] kernel |
| SURE(kernel) | SURE-optimized convolution kernel (proposed) |

## Datasets

- **EEG**: MNE sample dataset — auditory ERPs, 59 channels, 600 Hz
- **MEG**: MNE sample dataset — auditory ERFs, 204 gradiometers, 600 Hz
- **P300**: BNCI2014-009 — target/non-target BCI, 16 channels, 128 Hz
- **ECG**: MIT-BIH record 119 — Normal + PVC beats, 1 channel, 360 Hz

The MNE sample dataset (EEG + MEG) and the MIT-BIH ECG record download
automatically on first use. The P300 dataset downloads via MOABB on first use.
Datasets live in your home directory under `~/mne_data/` (the conventional MNE
location) — `~/mne_data/MNE-sample-data/` for EEG/MEG and `~/mne_data/mitdb/`
for the ECG record — not inside the repo. Override the base location with the
`MNE_DATA` environment variable.

## Installation

All scripts are plain Python and run directly from this folder — no build
system required. Create the conda environment from the provided file:

```bash
conda env create -f environment.yml
conda activate template-estimation
```

This installs numpy, scipy, pandas, matplotlib, scikit-learn, MNE (EEG/MEG),
MOABB (P300), and wfdb (ECG).

## Data setup

All datasets are stored **outside the repo**, in your home directory under
`~/mne_data/` (the conventional MNE location). Set the `MNE_DATA` environment
variable to use a different base directory.

| Dataset | Location | Download |
|---------|----------|----------|
| EEG / MEG (MNE sample) | `~/mne_data/MNE-sample-data/` | automatic on first use (`mne.datasets.sample.data_path`) |
| P300 (BNCI2014-009) | `~/mne_data/MNE-bnci-data/` | automatic on first use (MOABB) |
| ECG (MIT-BIH rec. 119) | `~/mne_data/mitdb/` | automatic on first use (`wfdb.dl_database`) |

`run_benchmarks.py` auto-downloads the ECG record via `wfdb.dl_database`, which
spawns a worker pool. In restricted/sandboxed shells that pool can fail with a
`PermissionError` from `multiprocessing`. If that happens, pre-download the
record once in a normal shell:

```bash
python -c "import wfdb; from pathlib import Path; \
d = Path.home()/'mne_data'/'mitdb'; d.mkdir(parents=True, exist_ok=True); \
wfdb.dl_database('mitdb', str(d), records=['119'])"
```

The loader skips the download when `~/mne_data/mitdb/119.{dat,hea,atr}` already
exist.

## Directory Structure

```
environment.yml                      # Conda environment definition
core.py                              # All estimation algorithms
data_loaders.py                      # EEG/MEG loader (MNE sample)
data_loaders_p300.py                 # P300 loader (BNCI2014-009)
data_loaders_ecg.py                  # ECG loader (MIT-BIH)
run_benchmarks.py                    # Reproduce all result CSVs (results/)
generate_metric_figures.py           # Distance-vs-K figures (Fig 3, 4, 6, 7, EEG)
generate_template_kernel_figures.py  # Synthetic templates, kernel frequency response
plot_sim_templates.py                # Fig 1b — simulated templates
plot_colored_noise_sim.py            # Fig 5 — colored-noise simulation
plot_template_eeg.py                 # EEG auditory L/R ERP template figures
plot_kernels_cross_dataset.py        # Cross-dataset kernel frequency response
plot_meg_combined.py                 # MEG combined (filtered / no-filter) example
plot_template_composite.py           # Cross-modality template composite
example_ecg_template.py              # ECG (Normal / PVC) template example
results/                             # Generated CSVs (created on first run)
figures/                             # Generated PDFs/PNGs (created on first run)
tests/test_core.py                   # Unit tests
supplementary/                       # Supplementary analyses (compute + plot)
    run_convergence_analysis.py      #   SURE kernel optimizer convergence (data)
    plot_convergence_kernels.py      #   convergence figure
    run_phi_sensitivity.py           #   AR(1) coefficient sensitivity (data)
    plot_phi_sensitivity.py          #   AR(1) coefficient sensitivity (figure)
    run_numerical_stability.py       #   numerical stability at high phi (data)
    plot_stability_condition_numbers.py  # numerical-stability figure
    run_multi_subject_p300.py        #   10-subject P300 validation table
```

All scripts use folder-local imports (`from core import ...`) and write their
outputs into `figures/` and `results/` next to this README. Run them from this
directory.

## Reproducing the paper figures

Run everything from inside this folder with the `template-estimation` env
active.

```bash
# 1. Unit tests
python -m pytest tests/test_core.py

# 2. Benchmarks — writes results/*_final2.csv (EEG, P300, ECG, simulations)
python run_benchmarks.py

# 3. Distance-vs-K figures — reads results/*_final2.csv; writes fig3, fig4,
#    fig6_p300, fig7_ecg, fig_eeg
python generate_metric_figures.py

# 4. Synthetic templates + kernel frequency response
python generate_template_kernel_figures.py

# 5. Fig 1b — simulated templates
python plot_sim_templates.py

# 6. Fig 5 — colored-noise simulation (computes results/colored_noise_sim_noise01.csv
#    on first run, then plots)
python plot_colored_noise_sim.py

# 7. EEG auditory L/R ERP template figures
python plot_template_eeg.py

# 8. Cross-dataset kernel frequency response
python plot_kernels_cross_dataset.py

# 9. MEG combined — broadband, and 1-30 Hz with --filtered
python plot_meg_combined.py
python plot_meg_combined.py --filtered

# 10. Cross-modality template composite
python plot_template_composite.py

# 11. ECG template example (choose the beat type used in the paper)
python example_ecg_template.py --beat-type PVC
```

### Paper figure map

Each script writes the figure under the exact name the paper uses. The
correspondence between each paper figure and the script that produces it:

| Paper figure | Script | Output in `figures/` |
|--------------|--------|----------------------|
| Fig. 1b — simulated templates | `plot_sim_templates.py` | `fig1b_templates.pdf` |
| Fig. 3 — noise sweep | `generate_metric_figures.py` | `fig3_noise_sweep.pdf` |
| Fig. 4 — n-events sweep | `generate_metric_figures.py` | `fig4_nevents_sweep.pdf` |
| Fig. 5 — colored-noise simulation | `plot_colored_noise_sim.py` | `fig5_colored_noise_sim.pdf` |
| Fig. 6 — P300 | `generate_metric_figures.py` | `fig6_p300.pdf` |
| Fig. 7 — ECG | `generate_metric_figures.py` | `fig7_ecg.pdf` |
| EEG distance | `generate_metric_figures.py` | `fig_eeg.pdf` |
| EEG auditory L/R templates | `plot_template_eeg.py` | `fig_template_eeg_aud_L.pdf`, `fig_template_eeg_aud_R.pdf` |
| Cross-dataset kernels | `plot_kernels_cross_dataset.py` | `fig_kernels_cross_dataset.pdf` |
| MEG combined (filtered / no-filter) | `plot_meg_combined.py` (`--filtered`) | `fig_meg_combined_filtered.pdf`, `fig_meg_combined_nofilter.pdf` |
| Template composite | `plot_template_composite.py` | `fig_template_composite.pdf` |
| ECG template (PVC) | `example_ecg_template.py --beat-type PVC` | `fig_template_ecg_pvc.pdf` |
| Convergence (supplementary) | `supplementary/run_convergence_analysis.py` → `plot_convergence_kernels.py` | `convergence_kernels.pdf` |
| φ sensitivity (supplementary) | `supplementary/run_phi_sensitivity.py` → `plot_phi_sensitivity.py` | `phi_sensitivity.pdf` |
| Numerical stability (supplementary) | `supplementary/plot_stability_condition_numbers.py` | `stability_condition_numbers.pdf` |

The supplementary analyses follow a **compute → plot** split: each `run_*.py`
writes its data CSV into `supplementary/`, and the matching `plot_*.py` renders
the figure into `figures/`. The plot scripts also recompute the CSV on the fly
if it is absent.

### Supplementary analyses (`supplementary/`)

```bash
# Convergence
python supplementary/run_convergence_analysis.py && python supplementary/plot_convergence_kernels.py
# phi sensitivity
python supplementary/run_phi_sensitivity.py && python supplementary/plot_phi_sensitivity.py
# Numerical stability
python supplementary/plot_stability_condition_numbers.py
# 10-subject P300 validation table
python supplementary/run_multi_subject_p300.py
```

### Notes / caveats

- `generate_metric_figures.py` consumes the `*_final2.csv` files produced by
  `run_benchmarks.py`; run the benchmarks first.
- Monte-Carlo results are seeded (`np.random.RandomState(42)`), so re-running
  `run_benchmarks.py` reproduces the committed numbers exactly.
- `supplementary/` `run_*.py` scripts write their data CSVs next to themselves
  inside `supplementary/`; the matching `plot_*.py` scripts write figures into
  `figures/`.
- The ECG download uses `wfdb.dl_database`, which spawns a worker pool. In some
  sandboxed shells that pool cannot be created; if you hit a `PermissionError`
  from `multiprocessing`, fetch record 119 manually from PhysioNet
  (`119.dat/.hea/.atr` into `~/mne_data/mitdb/`) and re-run — the loader skips
  the download when the files already exist.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for
details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md).
