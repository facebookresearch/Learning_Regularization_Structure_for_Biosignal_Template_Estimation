# Learning Regularization Structure for Biosignal Template Estimation

Companion code for:

> Y. Murin and A. Gramfort, "Learning Regularization Structure for Biosignal
> Template Estimation," IEEE Transactions on Biomedical Engineering, under review.

Event-locked templates are estimated by regularized least squares. SURE
selects both the regularization structure (a learned convolution kernel) and
its magnitude from the data, for white and AR(1) noise. The scripts below
produce the numbers, tables and figures of the paper and its supplement.

## Estimators

| Name | Description |
|------|-------------|
| Average | epoch averaging |
| OLS | ordinary least squares |
| GCV | generalized cross-validation choice of lambda |
| L-curve | maximum-curvature choice of lambda |
| Emp. Bayes | evidence maximization |
| SURE(lambda) | SURE choice of lambda with the fixed [1, -2, 1] kernel |
| SURE(kernel) | SURE choice of the convolution kernel (proposed) |
| Molina et al. | filtered DTW average (P300 comparator) |

## Library modules

- `core.py`: the estimators, noise variance and AR(1) estimation, the design matrix and the simulated data.
- `dtw_average.py`: the filtered DTW average of Molina et al. (2024).
- `heldout_p300.py`: P300 held-out block evaluation: folds, selections, scoring, AUC, Population control.
- `protocol.py`: estimator tables and the fixed settings of every analysis.
- `analysis_tools.py`: shared analysis code: caches and selections, the fit failure policy, distances, statistics (signed-rank, bootstrap, Holm), reference noise, L-curve verticality, the R2.9 rules, the worker pool, JSON output and `run_info.json`.
- `ar_tools.py`: AR(0)-AR(3) noise diagnostics (CLS fits, BIC, residual tests).
- `data_loaders.py`, `data_loaders_p300.py`, `data_loaders_ecg.py`: loaders of the MNE sample EEG, BNCI2014-009 and MIT-BIH record 119.
- `figure_style.py`: the paper's figure style.

The first six need only numpy, scipy and threadpoolctl. The scripts and
loaders also need MNE, MOABB, wfdb and matplotlib.

## Installation and data

```bash
conda env create -f environment.yml
conda activate template-estimation
```

The pins in `environment.yml` are the versions the results were reproduced
with. Run every script from this folder with one BLAS thread:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Data live outside the repo, under `--data-dir` (default `$MNE_DATA`, else
`~/mne_data`), and download on first use (except in
`supplementary/run_numerical_stability.py`; see below):

| Dataset | Location under the data dir | Download |
|---------|-----------------------------|----------|
| Auditory EEG (MNE sample) | `MNE-sample-data/` | `mne.datasets.sample` |
| P300 (BNCI2014-009) | `MNE-bnci-data/` | MOABB |
| ECG (MIT-BIH record 119) | `mitdb/` | `wfdb.dl_database` |

`wfdb.dl_database` starts a worker pool, which some sandboxed shells refuse
(`PermissionError` from `multiprocessing`). Then copy `119.dat`, `119.hea` and
`119.atr` from PhysioNet into `mitdb/`; the loader skips the download when they
exist.

## Analysis scripts

Each script writes `<out>/<analysis>/` (default `--out ./runs`) with
`per_fit.csv` (one row per fit), `summary.json` (every number the manuscript
cites from that analysis) and `run_info.json` (commit, arguments and settings,
package versions, BLAS threads, host, start and end time), plus the files
listed. `--workers N` sets the number of worker processes (default 1; not on
`run_timing.py`, which runs serially); `--subjects` (P300 scripts) and `--reps`
(selections per K, or simulation repetitions) select a quick slice.
`run_timing.py --reps` is the number of timed calls.

| Script | Inputs | Writes (besides the three files) | Serves |
|--------|--------|----------------------------------|--------|
| `run_p300.py` | BNCI2014-009 | `p300/`: `cache/`, `references/`, `population/`, `selections.json`, `fits/` | Table II; Figs. 4(a, b), 6, 7; Secs. III-D (phi), V-C; supplement: P300 statistical tests and per-subject results |
| `run_reference_noise.py` | `runs/p300/` | `reference_noise/`: `references_uV.npz` | Secs. V-C (rank, reference noise, Remark 2), VI-D; supplement: P300 protocol (conditioning), why a trivial template can be closer, squared-domain table |
| `run_p300_sensitivity.py` | `runs/p300/`, BNCI2014-009 | `p300_colored_factorial/`, `p300_q_symmetry/`, `p300_phi_variance/`: `templates.npz` (`p300_phi_variance/` also `bw1-30/`) | Secs. VI-A (ridge ablation), VI-B (phi); supplement: kernel length/symmetry and convergence, colored-noise P300 factorial, sensitivity to phi and the noise variance |
| `run_noise_diagnostics.py` | `runs/p300/selections.json`, BNCI2014-009, MNE sample, MIT-BIH | `noise_diagnostics/`: `inputs/` | Sec. VI-B; supplement: noise-model diagnostics and noise inputs |
| `run_r29.py` | `runs/p300/`, BNCI2014-009 | `r29/` | Remark 3 (Sec. V-C); supplement: what the held-out AUC measures, template-implied probabilities (post hoc) |
| `run_simulations.py` | none | `white_sims/`, `calibration/`, `colored_factorial/`, `q_symmetry/`, `known_covariance/`: `templates.npz` | Figs. 2, 3, 7; Sec. IV-C; supplement: noise level, kernel length/symmetry and convergence, colored-noise factorial and known covariance, reference rankings with a known truth |
| `run_real_data.py` | MNE sample, MIT-BIH | `real_data/`: `cache/`, `selections.json`, `references/`, `fits/` | Table I; Figs. 4(c, d), 5, 7; Secs. III-D (phi), V-B, V-D; supplement: EEG and ECG under all references, additional template waveforms |
| `run_timing.py` | MNE sample | `timing/` | Sec. VI-C; supplement: computation time |

The figure and table scripts read these folders and write `./figures` and
`./tables` (`--out`):

| Script | Writes |
|--------|--------|
| `make_figures.py` | `<name>.pdf`, `.png` for each figure it draws (below), and `fig_kernels_magnitude.json` (the kernel numbers the text cites) |
| `make_tables.py` | `<name>.tex` for each table it writes (below) |
| `plot_templates.py` | `fig_templates_real`, `figS_templates_eeg` (and `.json` with the subject/channel selection) |
| `plot_sim_templates.py` | `fig1b_templates` |

## Paper figures

| Figure | File | Drawn by | From |
|--------|------|----------|------|
| Fig. 1 | `fig1b_templates.pdf` | `plot_sim_templates.py` | none |
| Fig. 2 | `fig_sim_nevents_sweep.pdf` | `make_figures.py` | `white_sims` |
| Fig. 3 | `fig_sim_colored_simple.pdf` | `make_figures.py` | `colored_factorial` |
| Fig. 4 | `fig_templates_real.pdf` | `plot_templates.py` | `p300`, `real_data` |
| Fig. 5 | `fig_eeg_ecg_baselines.pdf` | `make_figures.py` | `real_data` |
| Fig. 6 | `fig_p300_kcurve_2panel.pdf` | `make_figures.py` | `p300` |
| Fig. 7 | `fig_kernels_magnitude.pdf` | `make_figures.py` | `p300`, `real_data`, `white_sims` |

Supplement:

| Figure | File | Drawn by | From |
|--------|------|----------|------|
| Fig. S1 | `figS_sim_noise_sweep.pdf` | `make_figures.py` | `white_sims` |
| Fig. S2 | `fig_sim_colored_factorial.pdf` | `make_figures.py` | `colored_factorial` |
| Fig. S3 | `figS_stability_condition_numbers.pdf` | `supplementary/plot_stability_condition_numbers.py` | `numerical_stability` |
| Fig. S4 | `figS_templates_eeg.pdf` | `plot_templates.py` | `real_data` |

## Paper tables

`make_tables.py` writes `<name>.tex`. Tables I and II and the squared-domain
table are the tabulars the manuscripts hold inline; the other files are the
supplement's `\input{supp_tables/<name>}` files.

| Table | File | From |
|-------|------|------|
| Table I | `tab_summary.tex` | `real_data` |
| Table II | `tab_p300_k5.tex` | `p300` |

Supplement, in order of appearance:

| Label | File | From |
|-------|------|------|
| `tab:sup_q5_conv` | `b3_q5_convergence.tex` | `q_symmetry`, `p300_q_symmetry` |
| `tab:sup_kernel_p300` | `b3_kernel_p300.tex` | `p300_q_symmetry` (Q1 row: `per_fit.csv`) |
| `tab:sup_colored_p300` | `b3_colored_p300.tex` | `p300_colored_factorial` |
| `tab:sup_known_cov` | `b3_known_cov.tex` | `known_covariance` |
| `tab:sup_calibration` | `b3_calibration.tex` | `calibration` |
| `tab:sup_eeg_ecg` | `b3_eeg_ecg_k5.tex` | `real_data` |
| `tab:sup_families` | `families.tex` | `p300` |
| `tab:sup_subjects` | `subjects.tex` | `p300` |
| `tab:sup_squared` | `tab_sup_squared.tex` | `reference_noise` |
| `tab:sup_gap` | `r29_gap.tex` | `r29` |
| `tab:sup_prob` | `r29_prob.tex` | `r29` |
| `tab:sup_diff` | `r29_diff.tex` | `r29` |
| `tab:sup_noise_model` | `b3_noise_model.tex` | `noise_diagnostics` |
| `tab:sup_phi` | `b3_phi_variance.tex` | `p300_phi_variance` |
| `tab:sup_timing` | `b3_timing.tex` | `timing` |

## Run order and commands

`run_p300.py` runs first: `run_reference_noise.py`, `run_p300_sensitivity.py`,
`run_noise_diagnostics.py` and `run_r29.py` read its output. `run_simulations.py`
and `run_real_data.py` are independent. `run_timing.py` measures wall-clock time:
run it alone on an idle machine, after the others. It reads the CPU model from
`/proc/cpuinfo`, so it runs on Linux only. The figures and tables come last.

```bash
python3 -B run_p300.py --workers 8
python3 -B run_reference_noise.py --workers 8
python3 -B run_p300_sensitivity.py --workers 8
python3 -B run_noise_diagnostics.py --selections runs/p300/selections.json --workers 8
python3 -B run_r29.py --workers 8
python3 -B run_simulations.py --workers 8
python3 -B run_real_data.py --workers 8
python3 -B run_timing.py

python3 -B make_figures.py
python3 -B make_tables.py
python3 -B plot_templates.py
python3 -B plot_sim_templates.py
```

Defaults: `--data-dir` `$MNE_DATA` or `~/mne_data`; `--out ./runs` for the
analysis scripts; `--p300` (`run_reference_noise.py`, `run_r29.py`)
`./runs/p300`; `--central` (`run_p300_sensitivity.py`) `<out>/p300`; `--runs
./runs` (`make_figures.py`, `make_tables.py`, `plot_templates.py`); `--out
./figures` for the figure scripts and `./tables` for `make_tables.py`.
`make_figures.py` and `make_tables.py` take figure or table names as positional
arguments (`--help` lists them); `run_simulations.py --analyses`,
`run_p300_sensitivity.py --analyses`, `run_real_data.py --datasets` and
`run_timing.py --configs` run a subset.

A quick slice of the P300 chain (2 selections per K; `run_r29.py` needs the
Population control, which `run_p300.py` builds from the other subjects):

```bash
python3 -B run_p300.py --subjects 1 2 --reps 2 --out /tmp/slice
python3 -B run_reference_noise.py --p300 /tmp/slice/p300 --subjects 1 --reps 2 --out /tmp/slice
python3 -B run_p300_sensitivity.py --central /tmp/slice/p300 --subjects 1 --reps 2 --out /tmp/slice
python3 -B run_noise_diagnostics.py --selections /tmp/slice/p300/selections.json --subjects 1 --reps 2 --channels 2 --out /tmp/slice
python3 -B run_r29.py --p300 /tmp/slice/p300 --subjects 1 --reps 2 --out /tmp/slice
```

Statistics that a slice leaves undefined are `null` or a nonfinite value
written as a string (`"nan"`, `"inf"`) in `summary.json`, and `n/a` or not
drawn in the tables and figures.

## Supplementary: numerical stability

```bash
python3 -B supplementary/run_numerical_stability.py --workers 4
python3 -B supplementary/plot_stability_condition_numbers.py
```

`supplementary/run_numerical_stability.py` does not download the EEG and ECG
data: run `run_real_data.py` first with the same `--data-dir` (it downloads the
MNE sample data and MIT-BIH record 119). MOABB downloads BNCI2014-009 when needed.
It writes `runs/numerical_stability/`
(`results/<config>.json`, `summary.json`, `run_info.json`);
`supplementary/plot_stability_condition_numbers.py` reads `results/EEG.json`
there and writes `runs/numerical_stability/figS_stability_condition_numbers.pdf`
(`--results`, `--out` override both paths). The eigenvalues depend in their last
bits on the workers' BLAS thread count (`--blas-threads`, default 1; the paper's
run used 4, and 16 for `--configs BRIDGE`).

## Tests

Standalone, from this folder:

```bash
OPENBLAS_NUM_THREADS=1 python3 -B -m unittest discover -s tests
OPENBLAS_NUM_THREADS=1 python3 -B -m pytest tests -q
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for
details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md).
