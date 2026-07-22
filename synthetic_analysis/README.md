# Synthetic Data Analysis

Scripts for generating synthetic fluorescence time-series data, training the DART model, and benchmarking against established binarization methods.

## Scripts

| Script | Description |
|--------|-------------|
| `gen_ideal.jl` | Generate idealized synthetic data |
| `gen_real.jl` | Generate realistic synthetic data |
| `dl_run_ideal.jl` | Train and evaluate DART on idealized data |
| `dl_run_real.jl` | Train and evaluate DART on realistic data |
| `methods_compare.jl` | Compare DART with HMM, MA, and SG |
| `apply_tel.jl` | Generate telegraph-model synthetic data and apply DART |
| `compare_tel.jl` | Apply HMM on telegraph-model synthetic data |
| `apply_ref.jl`, `apply_gen2.jl`, `apply_mm2.jl` | Apply DART to synthetic data from unseen model structures |

## Naming Conventions

Filenames encode burstiness level, noise condition, and method. The table below lists the abbreviations used throughout.

| Component | Values | Meaning |
|-----------|--------|---------|
| Burstiness | `b` / `nb1` / `nb0` | High / intermediate / low |
| Noise prefix | `m` / `n` | Noise-free / with noise |
| Noise level | `cv0` / `cv1` / `cv2` | 5% / 10% / 20% technical noise (coefficient of variation) |
| Seed | `seed_s` | Random seed index (`s` = 1, 2, …, 100) |

## Data

All generated data is stored in `synthetic_data/` (also available on [Zenodo](https://doi.org/10.5281/zenodo.21470892)).

### `base_compare/`

Binarization results from HMM, MA, and SG at three burstiness levels, named as `gen{burstiness}_{method}.jld2` (e.g., `genb_hmm.jld2`, `gennb0_sg.jld2`).

Also contains:
- `dl_eg_traces.jld2` — example traces for Fig. 2B–C.
- Telegraph-model comparison: HMM results (`tel{burstiness}_hmm.jld2`) and corresponding DART results in `ideal_data_res/` (`mtel_dl{burstiness}33_compare_seed_1.jld2`).

### `ideal_data_res/`

Idealized synthetic data and DART results (produced by `gen_ideal.jl` and `dl_run_ideal.jl`).

- **Synthetic data:** `gen{burstiness}_mtest.jld2`
- **Trained models:** `mtrained_model{burstiness}33_seed_s.bson`
- **Test results:** `mgen_dl{burstiness}33_compare_seed_s.jld2`
- **Hyperparameter tuning:** replace `33` → `333` in filenames above.
- **Cross-model generalization:** `m{model}_dl{burstiness}33_compare_seed_s.jld2`, where `{model}` ∈ {`ref`, `gen2`, `mm2`} and `s` is the seed corresponding to median performance for each burstiness level.

### `real_cv{0,1,2}_data_res/`

Realistic synthetic data at three noise levels (produced by `gen_real.jl` and `dl_run_real.jl`). Each folder contains:

- **Synthetic data:** `gen{burstiness}_ntest_cv{X}.jld2`
- **Trained models:** `{n|m}trained_model{burstiness}_seed_1.bson`
- **Test results:** `{n|m}gen_dl{burstiness}_compare_seed_1.jld2`

### `bI_compare_cv{0,2}/`

Comparison of DART, burstInfer, and standard HMM on realistic data (5% and 20% noise).

### `cp_gen{burstiness}_cv{0,1,2}/`

Realistic data formatted for cpHMM comparison, organized by burstiness level and noise level.
