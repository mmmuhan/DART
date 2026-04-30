# DART

This repository contains the code for the paper: DART: Deep learning for the Analysis and Reconstruction of Transcriptional dynamics from live-cell imaging data, Muhan Ma, Ramon Grima (2025)

Preprint: https://doi.org/10.1101/2025.09.02.673499

## Requirements

- Julia 1.10.4 (environment generated with this version)
- Dependencies listed in `Project.toml` / `Manifest.toml`
- GPU recommended for deep learning training (tested on NVIDIA A100 in our paper).

## Installation

Clone this repository and activate the Julia environment:

```bash
git clone https://github.com/mmmuhan/DART.git
cd DART
```
Then open julia and run:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Code for generating results from the paper
### Data Availability

Most synthetic datasets used in this work are too large to host on GitHub.  
They are archived separately on Zenodo:  
[Synthetic data](https://doi.org/10.5281/zenodo.17100879)

The GitHub repository contains **all analysis code** (see `synthetic_analysis/`) to generate and test on these datasets.  

### Structure

- `dl_train` - scripts for deep learning simulators
- `utils` - other Julia scripts for stochastic simulations, baseline binarization methods, utility functions
- `synthetic_analysis` - scripts to generate synthetic data, train DART, and evaluate the performance (**data `synthetic_data` hosted on Zenodo**)  
- `svm` - scipts for model selection using SVM classifier
- `eve_analysis` - application of DART on eve data from Berrocal et al., 2020
- `Figures` - code for generating figures in the paper (Fig 6, S5 were generated from scripts in the SVM folder)

## Running DART on your own MS2-MCP Data

Two main scripts can be run directly from the terminal on MS2-MCP data:

- `DART_gen_train.jl` – generates synthetic data and trains a model  
- `DART_trace.jl` – applies the trained model to experimental data  

---

## Example Workflow

Results will be stored in the folder `eve_data/`. Example data is provided in `eve_data/data/`.

Your input data should be a CSV file where **each row corresponds to the fluorescence time series of a single cell**.

---

### Arguments

#### `DART_gen_train.jl`

| Argument | Type | Description | Default / Requirement |
|----------|------|-------------|------------------------|
| `--L1` | Int | MS2 sequence length (bp) | **Required** |
| `--L` | Int | `L1 + L2`, where `L2` is gene length (bp) | **Required** |
| `--tau` | Float64 | Elongation time (min) | **Required** |
| `--num` | Int | Number of cells | **Required** |
| `--obst` | Float64 | Time resolution (min) | **Required** |
| `--tend` | Float64 | Total experiment time (min) | **Required** |
| `--n-level` | Float64 | Technical noise CV | `0.05` |
| `--seed` | Int | Random seed for generating synthetic data | `1` |
| `--train-seeds` | Int | Random seed(s) for training (comma-separated, e.g. `1,2,3`) | `1` |
| `--out-dir` | String | Output directory | **Required** |

#### `DART_trace.jl`

| Argument | Type | Description | Default / Requirement |
|----------|------|-------------|------------------------|
| `--csv-dir` | String | Directory containing your experimental CSV files | **Required** |
| `--out-dir` | String | Output directory (must match the one from Step 1) | **Required** |
| `--gene` | String | Gene name (used for labeling outputs) | **Required** |
| `--train-seeds` | Int | Random seed of the trained model to load | `1` |
| `--obst` | Float64 | Time resolution (min) | **Required** |
| `--tau` | Float64 | Elongation time (min) | **Required** |
| `--rn-header` | Flag | Pass if CSV files include a header row | `false` |

---

### Step 1 — Generate Synthetic Data and Train a Model

*Takes around 400 seconds on an A100 GPU.*

```bash
julia DART_gen_train.jl \
  --L1 1500 --L 6605 --tau 2.33 --num 40 --obst 0.33 --tend 50.0 \
  --train-seeds 1 \
  --out-dir eve_data/
```

#### Outputs

For each experimental setting, the following files are generated:

| File | Format | Description |
|------|--------|-------------|
| `gen_ntest.jld2` | JLD2 | Synthetic data generated for this setting |
| `trained_modelbnb_seed_i.bson` | BSON | Trained deep learning model (one per seed `i`) |

### Step 2 — Apply the Trained Model to Experimental Data

*Takes around 40 seconds.*

```bash
julia DART_trace.jl \
  --csv-dir eve_data/data \
  --out-dir eve_data/ \
  --gene eve \
  --train-seeds 1 \
  --obst 0.33 \
  --tau 2.33 \
  --rn-header
```

#### Outputs

For each input CSV file, the following results are generated (where `i` is the seed and `filename` is the input CSV's base name):

| File | Format | Description |
|------|--------|-------------|
| `results_seedi_filename_trace.csv` | CSV | Binarized promoter traces |
| `results_seedi_filename_rates.csv` | CSV | Effective transcription rates |
| `results_seedi_filename.jld2` | JLD2 | Both traces and rates, stored in Julia's JLD2 format |