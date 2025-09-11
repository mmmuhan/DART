# DART

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

This repository contains the code for the paper: DART: Deep learning for the Analysis and Reconstruction of Transcriptional dynamics from live-cell imaging data, Muhan Ma, Ramon Grima (2025)

Preprint: https://doi.org/10.1101/2025.09.02.673499s

## Requirements

- Julia 1.10.4 (environment generated with this version)
- Dependencies listed in `Project.toml` / `Manifest.toml`
- GPU recommended for training scripts

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
[Synthetic data](https://doi.org/10.5281/zenodo.DATASET_DOI)

The GitHub repository contains **all analysis code** (see `synthetic_analysis/`) to generate and test on these datasets.  

### Structure

- `dl_train` - scripts for deep learning simulators
- `utils` - other Julia scripts for stochastic simulations, baseline binarization methods, utility functions
- `synthetic_analysis` - scripts to generate synthetic data, train DART, and evaluate the performance (**data `synthetic_data` hosted on Zenodo**)  
- `svm` - scipts for model selection using SVM classifier
- `eve_analysis` - application of DART on eve data from Berrocal et al., 2020
- `Figures` - code for generating figures in the paper

## Running DART on your own MS2-MCP Data

Two main scripts can be run directly from the terminal on MS2-MCP data:

- `DART_gen_train.jl` – generates synthetic data and trains a model  
- `DART_trace.jl` – applies the trained model to experimental data  

---

### Example Workflow: `eve_data`

Suppose your data is stored in a CSV file, where **each row corresponds to the fluorescence time series of a single cell**.

---

### Arguments

#### For `DART_gen_train.jl`

| Argument | Type | Description | Default / Requirement |
|----------|------|-------------|------------------------|
| `L1` | Int | MS2 sequence length (bp) | **Required** |
| `L` | Int | `L1 + L2`, where `L2` is gene length | **Required** |
| `tau` | Float64 | Elongation time (min) | **Required** |
| `num` | Integer | Number of cells | **Required** |
| `obst` | Float64 | Time resolution (min) | **Required** |
| `tend` | Float64 | Total experiment time (min) | **Required** |
| `n_level` | Float64 | Technical noise CV | 0.05 |
| `seed` | Integer | Random seed for generating synthetic data | 1 |
| `train-seeds` | Integer | Random seed for training (can provide multiple seeds, e.g. `1,2,3`) | 1 |
| `out-dir` | String | Output directory | **Required** |

#### For `DART_trace.jl`

| Argument | Type | Description | Default / Requirement |
|----------|------|-------------|------------------------|
| `csv-dir` | String | Directory containing your experimental CSV files | **Required** |
| `out-dir` | String | Output directory (must match the one from Step 1) | **Required** |
| `gene` | String | Gene name (used for labeling outputs) | **Required** |
| `seed` | Integer | Random seed for selecting trained model | 1 |
| `obst` | Float64 | Time resolution (min) | **Required** |
| `rn-header` | Flag | Include if CSV files have a header row | `false` |

---

#### Step 1. Generate Synthetic Data and Train a Model 
```bash
julia DART_gen_train.jl \
  --L1 1500 --L 6605 --tau 2.33 --num 40 --obst 0.33 --tend 50.0 \
  --train-seeds 1 \
  --out-dir eve_data/
```

#### Step 2. Apply the Trained Model to Experimental Data
```bash
julia DART_trace.jl \
  --csv-dir eve_data/data \
  --out-dir eve_data/ \
  --gene eve \
  --seed 1 \
  --obst 0.33 \
  --rn-header 
```

### Outputs

For each input CSV file, the following results are generated:

| File | Format | Description |
|------|--------|-------------|
| `results_seed_i_filename_trace.csv` | CSV   | Binarized promoter traces |
| `results_seed_i_filename_trace.jld2` | JLD2  | Same as above, stored in Julia’s JLD2 format |
