# Application of DART on *eve* Data

Analysis of *eve* stripe expression data from Berrocal et al., 2020.

## Scripts

| Script | Description |
|--------|-------------|
| `gen_eve.jl` | Train DART model on *eve* data |
| `eve_bootstrap.jl` | Bootstrap inference of ON/OFF rates, their standard deviations, and fraction of ON time |
| `test.jl` | Binarize promoter states for each stripe and fluorescence-bin group |

## Data

- **`all_concat_traces10.csv`** — Reorganized data from Berrocal et al., ordered by stripe and fluorescence bins, with NaNs removed.
- **`w7_K3_t0_fluo_hmm_results_final.csv`** — cpHMM-inferred rates provided by Berrocal et al., 2020.

## Results

- **`ntrained_modelbnb_seed_1.bson`** — Trained DART model (from `gen_eve.jl`).
- **`boot_dl_results.jld2`** — Bootstrap results: mean and standard deviation of ON/OFF rates, fraction of ON time (from `eve_bootstrap.jl`).
- **`trained_trace10/`** — Binarized promoter states, named `dl_trace{i}{j}.jld2` where `i` = stripe and `j` = fluorescence-bin group (from `test.jl`).

---

# Application of DART+SVM on *eve* Data

Model selection on *eve* data using the DART+SVM pipeline. All files are in the `svm/` folder.

## Scripts

| Script | Description |
|--------|-------------|
| `gen_ms_eve.jl` | Generate synthetic data from 2-state and 3-state models |
| `dl_run_svm_eve.jl` | Apply the trained DART model to synthetic data from both models |
| `generate_svm_eve.jl` | Format data as input for the SVM classifier |
| `svm_eve.py` | Run the SVM classifier |
| `eve_times.jl` | Obtain off-times inferred by DART  |
| `ms_eve_py.ipynb` | Model selection results and analysis |

## Results
 
- **`ml_rev23_1000.npz`** — Input data for the SVM classifier (from `generate_svm_input_eve.jl`).
- **`svm_model_linear.pkl`** — Trained SVM model (from `svm_eve.py`).
- **`offts3_by_ij.jld2`** — Off-times inferred by DART (from `eve_times.jl`).
