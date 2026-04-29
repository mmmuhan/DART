# Model Selection with SVM (Support Vector Machine) Classifier

SVM-based inference of the number of promoter sub-states, applied to two cases: multiple off-states (1 on, 2–4 off) and multiple on-states (1 off, 2–4 on). Both use the DART model `ntrained_modelnb1_seed_1.bson` (trained by `dl_run_real.jl`). Results are stored in `synthetic_data/svm/` (off-states) and `synthetic_data/svmon/` (on-states).

## Scripts

| Step | Multiple off-states | Multiple on-states | Description |
|------|--------------------|--------------------|-------------|
| Generate data | `gen_ms.jl` | `gen_ms_on.jl` | Simulate promoter-switching models with varying numbers of sub-states |
| Apply DART | `dl_run_svm.jl` | `dl_run_on.jl` | Binarize synthetic traces with the trained DART model |
| Format for SVM | `generate_svm_input.jl` | `generate_svm_on.jl` | Convert to `.npz` format (ground-truth and DART-inferred) |
| Apply SVM (ground-truth) | `svm_true.py` | `svm_trueon.py` | SVM on ground-truth states; results in `svm/true/` and `svmon/true/` |
| Apply SVM (DART) | `svm_dart.py` | `svm_darton.py` | SVM on DART-inferred states; output confusion matrix and ROC curve |

## Comparison with Other Methods
 
| Method | Script | Description |
|--------|--------|-------------|
| DART | `svm_dart_bd.py` | Model selection between 2-state and 3-state models for comparison |
| BurstDeconv | `BD_analysis.ipynb` | BurstDECONV analysis, comparison, and plots |
| ABC-SMC | `ABC_analysis.ipynb` | ABC-SMC analysis, comparison, and plots |
 

## Data

### `synthetic_data/svm/` (multiple off-states)

- **Synthetic data:** `tel_ntest.jld2`, `perm_ntest.jld2`, `perm1_ntest.jld2`, `perm2_ntest.jld2`
- **Ground-truth dwell times:** `tel_times.jld2`, `perm_times.jld2`, `perm1_times.jld2`, `perm2_times.jld2`
- **DART results:** `ntel_dlnb1_compare_seed_1.jld2`, `nperm_dlnb1_compare_seed_1.jld2`, `nperm1_dlnb1_compare_seed_1.jld2`, `nperm2_dlnb1_compare_seed_1.jld2`
- **SVM input:** `true_rev2345.npz` (ground-truth), `ml_rev2345.npz` (DART)

### `synthetic_data/svmon/` (multiple on-states)

- **Synthetic data:** `on2_ntest.jld2`, `on3_ntest.jld2`, `on4_ntest.jld2`
- **Ground-truth dwell times:** `on2_times.jld2`, `on3_times.jld2`, `on4_times.jld2`
- **DART results:** `ntel_dlnb1_compare_seed_1.jld2`, `non2_dlnb1_compare_seed_1.jld2`, `non3_dlnb1_compare_seed_1.jld2`, `non4_dlnb1_compare_seed_1.jld2`
- **SVM input:** `true_revon2345.npz` (ground-truth), `ml_revon2345.npz` (DART)

### `synthetic_data/svm/bd/` (BurstDeconv)
 
- **`data_ms_all/`:** BurstDeconv results on synthetic data across 100 parameter sets (100 genes)

### `synthetic_data/svm/ABC/` (ABC-SMC)
 
- **`100_6_04/`:** Output from pyABC
