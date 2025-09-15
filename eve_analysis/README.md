# Application of DART on eve data from Berrocal et al., 2020

- `gen_eve.jl` - run to obtain:
  - trained DART model: `ntrained_modelbnb_seed_1.bson`

- `eve_bootstrap.jl` — run to obtain `boot_dl_results.jld2`, including results for:
  - mean on rate
  - mean off rate
  - standard deviation on rate
  - standard deviation off rate
  - fraction of on time

- `test.jl` — run to obtain binarized promoter states `trained_trace10/`: Each file follows dl_trace{i}{j}.jld2 (i: stripe; j: fluorescence-bin group).

- `all_concat_traces10.csv` — reorganized data from Berrocal et al., ordered by stripe and fluorescence bins, with NaNs removed.

- `w7_K3_t0_fluo_hmm_results_final.csv` — cpHMM-inferred rates provided by Berrocal et al., 2020.
