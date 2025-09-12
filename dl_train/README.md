## dl_train folder (requires GPU)

This folder contains scripts for training and testing the DART deep-learning models.

- `train33_ideal.jl` – Train DART with 3-3 structure on **idealized synthetic data**.
- `train333_ideal.jl` – Train DART with 3-3-3 structure on **idealized synthetic data**.
- `train33_noise.jl` – Train DART with 3-3 structure on **realistic synthetic data**.
- `train33_data.jl` – Train DART with 3-3 structure on **experimental eve data**.
- `test_model.jl` – Infer binarized promoter states and evaluate the performance of a trained model (for synthetic or experimental data).
