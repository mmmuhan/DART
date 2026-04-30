# `dl_train/` (requires GPU)

Scripts for training and testing DART deep-learning models. The model structure is denoted by its layers (e.g., 3-3 or 3-3-3).

## Training

| Script | Structure | Data | Transcription rate |
|--------|-----------|------|--------------------|
| `train33_ideal.jl` | 3-3 | Idealized synthetic | No |
| `train333_ideal.jl` | 3-3-3 | Idealized synthetic | No |
| `train33_noise.jl` | 3-3 | Realistic synthetic | No |
| `train33_data.jl` | 3-3 | Experimental | No |
| `train33_keff.jl` | 3-3 | Idealized synthetic | Yes |
| `train33_data_keff.jl` | 3-3 | Experimental | Yes |

## Testing

| Script | Description |
|--------|-------------|
| `test_model.jl` | Infer binarized promoter states and evaluate model performance |
| `test_model_keff.jl` | Infer binarized promoter states and effective transcription rates, and evaluate model performance |

Both testing scripts work with synthetic and experimental data.