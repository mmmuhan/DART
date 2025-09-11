# Apply trained model `ntrained_modelnb1_seed_1.bson` on promoter-switching model with 1,2,3,4 off-states to get binarized promoter states

using Pkg
Pkg.activate(normpath(@__DIR__, ".."))

using JLD2

include(joinpath(@__DIR__, "..", "dl_train", "train33_noise.jl"))
include(joinpath(@__DIR__, "..", "dl_train", "test_model.jl"))
include(joinpath(@__DIR__, "..", "utils", "utils.jl"))


# -----------------------
# Config
# -----------------------
const BASE_NDIR  = "synthetic_data/svm/n"  # input/output directory for dl_metrics
const blevel  = "nb1"                   # metric class name
const noise    = true                    # whether to save the results

# -----------------------
# Helper function
# -----------------------
"""
    run_case(file::String, varname::String, tag::String) -> Any

Load variable `varname` from `.jld2` file `file`,  
set global variables (`rn_data`, `rn_params`, `rn_data_true`)  
and call `dl_metrics(1, BASE_NDIR, BASE_NDIR, MET_CLASS, tag, noise)`.  
Returns the result of `dl_metrics`.
"""
function run_case(file::String, varname::String, tag::String)
    # Load the dataset from JLD2 file
    data = JLD2.load(file, varname)

    # These globals are required by dl_metrics
    global rn_data       = copy(data)
    global rn_params     = [d.p for d in rn_data]
    global rn_data_true  = copy(data)

    return dl_metrics(1, BASE_NDIR, BASE_NDIR, blevel, tag, noise)
end

# copy paste the trained DART for inference
cp("synthetic_data/real_cv0_data_res/ntrained_modelnb1_seed_1.bson", "synthetic_data/svm/ntrained_modelnb1_seed_1.bson")

# -----------------------
# Run all cases
# -----------------------
results = Dict{String,Any}()

# 2-state
results["tel"] = run_case("synthetic_data/svm/tel_ntest.jld2",   "tel_ntest",   "tel")

# 3-state
results["perm"] = run_case("synthetic_data/svm/perm_ntest.jld2", "perm_ntest",  "perm")

# 4-state
results["perm1"] = run_case("synthetic_data/svm/perm1_ntest.jld2", "perm1_ntest", "perm1")

# 5-state
results["perm2"] = run_case("synthetic_data/svm/perm2_ntest.jld2", "perm2_ntest", "perm2")
