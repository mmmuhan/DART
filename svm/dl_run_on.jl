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
const BASE_NDIR  = "synthetic_data/svmon/dart/n"  # input/output directory for dl_metrics
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
cp("synthetic_data/real_cv0_data_res/ntrained_modelnb1_seed_1.bson", "synthetic_data/svmon/dart/ntrained_modelnb1_seed_1.bson", force=true)

# -----------------------
# Run all cases
# -----------------------
results = Dict{String,Any}()

# 3-state
results["on2"] = run_case("synthetic_data/svmon/dart/on2_ntest.jld2", "on2_ntest",  "on2")

# 4-state
results["on3"] = run_case("synthetic_data/svmon/dart/on3_ntest.jld2", "on3_ntest", "on3")

# 5-state
results["on4"] = run_case("synthetic_data/svmon/dart/on4_ntest.jld2", "on4_ntest", "on4")