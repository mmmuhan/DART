using Pkg
Pkg.activate(normpath(@__DIR__, "..", ".."))

using JLD2

include(joinpath(@__DIR__, "..", "..", "dl_train", "train33_data_keff.jl"))
include(joinpath(@__DIR__, "..", "..", "dl_train", "test_model_keff.jl"))
include(joinpath(@__DIR__, "..", "..", "utils", "utils.jl"))


# -----------------------
# Config
# -----------------------
const BASE_NDIR  = "eve_analysis/svm/n"  # input/output directory for dl_metrics
const blevel  = "bnb"                   # metric class name
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
function run_case(file::String, varname::String, tag::String, rid::UnitRange{Int})
    # Load the dataset from JLD2 file
    data = JLD2.load(file, varname)

    # These globals are required by dl_metrics
    global rn_data       = copy(data)
    global rn_params     = [d.p for d in rn_data]
    global rn_data_true  = copy(data)

    return dl_metrics(1, BASE_NDIR, BASE_NDIR, blevel, tag, noise; rate_idx=rid)
end

# copy paste the trained DART for inference
cp("eve_analysis/ntrained_modelbnb_seed_1.bson", "eve_analysis/svm/ntrained_modelbnb_seed_1.bson", force=true)

# -----------------------
# Run all cases
# -----------------------
results = Dict{String,Any}()

# 2-state
results["tel"] = run_case("eve_analysis/svm/tel_ntest.jld2",   "tel_ntest",   "tel", 3:3)

# 3-state
results["perm"] = run_case("eve_analysis/svm/perm_ntest.jld2", "perm_ntest",  "perm", 5:5)
