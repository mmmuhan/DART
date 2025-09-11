using Pkg
Pkg.activate(normpath(@__DIR__, ".."))
using JLD2

# --- Includes ---
include(joinpath(@__DIR__, "..", "dl_train", "train33_noise.jl"))
include(joinpath(@__DIR__, "..", "dl_train", "test_model.jl"))
include(joinpath(@__DIR__, "..", "utils", "utils.jl"))

# ---- helpers ----
load_out(path) = JLD2.load(path, splitext(basename(path))[1])
slice_first(data, N=1000) = begin
    rn_data = copy(data[1:N])
    (rn_data = rn_data,
     rn_params = [d.p for d in rn_data],
     rn_data_true = copy(rn_data))
end

"""
    run_cv(cv::String; s=1)

Run training + slicing + metrics for a given cross-validation case (`"cv0"`, `"cv1"`, `"cv2"`, ...).
Returns a Dict with keys `"b"`, `"nb1"`, `"nb0"`.
"""
function run_cv(cv::String; s::Int=1)
    data_dir = joinpath(@__DIR__, "..", "synthetic_data", "real_" * cv * "_data_res")
    out_dir_ideal = joinpath(data_dir, "m")
    out_dir_noise  = joinpath(data_dir, "n")
    mkpath(data_dir)

    # datasets per fold: (filename, metrics tag)
    sets = [
        ("genb_ntest_$(cv).jld2",   "b"),
        ("gennb1_ntest_$(cv).jld2", "nb1"),
        ("gennb0_ntest_$(cv).jld2", "nb0"),
    ]

    results = Dict{String,Any}()

    for (fname, tag) in sets
        data = load_out(joinpath(data_dir, fname))

        # training 
        run_train(s, data, out_dir_noise, tag, true)
        run_train(s, data, out_dir_ideal, tag, false)

        # take first 1000
        sl = slice_first(data, 1000)

        global rn_data      = sl.rn_data
        global rn_params    = sl.rn_params
        global rn_data_true = sl.rn_data_true

        # metrics
        met_noise = dl_metrics(s, out_dir_noise, out_dir_noise, tag, "gen", true)
        met_ideal = dl_metrics(s, out_dir_ideal, out_dir_ideal, tag, "gen", false)

        results[tag] = (metrics = [met_noise,met_ideal], slice = sl)
    end

    return results
end

res0 = run_cv("cv0");

# uncomment to obtain results for cv1, cv2
#res1 = run_cv("cv1")  
#res2 = run_cv("cv2")