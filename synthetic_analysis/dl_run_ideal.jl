using Pkg
Pkg.activate(normpath(@__DIR__, ".."))

using JLD2

# --- Paths ---
DATA_DIR = joinpath(@__DIR__, "..", "synthetic_data", "ideal_data_res")
MROOT    = joinpath(DATA_DIR, "m")
mkpath(DATA_DIR)   # make sure output dir exists

# --- Includes ---
include(joinpath(@__DIR__, "..", "dl_train", "train33.jl"))
include(joinpath(@__DIR__, "..", "dl_train", "test_model.jl"))
include(joinpath(@__DIR__, "..", "utils", "utils.jl"))

# --- Load datasets ---
@load joinpath(DATA_DIR, "genb_mtest.jld2") genb_mtest 
@load joinpath(DATA_DIR, "gennb1_mtest.jld2") gennb1_mtest
@load joinpath(DATA_DIR, "gennb0_mtest.jld2") gennb0_mtest

datasets = Dict(
    :genb   => (genb_mtest,   "b33"),
    :gennb1 => (gennb1_mtest, "nb133"),
    :gennb0 => (gennb0_mtest, "nb033"),
)

# --- Helpers ---
function train_many(data; runs=100, mroot=MROOT, tag::AbstractString, verbose=false)
    models = Dict{Int,Any}()
    train_losses = Dict{Int,Any}()
    val_losses = Dict{Int,Any}()
    for s in 1:runs
        model, tl, vl = run_train(s, data, mroot, tag, verbose)
        models[s] = model
        train_losses[s] = tl
        val_losses[s] = vl
    end
    return (models=models, train_losses=train_losses, val_losses=val_losses)
end

slice_first(data, N=600) = begin
    rn_data = copy(data[1:N])
    (rn_data = rn_data,
     rn_params = [d.p for d in rn_data],
     rn_data_true = copy(rn_data))
end

compute_metrics(tag; runs=100, mroot=MROOT, droot=MROOT, kind="gen", verbose=false) =
    [dl_metrics(s, mroot, droot, tag, kind, verbose) for s in 1:runs]

# --- Run for each dataset ---
results = Dict{Symbol, Any}()

for (name, (data, tag)) in datasets
    s = slice_first(data)

    # make the vars that test_model.jl expects:
    global rn_data      = s.rn_data
    global rn_params    = s.rn_params
    global rn_data_true = s.rn_data_true

    results[name] = (
        training = train_many(data; runs=100, tag=tag, verbose=false),
        slice    = s,
        metrics  = compute_metrics(tag; runs=100, kind="gen", verbose=false), # 100 random seeds
        tag      = tag,
    )
end;
