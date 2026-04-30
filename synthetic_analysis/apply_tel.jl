using Pkg
Pkg.activate(normpath(@__DIR__, ".."))

using Random, JLD2, BSON, Statistics, Sobol
include(normpath(joinpath(@__DIR__, "..", "utils", "utils.jl")))
include(normpath(joinpath(@__DIR__, "..", "utils", "reactions.jl")))
include(normpath(joinpath(@__DIR__, "..", "utils", "ssa.jl")))
include(normpath(joinpath(@__DIR__, "..", "dl_train", "train33.jl")))
include(normpath(joinpath(@__DIR__, "..", "dl_train", "test_model.jl")))

# --- Paths ---
DATA_DIR = joinpath(@__DIR__, "..", "synthetic_data", "ideal_data_res")
MROOT    = joinpath(DATA_DIR, "m")
mkpath(DATA_DIR)

# -------------------------------------------------------
# Dataset generation
# -------------------------------------------------------
function make_dataset(which::Union{Symbol,String};
    target_n::Int = 3300,
    lb::AbstractVector = [1e-1, 1e-1, 1.],
    ub::AbstractVector = [10., 10., 50.],
    seed::Integer = 0,
    savefile::Union{Nothing,String} = nothing,
)
    Sobol_seq = SobolSeq(lb, ub)
    out = Any[]
    sym = which isa String ? Symbol(which) : which

    while length(out) < target_n
        k = Sobol.next!(Sobol_seq)

        ont  = 1/k[2]
        offt = 1/k[1]
        r    = offt / ont
        bs   = k[3] * ont

        ratio = 10^(-2 + (1 - (-2)) * rand())
        tau   = ratio * offt
        obst  = 0.1 * min(ont, offt)
        nums  = ceil(Int, min((offt + ont) * 1100, 30000) / 30)

        in_bucket = if sym === :b
            5 <= r <= 20
        elseif sym === :nb1
            1 <= r < 5
        elseif sym === :nb0
            0.1 <= r < 1
        else
            throw(ArgumentError("which must be :b, :nb1, or :nb0"))
        end

        if in_bucket && (1 <= bs <= 150) && (0.1 <= tau < 10)
            res = generate_synthetic(construct_prob_delaytel,
                vcat(k, [0.0, tau, 1.0, 30.0]),
                0.0, 1.0, 1.0,
                nums, obst, false, 0.05, [2]
            )

            if length(findall(x -> x == 0, sum.(res.syn))) <= 1
                on_time  = on_off_time.(res.true_trace, obst, 1.0)
                off_time = on_off_time.(res.true_trace, obst, 0.0)
                push!(out, (p = vcat(k, tau, obst), syn = res.syn, trace = res.true_trace,
                            even_time = [on_time, off_time]))
            end
        end
    end

    if savefile !== nothing
        mkpath(dirname(savefile))
        JLD2.save(savefile, splitext(basename(savefile))[1], out)
    end

    return out
end

# -------------------------------------------------------
# Generate (or load) datasets
# -------------------------------------------------------
DATASETS = Dict(
    :telb   => (joinpath(DATA_DIR, "telb_mtest.jld2"),   "b33",   :b),
    :telnb1 => (joinpath(DATA_DIR, "telnb1_mtest.jld2"), "nb133", :nb1),
    :telnb0 => (joinpath(DATA_DIR, "telnb0_mtest.jld2"), "nb033", :nb0),
)

for (name, (path, _, bucket)) in DATASETS
    make_dataset(bucket; target_n=600, savefile=path)
end

# -------------------------------------------------------
# Evaluate
# -------------------------------------------------------
function compute_metrics(tag; runs=100, mroot=MROOT, droot=MROOT, kind="tel", verbose=false)
    [dl_metrics(s, mroot, droot, tag, kind, verbose) for s in 1:runs]
end

function slice_first(data, N=600)
    rn_data = copy(data[1:N])
    (rn_data      = rn_data,
     rn_params    = [d.p for d in rn_data],
     rn_data_true = copy(rn_data))
end

results = Dict{Symbol, Any}()

for (name, (path, tag, _)) in DATASETS
    varname = String(name) * "_mtest"
    data = JLD2.load(path, varname)
    s    = slice_first(data)

    global rn_data      = s.rn_data
    global rn_params    = s.rn_params
    global rn_data_true = s.rn_data_true

    results[name] = (metrics = compute_metrics(tag; runs=1), tag = tag)
end
