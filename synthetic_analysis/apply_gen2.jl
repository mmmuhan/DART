using Pkg
Pkg.activate(normpath(@__DIR__, ".."))

using Random, JLD2, BSON, Statistics, Sobol

include(normpath(joinpath(@__DIR__, "..", "utils", "reactions.jl")))
include(normpath(joinpath(@__DIR__, "..", "utils", "ssa.jl")))
include(normpath(joinpath(@__DIR__, "..", "utils", "utils.jl")))

include(joinpath(@__DIR__,  "..", "dl_train", "train33.jl"))
include(joinpath(@__DIR__,  "..", "dl_train", "test_model.jl"))

function make_gen2_dataset(which::Union{Symbol,String};
    target_n::Int = 600,
    lb::AbstractVector = [2e-3,2e-3,2e-3,2e-3,2e-3,2e-3,1.,1.] ,
    ub::AbstractVector = [20.,20.,20.,20.,20.,20.,50.,50.],
    seed::Integer = 0,
    savefile::Union{Nothing,String} = nothing,
)

    seed != 0 && Random.seed!(seed)

    Sobol_seq = SobolSeq(lb, ub)
    out = Any[]

    sym = which isa String ? Symbol(which) : which  # allow "b" or :b

    while length(out) < target_n
        k = Sobol.next!(Sobol_seq)

        ont = compute_ont(k,3,"gen2") 
        offt = compute_offt(k,3,"gen2")
        bs = (k[6]*k[7]+k[5]*k[8])/(k[4]*k[6])
        r    = offt / ont

        ratio = 10^(-2 + (1 - (-2)) * rand())
        tau   = ratio * offt

        obst = 0.1 * min(ont, offt)
        nums = ceil(Int, min((offt + ont) * 1100, 30000) / 30)

        # Bucket selection (ONLY thing that differs between b/nb1/nb0)
        in_bucket = if sym === :b
            5 <= r <= 20
        elseif sym === :nb1
            1 <= r < 15
        elseif sym === :nb0
            0.1 <= r < 1
        else
            throw(ArgumentError("which must be :b, :nb1, or :nb0"))
        end

        if in_bucket && (1 <= bs <= 150) && (0.1 <= tau < 10) && (ont >= obst)
            res = generate_synthetic(
                construct_prob_delaygen2,
                vcat(k, [0.0, tau, 1.0, 30.0]),
                0.0, 1.0, 1.0,
                nums, obst, false, 0.05, [3,4]
            )

            even_syn   = res.syn
            even_trace = res.true_trace

            if mean(vcat(even_syn...)) >= 5 &&
               length(findall(x -> x == 0, sum.(even_syn))) <= 1

                on_time  = on_off_time.(even_trace, obst, 1.0)
                off_time = on_off_time.(even_trace, obst, 0.0)

                push!(out, (p = vcat(k, tau, obst),
                            syn = even_syn,
                            trace = even_trace,
                            even_time = [on_time, off_time]))
            end
        end
    end

    if savefile !== nothing
        mkpath(dirname(savefile))
        # save variable name derived from filename (same as your pattern)
        JLD2.save(savefile, splitext(basename(savefile))[1], out)
    end

    return out
end

#outdir = "synthetic_data/ideal_data_res"
#mkpath(outdir)

# Call like:
gen2b   = make_gen2_dataset(:b;   target_n=600, savefile="synthetic_data/ideal_data_res/gen2b_mtest.jld2")
rn_data = copy(gen2b[1:600]);
rn_params = [d.p for d in rn_data];
rn_data_true = copy(gen2b[1:600]);
metb_gen2 = dl_metrics(58, "synthetic_data/ideal_data_res/m", "synthetic_data/ideal_data_res/m", "b33", "gen2", false); 

gen2nb1 = make_gen2_dataset(:nb1; target_n=600, savefile="synthetic_data/ideal_data_res/gen2nb1_mtest.jld2")
rn_data = copy(gen2nb1[1:600]);
rn_params = [d.p for d in rn_data];
rn_data_true = copy(gen2nb1[1:600]);
metnb1_gen2 = dl_metrics(11, "synthetic_data/ideal_data_res/m", "synthetic_data/ideal_data_res/m", "nb133", "gen2", false); 

gen2nb0 = make_gen2_dataset(:nb0; target_n=600, savefile="synthetic_data/ideal_data_res/gen2nb0_mtest.jld2");
rn_data = copy(gen2nb0[1:600]);
rn_params = [d.p for d in rn_data];
rn_data_true = copy(gen2nb0[1:600]);
metnb0_gen2 = dl_metrics(49, "synthetic_data/ideal_data_res/m", "synthetic_data/ideal_data_res/m", "nb033", "gen2", false); 