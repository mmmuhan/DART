using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))

using Random, JLD2, BSON, Statistics, Sobol

include(normpath(joinpath(@__DIR__, "..", "utils", "reactions.jl")))
include(normpath(joinpath(@__DIR__, "..", "utils", "ssa.jl")))
include(normpath(joinpath(@__DIR__, "..", "utils", "utils.jl")))

function make_dataset(which::Union{Symbol,String};
    target_n::Int = 3700,
    lb::AbstractVector = [3e-3,3e-3,3e-3,3e-3,3e-3,3e-3,3e-3,3e-3,3e-3,3e-3,1.,1.,1.],
    ub::AbstractVector = [30.,30.,30.,30.,30.,30.,30.,30.,30.,30.,50.,50.,50.],
    obst::Float64 = 0.1, # observation time/ sampling time of live-cell imaging
    n_level::Float64 = 0.05,# technical noise level
    num::Integer = 300, # cell numbers
    seed::Integer = 0,
    savefile::Union{Nothing,String} = nothing,
)

    # Sobol
    Sobol_seq = SobolSeq(lb, ub)

    # Container (same structure as your push!)
    out = Any[]

    sym = which isa String ? Symbol(which) : which # works for both "b" and :b

    while length(out) < target_n
        k = Sobol.next!(Sobol_seq)

        ont  = compute_ont(k, 10, "gen") # mean on time
        offt = compute_offt(k, 10, "gen") # mean off time
        r    = offt / ont # burstiness level

        bs = (k[10]*k[12]*k[7] + k[10]*k[11]*k[8] + k[13]*k[7]*k[9]) / (k[10]*k[6]*k[8]) # burst size
        
        ratio = 10^(-2 + (1 - (-2)) * rand())
        tau   = ratio * offt

        # Bucket selection
        in_bucket = if sym === :b
            5 <= r <= 20                 # genb_mtest
        elseif sym === :nb1
            1 <= r < 5                   # gennb1_mtest
        elseif sym === :nb0
            0.1 <= r < 1                 # gennb0_mtest
        else
            throw(ArgumentError("which must be :b, :nb1, or :nb0"))
        end

        if in_bucket && (1 <= bs <= 150) && (0.1 <= tau < 10)
            res = generate_synthetic(construct_prob_delaygen,
                vcat(k, [0.0, tau, 1.0, 30.0]),
                0.0, 1.0, 1.0,
                num, obst, true, n_level, [4, 5, 6]
            )

            even_syn  = res.syn
            even_nsyn = res.nsyn
            even_trace = res.true_trace

            # at most one trace with no pulse
            if length(findall(x->x==0,sum.(even_syn)))<=1
                on_time  = on_off_time.(even_trace, obst, 1.0)
                off_time = on_off_time.(even_trace, obst, 0.0)
                push!(out, (p = vcat(k, tau, obst), syn = even_syn, nsyn = even_nsyn, trace = even_trace, even_time = [on_time, off_time]))
            end
        end
    end

    if savefile !== nothing
        mkpath(dirname(savefile))  
        JLD2.save(savefile, splitext(basename(savefile))[1], out)
    end

    return out
end


# generate data at 5% technical noise level:

outdir = "synthetic_data/real_cv0_data_res"
mkpath(outdir)   # makes sure the directory exists

genb   = make_dataset(:b;   target_n=3700, n_level = 0.05, savefile="synthetic_data/real_cv0_data_res/genb_ntest_cv0.jld2")
gennb1 = make_dataset(:nb1; target_n=3700, n_level = 0.05, savefile="synthetic_data/real_cv0_data_res/gennb1_ntest_cv0.jld2")
gennb0 = make_dataset(:nb0; target_n=3700, n_level = 0.05, savefile="synthetic_data/real_cv0_data_res/gennb0_ntest_cv0.jld2");


# generate data at 10% technical noise level:

#=
outdir = "synthetic_data/real_cv1_data_res"
mkpath(outdir)   

genb   = make_dataset(:b;   target_n=3700, n_level = 0.1, savefile="synthetic_data/real_cv1_data_res/genb_ntest_cv1.jld2")
gennb1 = make_dataset(:nb1; target_n=3700, n_level = 0.1, savefile="synthetic_data/real_cv1_data_res/gennb1_ntest_cv1.jld2")
gennb0 = make_dataset(:nb0; target_n=3700, n_level = 0.1, savefile="synthetic_data/real_cv1_data_res/gennb0_ntest_cv1.jld2");
=#

# generate data at 20% technical noise level:

#=
outdir = "synthetic_data/real_cv2_data_res"
mkpath(outdir)

genb   = make_dataset(:b;   target_n=3700, n_level = 0.2, savefile="synthetic_data/real_cv2_data_res/genb_ntest_cv2.jld2")
gennb1 = make_dataset(:nb1; target_n=3700, n_level = 0.2, savefile="synthetic_data/real_cv2_data_res/gennb1_ntest_cv2.jld2")
gennb0 = make_dataset(:nb0; target_n=3700, n_level = 0.2, savefile="synthetic_data/real_cv2_data_res/gennb0_ntest_cv2.jld2");
=#
