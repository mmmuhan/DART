using Pkg
Pkg.activate(normpath(@__DIR__, ".."))

using Random, JLD2, BSON, Statistics, Sobol

include(normpath(joinpath(@__DIR__, "..", "utils", "reactions.jl")))
include(normpath(joinpath(@__DIR__, "..", "utils", "ssa.jl")))
include(normpath(joinpath(@__DIR__, "..", "utils", "utils.jl")))

const NUMS   = 300
const T0, L1, L, I0 = 1.0, 0.0, 1.0, 1.0
const OBST   = 0.1
ratio10()    = 10^(-2 + (1-(-2))*rand())            

#  - name: save file name
#  - lb/ub: Sobol upper and lower bound
#  - bs_idx: index of initiation rate in parameter set
#  - on_idx: index of on-state
const CONFIGS = [
    (
        key=:on2,
        name="on2_ntest",
        lb=[1e-3,2e-3,2e-3,2e-3,1.0,1.0],
        ub=[10.0,20.0,20.0,20.0,50.0,50.0],
        bs= k -> (k[6]*k[3] + k[5]*k[4])/(k[2]*k[4]),
        construct=construct_prob_delayon2,
        noise=true, on_idx=[2,3],
        ont_offt = (k)->(compute_ont(k,4,"on2"), 1/k[1])
    ),
    (
        key=:on3,
        name="on3_ntest",
        lb=[1e-1,3e-3,3e-3,3e-3,3e-3,3e-3,1.0,1.0,1.0],
        ub=[10.0,30.0,30.0,30.0,30.0,30.0,50.0,50.0,50.0],
        bs= k ->  (k[9]*k[3]*k[5] + k[8]*k[3]*k[6] + k[7]*k[4]*k[6]) / (k[2]*k[4]*k[6]),
        construct=construct_prob_delayon3,
        noise=true, on_idx=[2,3,4],
        ont_offt = (k)->(compute_ont(k,4,"on3"), 1/k[1])
    ),
    (
        key=:on4,
        name="on4_ntest",
        lb=[1e-3,4e-3,4e-3,4e-3,4e-3,4e-3,4e-3,4e-3,1.0,1.0,1.0,1.0],
        ub=[10.0,40.0,40.0,40.0,40.0,40.0,40.0,40.0,50.0,50.0,50.0,50.0],
        bs=k -> (k[12]*k[3]*k[5]*k[7] + k[11]*k[3]*k[5]*k[8] + k[10]*k[3]*k[6]*k[8] + k[9]*k[4]*k[6]*k[8]) /
(k[2]*k[4]*k[6]*k[8]),
        construct=construct_prob_delayon4,
        noise=true, on_idx=[2,3,4,5],
        ont_offt = (k)->(compute_ont(k,4,"on4"), 1/k[1])
    ),
]

function make_dataset(cfg; target=4000, save_dir="synthetic_data/svmon/dart")
    sob = SobolSeq(cfg.lb, cfg.ub)

    out = Vector{Any}()
    out_size() = length(out)

    @time while out_size() < target
        k = Sobol.next!(sob)

        ont, offt = cfg.ont_offt(k)
        bs  = bs  = cfg.bs(k)
        tau = ratio10() * offt
        obst = OBST

        if 1 <= offt/ont < 5 && 1 <= bs <= 150 && 0.1 <= tau < 10 && ont >= obst
    
            params = vcat(k, [0.0, tau, 1.0, 30.0])

            res = generate_synthetic(
                cfg.construct, params,
                L1, L, I0, NUMS, obst, cfg.noise, 0.05, cfg.on_idx
            )

            even_syn, even_nsyn, even_trace = res.syn, res.nsyn, res.true_trace

            if length(findall(x -> x == 0, sum.(even_syn))) <= 1
                on_time  = on_off_time.(even_trace, obst, 1.0)
                off_time = on_off_time.(even_trace, obst, 0.0)

                push!(out, (
                    p          = vcat(k, tau, obst),
                    nsyn       = even_nsyn,
                    trace      = even_trace,
                    even_time  = [on_time, off_time]
                ))
            end
        end
    end
    
    # save
    mkpath(save_dir)
    save_path = joinpath(save_dir, "$(cfg.name).jld2")
    varname = String(cfg.name)             
    JLD2.save(save_path, varname, out)   
    println("Saved $(cfg.name) ($(length(out)) items) -> $save_path")
    return out
    
end

# generate multiple off-state synthetic data
mkpath("synthetic_data/svmon/true")
mkpath("synthetic_data/svmon/dart")

on2_ntest   = make_dataset(CONFIGS[1])
on2_times   = [m.even_time for m in on2_ntest]
@save "synthetic_data/svmon/true/on2_times.jld2" on2_times

on3_ntest = make_dataset(CONFIGS[2])
on3_times   = [m.even_time for m in on3_ntest]
@save "synthetic_data/svmon/true/on3_times.jld2" on3_times

on4_ntest = make_dataset(CONFIGS[3]);
on4_times   = [m.even_time for m in on4_ntest]
@save "synthetic_data/svmon/true/on4_times.jld2" on4_times;
