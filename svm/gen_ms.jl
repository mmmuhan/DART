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
        key=:tel,
        name="tel_ntest",
        lb=[1e-3, 1e-3, 1.0],
        ub=[10.0, 10.0, 50.0],
        bs_idx=3,
        construct=construct_prob_delaytel,
        noise=true, on_idx=2,
        ont_offt = (k)->(1/k[2], 1/k[1])                
    ),
    (
        key=:perm,
        name="perm_ntest",
        lb=[2e-3,2e-3,2e-3,1e-3,1.0],
        ub=[20.0,20.0,20.0,10.0,50.0],
        bs_idx=5,
        construct=construct_prob_delayperm,
        noise=true, on_idx=3,
        ont_offt = (k)->(compute_ont(k,4,"perm"), compute_offt(k,4,"perm"))
    ),
    (
        key=:perm1,
        name="perm1_ntest",
        lb=[3e-3,3e-3,3e-3,3e-3,3e-3,1e-3,1.0],
        ub=[30.0,30.0,30.0,30.0,30.0,10.0,50.0],
        bs_idx=7,
        construct=construct_prob_delayperm1,
        noise=true, on_idx=4,
        ont_offt = (k)->(compute_ont(k,6,"perm1"), compute_offt(k,6,"perm1"))
    ),
    (
        key=:perm2,
        name="perm2_ntest",
        lb=[4e-3,4e-3,4e-3,4e-3,4e-3,4e-3,4e-3,1e-3,1.0],
        ub=[40.0,40.0,40.0,40.0,40.0,40.0,40.0,10.0,50.0],
        bs_idx=9,
        construct=construct_prob_delayperm2,
        noise=true, on_idx=5,
        ont_offt = (k)->(compute_ont(k,8,"perm2"), compute_offt(k,8,"perm2"))
    ),
]

function make_dataset(cfg; target=4000, save_dir="synthetic_data/svm/dart")
    sob = SobolSeq(cfg.lb, cfg.ub)

    out = Vector{Any}()
    out_size() = length(out)

    @time while out_size() < target
        k = Sobol.next!(sob)

        ont, offt = cfg.ont_offt(k)
        bs  = k[cfg.bs_idx] * ont
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
                    syn        = even_syn,
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

mkpath("synthetic_data/svm/true")
mkpath("synthetic_data/svm/dart")

# generate multiple off-state synthetic data
tel_ntest   = make_dataset(CONFIGS[1])
tel_times   = [m.even_time for m in tel_ntest]
@save "synthetic_data/svm/true/tel_times.jld2" tel_times # save ground-truth time data

perm_ntest   = make_dataset(CONFIGS[2])
perm_times   = [m.even_time for m in perm_ntest]
@save "synthetic_data/svm/true/perm_times.jld2" perm_times

perm1_ntest = make_dataset(CONFIGS[3])
perm1_times   = [m.even_time for m in perm1_ntest]
@save "synthetic_data/svm/true/perm1_times.jld2" perm1_times

perm2_ntest = make_dataset(CONFIGS[4]);
perm2_times   = [m.even_time for m in perm2_ntest]
@save "synthetic_data/svm/true/perm2_times.jld2" perm2_times
