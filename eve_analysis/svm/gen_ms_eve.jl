using Pkg
Pkg.activate(normpath(@__DIR__, "..", ".."))

using Random, JLD2, BSON, Statistics, Sobol

include(normpath(joinpath(@__DIR__, "..", "..", "utils", "reactions.jl")))
include(normpath(joinpath(@__DIR__, "..", "..", "utils", "ssa_multi.jl")))
include(normpath(joinpath(@__DIR__, "..", "..", "utils", "utils.jl")))

NUMS   = 120
L1, L, I0 = 1500, 1500+5165, 1.0
OBST   = 0.33
TAU = 2.33

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
]

function make_dataset(cfg; target=4000, save_dir="eve_analysis/svm")
    sob = SobolSeq(cfg.lb, cfg.ub)

    out = Vector{Any}()
    out_size() = length(out)

    @time while out_size() < target
        k = Sobol.next!(sob)

        ont, offt = cfg.ont_offt(k)
        bs  = k[cfg.bs_idx] * ont
        tau = TAU
        obst = OBST

        if 0.1 <= offt/ont < 20 && 1 <= bs <= 150 && ont >= obst
    
            params = vcat(k, [0.0, tau, 1.0, 50.0])

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
                    #syn        = even_syn,
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
tel_ntest   = make_dataset(CONFIGS[1])
mkpath("eve_analysis/svm")
tel_times   = [m.even_time for m in tel_ntest]
@save "eve_analysis/svm/tel_times.jld2" tel_times # save ground-truth time data

perm_ntest   = make_dataset(CONFIGS[2])
perm_times   = [m.even_time for m in perm_ntest]
@save "eve_analysis/svm/perm_times.jld2" perm_times;
