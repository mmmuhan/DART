using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))

using Random, StatsBase, Statistics
include(normpath(joinpath(@__DIR__, "..", "dl_train", "test_model.jl")))
include(normpath(joinpath(@__DIR__, "..", "utils", "utils.jl")))

using CSV, DataFrames

# read traces files (7 stripes 10 fuo groups)
all_traces10 = CSV.read("eve_analysis/all_concat_traces10.csv", DataFrame);

# --- helper: effective rates from a set of binarized traces ---
# bin_tr: Vector of traces; each trace is a Vector{0/1}. dt = sampling interval
function eff_rates(bin_tr::AbstractVector,
                              dt::Float64; ϵ::Float64=1e-12)
    n_off_on = 0      # number of 0→1 transitions
    n_on_off = 0      # number of 1→0 transitions
    t_off = 0.0       # total time spent in 0
    t_on  = 0.0       # total time spent in 1

    for tr in bin_tr
        # total ON/OFF times
        t_off += count(==(0), tr) * dt
        t_on  += count(==(1), tr) * dt

        # transition counts
        @inbounds for i in 2:length(tr)
            prev, curr = tr[i-1], tr[i]
            n_off_on += (prev == 0 && curr == 1)
            n_on_off += (prev == 1 && curr == 0)
        end
    end

    kon_eff  = n_off_on / max(t_off, ϵ)
    koff_eff = n_on_off / max(t_on,  ϵ)
    return kon_eff, koff_eff
end

# --- bootstrap using effective-rate definitions ---
function boot_dl_rates(traces_test; reps::Int=100, obst::Float64=0.33, ϵ::Float64=1e-12)
    N = length(traces_test)
    onrate  = Vector{Float64}(undef, reps)
    offrate = Vector{Float64}(undef, reps)
    Pon_vec = Vector{Float64}(undef, reps)
    
    for r in 1:reps
        # 1) Bootstrap resample cells
        idxs = sample(1:N, N; replace=true)
        rn_vec = traces_test[idxs]

        # 2) Re-run DART on the resampled data
        res = dl_metrics(1, "eve_analysis/n", "eve_analysis/n", "bnb", "eve", false;
                         obst=obst, rn_vector=rn_vec)

        # 3) Extract binarized traces (Vector of 0/1 vectors)
        bin_tr = res[1].binar_trace

        # 4) Effective rates from pooled counts/times
        kon_eff, koff_eff = eff_rates(bin_tr, obst; ϵ=ϵ)

        onrate[r]  = kon_eff
        offrate[r] = koff_eff
        denom      = kon_eff + koff_eff
        Pon_vec[r] = kon_eff / max(denom, ϵ)
    end

    μ_onr,  sd_onr  = mean(onrate),  std(onrate)
    μ_offr, sd_offr = mean(offrate), std(offrate)
    μ_Pon = mean(Pon_vec)

    return μ_onr, sd_onr, μ_offr, sd_offr, μ_Pon
end

@time begin
mean_onrate  = zeros(7,10);  sd_onrate  = zeros(7,10)
mean_offrate = zeros(7,10);  sd_offrate = zeros(7,10)
Pon_mean     = zeros(7,10);   

for sid in 1:7, fgroup in 1:10
    trace_subset = filter(row -> row.StripeID == sid && row.fluobin_group == fgroup, all_traces10)
    if nrow(trace_subset) == 0
        @info "No data for StripeID=$sid and fluobin_group=$fgroup";  continue
    end
    grouped = groupby(trace_subset, :particle_id)
    traces_test = [collect(g.fluo) for g in grouped]

    μ_onr, sd_onr, μ_offr, sd_offr, μ_Pon =
        boot_dl_rates(traces_test; reps=100, obst=0.33)

    mean_onrate[sid,fgroup]  = μ_onr;  sd_onrate[sid,fgroup]  = sd_onr
    mean_offrate[sid,fgroup] = μ_offr; sd_offrate[sid,fgroup] = sd_offr
    Pon_mean[sid,fgroup]     = μ_Pon; 
end
    
end

using JLD2

@save "eve_analysis/boot_dl_results.jld2" mean_onrate sd_onrate mean_offrate sd_offrate Pon_mean;
