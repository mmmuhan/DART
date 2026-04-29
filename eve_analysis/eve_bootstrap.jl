using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))
using Random, StatsBase, Statistics
include(normpath(joinpath(@__DIR__, "..", "dl_train", "test_model_keff.jl")))
include(normpath(joinpath(@__DIR__, "..", "utils", "utils.jl")))
using CSV, DataFrames, JLD2

all_traces10 = CSV.read("eve_analysis/all_concat_traces10.csv", DataFrame);

# --- helper: effective switching rates from a set of binarized traces ---
function eff_rates(bin_tr::AbstractVector, dt::Float64; ϵ::Float64=1e-12)
    n_off_on = 0
    n_on_off = 0
    t_off = 0.0
    t_on  = 0.0
    for tr in bin_tr
        t_off += count(==(0), tr) * dt
        t_on  += count(==(1), tr) * dt
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

# --- bootstrap with effective rates + pred_r ---
function boot_dl_rates(traces_test;
                       reps::Int=100,
                       obst::Float64=0.33,
                       telong::Float64=2.33,
                       ϵ::Float64=1e-12)
    N = length(traces_test)
    onrate  = Vector{Float64}(undef, reps)
    offrate = Vector{Float64}(undef, reps)
    Pon_vec = Vector{Float64}(undef, reps)
    rate_vec = Vector{Float64}(undef, reps)   # ← pred_r across bootstrap reps

    for r in 1:reps
        # 1) Bootstrap resample cells
        idxs   = sample(1:N, N; replace=true)
        rn_vec = traces_test[idxs]

        # 2) Re-run DART on the resampled data
        res = dl_metrics(1, "eve_analysis/n", "eve_analysis/n", "bnb", "eve", false;
                         obst=obst, telong=telong,
                         rn_vector=rn_vec, rate_idx=11:13)

        # 3) Extract binarized traces and pred_r
        bin_tr   = res[1].binar_trace        # Vector of per-particle segment vectors
        pred_r   = res[1].pred_r            # scalar rate estimate

        # 4) Effective rates from pooled counts/times
        kon_eff, koff_eff = eff_rates(bin_tr, obst; ϵ=ϵ)
        onrate[r]  = kon_eff
        offrate[r] = koff_eff
        denom      = kon_eff + koff_eff
        Pon_vec[r] = kon_eff / max(denom, ϵ)
        rate_vec[r] = pred_r
    end

    μ_onr,  sd_onr  = mean(onrate),  std(onrate)
    μ_offr, sd_offr = mean(offrate), std(offrate)
    μ_pon, sd_pon = mean(Pon_vec), std(Pon_vec)
    μ_rate, sd_rate = mean(rate_vec), std(rate_vec)   # ← mean/std of pred_r

    return μ_onr, sd_onr, μ_offr, sd_offr, μ_pon, sd_pon, μ_rate, sd_rate
end

@time begin
    mean_onrate  = zeros(7,10);  sd_onrate  = zeros(7,10)
    mean_offrate = zeros(7,10);  sd_offrate = zeros(7,10)
    mean_Pon     = zeros(7,10); sd_Pon = zeros(7,10)
    mean_rate    = zeros(7,10);  sd_rate    = zeros(7,10)  # ← pred_r storage

    for sid in 1:7, fgroup in 1:10
        trace_subset = filter(row -> row.StripeID == sid && row.fluobin_group == fgroup, all_traces10)
        if nrow(trace_subset) == 0
            @info "No data for StripeID=$sid and fluobin_group=$fgroup"; continue
        end
        grouped     = groupby(trace_subset, :particle_id)
        traces_test = [collect(g.fluo) for g in grouped]

        μ_onr, sd_onr, μ_offr, sd_offr, μ_pon, sd_pon, μ_rate, sd_r =
            boot_dl_rates(traces_test; reps=100, obst=0.33, telong=2.33)

        mean_onrate[sid,fgroup]  = μ_onr;   sd_onrate[sid,fgroup]  = sd_onr
        mean_offrate[sid,fgroup] = μ_offr;  sd_offrate[sid,fgroup] = sd_offr
        mean_Pon[sid,fgroup]     = μ_pon; sd_Pon[sid,fgroup] = sd_pon
        mean_rate[sid,fgroup]    = μ_rate;  sd_rate[sid,fgroup]    = sd_r
    end
end

@save "eve_analysis/boot_dl_results.jld2" mean_onrate sd_onrate mean_offrate sd_offrate mean_Pon sd_Pon mean_rate sd_rate