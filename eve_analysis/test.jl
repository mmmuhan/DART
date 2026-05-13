using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))

using Random, StatsBase, Statistics
using CSV, DataFrames

include(normpath(joinpath(@__DIR__, "..", "dl_train", "test_model_keff.jl")))

# read traces files (7 stripes 10 fuo groups)
all_traces10 = CSV.read("eve_analysis/all_concat_traces10.csv", DataFrame);

save_dir = "eve_analysis/trained_trace10"
mkpath(save_dir) 

for sid in 1:7
    for fgroup in 1:10
        trace_subset = filter(row -> row.StripeID == sid && row.fluobin_group == fgroup, all_traces10)
        if nrow(trace_subset) == 0
            @info "No data for StripeID=$sid and fluobin_group=$fgroup"
            continue
        end
        grouped = groupby(trace_subset, :particle_id)
        traces_test = [collect(g.fluo) for g in grouped]
        test_result = dl_metrics(1, "eve_analysis/n", "eve_analysis/n", "bnb", "eve", false;
                                 obst=0.33, telong=2.33, rn_vector=traces_test, rate_idx=11:13)

        trace_sym = Symbol("dl_trace$(sid)$(fgroup)")
        rate_sym  = Symbol("dl_rate$(sid)$(fgroup)")

        trace_val = test_result[1].binar_trace
        rate_val  = test_result[1].pred_r

        filename = "eve_analysis/trained_trace10/$(String(trace_sym)).jld2"

        @eval $(trace_sym) = $trace_val
        @eval $(rate_sym)  = $rate_val
        @eval JLD2.@save $filename $(trace_sym) $(rate_sym)
    end
end

#===

using NPZ, JLD2


# Reload all 70
dl_results = Dict{Tuple{Int,Int}, NamedTuple}()
for sid in 1:7
    for fgroup in 1:10
        trace_sym = Symbol("dl_trace$(sid)$(fgroup)")
        rate_sym  = Symbol("dl_rate$(sid)$(fgroup)")
        filename  = joinpath(save_dir, "$(String(trace_sym)).jld2")
        if !isfile(filename)
            @info "File not found for StripeID=$sid fluobin_group=$fgroup, skipping"
            continue
        end
        data_jld = JLD2.load(filename)
        dl_results[(sid, fgroup)] = (
            trace = data_jld[String(trace_sym)],
            rate  = data_jld[String(rate_sym)],
        )
    end
end

# Per-(sid, fgroup) on/off times
onts  = Dict{Tuple{Int,Int}, Vector{Float64}}()
offts = Dict{Tuple{Int,Int}, Vector{Float64}}()
for sid in 1:7, fgroup in 1:10
    haskey(dl_results, (sid, fgroup)) || continue
    tr = dl_results[(sid, fgroup)].trace
    onts[(sid, fgroup)]  = on_off_time(tr, 0.33, 1.0)
    offts[(sid, fgroup)] = on_off_time(tr, 0.33, 0.0)
end

# Merge fg 1–3, 4–6, 7–10 into 3 bins for saving (7×3 = 21 entries)
fgroup_bins = [1:3, 4:6, 7:10]
data = Dict{String, Vector{Float64}}()
for sid in 1:7, (j, bin) in enumerate(fgroup_bins)
    merged = Float64[]
    for fg in bin
        haskey(offts, (sid, fg)) && append!(merged, offts[(sid, fg)])
    end
    isempty(merged) && continue
    data["$(sid)_$(j)"] = merged
end
npzwrite("eve_analysis/svm/offts3_by_ij.npz", data)
===#