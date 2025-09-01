using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))

using Random, StatsBase, Statistics
include(normpath(joinpath(@__DIR__, "..", "dl_train", "test_model.jl")))
#include(normpath(joinpath(@__DIR__, "..", "utils", "utils.jl")))

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
                                 obst=0.33, rn_vector=traces_test)

        varsym   = Symbol("dl_trace$(sid)$(fgroup)")
        val      = test_result[1].binar_trace
        filename = "eve_analysis/trained_trace10/$(String(varsym)).jld2"

        # save 
        @eval $(varsym) = $val
        @eval JLD2.@save $filename $(varsym)
    end
end
