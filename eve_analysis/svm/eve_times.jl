using Pkg
PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(PROJECT_ROOT)

using NPZ, JLD2

include(joinpath(PROJECT_ROOT, "utils", "utils.jl"))

save_dir = joinpath(PROJECT_ROOT, "eve_analysis", "trained_trace10")
outpath  = joinpath(PROJECT_ROOT, "eve_analysis", "svm", "offts3_by_ij.npz")
mkpath(dirname(outpath))

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

        data = JLD2.load(filename)
        dl_results[(sid, fgroup)] = (
            trace = data[String(trace_sym)],
            rate  = data[String(rate_sym)]
        )
    end
end

onts = []
offts = []
fgroup_bins = [1:3, 4:6, 7:10]
for sid in 1:7
    ons_bins = []
    offs_bins = []
    for bin in fgroup_bins
        fgs = [fg for fg in bin if haskey(dl_results, (sid, fg))]
        ons  = vcat([vcat(on_off_time.(dl_results[(sid, fg)].trace, 0.33, 1.0)...) for fg in fgs]...)
        offs = vcat([vcat(on_off_time.(dl_results[(sid, fg)].trace, 0.33, 0.0)...) for fg in fgs]...)
        push!(ons_bins, ons)
        push!(offs_bins, offs)
    end
    push!(onts, ons_bins)
    push!(offts, offs_bins)
end

data = Dict{String, Vector{Float64}}()   # adjust eltype if needed
for i in 1:7, j in 1:3
    data["$(i)_$(j)"] = offts[i][j]
end
npzwrite(outpath, data)