#!/usr/bin/env julia
module CSVToJLD2

using CSV, DataFrames
using JLD2
using Printf

export csv_to_jld2_rn_vector

"""
Convert a ragged CSV (rows with different lengths) into a JLD2 file with:
    rn_vector :: Vector{Vector{Float64}}
"""
function csv_to_jld2_rn_vector(csv_path::AbstractString,
                               jld2_path::AbstractString;
                               header::Bool=false,
                               delim::Union{Char,Nothing}=nothing,
                               keep_nan::Bool=false)

    # Read CSV; CSV.jl will pad short rows with `missing` to the max column count.
    if delim === nothing
        tbl = CSV.File(csv_path; header=header)  # auto-detect delimiter
    else
        tbl = CSV.File(csv_path; header=header, delim=delim, ignorerepeated=true)
    end
    df  = DataFrame(tbl)

    rn_vector = Vector{Vector{Float64}}(undef, nrow(df))
    for i in 1:nrow(df)
        vals = Float64[]
        @inbounds for c in 1:ncol(df)
            x = df[i, c]
            if x === missing
                keep_nan && push!(vals, NaN); continue
            elseif x isa AbstractString
                s = strip(x)
                if isempty(s)
                    keep_nan && push!(vals, NaN); continue
                end
                y = tryparse(Float64, s)
                y === nothing ? (keep_nan && push!(vals, NaN)) : push!(vals, y)
            elseif x isa Number
                push!(vals, float(x))
            else
                keep_nan && push!(vals, NaN)
            end
        end
        rn_vector[i] = vals
    end

    isdir(dirname(jld2_path)) || mkpath(dirname(jld2_path))
    @save jld2_path rn_vector
    @printf("Saved %d vectors to %s\n", length(rn_vector), jld2_path)
    return rn_vector
end

end # module
