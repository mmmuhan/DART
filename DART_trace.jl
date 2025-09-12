#!/usr/bin/env julia
using Pkg
Pkg.activate(".")

using DelimitedFiles  
using ArgParse
using Logging, Dates
using Printf
using JLD2
using Printf: @printf

# includes
include("csv_to_jld2.jl")
using .CSVToJLD2: csv_to_jld2_rn_vector

include("utils/utils.jl")
include("dl_train/test_model.jl")   # dl_metrics

# ── logging ────────────────────────────────────────────────────────────────────
function init_logger!(level::LogLevel = Info)
    global_logger(ConsoleLogger(stdout, level))
    @info "Started at $(Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))"
end

# ── wrapper: fixed "bnb" ───────────────────────────────────────────────────────
function compute_dl_metrics(seed::Integer,
                            in_root::AbstractString,
                            out_dir::AbstractString,
                            gene::AbstractString,
                            noise::Bool;
                            obst::Float64,
                            rn_vector)
    @info "Computing DL metrics (seed=$seed, gene=$gene)"
    t = @elapsed begin
        res = dl_metrics(seed, in_root, out_dir, "bnb", gene, noise; obst=obst, rn_vector=rn_vector)
        global last_metrics = res
        #savepath = joinpath(out_dir, "results_seed$(seed).jld2")  # optional generic save
        #@save savepath res
        #@info "Saved metrics to $savepath"
    end
    @info @sprintf("DL metrics finished in %.2f s", t)
    return last_metrics
end

# ── process one CSV ────────────────────────────────────────────────────────────
function run_one_csv(csv_path::AbstractString;
                     out_dir::AbstractString,
                     gene::AbstractString,
                     seed::Int,
                     obst::Float64,
                     noise::Bool,
                     header::Bool,
                     delim::Union{Nothing,Char},
                     keep_nan::Bool)

    isfile(csv_path) || error("CSV not found: $csv_path")

    mkpath(out_dir)
    base = replace(basename(csv_path), r"\.csv$" => "")
    jld2_path = joinpath(out_dir, base * ".jld2")

    @info "Converting CSV → JLD2" csv=csv_path out=jld2_path
    rn_vector = csv_to_jld2_rn_vector(csv_path, jld2_path;
                                      header=header, delim=delim, keep_nan=keep_nan)

    # run metrics; also save per-csv result
    res = compute_dl_metrics(seed, out_dir, out_dir, gene, noise; obst=obst, rn_vector=rn_vector)
    res = res[1].binar_trace
    per_out = joinpath(out_dir, "results_seed$(seed)_$(base).jld2")
    @save per_out res
    @info "Saved per-file metrics" path=per_out

    # Export binar_trace to CSV
    trace_csv = joinpath(out_dir, "results_seed$(seed)_$(base)_trace.csv")
    open(trace_csv, "w") do io
        ncols = maximum(length.(res))  # header: 1,2,3,...,N  where N = length of the longest trace
        writedlm(io, collect(1:ncols)', ',')
        writedlm(io, res, ',') # write data (each trace as a row)
    end
    #writedlm(trace_csv, res, ',')

    @info "Also saved CSVs" trace=trace_csv

end

# ── CLI ────────────────────────────────────────────────────────────────────────
function build_parser()
    s = ArgParseSettings(description = "Convert all CSVs in a folder and run dl_metrics using pre-trained bnb.")
    @add_arg_table! s begin
        "--csv-dir"; help="Directory containing CSV files"; arg_type=String; required=true
        "--out-dir"; help="Directory to write JLD2 and results"; arg_type=String; default="your_data"
        "--gene"; help="Gene name (e.g., eve)"; arg_type=String; default="gene"
        "--train-seeds"; help="Which trained seed to use"; arg_type=Int; default=1
        "--obst"; help="Time resolution (min)"; arg_type=Float64; required=true
        "--no-noise"; help="Disable noise inside dl_metrics"; action=:store_true
        "--log-level"; help="debug|info|warn|error"; arg_type=String; default="info"
        "--rn-header"; help="Treat first row as header"; action=:store_true
        "--rn-delimiter"; help="CSV delimiter (single char). If empty, auto-detect."; arg_type=String; default=""
        "--rn-keep-nan"; help="Keep empty cells as NaN"; action=:store_true
    end
    return s
end

function parse_level(s::String)
    s = lowercase(s)
    s=="debug" ? Debug : s=="info" ? Info : s=="warn" ? Warn : s=="error" ? Error : Info
end

function main()
    args = parse_args(build_parser())
    init_logger!(parse_level(args["log-level"]))

    noise = !args["no-noise"]

    delim = if isempty(args["rn-delimiter"])
        nothing
    elseif length(args["rn-delimiter"]) == 1
        args["rn-delimiter"][1]
    else
        error("--rn-delimiter must be a single character.")
    end

    isdir(args["csv-dir"]) || error("Not a directory: $(args["csv-dir"])")
    csvs = filter(f -> endswith(f, ".csv"), readdir(args["csv-dir"]; join=true))
    isempty(csvs) && error("No .csv files found in $(args["csv-dir"]).")

    @info "Plan" n_files=length(csvs) out_dir=args["out-dir"] gene=args["gene"] seed=args["train-seeds"] obst=args["obst"] noise=noise

    for csv_path in csvs
        run_one_csv(csv_path;
            out_dir=args["out-dir"],
            gene=args["gene"],
            seed=args["train-seeds"],
            obst=args["obst"],
            noise=noise,
            header=get(args, "rn-header", false),
            delim=delim,
            keep_nan=get(args, "rn-keep-nan", false))
    end

    @info "All done. Finished at $(Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
