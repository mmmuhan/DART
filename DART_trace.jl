#!/usr/bin/env julia
using Pkg
Pkg.activate(".")
using DelimitedFiles
using ArgParse
using Logging, Dates
using Printf
using JLD2
include("csv_to_jld2.jl")
using .CSVToJLD2: csv_to_jld2_rn_vector
include("utils/utils.jl")
include("dl_train/test_model_keff.jl")

# --- process one CSV ---
function run_one_csv(csv_path::AbstractString;
                     out_dir::AbstractString,
                     gene::AbstractString,
                     seed::Int,
                     obst::Float64,
                     tau::Float64,
                     noise::Bool,
                     header::Bool,
                     delim::Union{Nothing,Char},
                     keep_nan::Bool)
    isfile(csv_path) || error("CSV not found: $csv_path")
    mkpath(out_dir)
    base = replace(basename(csv_path), r"\.csv$" => "")
    jld2_path = joinpath(out_dir, base * ".jld2")
    @info "Converting $csv_path → $jld2_path"
    rn_vector = csv_to_jld2_rn_vector(csv_path, jld2_path;
                                      header=header, delim=delim, keep_nan=keep_nan)

    @info "Running DL metrics (seed=$seed, gene=$gene, tau=$tau)"
    res = dl_metrics(seed, out_dir, out_dir, "bnb", gene, noise;
                     obst=obst, telong=tau, rn_vector=rn_vector)

    traces = res[1].binar_trace
    rates  = res[1].pred_r

    # save JLD2 (traces + rates)
    out = joinpath(out_dir, "results_seed$(seed)_$(base).jld2")
    @save out traces rates
    @info "Saved $out"

    # export trace CSV
    trace_csv = joinpath(out_dir, "results_seed$(seed)_$(base)_trace.csv")
    open(trace_csv, "w") do io
        ncols = maximum(length.(traces))
        writedlm(io, collect(1:ncols)', ',')
        writedlm(io, traces, ',')
    end
    @info "Saved $trace_csv"

    # export rates CSV
    rates_csv = joinpath(out_dir, "results_seed$(seed)_$(base)_rates.csv")
    writedlm(rates_csv, rates, ',')
    @info "Saved $rates_csv"
end

# --- CLI ---
function build_parser()
    s = ArgParseSettings(description="Convert CSVs and run dl_metrics with pre-trained bnb model.")
    @add_arg_table! s begin
        "--csv-dir";      arg_type=String;  required=true;  help="Directory with CSV files"
        "--out-dir";      arg_type=String;  default="your_data"
        "--gene";         arg_type=String;  default="gene"
        "--train-seeds";  arg_type=Int;     default=1
        "--obst";         arg_type=Float64; required=true;  help="Time resolution (min)"
        "--tau";          arg_type=Float64; required=true;  help="Elongation time τ (min)"
        "--no-noise";     action=:store_true
        "--log-level";    arg_type=String;  default="info"
        "--rn-header";    action=:store_true
        "--rn-delimiter"; arg_type=String;  default=""
        "--rn-keep-nan";  action=:store_true
    end
    return s
end

function parse_level(s::String)
    s = lowercase(s)
    s == "debug" ? Debug : s == "info" ? Info : s == "warn" ? Warn : Error
end

function main()
    args = parse_args(build_parser())
    global_logger(ConsoleLogger(stdout, parse_level(args["log-level"])))

    noise = !args["no-noise"]
    delim = isempty(args["rn-delimiter"]) ? nothing : only(args["rn-delimiter"])

    csv_dir = args["csv-dir"]
    isdir(csv_dir) || error("Not a directory: $csv_dir")
    csvs = filter(f -> endswith(f, ".csv"), readdir(csv_dir; join=true))
    isempty(csvs) && error("No .csv files in $csv_dir")

    @info "Processing $(length(csvs)) files → $(args["out-dir"])"
    for csv_path in csvs
        run_one_csv(csv_path;
            out_dir  = args["out-dir"],
            gene     = args["gene"],
            seed     = args["train-seeds"],
            obst     = args["obst"],
            tau      = args["tau"],
            noise    = noise,
            header   = args["rn-header"],
            delim    = delim,
            keep_nan = args["rn-keep-nan"])
    end
    @info "Done."
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end