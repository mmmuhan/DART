#!/usr/bin/env julia
using Pkg
Pkg.activate(".")

using Random, Statistics
using Sobol
using JLD2, BSON
using ArgParse
using ProgressMeter
using Logging, Dates
using Printf

include("utils/reactions.jl")
include("utils/utils.jl")
include("utils/ssa_multi.jl")
include("dl_train/train33_data_keff.jl")

# --- dataset generation ---

function make_dataset(;
    target_n::Int = 2700,
    lb = [3e-3,3e-3,3e-3,3e-3,3e-3,3e-3,3e-3,3e-3,3e-3,3e-3,1.,1.,1.],
    ub = [30.,30.,30.,30.,30.,30.,30.,30.,30.,30.,50.,50.,50.],
    L1::Int, L::Int, tau::Float64,
    num::Int, obst::Float64, tend::Float64,
    n_level::Float64 = 0.05,
    seed::Int = 1,
    savefile::Union{Nothing,String} = nothing,
)
    L > L1 || @warn "L=$L is not greater than L1=$L1"

    Random.seed!(seed)
    sobol_seq = SobolSeq(lb, ub)

    out = Any[]
    pbar = Progress(target_n; desc="Generating traces")

    while length(out) < target_n
        k = Sobol.next!(sobol_seq)

        bs   = (k[10]*k[12]*k[7] + k[10]*k[11]*k[8] + k[13]*k[7]*k[9]) / (k[10]*k[6]*k[8])
        ont  = compute_ont(k, 10, "gen")
        offt = compute_offt(k, 10, "gen")
        r    = offt / ont

        if !(1 <= bs <= 150 && 1 <= r <= 15 && ont >= obst)
            continue
        end

        res = generate_synthetic(construct_prob_delaygen,
            vcat(k, [0.0, tau, 1.0, tend]),
            L1, L, 1.0, num, obst, true, n_level, [4, 5, 6])

        # skip if more than one cell has zero signal
        if count(s -> sum(s) == 0, res.syn) > 1
            continue
        end

        on_time  = on_off_time.(res.true_trace, obst, 1.0)
        off_time = on_off_time.(res.true_trace, obst, 0.0)

        push!(out, (p = vcat(k, tau, obst),
                    syn = res.syn, nsyn = res.nsyn,
                    trace = res.true_trace,
                    even_time = [on_time, off_time]))
        ProgressMeter.next!(pbar)
    end
    finish!(pbar)

    if savefile !== nothing
        mkpath(dirname(savefile))
        @info "Saving dataset → $savefile"
        @save savefile out
    end

    return out
end

# --- training ---

function run_training_suite(seeds, dataset, out_dir; noise=true)
    models = Dict{Int,Any}()
    metrics = Dict{Int,Any}()
    preds  = Dict{Int,Any}()

    for s in seeds
        @info "Training seed=$s"
        t = @elapsed begin
            m, met, pr = run_train_rates(s, dataset, out_dir, "bnb", noise)
            models[s]  = m
            metrics[s] = met
            preds[s]   = pr
        end
        @info @sprintf("  seed=%d done in %.1f s", s, t)
    end
    return models, metrics, preds
end

# --- CLI ---

function build_parser()
    s = ArgParseSettings(description="Generate synthetic data and train DL model.")
    @add_arg_table! s begin
        "--L1";          arg_type=Int;     required=true;  help="MS2 sequence length (bp)"
        "--L";           arg_type=Int;     required=true;  help="Total length L1+L2 (bp)"
        "--tau";         arg_type=Float64; required=true;  help="Elongation time (min)"
        "--num";         arg_type=Int;     required=true;  help="Number of cells"
        "--obst";        arg_type=Float64; required=true;  help="Time resolution (min)"
        "--tend";        arg_type=Float64; required=true;  help="Experiment duration (min)"
        "--n-level";     arg_type=Float64; default=0.05
        "--target-n";    arg_type=Int;     default=2700
        "--seed";        arg_type=Int;     default=1
        "--train-seeds"; arg_type=String;  default="1";    help="Comma-separated seeds"
        "--out-dir";     arg_type=String;  default="your_data"
        "--log-level";   arg_type=String;  default="info"
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

    out_dir = args["out-dir"]
    mkpath(out_dir)
    savefile = joinpath(out_dir, "gen_ntest.jld2")

    @info "Generating dataset..."
    t = @elapsed dataset = make_dataset(
        target_n = args["target-n"],
        L1 = args["L1"], L = args["L"], tau = args["tau"],
        num = args["num"], obst = args["obst"], tend = args["tend"],
        n_level = args["n-level"], seed = args["seed"],
        savefile = savefile)
    @info @sprintf("Generated %d samples in %.1f s", length(dataset), t)

    seeds = parse.(Int, split(args["train-seeds"], ","))
    run_training_suite(seeds, dataset, out_dir; noise=true)

    @info "All done."
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end