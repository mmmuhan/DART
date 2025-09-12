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
using Printf: @printf

# includes
include("utils/reactions.jl")
include("utils/utils.jl")
include("utils/ssa.jl")
include("dl_train/train33_data.jl")
include("dl_train/test_model.jl")   # run_train

# ── logging ─────────────────────────────────────────────────────────────────────
function init_logger!(level::LogLevel = Info)
    global_logger(ConsoleLogger(stdout, level))
    @info "Started at $(Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))"
end

# ── dataset generation ─────────────────────────────────────────────
"""
make_dataset

Generate a dataset by Sobol sampling for downstream DL training.

Arguments (keywords):
- target_n::Int           : number of accepted parameter sets to generate (default 2700)
- lb::AbstractVector      : lower bounds for 13 parameters
- ub::AbstractVector      : upper bounds for 13 parameters
- L1                      : MS2 sequence length (bp)          [required]
- L                       : L1 + L2 where L2 is gene length   [required]
- tau::Float64            : elongation time (min)             [required]
- num::Integer            : number of cells                   [required]
- obst::Float64           : time resolution (min)             [required]
- tend::Float64           : total experiment time (min)       [required]
- n_level::Float64        : noise CV (default 0.05)
- seed::Integer           : RNG seed (default 1)
- savefile::Union{Nothing,String} : optional path to JLD2 file
"""
function make_dataset(; 
    target_n::Int = 2700,
    lb::AbstractVector = [3e-3,3e-3,3e-3,3e-3,3e-3,3e-3,3e-3,3e-3,3e-3,3e-3,1.,1.,1.],
    ub::AbstractVector = [30.,30.,30.,30.,30.,30.,30.,30.,30.,30.,50.,50.,50.],
    L1::Integer, 
    L::Integer,
    tau::Float64,               
    num::Integer,               
    obst::Float64,              
    tend::Float64,              
    n_level::Float64 = 0.05,    
    seed::Integer = 1,          
    savefile::Union{Nothing,String} = nothing,
)
    # Basic sanity checks
    L > L1             || @warn "L (=$L) is not greater than L1 (=$L1). Is that intentional?"

    Random.seed!(seed)
    sobol_seq = SobolSeq(lb, ub)

    out = Any[]
    pbar = Progress(target_n; desc="Generating synthetic traces", barglyphs=BarGlyphs("[=> ]"))

    while length(out) < target_n
        k = Sobol.next!(sobol_seq)

        # Burst size constraint
        bs = (k[10]*k[12]*k[7] + k[10]*k[11]*k[8] + k[13]*k[7]*k[9]) / (k[10]*k[6]*k[8])
        ont  = compute_ont(k, 10, "gen") # mean on time
        offt = compute_offt(k, 10, "gen") # mean off time
        r    = offt / ont # burstiness level

        # Accept/reject
        if 1 <= bs <= 150 && 1 <= r <= 15 && ont >= obst
            res = generate_synthetic(construct_prob_delaygen,
                vcat(k, [0.0, tau, 1.0, tend]),
                L1, L, 1.0,
                num, obst, true, n_level, [4, 5, 6]
            )

            even_syn   = res.syn
            even_nsyn = res.nsyn
            even_trace = res.true_trace

            # At most one trace with no pulse
            if length(findall(x -> x == 0, sum.(even_syn))) <= 1
                on_time  = on_off_time.(even_trace, obst, 1.0)
                off_time = on_off_time.(even_trace, obst, 0.0)
                push!(out, (p = vcat(k, tau, obst), syn = even_syn, nsyn = even_nsyn, trace = even_trace, even_time = [on_time, off_time]))
                ProgressMeter.next!(pbar)
            end
        end
    end
    finish!(pbar)

    if savefile !== nothing
        # Ensure directory exists
        dir = dirname(savefile)
        if !isdir(dir)
            mkpath(dir)
        end
        @info "Saving dataset to $savefile"
        @save savefile out
    end

    return out
end


# ── training  ──────────────────────────────────────────
function run_training_suite(seed_list::AbstractVector{<:Integer},
                            dataset,
                            out_dir::AbstractString;
                            noise::Bool=true)
    trained_models = Dict{Int,Any}()
    tlss = Dict{Int,Any}()
    vlss = Dict{Int,Any}()
    for s in seed_list
        @info "Training start (seed=$s)"
        t = @elapsed begin
            model, train_losses, val_losses = run_train(s, dataset, out_dir, "bnb", noise)
            trained_models[s] = model
            tlss[s] = train_losses
            vlss[s] = val_losses
        end
        @info @sprintf("Training done (seed=%d) in %.2f s", s, t)
    end
    return trained_models, tlss, vlss
end

# ── CLI ────────────────────────────────────────────────────────────────────────
function build_parser()
    s = ArgParseSettings(description = "Generate synthetic data and (optionally) train DL model.")
    @add_arg_table! s begin
        "--L1"; help="MS2 sequence length (bp)"; arg_type=Int; required=true
        "--L";  help="Total length L1 + L2 (bp)"; arg_type=Int; required=true
        "--tau"; help="Elongation time (min)"; arg_type=Float64; required=true
        "--num"; help="Number of cells"; arg_type=Int; required=true
        "--obst"; help="Time resolution (min)"; arg_type=Float64; required=true
        "--tend"; help="Experiment duration (min)"; arg_type=Float64; required=true
        "--n-level"; help="Noise CV"; arg_type=Float64; default=0.05
        "--target-n"; help="Target accepted samples"; arg_type=Int; default=2700
        "--seed"; help="RNG seed"; arg_type=Int; default=1
        "--train-seeds"; help="Comma-separated seeds"; arg_type=String; default="1"
        "--out-dir"; help = "Output directory (for JLD2 data and trained models)"; arg_type = String; default = "your_data"
        "--log-level"; help="debug|info|warn|error"; arg_type=String; default="info"
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
    
    # make output path
    out_dir = args["out-dir"]
    mkpath(out_dir)
    savefile = joinpath(out_dir, "gen_ntest.jld2")

    # generate
    t_gen = @elapsed begin
        gen_ntest = make_dataset(
            target_n=args["target-n"], L1=args["L1"], L=args["L"], tau=args["tau"],
            num=args["num"], obst=args["obst"], tend=args["tend"],
            n_level=args["n-level"], seed=args["seed"], savefile=savefile,
        )
        global last_dataset = gen_ntest
    end
    @info @sprintf("Synthetic data generated in %.2f s → %s", t_gen, savefile)

    # optional train
    seeds = parse.(Int, split(args["train-seeds"], ","))
    run_training_suite(seeds, last_dataset, out_dir; noise=true)
    @info "Training complete" num_seeds=length(seeds)

    @info "All done. Finished at $(Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
