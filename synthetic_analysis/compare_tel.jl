using Pkg
Pkg.activate(normpath(@__DIR__, ".."))

using Random, JLD2, BSON, Statistics, Sobol
include(normpath(joinpath(@__DIR__, "..", "utils", "utils.jl")))
include(normpath(joinpath(@__DIR__, "..", "utils", "binar_methods.jl")))

# load all idealized synthetic data needed for this analysis
@load "../synthetic_data/ideal_data_res/telb_mtest.jld2" telb_mtest;
@load "../synthetic_data/ideal_data_res/telnb1_mtest.jld2" telnb1_mtest;
@load "../synthetic_data/ideal_data_res/telnb0_mtest.jld2" telnb0_mtest;

function fix_ratio_binar(binar_method::Function,data)
    
    t0, L1, L, I0 = 1.0, 0., 1., 1.0

    ratio_res = []; count = 0; error_counts = [];

    while count < length(data) #length(ratio_res) < 500 && 
        count += 1  
        try
            k = data[count].p                     
            obst = k[end] # obs time is smaller than min on/off time to make sure every burst can be captured

            if binar_method === synthetic_traces_hmm
                res_binar = synthetic_traces_hmm(vcat(data[count].syn...), obst, 2, 2) # use binarization method
            else
                res_binar = binar_method(vcat(data[count].syn...), obst)
            end
            
            res = (p = k, binar_trace = res_binar[1], binar_time = [res_binar[2], res_binar[3]])
            push!(ratio_res, res)
            
        catch e
            push!(error_counts, count)
            if occursin("isprobvec(hmm.a) must hold", string(e))
                println("Skipping due to hmm.a probability vector error")
                continue  # Skip this iteration and move to the next  
            #elseif occursin("maximum(logb) => Inf", string(e))
                #println("Skipping count $count due to Inf in log-likelihood (logb)")
                #continue
            elseif isa(e, HMMRetryError)
                println("Skipping index $count due to repeated HMM fitting failure")
                continue
            else
                rethrow()  # Rethrow if it's another type of error
            end
        end
    end

    return ratio_res,error_counts
end;

function save_data_methods(name::AbstractString, level::AbstractString)
    # name: methods name, should be one of "hmm", "ma" and "sg"
    # level: transcription bursting level, shoud be one of "nb0", "nb1", "b"
    
    methods_map = Dict(
        "hmm" => synthetic_traces_hmm,
        "ma"  => synthetic_traces_ma,
        "sg"  => synthetic_traces_sg,
    )
    haskey(methods_map, name) || throw(ArgumentError("name must be one of: hmm, ma, sg"))
    methods = methods_map[name]

    # only accept: "nb0", "nb1", "b"
    lvl = lowercase(level)
    lvl in ("nb0","nb1","b") || throw(ArgumentError("level must be one of: nb0, nb1, b"))

    # dataset by level
    level_map = Dict(
        "nb0" => telnb0_mtest,
        "nb1" => telnb1_mtest,
        "b"   => telb_mtest,
    )
    data = level_map[lvl]

    # run binarization
    @time rn_data_binar, error_from_method = fix_ratio_binar(methods, data[1:600])

    # collect additional short-trace errors
    error_ids = Int[]
    for i in 1:length(rn_data_binar)
        bt = rn_data_binar[i].binar_time
        if length(bt[1]) <= 20 && length(bt[2]) <= 20
            push!(error_ids, i)
        end
    end
    error_ids = sort(unique(vcat(error_ids, error_from_method)))
    keep = setdiff(1:600, error_ids)

    # align GT and predictions
    rn_data_revised = data[keep]
    if name == "hmm"
        rn_data_binar = copy(rn_data_binar)  
    else
        rn_data_binar   = rn_data_binar[keep]
    end
    
    rn_met_binar = compare_metric(rn_data_revised, rn_data_binar; compute_acc = true)

    # drop heavy field(s) before saving
    rn_data_filter = copy(rn_data_binar)
    to_delete = Set([:binar_trace])
    rn_data_filter = [(; filter(kv -> !(kv[1] in to_delete), pairs(v))...) for v in rn_data_filter]
    rn_binar_filter = copy(rn_data_filter)

    # save with long tag for filename
    save_tag = Dict("nb0"=>"telnb0","nb1"=>"telnb1","b"=>"telb")[lvl]
    outdir = "synthetic_data/base_compare"
    mkpath(outdir)
    save_path = joinpath(outdir, "$(save_tag)_$(name).jld2")
    @save save_path rn_binar_filter rn_met_binar error_ids

end;

# run each transcriptional bursting and binarization method separately
# for example, "ma" method at low burstiness level:
save_data_methods("hmm",  "b");
save_data_methods("hmm",  "nb1");
save_data_methods("hmm",  "nb0");