# =============================================================================
# test_model_keff.jl
# General inference script for binary state + k_eff_on prediction
#
# Matches train33_keff.jl (run_train_rates):
#   - State classification: binary (OFF vs ON)
#   - Rate prediction: global mean pool + elongation time → 1 scalar (k_eff_on)
#   - Works for any number of ON states
#   - rate_idx: which indices in p hold the loading rates
#
# Usage:
#   rn_data mode:   dl_metrics(seed, folder, obs, model, rn, noise; rate_idx=5:6)
#   rn_vector mode: dl_metrics(seed, folder, obs, model, rn, noise;
#                              obst=0.007, telong=0.4, rn_vector=vecs)
# =============================================================================

ENV["JULIA_CUDA_USE_BINARYBUILDER"] = "true"

using BSON, JLD2, Statistics
using Flux
using CUDA, ProgressMeter

# -----------------------------
# Helpers
# -----------------------------
to_device(x) = CUDA.functional() ? cu(x) : x

mutable struct Metrics
    data::Dict{Symbol, Vector{Float64}}
end

function global_mean_pool(h)
    dropdims(mean(h, dims=2); dims=2)
end

function compute_keff_on(trace, rates)
    n_on = length(rates)
    num = 0.0
    denom = 0.0
    for i in 1:n_on
        πi = mean(trace .== i)
        num += πi * rates[i]
        denom += πi
    end
    denom < 1e-8 && return 0.0
    return num / denom
end

function logits_to_binary_trace(y_logits)
    cls = dropdims(Array(argmax(y_logits, dims=1)); dims=1)
    cls = first.(Tuple.(cls))
    return Float64.(cls .- 1)
end

function segment_signals(signals, segment_length; pad_last::Bool=false)
    segmented_signals = []
    nums = Int[]
    for signal in signals
        sig = signal
        if pad_last && length(sig) % segment_length != 0
            pad_len = segment_length - (length(sig) % segment_length)
            sig = vcat(sig, zeros(pad_len))
        end
        num_segments = div(length(sig), segment_length)
        push!(nums, num_segments)
        for i in 0:(num_segments - 1)
            push!(segmented_signals, sig[(i * segment_length + 1):(i * segment_length + segment_length)])
        end
    end
    return segmented_signals, nums
end

# -----------------------------
# Inference
# -----------------------------
function infer_states_and_rate(model_parts, x, elong_batch)
    h = Base.invokelatest(model_parts.trunk, x)
    y_logits = Base.invokelatest(model_parts.state_head, h)

    h_pool = global_mean_pool(h)
    h_cat = vcat(h_pool, elong_batch)
    r_hat = Base.invokelatest(model_parts.rate_head, h_cat)

    return y_logits, r_hat
end

# -----------------------------
# Compare metric
# -----------------------------
function compare_metric(true_data, binar_data; compute_acc=false)
    results = []

    if length(true_data) != length(binar_data)
        error("true_data and binar_data must have the same length.")
    end

    for (td, bd) in zip(true_data, binar_data)
        try
            k = bd.p

            true_trace_full = vcat(td.trace...)
            binar_trace_full = bd.binar_trace

            lth = min(length(true_trace_full), length(binar_trace_full))
            y_true = Float64.(true_trace_full[1:lth] .> 0)
            y_binar = Float64.(binar_trace_full[1:lth])

            o = k[end]
            true_on_t  = on_off_time(y_true, o, 1.0)
            true_off_t = on_off_time(y_true, o, 0.0)
            binar_on_t  = on_off_time(y_binar, o, 1.0)
            binar_off_t = on_off_time(y_binar, o, 0.0)

            true_ont  = isempty(true_on_t)  ? NaN : mean(true_on_t)
            true_offt = isempty(true_off_t) ? NaN : mean(true_off_t)
            binar_ont  = isempty(binar_on_t)  ? NaN : mean(binar_on_t)
            binar_offt = isempty(binar_off_t) ? NaN : mean(binar_off_t)

            true_oncv  = isempty(true_on_t)  ? NaN : std(true_on_t) / mean(true_on_t)
            true_offcv = isempty(true_off_t) ? NaN : std(true_off_t) / mean(true_off_t)
            binar_oncv  = isempty(binar_on_t)  ? NaN : std(binar_on_t) / mean(binar_on_t)
            binar_offcv = isempty(binar_off_t) ? NaN : std(binar_off_t) / mean(binar_off_t)

            if hasproperty(bd, :pred_r) && hasproperty(bd, :true_r) &&
               bd.pred_r !== nothing && bd.true_r !== nothing
                rel_r = abs((bd.pred_r - bd.true_r) / (bd.true_r + 1e-8))
            else
                rel_r = nothing
            end

            if compute_acc
                acc = mean(y_binar .== y_true)

                result = (
                    p = k,
                    binart = [binar_on_t, binar_off_t],
                    truet_mean = [true_ont, true_offt],
                    binart_mean = [binar_ont, binar_offt],
                    truet_cv = [true_oncv, true_offcv],
                    binart_cv = [binar_oncv, binar_offcv],
                    acc = acc,
                    pred_r = hasproperty(bd, :pred_r) ? bd.pred_r : nothing,
                    true_r = hasproperty(bd, :true_r) ? bd.true_r : nothing,
                    rel_r = rel_r
                )
            else
                result = (
                    p = k,
                    binart = [binar_on_t, binar_off_t],
                    truet_mean = [true_ont, true_offt],
                    binart_mean = [binar_ont, binar_offt],
                    truet_cv = [true_oncv, true_offcv],
                    binart_cv = [binar_oncv, binar_offcv],
                    pred_r = hasproperty(bd, :pred_r) ? bd.pred_r : nothing,
                    true_r = hasproperty(bd, :true_r) ? bd.true_r : nothing,
                    rel_r = rel_r
                )
            end

            push!(results, result)

        catch e
            if isa(e, ArgumentError) && occursin("PhaseType", string(e))
                println("Skipping due to error: $(e)")
                continue
            else
                rethrow()
            end
        end
    end

    return results
end

# -----------------------------
# Main inference function
# -----------------------------
function dl_metrics(
    seed::Int,
    folder_name::String,
    obs_name::String,
    model_name::String, # burstiness level
    rn::String, # model type
    noise::Bool;
    obst::Union{Nothing,Real}=nothing,
    telong::Union{Nothing,Real}=nothing,
    rn_vector::Union{Nothing, Vector{Vector{Float64}}}=nothing,
    rate_idx::UnitRange{Int} = 0:-1
)
    # --- Load model ---
    model_blob = BSON.load("$(folder_name)trained_model$(model_name)_seed_$(seed).bson")
    model_parts_cpu = model_blob[:model_parts_cpu]

    # Load rate_idx from saved model if not provided
    if isempty(rate_idx) && haskey(model_blob, :rate_idx)
        rate_idx = model_blob[:rate_idx]
    end

    model_parts = (
        trunk = to_device(model_parts_cpu.trunk),
        state_head = to_device(model_parts_cpu.state_head),
        rate_head = to_device(model_parts_cpu.rate_head)
    )

    # --- Build input ---
    if rn_vector !== nothing
        x = vcat(rn_vector...)
        orig_len = length(x)

        padded_len = ceil(Int, orig_len / 5000) * 5000
        pad_amount = padded_len - orig_len
        x_padded = vcat(x, zeros(pad_amount))

        x_norm = min_max(x_padded)
        signals = [x_norm]
        seg_x, seg_nums = segment_signals(signals, 5000; pad_last=false)
    else
        x = [vcat((noise ? r.nsyn : r.syn)...) for r in rn_data]
        x_norm = min_max.(x)
        signals = x_norm
        seg_x, seg_nums = segment_signals(signals, 5000)
    end

    X = hcat(seg_x...)
    xb = Float32.(permutedims(reshape(X, 1, size(X, 1), size(X, 2)), (2, 1, 3)))
    xb = to_device(xb)

    total_segs = size(xb, 3)

    # --- Build elongation time tensor ---
    if rn_vector !== nothing
        if telong === nothing
            error("elongation time must be provided when using rn_vector")
        end
        elong_vec = fill(Float32(telong), total_segs)
    else
        elong_vec = Float32[]
        for (i, n) in enumerate(seg_nums)
            e = Float32(rn_params[i][end-1])
            append!(elong_vec, fill(e, n))
        end
    end
    elong_all = to_device(reshape(elong_vec, 1, :))

    # --- Inference ---
    if rn_vector !== nothing
        y_logits, r_pred = infer_states_and_rate(model_parts, xb, elong_all)
    else
        batch = 100
        y_logits = nothing
        r_pred = nothing
        last_full = floor(Int, total_segs / batch) * batch

        for i in 1:batch:last_full
            xchunk = xb[:, :, i:i+batch-1]
            echunk = elong_all[:, i:i+batch-1]
            yslice, rslice = infer_states_and_rate(model_parts, xchunk, echunk)
            y_logits = y_logits === nothing ? yslice : cat(y_logits, yslice; dims=3)
            r_pred   = r_pred   === nothing ? rslice : cat(r_pred, rslice; dims=2)
        end

        rem = last_full + 1
        if rem <= total_segs
            xchunk = xb[:, :, rem:total_segs]
            echunk = elong_all[:, rem:total_segs]
            yslice, rslice = infer_states_and_rate(model_parts, xchunk, echunk)
            y_logits = y_logits === nothing ? yslice : cat(y_logits, yslice; dims=3)
            r_pred   = r_pred   === nothing ? rslice : cat(r_pred, rslice; dims=2)
        end
    end

    # --- Decode states (binary) ---
    y_binar_states = logits_to_binary_trace(y_logits)
    binar_traces = [vec(y_binar_states[:, i]) for i in 1:size(y_binar_states, 2)]

    # --- Aggregate rate predictions per trace (median across segments) ---
    r_pred_orig = vec(Array(cpu(r_pred)))
    y_binar_trace = Vector{Vector{Float64}}()
    y_binar_rates = Vector{Float32}()

    idx = 1
    for n in seg_nums
        if idx + n - 1 <= length(binar_traces)
            trace = vcat(binar_traces[idx:idx+n-1]...)
            if rn_vector !== nothing
                trace = trace[1:orig_len]
            end
            push!(y_binar_trace, trace)
            push!(y_binar_rates, Float32(median(r_pred_orig[idx:idx+n-1])))
            idx += n
        else
            @warn "Skipping segment due to out-of-bounds access: idx=$(idx), n=$(n)"
            break
        end
    end

    # --- Build output ---
    rn_data_dl = []

    if obst !== nothing
        for (i, yt) in enumerate(y_binar_trace)
            raw_lengths = length.(rn_vector)
            split_idxs = cumsum(vcat(1, raw_lengths[1:end-1]))

            seg_list = []
            on_t_list = []
            off_t_list = []

            for (j, start_idx) in enumerate(split_idxs)
                end_idx = start_idx + raw_lengths[j] - 1
                if end_idx > length(yt)
                    continue
                end

                seg = yt[start_idx:end_idx]
                push!(seg_list, seg)
                push!(on_t_list, on_off_time(seg, obst, 1.0))
                push!(off_t_list, on_off_time(seg, obst, 0.0))
            end

            on_t = vcat(on_t_list...)
            off_t = vcat(off_t_list...)

            push!(rn_data_dl, (
                binar_trace = seg_list,
                binar_time = [on_t, off_t],
                pred_r = y_binar_rates[i]
            ))
        end

        return rn_data_dl

    else
        for (i, yt) in enumerate(y_binar_trace)
            k = rn_params[i]
            o = k[end]

            raw_lengths = length.((noise ? rn_data[i].nsyn : rn_data[i].syn))
            split_idxs = cumsum(vcat(1, raw_lengths[1:end-1]))

            on_t_list = []
            off_t_list = []

            for (j, start_idx) in enumerate(split_idxs)
                end_idx = start_idx + raw_lengths[j] - 1
                if end_idx > length(yt)
                    continue
                end

                seg = yt[start_idx:end_idx]
                push!(on_t_list, on_off_time(seg, o, 1.0))
                push!(off_t_list, on_off_time(seg, o, 0.0))
            end

            on_t = vcat(on_t_list...)
            off_t = vcat(off_t_list...)

            # true k_eff_on from ground truth
            true_keff = Float32(compute_keff_on(
                vcat(rn_data[i].trace...),
                Float64.(k[rate_idx])
            ))

            push!(rn_data_dl, (
                p = k,
                binar_trace = yt,
                binar_time = [on_t, off_t],
                pred_r = y_binar_rates[i],
                true_r = true_keff
            ))
        end

        rn_met_dl = compare_metric(rn_data_true, rn_data_dl; compute_acc=true)
        @save "$(obs_name)$(rn)_dl$(model_name)_compare_seed_$(seed).jld2" rn_met_dl
        return rn_met_dl, rn_data_dl
    end
end