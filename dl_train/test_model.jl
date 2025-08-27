using BSON, JLD2
using Flux
using Flux: train!
using Flux: onehotbatch, logitcrossentropy
using CUDA,ProgressMeter #cuDNN

function compare_metric(true_data, binar_data; compute_fit=false, compute_acc=false)
    results = []

    if length(true_data) != length(binar_data)
        error("true_data and binar_data must have the same length.")
    end

    for (td, bd) in zip(true_data, binar_data)
        try
            k = bd.p

            # Compute mean times
            true_ont, true_offt = mean.(Iterators.flatten.([td.even_time[1], td.even_time[2]]))
            binar_ont, binar_offt = mean.(bd.binar_time)

            # Compute coefficient of variation (CV)
            true_oncv, true_offcv = std.(Iterators.flatten.([td.even_time[1], td.even_time[2]])) ./ mean.(Iterators.flatten.([td.even_time[1], td.even_time[2]]))
            binar_oncv, binar_offcv = std.(bd.binar_time) ./ mean.(bd.binar_time)

            # Compute fit-related metrics if enabled
            if compute_fit
                id = 3
                bic_binar = time2bic(bd.binar_time[2])
                id_binar = find_min_idx(bic_binar.bic, 5)
                err_binar = errors(id - 1, bd.p, bic_binar)
                result = (
                    p = k, binart = [bd.binar_time[1], bd.binar_time[2]],
                    truet_mean = [true_ont, true_offt], binart_mean = [binar_ont, binar_offt],
                    truet_cv = [true_oncv, true_offcv], binart_cv = [binar_oncv, binar_offcv], 
                    bic = bic_binar, id = id_binar, err = err_binar)

            # Compute accuracy if enabled
            elseif compute_acc
                #y_true = vcat(td.trace...)[1:length(bd.binar_trace)]
                #y_binar = bd.binar_trace
                lth = min(length(vcat(td.trace...)),length(bd.binar_trace))
                y_true = vcat(td.trace...)[1:lth]; y_binar = bd.binar_trace[1:lth]
                acc = mean(y_binar .== y_true)
                
                
                result = (p = k, binart = [bd.binar_time[1], bd.binar_time[2]],  # Vector of vectors
                    truet_mean = [true_ont, true_offt],
                    binart_mean = [binar_ont, binar_offt],
                    truet_cv = [true_oncv, true_offcv],
                    binart_cv = [binar_oncv, binar_offcv], acc = acc)
                    
            else 
                result = (
                p = k,
                binart = [bd.binar_time[1], bd.binar_time[2]],  # Vector of vectors
                truet_mean = [true_ont, true_offt],
                binart_mean = [binar_ont, binar_offt],
                truet_cv = [true_oncv, true_offcv],
                binart_cv = [binar_oncv, binar_offcv])
            end

            push!(results, result)
        catch e
            if isa(e, ArgumentError) && occursin("PhaseType: the condition all(π .>= zero(π[1])) is not satisfied", string(e))
                println("Skipping due to error: $(e)")
                continue  # Skip this iteration
            else
                rethrow()  # Rethrow unexpected errors
            end
        end
    end

    return results
end;


function segment_signals(signals, segment_length; pad_last::Bool=false)
    segmented_signals = []
    nums = []

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

function dl_metrics(
    seed::Int,
    folder_name::String,
    obs_name::String,
    model_name::String,
    rn::String,
    noise::Bool;
    obst::Union{Nothing,Real}=nothing,
    rn_vector::Union{Nothing, Vector{Vector{Float64}}} = nothing
)
    # —– Load & move model to GPU
    BSON.@load "$(folder_name)trained_model$(model_name)_seed_$(seed).bson" model
    model = gpu(model)

    # —– Build or accept input signals
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

    # —– Prepare GPU tensor
    X = hcat(seg_x...)
    xb = reshape(X, 1, size(X,1), size(X,2))
    xb = Float32.(xb)
    xb = permutedims(xb, (2,1,3))
    xb = cu(xb)

    # —– Inference
    if rn_vector !== nothing
        y_pred_binary = Base.invokelatest(model, xb) .> 0.5
    else
        batch = 100
        total = size(xb, 3)
        y_pred_binary = nothing
        last_full = floor(Int, total / batch) * batch

        for i in 1:batch:last_full
            xchunk = xb[:, :, i:i+batch-1]
            yslice = Base.invokelatest(model, xchunk) .> 0.5
            y_pred_binary = y_pred_binary === nothing ? yslice : cat(y_pred_binary, yslice; dims=3)
        end

        rem = last_full + 1
        if rem <= total
            xchunk = xb[:, :, rem:total]
            yslice = Base.invokelatest(model, xchunk) .> 0.5
            y_pred_binary = cat(y_pred_binary, yslice; dims=3)
        end
    end

    # —– Reassemble full-length traces
    pred_traces = [Float64.(vec(Array(y_pred_binary[:, :, i]))) for i in 1:size(y_pred_binary, 3)]
    y_pred_trace = Vector{Vector{Float64}}()
    idx = 1
    for n in seg_nums
        if idx + n - 1 <= length(pred_traces)
            trace = vcat(pred_traces[idx:idx+n-1]...)
            if rn_vector !== nothing
                trace = trace[1:orig_len]
            end
            push!(y_pred_trace, trace)
            idx += n
        else
            @warn "Skipping segment due to out‐of‐bounds access: idx=$(idx), n=$(n)"
            break
        end
    end

    # —– Build output
    rn_data_dl = []
    if obst !== nothing
        for (i, yt) in enumerate(y_pred_trace)
            raw_lengths = length.(rn_vector)
            split_idxs = cumsum(vcat(1, raw_lengths[1:end-1]))
            on_t_list = []
            off_t_list = []
            seg_list = []
            for (j, start_idx) in enumerate(split_idxs)
                end_idx = start_idx + raw_lengths[j] - 1
                if end_idx > length(yt)
                    #@warn "Segment end index $end_idx exceeds trace length $(length(yt)). Skipping."
                    continue
                end
                seg = yt[start_idx:end_idx]
                push!(on_t_list, on_off_time(seg, obst, 1.0))
                push!(off_t_list, on_off_time(seg, obst, 0.0))
                push!(seg_list, seg)
            end
            on_t = vcat(on_t_list...)
            off_t = vcat(off_t_list...)
            push!(rn_data_dl, (binar_trace=seg_list, binar_time=[on_t, off_t])) #binar_trace = yt
        end
        return rn_data_dl
    else
        for (i, yt) in enumerate(y_pred_trace)
            k = rn_params[i]
            o = k[end]
            raw_lengths = length.((noise ? rn_data[i].nsyn : rn_data[i].syn)) #length.(rn_data[i].nsyn)
            split_idxs = cumsum(vcat(1, raw_lengths[1:end-1]))
            on_t_list = []
            off_t_list = []
            for (j, start_idx) in enumerate(split_idxs)
                end_idx = start_idx + raw_lengths[j] - 1
                if end_idx > length(yt)
                    #@warn "Segment end index $end_idx exceeds trace length $(length(yt)). Skipping."
                    continue
                end
                seg = yt[start_idx:end_idx]
                push!(on_t_list, on_off_time(seg, o, 1.0))
                push!(off_t_list, on_off_time(seg, o, 0.0))
            end
            on_t = vcat(on_t_list...)
            off_t = vcat(off_t_list...)
            push!(rn_data_dl, (p=k, binar_trace=yt, binar_time=[on_t, off_t]))
        end
        rn_met_dl = compare_metric(rn_data_true, rn_data_dl; compute_acc=true)
        @save "$(obs_name)$(rn)_dl$(model_name)_compare_seed_$(seed).jld2" rn_met_dl
        return rn_met_dl, rn_data_dl
    end
end
