# =============================================================================
# Works for any number of ON states:
#   - State classification: binary (OFF vs ON), all ON states collapsed to 1
#   - Rate prediction: k_eff_on = Σ(πᵢkᵢ) / Σ(πᵢ) for all ON states
#
# Key parameters:
#   - n_on: number of ON states (determines loading rate indices)
#   - rate_idx: which indices in p hold the loading rates
#     default: p[end-4:end-2] for 3 ON case
#   - Elongation time: p[end-1]
#   - Trace labels: 0 = OFF, 1..n_on = ON states
# =============================================================================

ENV["JULIA_CUDA_USE_BINARYBUILDER"] = "true"

using Random, BSON, Statistics
using Flux
using Flux: onehotbatch, logitcrossentropy
using CUDA, ProgressMeter

# -----------------------------
# Helpers
# -----------------------------
to_device(x) = CUDA.functional() ? cu(x) : x

function decode_class_labels(y)
    idx = dropdims(Array(argmax(cpu(y), dims=1)); dims=1)
    return first.(Tuple.(idx)) .- 1
end

state_accuracy(y_hat, y_true) = mean(decode_class_labels(y_hat) .== decode_class_labels(y_true))

function huber_loss(yhat, y; δ=1.0f0)
    ad = abs.(yhat .- y)
    quad = min.(ad, δ)
    return mean(0.5f0 .* quad.^2 .+ δ .* (ad .- quad))
end

function rate_mrelerr(r_hat, r_true, rate_scale; eps=1f-8)
    rh, rt = Array(cpu(r_hat .* rate_scale)), Array(cpu(r_true .* rate_scale))
    return median(abs.((rh .- rt) ./ (rt .+ eps)))
end

function global_mean_pool(h)
    dropdims(mean(h, dims=2); dims=2)
end

# -----------------------------
# Compute k_eff_on for any number of ON states
# trace: vector with labels 0 (OFF), 1..n_on (ON states)
# rates: vector of loading rates [k1, k2, ..., k_n]
# -----------------------------
function compute_keff_on(trace, rates)
    n_on = length(rates)
    num = 0.0
    denom = 0.0
    for i in 1:n_on
        πi = mean(trace .== i)
        num += πi * rates[i]
        denom += πi
    end
    denom < 1e-8 && return 0f0
    return Float32(num / denom)
end

# -----------------------------
# Forward
# -----------------------------
function forward(model_parts, x, elong)
    h     = model_parts.trunk(x)
    y_hat = model_parts.state_head(h)       # (2, time, batch)

    h_pool = global_mean_pool(h)            # (64, batch)
    h_cat  = vcat(h_pool, elong)            # (65, batch)
    r_hat  = model_parts.rate_head(h_cat)   # (1, batch)

    return y_hat, r_hat
end

# -----------------------------
# Collect rate predictions
# -----------------------------
function collect_rate_predictions(model_parts, data_loader, rate_scale)
    true_r = Float32[]; pred_r = Float32[]

    for (x, _, r_true, elong) in data_loader
        x, elong = to_device(x), to_device(elong)
        _, r_hat = forward(model_parts, x, elong)

        append!(true_r, vec(Array(cpu(r_true .* rate_scale))))
        append!(pred_r, vec(Array(cpu(r_hat  .* rate_scale))))
    end
    return (true_r=true_r, pred_r=pred_r)
end

# -----------------------------
# Metric accumulator
# -----------------------------
mutable struct Metrics
    data::Dict{Symbol, Vector{Float64}}
end
Metrics(keys::Vector{Symbol}) = Metrics(Dict(k => Float64[] for k in keys))
Base.push!(m::Metrics, k::Symbol, v) = push!(m.data[k], v)
Base.getindex(m::Metrics, k::Symbol) = m.data[k]

# -----------------------------
# Main training function
# -----------------------------
function run_train_rates(
    seed::Int, ref_data, folder_name::String, model_name::String, noise::Bool;
    λ_rate::Float32 = 0.1f0, batchsize::Int = 20, maxlen::Int = 5000,
    nepochs::Int = 200,
    rate_idx::UnitRange{Int} = 0:-1    # default: auto-detect as p[end-4:end-2]
)
    Random.seed!(seed)

    # --- Data preparation ---
    lengths = [length(vcat(ref_data[i].syn...)) for i in 1:3300]
    ids = findall(>(maxlen), lengths)

    # Auto-detect rate indices if not provided
    if isempty(rate_idx)
        rate_idx = (length(ref_data[ids[1]].p) - 4):(length(ref_data[ids[1]].p) - 2)
    end
    n_on = length(rate_idx)
    println("Detected $n_on ON states, loading rates at p[$rate_idx]")

    x = [Float32.(vcat((noise ? ref_data[i].nsyn : ref_data[i].syn)...)[1:maxlen]) for i in ids]

    # Binary labels: collapse all ON states to 1
    y = [Int.(vcat(ref_data[i].trace...)[1:maxlen] .> 0) for i in ids]

    # k_eff_on: weighted average loading rate across all ON states
    keff = [compute_keff_on(
                vcat(ref_data[i].trace...),
                Float64.(ref_data[i].p[rate_idx])
            ) for i in ids]

    # Elongation time: p[end-1]
    elong = [Float32(ref_data[i].p[end-1]) for i in ids]

    x_norm = min_max.(x)
    x1    = hcat(x_norm...)
    y1    = hcat(y...)
    rmat  = reshape(Float32.(keff), 1, :)
    emat  = reshape(Float32.(elong), 1, :)

    rate_scale = 1f0

    # Tensors (binary: 2-class one-hot)
    x_tensor = to_device(permutedims(reshape(Float32.(x1), 1, size(x1)...), (2, 1, 3)))
    y_oh     = [Flux.onehotbatch(y1[:, i] .+ 1, 1:2) for i in 1:size(y1, 2)]
    y_tensor = to_device(Float32.(cat(Array.(y_oh)..., dims=3)))
    r_tensor = to_device(Float32.(rmat))
    e_tensor = to_device(Float32.(emat))

    # Train/val split
    train_idx, val_idx = 601:3000, 3001:size(x_tensor, 3)
    slice(t, idx) = selectdim(t, ndims(t), idx)

    train_loader = Flux.DataLoader(
        map(t -> slice(t, train_idx), (x_tensor, y_tensor, r_tensor, e_tensor)),
        batchsize=batchsize, shuffle=true)
    val_loader = Flux.DataLoader(
        map(t -> slice(t, val_idx), (x_tensor, y_tensor, r_tensor, e_tensor)),
        batchsize=batchsize, shuffle=false)

    # --- Model ---
    trunk = Chain(
        Conv((3,), 1 => 32, pad=(1,), relu),
        MaxPool((3,), pad=SamePad(), stride=(1,)),
        Conv((3,), 32 => 64, pad=(1,), relu),
        MaxPool((3,), pad=SamePad(), stride=(1,)),
        x -> permutedims(x, (2, 1, 3)),
        x -> reshape(x, size(x, 1), size(x, 2), :),
        LSTM(64, 64)
    ) |> to_device

    state_head = Dense(64, 2) |> to_device    # binary: OFF vs ON

    # Rate head: 64 (mean pooled) + 1 (elongation time) = 65 → 1 scalar
    rate_head = Chain(
        Dense(65, 128, relu),
        Dense(128, 64, relu),
        Dense(64, 1),
        softplus
    ) |> to_device

    model_parts = (trunk=trunk, state_head=state_head, rate_head=rate_head)

    # --- Loss ---
    function total_loss(m, x, y, r_true, elong)
        y_hat, r_hat = forward(m, x, elong)
        l_cls  = logitcrossentropy(y_hat, y)
        l_rate = huber_loss(r_hat, r_true)
        return l_cls + λ_rate * l_rate, l_cls, l_rate, y_hat, r_hat
    end

    # --- Optimizer & scheduling ---
    η = 0.01
    optim = Flux.setup(Flux.Adam(η), model_parts)
    lr_halve_count = 0

    metric_keys = [
        :train_total, :val_total,
        :train_acc, :val_acc,
        :train_rate, :val_rate,
        :train_mrel, :val_mrel
    ]
    met = Metrics(metric_keys)

    # --- Training loop ---
    @time for epoch in 1:nepochs

        # -- Train --
        accum = Dict(k => 0.0 for k in [:loss, :rate, :acc, :mrel])
        for (x, y, r_true, elong) in train_loader
            x, y, r_true, elong = to_device.((x, y, r_true, elong))

            local l_rate, y_hat, r_hat
            loss_val, grads = Flux.withgradient(model_parts) do m
                l_tot, _, l_r, yh, rh = total_loss(m, x, y, r_true, elong)
                l_rate = l_r
                y_hat = yh
                r_hat = rh
                l_tot
            end
            Flux.update!(optim, model_parts, grads[1])

            accum[:loss] += loss_val
            accum[:rate] += l_rate
            accum[:acc]  += state_accuracy(y_hat, y)
            accum[:mrel] += rate_mrelerr(r_hat, r_true, rate_scale)
        end
        for k in keys(accum); accum[k] /= length(train_loader); end

        # -- Val --
        vaccum = Dict(k => 0.0 for k in [:tot, :rate, :acc, :mrel])
        for (x, y, r_true, elong) in val_loader
            x, y, r_true, elong = to_device.((x, y, r_true, elong))
            y_hat, r_hat = forward(model_parts, x, elong)

            l_cls  = logitcrossentropy(y_hat, y)
            l_rate = huber_loss(r_hat, r_true)

            vaccum[:tot]  += l_cls + λ_rate * l_rate
            vaccum[:rate] += l_rate
            vaccum[:acc]  += state_accuracy(y_hat, y)
            vaccum[:mrel] += rate_mrelerr(r_hat, r_true, rate_scale)
        end
        for k in keys(vaccum); vaccum[k] /= length(val_loader); end

        # -- Record metrics --
        push!(met, :train_total, accum[:loss])
        push!(met, :val_total,   vaccum[:tot])
        push!(met, :train_acc,   accum[:acc])
        push!(met, :val_acc,     vaccum[:acc])
        push!(met, :train_rate,  accum[:rate])
        push!(met, :val_rate,    vaccum[:rate])
        push!(met, :train_mrel,  accum[:mrel])
        push!(met, :val_mrel,    vaccum[:mrel])

        println("Seed $seed  Epoch $epoch")
        println("  Train: tot=$(accum[:loss])  rate=$(accum[:rate])  acc=$(accum[:acc])  mrel=$(accum[:mrel])")
        println("  Val  : tot=$(vaccum[:tot])  rate=$(vaccum[:rate])  acc=$(vaccum[:acc])  mrel=$(vaccum[:mrel])")

        # -- LR schedule --
        if epoch > 25
            vls = met[:val_total]
            avg_prev = mean(vls[epoch-25:epoch-1])
            if (avg_prev - vaccum[:tot]) / max(avg_prev, 1e-8) < 0.005
                η /= 2
                optim = Flux.setup(Flux.Adam(η), model_parts)
                lr_halve_count += 1
                println("  Halving LR → $η  ($lr_halve_count/6)")
            end
        end
        lr_halve_count >= 6 && (println("Early stop: LR halved 6 times."); break)
    end

    # --- Save ---
    preds = collect_rate_predictions(model_parts, val_loader, rate_scale)

    mkpath(normpath(dirname(folder_name)))
    model_filename = "$(folder_name)trained_model$(model_name)_seed_$(seed).bson"
    model_parts_cpu = map(cpu, model_parts)

    BSON.@save model_filename model_parts_cpu met preds λ_rate rate_idx
    println("Saved: $(abspath(model_filename))")

    return model_parts_cpu, met, preds
end