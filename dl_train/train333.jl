function run_train(seed::Int,ref_data,folder_name::String,model_name::String,noise::Bool)
    # Set random seed
    Random.seed!(seed)

    # Data Preparation
    lengths = [length(vcat(ref_data[i].syn...)) for i in 1:3300] # change!
    ids = findall(x -> x > 5000, lengths)

    x = [vcat((noise ? ref_data[i].nsyn : ref_data[i].syn)...)[1:5000] for i in ids]  
    y = [vcat(ref_data[i].trace...)[1:5000] for i in ids]

    x_norm = min_max.(x)
    x1 = hcat(x_norm...)
    y1 = hcat(y...)

    # Stack into tensors for Flux
    x_tensor = reshape(x1, 1, size(x1, 1), size(x1, 2))
    x_tensor = Float32.(x_tensor)
    y_tensor = reshape(y1, 1, size(y1, 1), size(y1, 2))
    y_tensor = Float32.(y_tensor)

    x_tensor = permutedims(x_tensor, (2, 1, 3))  # Convert to input shape for CNN
    x_tensor = cu(x_tensor)  # Transfer input tensor to GPU
    y_tensor = cu(y_tensor)  # Transfer labels to GPU

    # Split data into training and validation sets
    x_train, y_train = x_tensor[:, :, 601:3000], y_tensor[:, :, 601:3000] #change!
    x_val, y_val = x_tensor[:, :, 3001:end], y_tensor[:, :, 3001:end]

    # Data loaders
    train_loader = Flux.DataLoader((x_train, y_train), batchsize=20, shuffle=true)
    val_loader = Flux.DataLoader((x_val, y_val), batchsize=20, shuffle=true)

    # Define the model
    model = Chain(
        Conv((3,), 1 => 16, pad=(1,), relu),
        MaxPool((3,), pad=SamePad(), stride=(1,)),
        Conv((3,), 16 => 32, pad=(1,), relu),
        MaxPool((3,), pad=SamePad(), stride=(1,)),
        Conv((3,), 32 => 64, pad=(1,), relu),
        MaxPool((3,), pad=SamePad(), stride=(1,)),
        x -> permutedims(x, (2, 1, 3)),
        x -> reshape(x, size(x, 1), size(x, 2), :),
        LSTM(64, 64),
        Dense(64, 1),
        σ
    ) |> gpu

    # Optimizer with initial learning rate
    η = 0.01
    optim = Flux.setup(Flux.Adam(η), model)

    # Hyperparameters for learning rate decay
    lr_halving_limit = 6; lr_halve_count = 0;
    thresh = 0.005  # 0.5% improvement threshold

    tls, vls = Float64[], Float64[]
    @time begin
        for epoch in 1:200
            train_loss = 0.0
            for (x, y) in train_loader
                x, y = gpu(x), gpu(y)
                loss1, grads1 = Flux.withgradient(model) do m
                    y_hat = m(x)
                    Flux.binarycrossentropy(y_hat, y)
                end
                Flux.update!(optim, model, grads1[1])
                train_loss += loss1
            end

            val_loss = 0.0
            for (x, y) in val_loader
                x, y = gpu(x), gpu(y)
                val_loss += Flux.binarycrossentropy(model(x), y)
            end

            train_loss /= length(train_loader)
            val_loss /= length(val_loader)

            push!(tls, train_loss)
            push!(vls, val_loss)

            println("Seed $seed - Epoch $epoch: Train Loss = $train_loss, Val Loss = $val_loss")

            # Check average improvement for the last 25 epochs (if enough data exists)
            if epoch > 25
                avg_prev_25 = mean(vls[epoch-25:epoch-1])
                improvement = (avg_prev_25 - val_loss) / avg_prev_25
                if improvement < thresh
                    η /= 2
                    optim = Flux.setup(Flux.Adam(η), model)
                    lr_halve_count += 1
                    println("Validation loss improvement below threshold. Halving learning rate to $η.")
                end
            end

            # Stop if learning rate is halved too many times
            if lr_halve_count >= lr_halving_limit
                println("Stopping training: Learning rate halved $lr_halve_count times.")
                break
            end
        end
    end

    # Save the trained model
    model_filename = "$(folder_name)trained_model$(model_name)_seed_$(seed).bson"
    model = cpu(model)
    BSON.@save model_filename model
    println("Model saved as $model_filename")

    return model, tls, vls
end;
