using StatsBase, LinearAlgebra, Distributions, DelaySSAToolkit


function generate_synthetic(construct_delay::Function, params, L1, L, I, numofrun, inter, noise::Bool, ex_cv, ids)
    delay_constructor = construct_delay

    # Generate delay problem and solve
    djprob = delay_constructor(params)
    djsol = solve(djprob, SSAStepper(); saveat = inter)
    ens_prob = EnsembleProblem(djprob)
    ens = solve(ens_prob, SSAStepper(), EnsembleThreads(), trajectories = numofrun, saveat = inter) #EnsembleThreads()

    # Generate the binary vectors for each id
    binary_vector = [[map(x -> x[id], ens[i].u) for i in eachindex(ens)] for id in ids]

    # Apply element-wise maximum across all binary vectors
    final_binary_vector = [reduce((a, b) -> max.(a, b), (binary_vector[j][i] for j in eachindex(ids)))
        for i in eachindex(ens)]

    # Process filtered solutions
    filter_djsol = [reduce(vcat, ens[i].channel) for i in eachindex(ens)]
    jump_dres = [[signal_function.(filter_djsol[i][j]; τ = params[end-2], L1 = L1, L = L, I = I) 
                  for j in 1:length(filter_djsol[i])] for i in 1:numofrun]
    nascent_syn = [sum_with_non.(jump_dres[i]) for i in 1:numofrun]

    noise_syn = Vector{Vector{Float64}}()
    for vec in nascent_syn
        noise_syn1 = Vector{Float64}()
        for val in vec
            p1, p2 = lognorm_param(val + 1e-4, ex_cv * (val + 1e-4))
            push!(noise_syn1, rand(LogNormal(p1, p2)))
        end
        push!(noise_syn, noise_syn1)
    end

    # Return based on noise
    if noise
        return (syn = nascent_syn, nsyn = noise_syn, true_trace = convert.(Vector{Float64}, final_binary_vector))
    else
        return (syn = nascent_syn, ens = ens, true_trace = convert.(Vector{Float64}, final_binary_vector))# remove ens
    end
end


# given the mean, std, find the parameters u,d for lognorm, transform the synthetic data to noise data
function lognorm_param(mean_val,std_val)
    u = log(mean_val^2/(sqrt(mean_val^2+std_val^2)))
    d = sqrt(log(1+std_val^2/mean_val^2))

    return u,d
end;
