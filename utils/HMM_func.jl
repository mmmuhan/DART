using Distributions,HMMBase,StatsBase
# this was taken from: https://github.com/maxmouchet/HMMBase.jl/blob/master/examples/fit_map.jl
# avoid "isprobvec(hmm.a) must hold. Got\nhmm.a => [NaN, NaN]" by putting a prior on the variance
# An MLE approach for the observations distributions parameters may fail with a singularity (variance = 0) if an outlier
# becomes the only observation associated to some state:
# even with this the same issue can still happen as the signal data can still cause the problem that there is only one observation assigened to the state
import ConjugatePriors: InverseGamma, NormalKnownMu, posterior_canon
import StatsBase: Weights

function fit_map(::Type{<:Normal}, observations, responsibilities)
    μ = mean(observations, Weights(responsibilities))

    ss = suffstats(NormalKnownMu(μ), observations, responsibilities)
    prior = InverseGamma(2, 1)
    posterior = posterior_canon(prior, ss)
    σ2 = mode(posterior)

    Normal(μ, sqrt(σ2))
end

function HMM_traces(states::Int64,data::Vector{Float64})
    hmm = HMM(randtransmat(states), [Normal(0,1) for i in 1:states]); # initial HMM
    hmm_revised, hist = fit_mle(hmm, data, estimator = fit_map, display = :none, init = :kmeans, maxiter = 1e5, tol = 1e-5); # Estimate the HMM parameters using the EM (Baum-Welch)
    traces=viterbi(hmm_revised,data); # get idealized traces using vierbi algorithm

    # assign revised ranking to states number based on mean value of each group (by states)
    traces_revised = copy(traces)
    indices = [findall(x->x==i,traces_revised) for i in 1:states]
    
    # Calculate means of data for each group
    mean_indices = [(mean(data[indices[i]]), i) for i in 1:states]
    sorted_means_indices = sort(mean_indices, rev=true)

    # Assign values based on mean ranking
    for k in 1:states
        for (mean_val, group_val) in sorted_means_indices
            if mean_val == sorted_means_indices[k][1]
                traces_revised[indices[group_val]] .= states - k + 1
            end
        end
    end
    return traces_revised,hist     
end