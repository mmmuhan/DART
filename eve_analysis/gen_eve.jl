using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))

using Random, StatsBase, Statistics, Sobol, JLD2, Printf
include(normpath(joinpath(@__DIR__, "..", "dl_train", "train33_data.jl")))
include(normpath(joinpath(@__DIR__, "..", "dl_train", "test_model.jl")))
include(normpath(joinpath(@__DIR__, "..", "utils", "utils.jl")))
include(normpath(joinpath(@__DIR__, "..", "utils", "reactions.jl")))
include(normpath(joinpath(@__DIR__, "..", "utils", "ssa.jl")))

gen_ntest_eve = [];

lb = [3e-3,3e-3,3e-3,3e-3,3e-3,3e-3,3e-3,3e-3,3e-3,3e-3,1.,1.,1.] 
ub = [30.,30.,30.,30.,30.,30.,30.,30.,30.,30.,50.,50.,50.];

Sobol_seq = SobolSeq(lb,ub);

@time begin
while length(gen_ntest_eve)<2700
    k = Sobol.next!(Sobol_seq)
    ont = compute_ont(k,10,"gen"); offt =compute_offt(k,10,"gen");
    bs = (k[10]*k[12]*k[7] + k[10]*k[11]*k[8] + k[13]*k[7]*k[9])/(k[10]*k[6]*k[8])
    tau = 2.33 
    obst = 0.33 
    
    nums = 40 
    
    t0, L1, L, I0 = 1.0, 1500, 1500+5165, 1.0
    
    if 1<= offt/ont <= 15  && 1 <= bs <= 150 && ont >= obst
        res_test = generate_synthetic(construct_prob_delaygen,vcat(k,[0,tau,1.,50]),L1,L,I0,nums,obst,true,0.05,[4,5,6])
        even_syn, even_nsyn, even_trace = res_test.syn, res_test.nsyn, res_test.true_trace
        if length(findall(x->x==0,sum.(even_syn)))<=1 #at most one trace that doesn't have any pulse
            on_time, off_time = on_off_time.(even_trace,obst,1.0), on_off_time.(even_trace,obst,0.0);
            push!(gen_ntest_eve,(p = vcat(k,tau,obst), syn=even_syn, nsyn = even_nsyn, trace = even_trace, even_time = [on_time, off_time]))
        end
    end
        
end
end

save_path = normpath(joinpath(@__DIR__, "..", "eve_analysis", "gen_ntest_eve.jld2"))
mkpath(dirname(save_path))
@save save_path gen_ntest_eve
@info @sprintf("Saved %d samples to %s", length(gen_ntest_eve), save_path)


model, train_losses, val_losses = run_train(1,gen_ntest_eve,"eve_analysis/n","bnb",true);