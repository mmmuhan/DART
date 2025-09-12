using Pkg
Pkg.activate(normpath(@__DIR__, ".."))

using Random, JLD2, BSON, Statistics, Sobol

include(normpath(joinpath(@__DIR__, "..", "utils", "reactions.jl")))
include(normpath(joinpath(@__DIR__, "..", "utils", "ssa.jl")))
include(normpath(joinpath(@__DIR__, "..", "utils", "utils.jl")))

include(joinpath(@__DIR__,  "..", "dl_train", "train33.jl"))
include(joinpath(@__DIR__,  "..", "dl_train", "test_model.jl"))

refb_mtest = [];

lb = [2e-3,2e-3,1e-3,1.] 
ub = [20.,20.,10.,50.];

Sobol_seq = SobolSeq(lb,ub);

@time begin
while length(refb_mtest)<3300
    k = Sobol.next!(Sobol_seq)
    ont = compute_ont(k,3,"ref"); offt =compute_offt(k,3,"ref");
    bs = k[4]*ont
    ratio = 10^(-2 + (1 - (-2)) * rand()); 
    tau = ratio * offt; 
    obst = 0.1*min(ont,offt)
    
    nums = ceil(Int,min((offt+ont) * 1100, 30000)/30)
    
    t0, L1, L, I0 = 1.0, 0., 1., 1.0
    
    if 5<= offt/ont <= 20  && 1 <= bs <= 150 && 0.1<= tau < 10 && ont >= obst
        res_test = generate_synthetic(construct_prob_delayref,vcat(k,[0,tau,1.,30]),L1,L,I0,nums,obst,false,0.05,3)
        even_syn, even_trace = res_test.syn, res_test.true_trace
        if mean(vcat(even_syn...))>=5 && length(findall(x->x==0,sum.(even_syn)))<=1 #at most one trace that doesn't have any pulse
            on_time, off_time = on_off_time.(even_trace,obst,1.0), on_off_time.(even_trace,obst,0.0);
            push!(refb_mtest,(p = vcat(k,tau,obst), syn=even_syn, trace = even_trace, even_time = [on_time, off_time]))
        end
    end
        
end
end

@save "synthetic_data/ideal_data_res/refb_mtest.jld2" refb_mtest;

rn_data = copy(refb_mtest[1:600]);
rn_params = [d.p for d in rn_data];

rn_data_true = copy(refb_mtest[1:600]);

metb_ref = dl_metrics(58, "synthetic_data/ideal_data_res/m", "synthetic_data/ideal_data_res/m", "b33", "ref", false); 