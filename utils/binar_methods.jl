using SavitzkyGolay,Clustering,DSP
include("HMM_func.jl")

function synthetic_traces_hmm(syn,inter,thresh,states::Int64)
    
    traces,_ = HMM_traces(states,syn);

    traces_revised = binarize(traces,thresh)
    
    on_times = on_off_time(traces_revised,inter,1.0); off_times = on_off_time(traces_revised,inter,0.0);
    on_times = set_default_empty!(on_times, 0); off_times = set_default_empty!(off_times, 0)

    return traces_revised,convert(Vector{Float64},on_times),convert(Vector{Float64},off_times)

end;

function synthetic_traces_ma(syn,inter)
    
    kernel = ones(3) / 3;
    smooth_syn = filt(kernel, 1, syn)
    #smooth_syn = conv(syn, kernel);

    thresh = maximum(smooth_syn)*0.1
    traces = binarize(smooth_syn,thresh)
    #thresh = (maximum(smooth_syn)*0.1-minimum(smooth_syn))/(maximum(smooth_syn)-minimum(smooth_syn))

    on_times = on_off_time(traces,inter,1.0); off_times = on_off_time(traces,inter,0.0);
    on_times = set_default_empty!(on_times, 0); off_times = set_default_empty!(off_times, 0);
    
    return traces,convert(Vector{Float64},on_times),convert(Vector{Float64},off_times)
end;

function synthetic_traces_sg(syn,inter)
    
    sg = savitzky_golay(syn, 11, 3;); 
    smooth_syn = sg.y;

    thresh = maximum(smooth_syn)*0.1
    traces = binarize(smooth_syn,thresh)

    on_times = on_off_time(binarize(smooth_syn,thresh),inter,1.0); off_times = on_off_time(binarize(smooth_syn,thresh),inter,0.0);
    on_times = set_default_empty!(on_times, 0); off_times = set_default_empty!(off_times, 0);
    
    return traces,convert(Vector{Float64},on_times),convert(Vector{Float64},off_times)
end;