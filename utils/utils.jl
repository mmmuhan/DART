using StatsBase, PyPlot, Statistics
const np = PyPlot.pyimport("numpy"); # to obtain probability of histogram through np.histogram

function sum_with_non(vec::Vector{T}) where {T}
    length(vec) == 0 ? zero(T) : sum(vec)
end


function signal_function(x; τ=τ, L1=L1, L=L, I=I)
    return min(x/τ*L/L1*I,I) # x/tau, the normalized position of RNAP on the gene (in terms of time as unit)
end

# min_max normalization, same as value/maximum normalizaion if the minimum is 0.
function min_max(vector)
    return (vector .- minimum(vector))./(maximum(vector) .- minimum(vector))
end;

function set_default_empty!(variable::Vector, default_value)
    if isempty(variable)
        variable = [default_value]
    end
    return variable
end

function CV2(data)
    return var(data)/(mean(data))^2
end;

# binarize data
function binarize(vector,threshold)
    vector_new = zeros(length(vector))
    for i in 1:length(vector)
        if vector[i]>=threshold
            vector_new[i]=1
        end           
    end
    return vector_new
end

# find on/off time distribution
# on time: k=1.0, off time: k=0.0
function on_off_time(binarized_data::Vector{Float64},inter::Float64,k::Float64)
    on_intervals = Vector{Float64}()
    start_time = 0
    for i in 1:length(binarized_data)
        if binarized_data[i]==k
            if start_time == 0
                start_time = i
            end
        else
            if start_time != 0
                push!(on_intervals, i - start_time)
                start_time = 0
            end
        end
    end
    if start_time != 0
        push!(on_intervals, length(binarized_data) - start_time + 1)
    end
    
    if isempty(on_intervals)
        return [0.0]
    else
        return on_intervals .* inter
    end
end


function quantile_idx(arr, q)
    sorted_arr = sort(arr)
    n = length(arr)
    quantile_pos = clamp(round(Int, q * (n - 1) + 1), 1, n)
    quantile_val = sorted_arr[quantile_pos]
    return findall(x -> x == quantile_val, arr)
end;

function median_idx(arr)
    sorted_arr = sort(arr)
    n = length(arr)
    if isodd(n)
        med = sorted_arr[div(n+1, 2)]
        return findfirst(x -> x == med, arr)
    else
        med1, med2 = sorted_arr[n ÷ 2], sorted_arr[n ÷ 2 + 1]
        return findall(x -> x == med1 || x == med2, arr)
    end
end;

function FD_bins(data) #freedman_diaconis
    n = length(data)
    q25, q75 = quantile(data, [0.25, 0.75])  # Compute the 25th and 75th percentiles
    iqr = q75 - q25  # Interquartile Range
    bin_width = 2 * iqr / n^(1/3)  # Freedman–Diaconis rule

    if bin_width == 0  # Prevent division by zero in case of constant data
        return 1
    end

    num_bins = ceil(Int, (maximum(data) - minimum(data)) / bin_width)  # Compute number of bins
    return max(num_bins, 1)  # Ensure at least 1 bin
end;

function plot_traces(data,traces)
    fig, ax1 = plt.subplots(figsize=(15,4))
    y1 = data; y2 =traces;
   
    ax1.plot(y1, label="Line 1")
    ax1.set_xlabel("time")
    ax1.set_ylabel("intensity")
    ax1.tick_params(axis="y")
    
    ax2 = ax1.twinx()
    ax2.plot(y2, color="orange", label="Line 2")
    ax2.set_ylabel("states")
    ax2.tick_params(axis="y")
    return fig
end

function scatter_plot(ax, rn_met, x_data, y_data, label_pos; axis_log=true)
    
    # Compute correlation
    
    # compute mean accuracy
    rn_acc = [m.acc for m in rn_met]   
    macc = round(mean(rn_acc), digits=3)
    
    ax.scatter(x_data, y_data, s=8, c="#6baed6", alpha=0.6, label=L"\langle \alpha \rangle=" * string(macc))
    
    ax.legend(borderpad=0.3, fontsize=10, loc="lower right")

    x_min, x_max = minimum(vcat(x_data,y_data)), maximum(vcat(x_data,y_data))
    ax.plot(x_min*0.9:0.1:x_max*1.1, x_min*0.9:0.1:x_max*1.1, c="#2c7bb6", linestyle="--", linewidth=1.5)
    
    ax.set_xlim(x_min * 0.8, x_max * 1.15)
    ax.set_ylim(x_min * 0.8, x_max * 1.15)
    
    if axis_log
        ax.set_xscale("log")
        ax.set_yscale("log")
    end
end

function plot_rn(rn_met)
    cm = 1/2.54;
    fig = plt.figure(figsize=(33*cm, 15*cm))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
    
    gs=plt.GridSpec(7, 11, wspace=2, hspace=0)
    ax1 = subplot(gs.new_subplotspec((0,0), rowspan=3,colspan=3)) # mean on
    ax2 = subplot(gs.new_subplotspec((0,3), rowspan=3,colspan=3)) # mean off
    ax3 = subplot(gs.new_subplotspec((0,6), rowspan=3,colspan=3)) # hist

    ax5 = subplot(gs.new_subplotspec((4,1), rowspan=3,colspan=3)) # P-P on
    ax6 = subplot(gs.new_subplotspec((4,5), rowspan=3,colspan=3)) # P-P off
    
    box = ax5.get_position()
    ax5.set_position([box.x0 + 0.03, box.y0, box.width, box.height])

    box = ax6.get_position()
    ax6.set_position([box.x0 - 0.03, box.y0, box.width, box.height])

    accs = [m.acc for m in rn_met]
    
    scatter_plot(ax1, rn_met, [m.truet_mean[1] for m in rn_met], [m.binart_mean[1] for m in rn_met], (0.6, 0.1))
    scatter_plot(ax2, rn_met, [m.truet_mean[2] for m in rn_met], [m.binart_mean[2] for m in rn_met], (0.6, 0.1))
    
    fs1=11
    ax1.set_xlabel("True mean on-time", fontsize=fs1); ax1.set_ylabel("DART mean on-time", fontsize=fs1);
    ax2.set_xlabel("True mean off-time", fontsize=fs1); ax2.set_ylabel("DART mean off-time", fontsize=fs1);
    
    num_bins = round(Int, sqrt(length(accs))) 
    ax3.set_axisbelow(true)
    
    counts, bins = np.histogram(accs, bins=15)
    ncounts = counts ./ sum(counts)
    ax3.bar(bins[1:end-1], ncounts, width=diff(bins), color="#74add1", edgecolor="white", lw=0.3);
    
    y_ticks = 0:0.02:maximum(ncounts)  # Define the range of y-axis ticks
    ax3.set_yticks(y_ticks)  
    ax3.set_xlabel(L"Accuracy $\alpha$", fontsize=fs1); ax3.set_ylabel("Probability",fontsize=fs1);
    
    axs = [ax5, ax6]; 
    colors = ["#fc8d59","#4575b4","#fee090"]; labels = ["Q1","Q2","Q3"]; inset_labels = ["on", "off"]

    for i in 1:2
        re = [abs.((m.binart_mean[i] - m.truet_mean[i]) / m.truet_mean[i]) for m in rn_met]
        data_id1 = quantile_idx(re, 0.25)[1]
        data_id2 = quantile_idx(re, 0.5)[1]
        data_id3 = quantile_idx(re, 0.75)[1]
        data_ids = [data_id1, data_id2, data_id3]
        
        for (q_idx, data_id) in enumerate(data_ids)
            truet = vcat(rn_data[data_id].even_time[i]...)
            binart = rn_met[data_id].binart[i]
            len_min = min(length(truet), length(binart))
        
            sorted_true = sort(truet[1:len_min])
            sorted_binar = sort(binart[1:len_min])
        
            ecdf1, ecdf2 = ecdf(truet), ecdf(binart)
            cdf1_values, cdf2_values = ecdf1(sorted_true), ecdf2(sorted_binar)
            
            axs[i].scatter(cdf1_values, cdf2_values,s=10,c=colors[q_idx],alpha=0.3,edgecolors="none",label=labels[q_idx])
            
            Line2D = PyPlot.matplotlib.lines.Line2D # Set custom legend
            legend_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8) for c in colors]
            
            axs[i].legend(legend_handles, ["Q1", "Q2", "Q3"], scatterpoints=1, borderpad=0.3, fontsize=9)
        
            axs[i].plot([0,1], [0,1], c="#cb181d",lw = 1.0)
            
            q1, q2, q3 = quantile(re, [0.25, 0.5, 0.75])  # Compute quantiles

            # === Manually Add Inset ===
            inset_ax = axs[i].inset_axes([0.6, 0.1, 0.38, 0.38], transform=axs[i].transAxes) 

            # Add histogram to inset
            inset_ax.hist(re, bins=15, color="#bdbdbd", alpha=0.6, edgecolor="white", lw = 0.3, density=false)
            inset_ax.axvline(q1, color=colors[1], linestyle="--", lw=1.2, label="Q1")
            inset_ax.axvline(q2, color=colors[2], linestyle="--", lw=1.2, label="Q2")
            inset_ax.axvline(q3, color=colors[3], linestyle="--", lw=1.2, label="Q3")

            # Remove ticks and labels from inset
            inset_ax.set_xticks([]); inset_ax.set_yticks([])
            #inset_ax.tick_params(axis="both", which="major", labelsize=7)
            label1 = inset_labels[i]
            inset_ax.set_xlabel("RE of mean $label1", fontsize=7)
            # Change the border color and thickness of the inset
            for spine in keys(inset_ax.spines)
                inset_ax.spines[spine].set_edgecolor("gray")  # Change to desired color
                inset_ax.spines[spine].set_linewidth(1.2)      # Adjust thickness
            end

        end
    end
    
    ax5.set_xlabel("CDF (true on-time)", fontsize=fs1); ax6.set_xlabel("CDF (true off-time)", fontsize=fs1)
    ax5.set_ylabel("CDF (DART on-time)", fontsize=fs1); ax6.set_ylabel("CDF (DART off-time)", fontsize=fs1)
    
    wd = 1.2
    for ax in [ax1,ax2,ax3,ax5,ax6]
        ax.tick_params(axis="both", labelsize=11, width=wd)
        for spine in values(ax.spines)
            spine.set_linewidth(wd)
        end
    end
    
    return fig
    
end;


function plot_time(rn_met) # plot without accuracy
    
    fig, axs = plt.subplots(2,2,figsize=(7,6))
    
    scatter_plot(axs[1,1], [m.truet_mean[1] for m in rn_met], [m.binart_mean[1] for m in rn_met], (0.6, 0.1))
    scatter_plot(axs[1,2], [m.truet_mean[2] for m in rn_met], [m.binart_mean[2] for m in rn_met], (0.6, 0.1))
    
    axs[1,1].set_xlabel("true mean on time"); axs[1,1].set_ylabel("DL mean on time");
    axs[1,2].set_xlabel("true mean off time"); axs[1,2].set_ylabel("DL mean off time");
    
    colors = ["#fc8d59","#4575b4","#fee090"]; labels = ["Q1","Q2","Q3"]; inset_labels = ["on", "off"]

    for i in 1:2
        re = [abs.((m.binart_mean[i] - m.truet_mean[i]) / m.truet_mean[i]) for m in rn_met]
        data_id1 = quantile_idx(re, 0.25)[1]
        data_id2 = quantile_idx(re, 0.5)[1]
        data_id3 = quantile_idx(re, 0.75)[1]
        data_ids = [data_id1, data_id2, data_id3]
        
        for (q_idx, data_id) in enumerate(data_ids)
            truet = vcat(rn_data[data_id].even_time[i]...)
            binart = rn_met[data_id].binart[i]
            len_min = min(length(truet), length(binart))
        
            sorted_true = sort(truet[1:len_min])
            sorted_binar = sort(binart[1:len_min])
        
            ecdf1, ecdf2 = ecdf(truet), ecdf(binart)
            cdf1_values, cdf2_values = ecdf1(sorted_true), ecdf2(sorted_binar)
            
            axs[2,i].scatter(cdf1_values, cdf2_values,s=10,c=colors[q_idx],alpha=0.3,edgecolors="none",label=labels[q_idx])
            
            Line2D = PyPlot.matplotlib.lines.Line2D # Set custom legend
            legend_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8) for c in colors]
            
            axs[2,i].legend(legend_handles, ["Q1", "Q2", "Q3"], scatterpoints=1, borderpad=0.3, fontsize=9)
        
            axs[2,i].plot([0,1], [0,1], c="#cb181d",lw = 1.0)
            
            q1, q2, q3 = quantile(re, [0.25, 0.5, 0.75])  # Compute quantiles

            # === Manually Add Inset ===
            inset_ax = axs[2,i].inset_axes([0.6, 0.1, 0.38, 0.38], transform=axs[2,i].transAxes) 

            # Add histogram to inset
            inset_ax.hist(re, bins=15, color="#bdbdbd", alpha=0.6, edgecolor="white", lw = 0.3, density=false)
            inset_ax.axvline(q1, color=colors[1], linestyle="--", lw=1.2, label="Q1")
            inset_ax.axvline(q2, color=colors[2], linestyle="--", lw=1.2, label="Q2")
            inset_ax.axvline(q3, color=colors[3], linestyle="--", lw=1.2, label="Q3")

            # Remove ticks and labels from inset
            inset_ax.set_xticks([]); inset_ax.set_yticks([])
            label1 = inset_labels[i]
            inset_ax.set_xlabel("RE of mean $label1", fontsize=7)
            # Change the border color and thickness of the inset
            for spine in keys(inset_ax.spines)
                inset_ax.spines[spine].set_edgecolor("gray")  # Change to desired color
                inset_ax.spines[spine].set_linewidth(1.2)      # Adjust thickness
            end

        end
    end
    
    axs[2,1].set_xlabel("CDF (true on time)"); axs[2,2].set_xlabel("CDF (true off time)")
    axs[2,1].set_ylabel("CDF (DL on time)"); axs[2,2].set_ylabel("CDF (DL off time)")
    
    for i in 1:2, j in 1:2
        axs[i,j].minorticks_on()  # Enable minor ticks
        axs[i,j].grid(which="minor", linestyle="--", linewidth=0.3, alpha=0.3)
    end
    
    plt.tight_layout()
    
    return fig
    
end;


"""
    fit_line(x, y)

Fit a simple linear regression y ≈ m*x + b
Inputs:
- x, y: numeric vectors (AbstractVector{<:Real})

Returns:
- m (slope)
- b (intercept)
- R² (coefficient of determination)
"""
function fit_line(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    # remove NaNs
    keep = .!(isnan.(x) .| isnan.(y))
    x = float.(x[keep]); y = float.(y[keep])
    @assert length(x) > 1 "Not enough valid points to fit."

    # slope & intercept (least squares closed form)
    m = cov(x, y) / var(x)
    b = mean(y) - m * mean(x)

    # fitted values
    ŷ = m .* x .+ b

    # R²
    ss_res = sum((y .- ŷ).^2)
    ss_tot = sum((y .- mean(y)).^2)
    r2 = 1 - ss_res/ss_tot

    return m, b, r2
end;