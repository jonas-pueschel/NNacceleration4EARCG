include("./dftk_setup.jl")
include("./util.jl")
include("./plot_nn_examples.jl")

using DFTK
using .DFTKSetup
using .Util
using .PlotExample
using PyCall
using CSV
using DataFrames
using Dates
using ProgressBars
using LinearAlgebra: norm
using BSON
using Plots
using Statistics
using Printf
using LinearAlgebra

n_comparisons = 500
path_comparison = "./comparisons/comp-paper"
path_figures = "./figs/"

mkpath(path_figures)

generate_example_plots = false
use_adjusted_mean = true

df = nothing

struct ComparisonResult
    path::String
    index::Int
    enhanced::Bool
    nn_aplications::Int
	steps₁::Int
	res₁::Float64
	total_steps_neural::Int
	total_steps_classical::Int
	energy₁::Float64
	energy₂::Float64
	energy₃::Float64
	energy_classical::Float64
    impr_ρ::Float64
    time_classical::Int64
    time_neural::Int64
    it_diff::Int
    it_diff_perc::Float64
    impr_time::Float64

    function ComparisonResult(path::String, index::Int, enhanced::Bool)
        csv_path = joinpath(path, "example-$index", "statistics.csv")
        df = DataFrame(CSV.File(csv_path))
        fields = !enhanced ? ["steps₁", "res₁", "total_steps_neural", "total_steps_classical", "energy₁", "energy₂", "energy₃", "energy_classical", "impr_ρ", "runtime_classical", "runtime_neural"] : ["steps₁_enhanced", "res₁_enhanced", "total_steps_neural_enhanced", "total_steps_classical", "energy₁_enhanced", "energy₂_enhanced", "energy₃_enhanced", "energy_classical", "impr_ρ_enhanced", "runtime_classical", "runtime_neural_enhanced"]
        vals = [df[!, field][1] for field = fields]
        nn_applications = enhanced ? df.nn_applications[1] : 1
        return new(joinpath(path, "example-$index"), index, enhanced, nn_applications, vals..., vals[4] - vals[3], (vals[4] - vals[3])/vals[4], (vals[end-1] - vals[end])/vals[end-1])
    end
end

results = [ComparisonResult(path_comparison, i, false) for i = 1:n_comparisons]
results_enhanced = [ComparisonResult(path_comparison, i, true) for i = 1:n_comparisons]

function vals_to_bins(vals, bins)
    bin_vals = zeros(length(bins) - 1)
    for val = vals
        for i = 1:length(bin_vals)
            if ( val >= bins[i] && val < bins[i + 1])
                bin_vals[i] += 1
                break
            end
        end
    end
    return bin_vals

end

function generate_hist(bins, vals_same, vals_diff, is_perc, mean_val, adj_mean_val, median_val, fpath)
    mean_val = use_adjusted_mean ? adj_mean_val : mean_val
    
    xtick = [ bins[2], bins[2]/2, 0, bins[end-1]/2, bins[end-1] ]
    xtick_str = prod("$(Int(xt)), " for xt = xtick)[1:end-2]
    if (!is_perc)
        xlabels = prod(xt < 0 ? "$(Int(xt))\\phantom{-}, " : "$(Int(xt)), " for xt = xtick)
    else
        xlabels = prod(xt >= 0 ? "\\phantom{\\%}$(Int(xt))\\%, " : "$(Int(xt))\\%, " for xt = xtick)
    end 
    xlabels = xlabels[1:end-2]
    xmin = bins[1]
    xmax = bins[end]
    ymax = max((vals_same + vals_diff)...)

    mean_str = @sprintf("%.2f", mean_val)
    adj_mean_str = @sprintf("%.2f", adj_mean_val)
    median_str = @sprintf("%.2f", median_val)
    if (is_perc) median_str = "$median_str\\%" end
    if (is_perc) adj_mean_str = "$adj_mean_str\\%" end
    if (is_perc) mean_str = "$mean_str\\%" end

    xs = [Int((bins[i] + bins[i+1])/2) for i = 1:length(bins)-1]

    function get_values_string(vals, bins)
        vals_int = "($(xs[1]), 0)\n" * prod("($x,$y)\n" for (x,y) = zip(xs[2:end-1], vals[2:end-1]))*"($(xs[end]), 0)"
        vals_ext = "($(xs[1]), $(vals[1]))\n" * prod("($x,0)\n" for x = xs[2:end-1]) *"($(xs[end]), $(vals[end]))"
        return vals_int, vals_ext
    end

    same_str_int, same_str_ext = get_values_string(vals_same, bins)
    diff_str_int, diff_str_ext = get_values_string(vals_diff, bins)

    bar_width = xs[end] - xs[end-1]

    st = """\\begin{tikzpicture}
    \\begin{axis}[
        width = \\textwidth,
        height = \\textwidth,
        ymin=0, ymax=$(ymax * 1.1),
        xmin = $xmin, xmax = $xmax,
        minor y tick num = 3,
        xtick = {$xtick_str},
        xticklabels = {{$xlabels}},
        area style,
        ]
    \\addplot+[ybar stacked, bar width=$bar_width, color = blue!30, draw = blue] coordinates { 
    $same_str_int
    };
    \\addplot+[ybar stacked, bar width=$bar_width, color = blue!30, draw = blue, pattern = north east lines, pattern color = blue] coordinates { 
    $same_str_ext
    };
    \\addplot+[ybar stacked, bar width=$bar_width, color = orange!30, draw = orange] coordinates { 
    $diff_str_int
    };
    \\addplot+[ybar stacked, bar width=$bar_width, color = orange!30, draw = orange, pattern = north east lines, pattern color = orange] coordinates { 
    $diff_str_ext
    };

    \\addplot[mark=none, red ,dashed, thick] coordinates {($mean_val,0) ($mean_val,$(ymax * 2))};
    \\addplot [red,, nodes near coords={\\small Mean: $mean_str},every node near coord/.style={anchor=180}] coordinates {( $(bins[2]), $(ymax* 0.6))};
    %\\addplot[mark=none, orange ,dashed, thick] coordinates {($adj_mean_val,0) ($adj_mean_val,$(ymax * 2))};
    %\\addplot [orange,, nodes near coords={\\small Adj. mean: $adj_mean_str},every node near coord/.style={anchor=180}] coordinates {( $(bins[2]), $(ymax* 0.55))};
    \\addplot[mark=none, black ,dashed ,thick] coordinates {($median_val,0) ($median_val,$(ymax * 2))};
    \\addplot [black,, nodes near coords={\\small Median: $median_str},every node near coord/.style={anchor=180}] coordinates {($(bins[2]), $(ymax * 0.55))};
    \\end{axis}
    \\end{tikzpicture}
    """

    open(fpath, "w") do file
        write(file, st)
    end
end

means = zeros(8)
medians = zeros(8)

function get_adjusted_mean(vals)
    sorted_vals = sort(vals)
    n_vals = length(vals) 
    half_perc = Int(round(n_vals/200))
    adj_vals = sorted_vals[half_perc+1 : n_vals - half_perc]
    return mean(adj_vals)
end

function generate_hist_iterations(prefix, fname, enhanced, fieldname, is_perc; resls = nothing, index = nothing)
    if (isnothing(resls))
        resls = enhanced ? results_enhanced : results
    end
    resls = copy(resls)
    if (fname == "hist_rhos")
        resls = [res for res in resls if res.total_steps_classical < 30_000]
    end

    vals = [getfield(result, fieldname) for result = resls]
    if (is_perc) vals .*= 100 end
    energy_diffs = [abs(result.energy₃ - result.energy_classical) for result = resls]

    if (!is_perc)
        bins = [i for i = -1000:100:1000]
        bins2 = [i for i = -1100:100:1100]
        bins = [-Inf ,bins..., Inf]
    else
        bins = [i for i = -100:10:100]
        bins2 = [i for i = -110:10:110]
        bins = [-Inf ,bins..., Inf]
    end

    median_val = median(vals)
    mean_val = mean(vals)

    adj_mean_val = get_adjusted_mean(vals)
    if !isnothing(index)
        medians[index] = median_val
        means[index] = mean_val
    end

    n_impr = length([v for v = vals if v > 0])

    vals_same_e = [vals[i] for i = 1:length(resls) if abs(energy_diffs[i]) < 1e-6]
    vals_diff_e = [vals[i] for i = 1:length(resls) if abs(energy_diffs[i]) >= 1e-6]
    h1 = histogram(vals ; bins = bins,plot_title = "$prefix $(enhanced ? " enhanced" : "")")
    display(h1)
    fname_full = joinpath(path_figures, "histograms", "$fname$(enhanced ? "_enhanced" : "").tex")
    binvals_same_e = vals_to_bins(vals_same_e, bins)
    binvals_diff_e = vals_to_bins(vals_diff_e, bins)
    generate_hist(bins2, binvals_same_e, binvals_diff_e, is_perc, mean_val, adj_mean_val, median_val, fname_full)

    if (enhanced && fname == "hist_rhos" && generate_example_plots)
        #plot three best and worst

        resls_sorted = sort(resls, by = x -> x.impr_ρ)


        for i = 1:3
            v_best = resls_sorted[end-(i-1)]
            v_worst = resls_sorted[i]

            PlotExample.generate_example_fig(v_best.path, "best-rho-$i"; gen_pngs = true)
            PlotExample.generate_example_fig(v_worst.path, "worst-rho-$i"; gen_pngs = true)
        end
    end

    if (enhanced && fname == "hist_perc_its" && generate_example_plots)
        #plot three best and worst

        resls_sorted = sort(resls, by = x -> x.it_diff_perc)

        idx_best = length(resls_sorted)
        idx_worst = 1

        for i = 1:3
            v_best = resls_sorted[idx_best]
            v_worst = resls_sorted[idx_worst]

            # filter out cases where classical and accelerated converged to different local minima
            # in these cases, iteration difference is not necessarily equivalent to NN guess quality
            while (abs(v_best.energy₃ - v_best.energy_classical) > 1e-6)
                idx_best -= 1 
                v_best = resls_sorted[idx_best]
            end
            while (abs(v_worst.energy₃ - v_worst.energy_classical) > 1e-6)
                idx_worst += 1 
                v_worst = resls_sorted[idx_worst]
            end

            idx_worst += 1
            idx_best -= 1 

            PlotExample.generate_example_fig(v_best.path, "best-it-$i"; gen_pngs = true)
            PlotExample.generate_example_fig(v_worst.path, "worst-it-$i"; gen_pngs = true)
        end
    end

    return fname_full, vals_same_e, length(resls), n_impr
end

function generate_hist_figure(prefix, fname_fig, fullname_left, fullname_right, same_e_left, same_e_right, interval, n_cases, n_impr)
    n_impr_l, n_impr_r = n_impr

    p_impr_l = n_impr_l/n_cases
    impr_l_str = @sprintf("%.1f", p_impr_l * 100) * "\\%"

    p_impr_r = n_impr_r/n_cases
    impr_r_str = @sprintf("%.1f", p_impr_r * 100) * "\\%"

    if n_cases != 500
        st_cases = "evaluated on the $n_cases relevant test cases."
    else
        st_cases = "evaluated on all $n_cases test cases."
    end
    st = """
\\begin{figure}
\\centering
\\begin{subfigure}{.5\\textwidth}
  \\centering
    \\captionsetup{width=.9\\textwidth}
  \\input{$fullname_left}
  \\label{fig:sub1}
    \\subcaption{$prefix by neural network enhancement of the algorithm when neural network was applied randomly. 
    Improvement was achieved in $impr_l_str of cases.
    \\\\ \\phantom{.}}
\\end{subfigure}%
\\begin{subfigure}{.5\\textwidth}
  \\centering
  \\captionsetup{width=.9\\textwidth}
  \\input{$fullname_right}  
  \\label{fig:sub2}
  \\subcaption{$prefix by neural network enhancement of the algorithm when neural network was applied via the previously described acceleration strategy.
  Improvement was achieved in $impr_r_str of cases.
  }
\\end{subfigure}
    \\caption{Histograms of the $(lowercase(prefix)) by the neural network-accelerated algorithms, $st_cases 
    In blue are the cases, where the enhanced and classical algorithm converged to the same local minimum, in orange the cases of different minima. 
    The dashed bars denote the cases $interval.
    }
\\label{fig:$fname_fig}
\\end{figure}
    """

    fname_full = joinpath(path_figures, "$fname_fig.tex")
    open(fname_full, "w") do file
        write(file, st)
    end
end

same_e_left = 0
same_e_right = 0
n_cases = 0

prefixes = ["Number of iterations saved", "Percentage of iterations saved", "Reduction in density error", "Reduction in wall time"]
fnames = ["hist_its", "hist_perc_its", "hist_rhos", "hist_times"]
is_percs = [false, true, true, true]
fieldnames = [:it_diff, :it_diff_perc, :impr_ρ, :impr_time]
index = 0
for (prefix, fname, is_perc, fieldname) = zip(prefixes, fnames, is_percs, fieldnames)
    fullname_left, same_e_left, n_cases, n_impr_l = generate_hist_iterations(prefix, fname, false, fieldname, is_perc; index = (index += 1))
    fullname_right, same_e_right, ~ , n_impr_r = generate_hist_iterations(prefix, fname, true, fieldname, is_perc; index = (index += 1))
    interval = is_perc ? "with increase bigger than 100\\%" : "falling outside the interval [-1000,1000]" 

    fname_fig = "figure_$fname"
    n_impr = (n_impr_l, n_impr_r)
    generate_hist_figure(prefix, fname_fig, fullname_left, fullname_right, length(same_e_left), length(same_e_right), interval, n_cases, n_impr)
end
