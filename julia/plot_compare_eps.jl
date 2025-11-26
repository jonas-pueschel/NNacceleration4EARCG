include("./dftk_setup.jl")
include("./util.jl")
include("./plot_nn_examples.jl")

using CSV
using DataFrames
using Dates
using ProgressBars
using Plots
using Statistics
using Printf

n_comparisons = 500
path_comparison = "./comparisons/comp-nntol-"
is = [i for i in 0:8]
tols = [10^(-1 - i * 0.25) for i = is]

compare_paper_results = false

if compare_paper_results
    is = [is..., 5.204119982655923]
    tols = [tols..., 5 * 10^(-3)]
    path_figures = "./figs/"
end

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

mean_steps = [0.0 for i = is]
mean_rhos = [0.0 for i = is]


median_steps = [0.0 for i = is]
median_rhos = [0.0 for i = is]


impr_steps = [0.0 for i = is]
impr_rhos = [0.0 for i = is]


function get_adjusted_mean(vals)
    #return mean(vals)
    sorted_vals = sort(vals)
    n_vals = length(vals) 
    half_perc = Int(round(n_vals/200))
    adj_vals = sorted_vals[half_perc+1 : n_vals - half_perc]
    return mean(adj_vals)
end

step_vals = nothing
for i = 0:8
    path = "$path_comparison$i"
    results_enhanced = [ComparisonResult(path, j, true) for j = 1:n_comparisons if isfile("$path/example-$(j)/statistics.csv")]

    step_vals = [result.it_diff_perc for result = results_enhanced]
    rho_vals = [result.impr_ρ for result = results_enhanced]
    time_vals = [result.impr_time for result = results_enhanced]

    mean_steps[i+1] = get_adjusted_mean(step_vals)
    median_steps[i+1] = median(step_vals)
    impr_steps[i+1] = length([v for v = step_vals if v > 0])/n_comparisons
    mean_rhos[i+1] = get_adjusted_mean(rho_vals)
    median_rhos[i+1] = median(rho_vals)
    impr_rhos[i+1] = length([v for v = rho_vals if v > 0])/n_comparisons
end

begin
    path = "./comparisons/comp-paper"
    results_enhanced = [ComparisonResult(path, j, true) for j = 1:n_comparisons if isfile("$path/example-$(j)/statistics.csv")]

    step_vals = [result.it_diff_perc for result = results_enhanced]
    rho_vals = [result.impr_ρ for result = results_enhanced]
    time_vals = [result.impr_time for result = results_enhanced]

    mean_steps[10] = get_adjusted_mean(step_vals)
    median_steps[10] = median(step_vals)
    impr_steps[10] = length([v for v = step_vals if v > 0])/n_comparisons
    mean_rhos[10] = get_adjusted_mean(rho_vals)
    median_rhos[10] = median(rho_vals)
    impr_rhos[10] = length([v for v = rho_vals if v > 0])/n_comparisons
end


if compare_paper_results
    perm = [1,2,3,4,5,6,10,7,8,9]
    tols = tols[perm]
    mean_steps = mean_steps[perm]
    mean_rhos = mean_rhos[perm]
    median_steps = median_steps[perm]
    median_rhos = median_rhos[perm]
    impr_steps = impr_steps[perm]
    impr_rhos = impr_rhos[perm]
end

plt1 =plot(tols[perm], mean_steps[perm] ;  xaxis = :log10, label = "steps", title = "means")
plot!(tols[perm], mean_rhos[perm], label = "rhos")
#plot!(tols, mean_times)
display(plt1)


plt3 =plot(tols[perm], median_steps[perm] ;  xaxis = :log10, label = "steps", title = "medians")
plot!(tols[perm], median_rhos[perm], label = "rhos")
#plot!(tols, mean_times)
display(plt3)


plt2 = plot(tols[perm], impr_steps[perm] ;  xaxis = :log10, label = "steps", title = "percentage improvement")
plot!(tols[perm], impr_rhos[perm], label = "rhos")
display(plt2)

# function to_tikz(vals)
#     for i = 1:10
#         println("$(8 - is[perm[i]]) $(vals[perm[i]]) \\\\")
#     end
# end

# to_tikz(mean_steps)
# to_tikz(mean_rhos)

# to_tikz(impr_steps)
# to_tikz(impr_rhos)