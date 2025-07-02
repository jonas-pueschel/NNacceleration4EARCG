using Pkg
Pkg.activate("./")

include("./dftk_setup.jl")
include("./util.jl")

using .DFTKSetup
using .Util
using DFTK
using NPZ
using Plots
using ProgressBars

data_folder_path = "./data"

# make sure output folder exists
mkpath(data_folder_path)

function make_plots(
    ψ_history::Vector{Vector{Matrix{ComplexF64}}},
    num_images::Int64,
    last_index::Int64,
    basis,
    filename::String,
)
    # # extract num_images evenly distributed indices
    indices = collect(round.(Int, range(1, stop = last_index, length = num_images)))
    ψ_list = ψ_history[indices]
    grad_finite_difference = [ψ_history[i] - ψ_history[i-1] for i in indices[2:end]]
    ψ_matrix = Util.ifft_to_2d_matrix.(basis, ψ_list)
    grad_finite_difference = Util.ifft_to_2d_matrix.(basis, grad_finite_difference)
    plot_images_with_gradient(ψ_matrix[2:end], grad_finite_difference, filename)
end

# Convergence we desire in density
tol = 1e-8

#number of examples 
num_examples = 1000


#set to true to additionally generate challenging data data
n_samples = 20
exponent_max = -1.0
exponent_min = -4.0
max_iterations = 30_000

function generate_data(challenging)
    for i in ProgressBar(1:num_examples)
        ψ_history = Vector{Vector{Matrix{ComplexF64}}}(undef, max_iterations)
        grad_history = Vector{Vector{Matrix{ComplexF64}}}(undef, max_iterations)
        H_history = Vector{Hamiltonian}(undef, max_iterations)
        norm_grad_history = ones(Float64, max_iterations) 

        # add a default callback if wanted
        default_cb = (info) -> info
        function step_callback(info)
            ψ_history[info.n_iter] = info.ψ
            grad_history[info.n_iter] = info.grad
            H_history[info.n_iter] = info.ham
            norm_grad_history[info.n_iter] = info.norm_grad
            if info.n_iter % 100 == 1
                default_cb(info)
            else
                info
            end
        end

        model, basis, κ, v, ω = random_gp2D(;Ecut = 100, challenging)
        i_save = i + (challenging ? num_examples : 0)

        scfres = DFTKSetup.earcg(basis, tol, max_iterations; callback = step_callback)
        if scfres.n_iter == max_iterations
            # convergence isn't reached
            continue
        end

        # This is only executed when dftk terminated in the given number of steps
        # TODO: make these tolerances instead of iteration percentages? Also is 50% too far, only 25% or tol = 1e-2?
        index = 1
        
        for j in 1:n_samples
            # index at j% of total number of iterations
            norm_grad_aim = 10^(exponent_max + j/n_samples * (exponent_min - exponent_max))
            
            while (norm_grad_history[index] > norm_grad_aim && index < max_iterations)
                index += 1
            end

            if (index == max_iterations && norm_grad_history[index] > 1e-6)
                #keep data if it didnt converge but convergence is "close enough"
                break
            end

            ψⱼ = ψ_history[index]
            gradⱼ = grad_history[index]
            Hⱼ = H_history[index]
            partial_solution = Util.ifft_to_2d_matrix(basis, ψⱼ)
            partial_grad = Util.ifft_to_2d_matrix(basis, gradⱼ)
            npzwrite(joinpath(data_folder_path, "partial_$(i_save)_$(j).npy"), partial_solution)
            npzwrite(joinpath(data_folder_path, "grad_$(i_save)_$(j).npy"), partial_grad)
        end
        last_index = scfres.n_iter
        final = Util.ifft_to_2d_matrix(basis, ψ_history[scfres.n_iter])
        npzwrite(joinpath(data_folder_path, "final_$(i_save).npy"), final)
        # make_plots(
        #     ψ_history,
        #     16,
        #     last_index,
        #     basis,
        #     "progression_with_finite_difference_grad-$(i_save).png",
        # )
    end
end

# generate data with params spread uniformly on the whole domain
generate_data(false)
# generate data with params in the challenging domain
generate_data(true)