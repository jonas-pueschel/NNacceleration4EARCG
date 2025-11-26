module Util

using Plots
using DFTK
using LinearAlgebra: norm
using PyCall
using ArrayInitializers

export plot_images,
    plot_images_with_gradient,
    complex_to_density,
    ifft_to_2d_matrix,
    inverse_fft_reshape,
    TorchUNet,
    PlotDensityCallback,
    plot_density,
    compute_ρ,
    compute_e,
    random_ψ


function plot_images(images::Vector{Array{Float64,3}}, filename::String)
    n = length(images)  # number of images

    # Find the best layout (as close to square as possible)
    nrows = floor(Int, sqrt(n))
    ncols = ceil(Int, n / nrows)

    # Create the plot with a grid layout
    plot_list = []
    for i = 1:n
        density = complex_to_density(images[i])
        pl = heatmap(
            density,
            aspect_ratio = :equal,
            grid = false,
            legend = false,
            showaxis = false,
        )
        push!(plot_list, pl)
    end

    # Save the plot as a PNG file
    plt = plot(plot_list..., layout = (nrows, ncols), size = (8000, 8000))  # adjust size if needed
    savefig(plt, filename)
end

function complex_to_density(arr)
    return arr[:, :, 1] .^ 2 .+ arr[:, :, 2] .^ 2
end

function plot_images_with_gradient(
    images::Vector{Array{Float64,3}},
    gradients::Vector{Array{Float64,3}},
    filename::String,
)
    n = length(images)  # number of images

    # Find the best layout (as close to square as possible), considering we double columns
    ncols = ceil(Int, sqrt(n * 2))  # Total columns accounting for density and gradient
    nrows = ceil(Int, n * 2 / ncols)  # Total rows needed

    # Create the plot with a grid layout
    plot_list = []
    for i = 1:n
        # Compute density and its gradient density
        density = complex_to_density(images[i])
        gradient_density = complex_to_density(gradients[i])

        # Plot each density and gradient density
        pl_density = heatmap(
            density,
            aspect_ratio = :equal,
            grid = false,
            legend = false,
            showaxis = false,
        )
        pl_gradient = heatmap(
            gradient_density,
            aspect_ratio = :equal,
            grid = false,
            legend = false,
            showaxis = false,
        )

        # Add both plots to the list
        push!(plot_list, pl_density)
        push!(plot_list, pl_gradient)
    end

    # Save the plot as a PNG file with adjusted layout
    plt = plot(plot_list..., layout = (nrows, ncols), size = (800, 800))
    savefig(plt, filename)
end

function ifft_to_2d_matrix(basis, ψ)
    ψ = ifft(basis, basis.kpoints[1], ψ[1][:, 1])[:, :, 1]
    ψ_real = real(ψ)
    ψ_imag = imag(ψ)
    m, n = size(ψ)

    ψ_matrix = zeros(Float64, m, n, 2)
    ψ_matrix[:, :, 1] = ψ_real
    ψ_matrix[:, :, 2] = ψ_imag

    return ψ_matrix
end


function inverse_fft_reshape(basis, ψ)
    ϕ = ifft(basis, basis.kpoints[1], ψ[1][:, 1])

    # Combine real and imaginary parts into a 3D array
    Φ_2d = Float32.(cat(real(ϕ), imag(ϕ), dims = 3))

    return Φ_2d
end

struct TorchUNet
    torch
    unet_model
    function TorchUNet(;model_path = "./models/model_paper.pth")
        torch = pyimport("torch")
        # make local modules discoverable py pyimport. See GitHub README of pycall
        pushfirst!(PyVector(pyimport("sys")["path"]), "")
        unet = pyimport("python.unet.model")

        unet_model = unet.UNet(out_channels = 2, in_channels = 4)
        unet_model.load_state_dict(torch.load(model_path, weights_only=true, map_location=torch.device("cpu")))
        unet_model.eval()

        return new(torch, unet_model)
    end
end


function (unet::TorchUNet)(basis, ψ, grad)
    Φ_2d = Util.inverse_fft_reshape(basis, ψ)
    grad_2d = Util.inverse_fft_reshape(basis, grad)

    input₁ = permutedims(Φ_2d, (3, 1, 2))
    input₂ = permutedims(grad_2d, (3, 1, 2))
    t₁ = unet.torch.tensor(input₁)
    t₂ = unet.torch.tensor(input₂)
    input = unet.torch.cat((t₁, t₂), dim = 0)
    output = unet.unet_model(unet.torch.reshape(input, (1, 4, 192, 192))).detach().numpy()

    # Reshape and convert to ComplexF64, maintaining 3D structure
    complex_array = ComplexF64.(complex.(output[1, 1, :, :], output[1, 2, :, :]))
    complex_array = reshape(complex_array, (192, 192, 1))

    # Perform FFT and ensure ψ is a Vector of matrices
    fft_result = fft(basis, basis.kpoints[1], complex_array)
    ψ_neural = [reshape(fft_result, (size(fft_result, 1), 1))]
    norm_ψ = norm(ψ_neural)
    ψ_neural = ψ_neural / norm_ψ
    @assert norm(ψ_neural) ≈ 1.0

    return ψ_neural, norm_ψ
end

function random_ψ(basis)
    model = basis.model
    filled_occ = DFTK.filled_occupation(model);
    n_spin = model.n_spin_components;
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp);
    return [DFTK.random_orbitals(basis, kpt, n_bands) for kpt in basis.kpoints];
end

function compute_ρ(basis, ψ)
    model = basis.model
    filled_occ = DFTK.filled_occupation(model);
    n_spin = model.n_spin_components;
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp);
    occupation = [filled_occ * ones(Float64, n_bands)  for kpt in basis.kpoints];
    return DFTK.compute_density(basis, ψ, occupation)
end

function compute_e(basis, ψ, ρ)
    model = basis.model
    filled_occ = DFTK.filled_occupation(model);
    n_spin = model.n_spin_components;
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp);
    occupation = [filled_occ * ones(Float64, n_bands)  for kpt in basis.kpoints];
    return DFTK.energy(basis, ψ, occupation; ρ)
end

mutable struct PlotDensityCallback
    default_callback
    save_img_path
    unet
    iter_offset
    function PlotDensityCallback(;default_callback = (info -> nothing), save_img_path = "", unet = nothing)
        new(default_callback, save_img_path, unet, 0)
    end
end

function (callback::PlotDensityCallback)(info)
    mod = info.norm_res > 1e-1 ? 1 : (info.norm_res > 1e-2 ? 2 : (info.norm_res > 0.5e-2 ? 5 : (info.norm_res > 1e-4 ? 10 : 100)))

    if  ((info.n_iter + callback.iter_offset) % mod == 0 && hasproperty(info, :ρout)) || info.stage == :finalize
        ρ = hasproperty(info, :ρout) ? info.ρout : info.ρ   
        save = callback.save_img_path != "" ? joinpath(callback.save_img_path, "hmp-$(info.n_iter + callback.iter_offset).png") : ""
        title = "iteration  $(info.n_iter + callback.iter_offset)"
        
        if !isnothing(callback.unet)
            ψ_neural, norm_nn = callback.unet(info.basis, info.ψ, info.grad)
            ρ_neural = compute_ρ(info.basis, ψ_neural)

            err = abs(norm_nn-1)
            save_iter = callback.save_img_path != "" ? joinpath(callback.save_img_path, "hmp-$(info.n_iter+ callback.iter_offset)-nn.png") : ""
            title_iter = "NN of iter $(info.n_iter + callback.iter_offset), err = $err"
            plot_density([ρ, ρ_neural] ;titles = [title, title_iter], saves = [save, save_iter])
        else
            plot_density(ρ ;title, save)
        end
    end

    if info.stage == :finalize
        callback.iter_offset += info.n_iter
    end

    callback.default_callback(info)    
end




function plot_density(ρs::Union{Vector{Array{Float64, 4}}, Vector{Matrix{Float64}}}; 
        titles = Array(ArrayInitializers.init(""), length(ρs)), 
        saves = Array(ArrayInitializers.init(""), length(ρs)))
    plts = []

    for (ρ, save, title) = zip(ρs, saves, titles)
        if length(size(ρ)) == 4
        hmp = heatmap(ρ[:, :, 1, 1], title = title, 
            xaxis=false, yaxis=false, aspect_ratio = :equal, colorbar=:none,
            showaxis = false, ticks= false) #, c=:blues)
        elseif length(size(ρ)) == 2
            maxval = max([abs(x) for x = ρ]...)
            hmp = heatmap(ρ[:, :], title = title, 
                xaxis=false, yaxis=false, aspect_ratio = :equal, colorbar=:none,
                showaxis = false, ticks= false, c=:RdBu, clim = (-maxval, maxval))
        else
            #TODO what happens here?
            continue
        end
        push!(plts, hmp)
        if save != ""
            png(hmp, save)
        end
    end

    plt = plot(plts..., layout = (1, length(ρs)))
    display(plt)
end

function plot_density(ρ::Union{Array, Matrix}; title::String = "", save::String = "")
    plot_density([ρ]; titles = [title], saves = [save])
end

end
