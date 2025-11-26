include("./util.jl")
include("./dftk_setup.jl")

using DFTK
using RCG_DFTK
using .DFTKSetup
using .Util
using PyCall
using CSV
using Dates
using DataFrames
using ProgressBars
using CairoMakie
#using Plots
using LinearAlgebra: norm
using BSON

# path where the model is saved
model_path = "./models/model_paper.pth"

# path where the results will be saved to
result_dir = "./comparisons/comp-" * Dates.format(now(), "yyyy-mm-dd-HH:MM")

# directory of a comparison, whose data shall be used
# i.e. paramters of gpe and initial value
# leave empty for random generated params and initial values
load_dir = ""

num_test_runs = 500
max_iter_direct = 30_000
tol₁_max = 1e-1
tol₁_min = 1e-4
tol₂ = 1e-8

# interval of nn applications when gradient norm is in [tol₁_min, tol₁_max]
interval = 5
# tolerance of accepting nn applications
nn_tol = 5e-3




unet = Util.TorchUNet(;model_path)


function calcInfo(classical, neural₁, neural₂, ψ_neural, neural₁_enhanced, neural₂_enhanced, ψ_neural_enhanced, nn_applications, κ, v, ω, basis :: PlaneWaveBasis)
    ρ = Util.compute_ρ(basis, ψ_neural)
    ρ_enhanced = Util.compute_ρ(basis, ψ_neural_enhanced)
    
    runtime_classical = Int(classical.runtime_ns)
    runtime_neural = Int(neural₁.runtime_ns) + Int(neural₂.runtime_ns)
    runtime_neural_enhanced = Int(neural₁_enhanced.runtime_ns) + Int(neural₂_enhanced.runtime_ns)

    norm1(X) = sum(abs(x) for x = X) 
    diff_ρ_pre = norm1(classical.ρ - neural₁.ρ)
    diff_ρ_post = norm1(classical.ρ - ρ)
    diff_ρ_enhanced_pre = norm1(classical.ρ - neural₁_enhanced.ρ)
    diff_ρ_enhanced_post = norm1(classical.ρ - ρ_enhanced)
    
    impr_ρ = 1 - diff_ρ_post/diff_ρ_pre
    impr_ρ_enhanced = 1 - diff_ρ_enhanced_post/diff_ρ_enhanced_pre

    energy_neural_out = Util.compute_e(basis, ψ_neural, ρ)
    energy_neural_enhanced_out = Util.compute_e(basis, ψ_neural_enhanced, ρ_enhanced)
    info = (
        steps₁ = neural₁.n_iter,
        steps₁_enhanced = neural₁_enhanced.n_iter,
        res₁ = neural₁.norm_res,
        res₁_enhanced = neural₁_enhanced.norm_res,
        total_steps_neural = neural₁.n_iter + neural₂.n_iter,
        total_steps_neural_enhanced = neural₁_enhanced.n_iter + neural₂_enhanced.n_iter,
        nn_applications = nn_applications,
        total_steps_classical = classical.n_iter,
        energy₁ = neural₁.energies.total,
        energy₂ = energy_neural_out.energies.total,
        energy₃ = neural₂.energies.total,
        energy₁_enhanced = neural₁_enhanced.energies.total,
        energy₂_enhanced = energy_neural_enhanced_out.energies.total,
        energy₃_enhanced = neural₂_enhanced.energies.total,
        energy_classical = classical.energies.total,
        impr_ρ,
        impr_ρ_enhanced,
        runtime_classical,
        runtime_neural,
        runtime_neural_enhanced,
        κ, v, ω,
    )
    return info
end

function updateInfoDf(df, neural₁_enhanced, neural₂_enhanced, ψ_neural_enhanced, nn_applications, ρ_classical, basis :: PlaneWaveBasis)
    ρ_enhanced = Util.compute_ρ(basis, ψ_neural_enhanced)
    runtime_neural_enhanced = Int(neural₁_enhanced.runtime_ns) + Int(neural₂_enhanced.runtime_ns)

    norm1(X) = sum(abs(x) for x = X) 
    diff_ρ_enhanced_pre = norm1(ρ_classical - neural₁_enhanced.ρ)
    diff_ρ_enhanced_post = norm1(ρ_classical - ρ_enhanced)
    
    impr_ρ_enhanced = 1 - diff_ρ_enhanced_post/diff_ρ_enhanced_pre

    energy_neural_enhanced_out = Util.compute_e(basis, ψ_neural_enhanced, ρ_enhanced)
  
    df.steps₁_enhanced[1] = neural₁_enhanced.n_iter
    df.res₁_enhanced[1] = neural₁_enhanced.norm_res
    df.total_steps_neural_enhanced[1] = neural₁_enhanced.n_iter + neural₂_enhanced.n_iter
    df.nn_applications[1] = nn_applications
    df.energy₁_enhanced[1] = neural₁_enhanced.energies.total
    df.energy₂_enhanced[1] = energy_neural_enhanced_out.energies.total
    df.energy₃_enhanced[1] = neural₂_enhanced.energies.total
    df.runtime_neural_enhanced[1] = runtime_neural_enhanced
    df.impr_ρ_enhanced[1] = impr_ρ_enhanced
end


function plot_densities(densities, filename, titles=nothing, colormap=:inferno)
    max_val = maximum(map(maximum, densities))
    common_range = (0, max_val)

    n = length(densities)
    fig = Figure()
    for i in 1:n
        ax, _ = CairoMakie.heatmap(fig[1, i],
            densities[i],
            colorrange=common_range,
            colormap=colormap,
        )
        ax.aspect = DataAspect()
        if !isnothing(titles)
            ax.title = titles[i]
        end
        hidedecorations!(ax)
    end

    rowsize!(fig.layout, 1, Aspect(n, 1))
    Colorbar(fig[:, end+1], colorrange=common_range, colormap=colormap)
    resize_to_layout!(fig)
    save(filename, fig)
end

function c(info)
    info
end

function with_neural_network(
    basis;
    ψ_init,
    tol₁_min::Float64,
    tol₁_max::Float64,
    tol₂::Float64,
    max_steps::Int64,
    index::Int64,
    result_dir,
    interval = 5,
    nn_tol = 5e-3,
    do_random = false
)
    function step_callback(info)
        info
    end
    
    if do_random
        # random tolerance for nn enhancemet
        exponent = log10(tol₁_min) + (log10(tol₁_max) - log10(tol₁_min)) * rand()
        tol₁ = 10^exponent
        scfres₁ = DFTKSetup.earcg(basis, tol₁, max_steps; callback = step_callback, ψ0 = ψ_init)

        ψ_neural,~ = unet(basis, scfres₁.ψ, scfres₁.grad)
        ρ_neural = Util.compute_ρ(basis, ψ_neural)

        neural₂ = DFTKSetup.earcg(basis, tol₂, max_steps - scfres₁.n_iter; callback = c, ψ0 = ψ_neural, ρ0 = ρ_neural) 

        # plot densities
        d1 = scfres₁.ρ[:, :, 1, 1]
        d2 = ρ_neural[:, :, 1, 1]
        d3 = neural₂.ρ[:, :, 1, 1]

        fig_path = joinpath(result_dir, "neural_progression_$(index).png")


        titles = ["Initial", "NN Out", "ρ_neural"]
        plot_densities([d1,d2,d3], fig_path, titles)
    end

    # use enhancement strategy
    scfres₁_enhanced, neural₂_enhanced, ψ_neural_enhanced, ρ_neural_enhanced, nn_applications = 
    DFTKSetup.enhanced_earcg(basis, tol₂, max_steps, unet; 
        ψ0 = ψ_init,
        tol₁_min, tol₁_max,
        interval,
        nn_tol,
        callback = step_callback
        );

    ρ_neural_enhanced = Util.compute_ρ(basis, ψ_neural_enhanced)


    d1e = scfres₁_enhanced.ρ[:, :, 1, 1]
    d2e = ρ_neural_enhanced[:, :, 1, 1]
    d3e = neural₂_enhanced.ρ[:, :, 1, 1]

    fig_path_e = joinpath(result_dir, "neural_progression_$(index)_enhanced.png")
    titles_e = ["Initial_enh", "NN Out", "ρ_neural_enh"]
    plot_densities([d1e,d2e,d3e], fig_path_e, titles_e)

    if do_random
        return scfres₁, neural₂, ψ_neural, scfres₁_enhanced, neural₂_enhanced, ψ_neural_enhanced, nn_applications
    else 
        return scfres₁_enhanced, neural₂_enhanced, ψ_neural_enhanced, nn_applications
    end
end

function without_neural_network(basis; ψ, tol::Float64, max_steps::Int64)
    scfres = DFTKSetup.earcg(basis, tol, max_steps; callback = c, ψ0 = ψ)
    return scfres
end

function compare_algorithms(interval, nn_tol, result_dir, load_dir)
    mkpath(result_dir)

    for i in ProgressBar(1:num_test_runs)

        #skip if already done
        if (isfile("$result_dir/neural_progression_$(i)_enhanced.png"))
            continue
        end

        mkpath(joinpath(result_dir, "example-$(i)"))

        if (load_dir == "")
            model, basis, κ, v, ω  = DFTKSetup.random_gp2D(;Ecut = 100)
            # initialize random ψ₀
            ψ₀ = Util.random_ψ(basis)

            _, elapsed_time_neural_net, _, _, _ = @timed begin
                neural₁, neural₂, ψ_neural, neural₁_enhanced, neural₂_enhanced, ψ_neural_enhanced, nn_applications  = with_neural_network(
                    basis,
                    ψ_init = ψ₀,
                    tol₁_min = tol₁_min,
                    tol₁_max = tol₁_max,
                    tol₂ = tol₂,
                    max_steps = max_iter_direct,
                    index = i,
                    result_dir = result_dir,
                    interval = interval,
                    nn_tol = nn_tol,
                    do_random = true
                )
            end
        
            _, elapsed_time_direct, _, _, _ = @timed begin
                classical = without_neural_network(
                    basis,
                    ψ = ψ₀,
                    tol = tol₂,
                    max_steps = max_iter_direct,
                )
            end

            info = calcInfo(classical, neural₁, neural₂, ψ_neural, neural₁_enhanced, neural₂_enhanced, ψ_neural_enhanced, nn_applications, κ, v, ω, basis)
            stats = [info]
            df = DataFrame(stats)

            ρ_classical = classical.ρ

        else
            load_path = joinpath(load_dir, "example-$(i)")
            κ, ω, v =  DFTKSetup.load_gp2D_params(load_path)

            #TODO load ρ_classical
            model, basis, ψ₀, ρ_classical = DFTKSetup.load_gp2D(load_path)

            # _, elapsed_time_direct, _, _, _ = @timed begin
            #     classical = without_neural_network(
            #         basis,
            #         ψ = ψ₀,
            #         tol = tol₂,
            #         max_steps = max_iter_direct,
            #     )
            # end
            # ρ_classical = classical.ρ
            
            _, elapsed_time_neural_net, _, _, _ = @timed begin
                neural₁_enhanced, neural₂_enhanced, ψ_neural_enhanced, nn_applications  = with_neural_network(
                    basis,
                    ψ_init = ψ₀,
                    tol₁_min = tol₁_min,
                    tol₁_max = tol₁_max,
                    tol₂ = tol₂,
                    max_steps = max_iter_direct,
                    index = i,
                    result_dir = result_dir,
                    interval = interval,
                    nn_tol = nn_tol,
                    do_random = false
                )
            end

            csv_path = joinpath(load_path, "statistics.csv")
            df = DataFrame(CSV.File(csv_path))

            updateInfoDf(df, neural₁_enhanced, neural₂_enhanced, ψ_neural_enhanced, nn_applications, ρ_classical, basis)
        end

        csv_path = joinpath(result_dir, "example-$(i)/statistics.csv")
        CSV.write(csv_path, df)
        
        bson_path = joinpath(result_dir, "example-$(i)/psi0.bson")
        ψ1 =  ψ₀
        BSON.@save bson_path ψ1 ρ_classical
    end
end

if abspath(PROGRAM_FILE) == @__FILE__ 
    #if this file is called as the main program
    compare_algorithms(interval, nn_tol, result_dir, load_dir)
end