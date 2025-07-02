module DFTKSetup

using DFTK
using StaticArrays
using CSV
using BSON
using DataFrames
using Plots

include("/PATH/TO/RCG_DFTK/src/rcg.jl")
include("./util.jl")
using .Util

export gp2D_setup, random_gp2D, random_ch_gp2D, load_gp2D, load_gp2D_params,  
        earcg,  random_ψ,  compute_ρ,  ea_gradient,  enhanced_earcg, load_gp2D_performance, 
        get_default_callback

function gp2D_setup(;Ecut = 100, a = 20, κ = 50, ω = 1.6, v = (1.1, 1.0))
    lattice = a .* [[1 0 0.]; [0 1 0]; [0 0 0]];

    # Confining scalar potential, and magnetic vector potential
    pot(x, y, z) = (v[1] * (x - a/2)^2 + v[2] * (y - a/2)^2)/2
    Apot(x, y, z) = 0.5 * ω * @SVector [y - a/2, -(x - a/2), 0]
    Apot(X) = Apot(X...);

    # Parameters
    α = 2
    n_electrons = 1;  # Increase this for fun

    # Collect all the terms, build and run the model
    terms = [Kinetic(),
            ExternalFromReal(X -> pot(X...)),
            LocalNonlinearity(ρ -> 0.25 * κ * ρ^α),
            Magnetic(Apot),
    ]
    model = Model(lattice; n_electrons, terms, spin_polarization=:spinless)  # spinless electrons
    basis = PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1))
    return[model, basis]
end


function random_gp2D(;Ecut = 100, challenging = false)
    if challenging
        κ = rand(600:1000)
        ω =  1.2 + rand() * 0.4
    else
        κ = rand(200:1000)
        ω =  0.8 + rand() * 0.8
    end
    v1 = 1.0 + rand() * 1.0
    v = (v1, 1.0)
    return [gp2D_setup(;Ecut, κ, v, ω)..., κ, v, ω]
end


# function load_classical_result()
    
# end

# function load_neural_result(path)
#     csv_path = joinpath(path, "statistics.csv")

#     df = DataFrame(CSV.File(csv_path))


#     return (;steps₁, res₁, total_steps_neural, energy₁, energy₂, energy₃, impr_ρ)
# end

function load_gp2D_params(path)
    csv_path = joinpath(path, "statistics.csv")

    df = DataFrame(CSV.File(csv_path))

    κ = df.κ[1]
    v1, v2 = split(strip(df.v[1], ('(',')')),",")
    v = (parse(Float64, v1), parse(Float64, v2))

    ω = df.ω[1]

    return κ, ω, v
end

function load_gp2D_performance(path; enhanced = true)
    csv_path = joinpath(path, "statistics.csv")
    df = DataFrame(CSV.File(csv_path))

    steps_neural = enhanced ? df.total_steps_neural_enhanced[1] : df.total_steps_neural[1]  
    steps_classical = df.total_steps_classical[1]
    impr_ρ = enhanced ? df.impr_ρ_enhanced[1] : df.impr_ρ[1]

    return steps_neural, steps_classical, impr_ρ
end

function load_gp2D(path; a = 20)
    ψ_path = joinpath(path, "psi0.bson")

    κ, ω, v = load_gp2D_params(path)

    model, basis = gp2D_setup(; κ, ω, v, a)

    BSON.@load ψ_path ψ1

    (model, basis, ψ1)
end

function get_default_callback()
    return RcgDefaultCallback(;show_grad_norm = true)
end

#EARCG
function earcg(basis, tol, max_iterations; ψ0 = nothing, ρ0 = nothing, callback = get_default_callback(), inner_it = 25)
    #EARCG setup
    shift = ConstantShift(0.0)

    gradient = EAGradient(basis, shift; 
        tol = 0.01,
        itmax = inner_it,
        h_solver = NaiveHSolver(;krylov_solver = Krylov.cg),
        Pks = [PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
        ) 

    iteration_strat = AdaptiveBacktracking(
        ModifiedSecantRule(0.05, 0.1, 1e-12, 0.5),
        ConstantStep(1.0), 10);

    is_converged = RcgConvergenceGradient(tol)

    return riemannian_conjugate_gradient(
        basis;
        ψ = ψ0,
        ρ = ρ0,
        tol,
        maxiter = max_iterations,
        callback,
        gradient,
        iteration_strat,
        is_converged,
        check_convergence_early = false
    )
end

#EARCG
function enhanced_earcg(basis, tol, max_iterations, unet; ψ0 = nothing, ρ0 = nothing, callback = get_default_callback(), 
                tol₁_max = 1e-1, tol₁_min = 1e-4, interval = 5, inner_it = 25)
    #EARCG setup
    shift = ConstantShift(0.0)
 
    gradient = EAGradient(basis, shift; 
        tol = 0.01,
        itmax = inner_it,
        h_solver = NaiveHSolver(;krylov_solver = Krylov.cg),
        Pks = [PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
        ) 

    iteration_strat = AdaptiveBacktracking(
                    ModifiedSecantRule(0.05, 0.1, 1e-12, 0.5),
                    ConstantStep(1.0), 10);
        
    apply_nn = false
    nn_applications = 0
    ψ_neural = nothing
    ρ_neural = nothing


    function cb(info)
        if (info.n_iter % interval != 0 || info.norm_grad > tol₁_max || info.norm_grad < tol₁_min)
            return callback(info)
        end

        ψ_neural, norm_nn = unet(info.basis, info.ψ, info.grad)
        nn_applications += 1

        err = abs(norm_nn-1)

        #if (err > min(0.1 * info.norm_grad, 1e-3))
        if (err > 5e-3)
            return callback(info)
        end

        ρ_neural = Util.compute_ρ(info.basis, ψ_neural)
        apply_nn = true

        return callback(info)
    end

    default_converged = RcgConvergenceGradient(tol₁_min)
    function is_converged(info)
        default_converged(info) || apply_nn
    end
    rcg_1 = riemannian_conjugate_gradient(
        basis;
        ψ = ψ0,
        ρ = ρ0,
        tol = tol₁_min,
        maxiter = max_iterations,
        is_converged,
        callback = cb,
        gradient,
        iteration_strat,
        check_convergence_early = false
    )

    if !apply_nn
        #apply nn manually if nn was never applied
        ψ_neural, ~ = unet(basis, rcg_1.ψ, rcg_1.grad)
        ρ_neural = Util.compute_ρ(basis, ψ_neural)
        nn_applications += 1
    end

    rcg_2 = earcg(basis, tol, max_iterations - rcg_1.n_iter ; ψ0 = ψ_neural, ρ0 = ρ_neural, callback, inner_it)

    return (rcg_1, rcg_2, ψ_neural, ρ_neural, nn_applications)
 end

function ea_gradient(basis, ψ, ρ)
    model = basis.model
    filled_occ = DFTK.filled_occupation(model);
    n_spin = model.n_spin_components;
    n_bands = div(model.n_electrons, n_spin * filled_occ, RoundUp);
    occupation = [filled_occ * ones(Float64, n_bands)  for kpt in basis.kpoints];

    energies, H = DFTK.energy_hamiltonian(basis, ψ, occupation; ρ)

    shift = ConstantShift(0.0)

    gradient = EAGradient(basis, shift; 
        tol = 0.01,
        itmax = 25,
        h_solver = NaiveHSolver(;krylov_solver = Krylov.cg),
        Pks = [PreconditionerTPA(basis, kpt) for kpt in basis.kpoints]
        ) 
    # compute first residual 
    Nk = length(ψ)

    Hψ = H * ψ
    Λ = [ψ[ik]'Hψ[ik] for ik = 1:Nk]
    Λ = 0.5 * [(Λ[ik] + Λ[ik]') for ik = 1:Nk]
    res = [Hψ[ik] - ψ[ik] * Λ[ik] for ik = 1:Nk]

    calculate_gradient(ψ, Hψ, H, Λ, res, gradient)
end

end