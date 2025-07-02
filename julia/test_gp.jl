using Pkg
Pkg.activate("./")

using DFTK
using LinearAlgebra
using Krylov
using Plots

include("./dftk_setup.jl")
include("./util.jl")
using .Util
using .DFTKSetup

# path to an example from the comparions to re-run/try it
# initial value and gpe params get taken from that experiment
# if left empty, random initial value and params will be used
# path = "./comparisons/comp-paper/example-123"
path = ""

# path to the trained unet
model_path = "./models/model_paper.pth"

unet = Util.TorchUNet(; model_path)

maxiter = 20_000
tol = 1e-8;

if (path == "")
    model, basis, κ, v, ω = DFTKSetup.random_gp2D()
    ψ1 = Util.random_ψ(basis)
    ρ1 = guess_density(basis)
    steps1 = maxiter
else
    model, basis, ψ1 = DFTKSetup.load_gp2D(path)
    ρ1 = guess_density(basis)
end

earcg_init, earcg_enhanced, ψ_neural_enh, ρ_neural_enh, nn_applications = DFTKSetup.enhanced_earcg(basis, tol, maxiter, unet; 
    ψ0 = ψ1, ρ0 = ρ1, 
    interval = 5
    );

#Util.plot_density(earcg_enhanced.ρ)

# classical
earcg_classical = DFTKSetup.earcg(basis, tol, maxiter; 
    ψ0 = ψ1, ρ0 = ρ1, 
    );

# runtime_classical = Int(earcg_classical.runtime_ns)
# runtime_classical = Int(earcg_init.runtime_ns) + Int(earcg_enhanced.runtime_ns)