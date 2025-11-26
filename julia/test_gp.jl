include("./dftk_setup.jl")
include("./util.jl")
using .Util
using .DFTKSetup
using DFTK

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
κ = 0.0
v = (1.0, 1.0)
ω = 0.0 

if (path == "")
    model, basis, κ, v, ω = DFTKSetup.random_gp2D()
    ψ1 = Util.random_ψ(basis)
    ρ1 = DFTK.guess_density(basis)
    steps1 = maxiter
else
    model, basis, ψ1 = DFTKSetup.load_gp2D(path)
    ρ1 = DFTK.guess_density(basis)
end

callback = Util.PlotDensityCallback(;default_callback = DFTKSetup.get_default_callback(), save_img_path = "./test")

earcg_init, earcg_enhanced, ψ_neural_enh, ρ_neural_enh, nn_applications = DFTKSetup.enhanced_earcg(basis, tol, maxiter, unet; 
    ψ0 = ψ1, ρ0 = ρ1, 
    interval = 5,
    nn_tol = 5e-3,
    callback
    );

title = "iteration $(earcg_init.n_iter) post nn"
save = "test/hmp-$(earcg_init.n_iter)-nn.png"
Util.plot_density(ρ_neural_enh; title, save)

partial_solution = Util.ifft_to_2d_matrix(basis, earcg_init.ψ)
partial_grad = Util.ifft_to_2d_matrix(basis, earcg_init.grad)
title = "iteration $(earcg_init.n_iter) iterate re"
save = "test/hmp-$(earcg_init.n_iter)-iterate-re.png"
Util.plot_density(partial_solution[:,:,1]; title, save)
title = "iteration $(earcg_init.n_iter) iterate im"
save = "test/hmp-$(earcg_init.n_iter)-iterate-im.png"
Util.plot_density(partial_solution[:,:,2]; title, save)
title = "iteration $(earcg_init.n_iter) gradient re"
save = "test/hmp-$(earcg_init.n_iter)-gradient-re.png"
Util.plot_density(partial_grad[:,:,1]; title, save)
title = "iteration $(earcg_init.n_iter) gradient im"
save = "test/hmp-$(earcg_init.n_iter)-gradient-im.png"
Util.plot_density(partial_grad[:,:,2]; title, save)


partial_solution = Util.ifft_to_2d_matrix(basis, earcg_enhanced.ψ)
title = "iteration $(earcg_enhanced.n_iter + earcg_init.n_iter) iterate re"
save = "test/hmp-$(earcg_enhanced.n_iter + earcg_init.n_iter)-iterate-re.png"
Util.plot_density(partial_solution[:,:,1]; title, save)
title = "iteration $(earcg_enhanced.n_iter + earcg_init.n_iter) iterate im"
save = "test/hmp-$(earcg_enhanced.n_iter + earcg_init.n_iter)-iterate-im.png"
Util.plot_density(partial_solution[:,:,2]; title, save)

title = "iteration $(earcg_init.n_iter) gradient"
save = "test/hmp-$(earcg_init.n_iter)-gradient.png"
ρ_grad = Util.compute_ρ(basis, earcg_init.grad)
Util.plot_density(ρ_grad; title, save)




callback = Util.PlotDensityCallback(;default_callback = DFTKSetup.get_default_callback(), unet, save_img_path = "./test2")
# classical
earcg_classical = DFTKSetup.earcg(basis, tol, maxiter; 
    ψ0 = ψ1, ρ0 = ρ1, 
    callback
    );

# runtime_classical = Int(earcg_classical.runtime_ns)
# runtime_classical = Int(earcg_init.runtime_ns) + Int(earcg_enhanced.runtime_ns)