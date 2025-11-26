using DFTK
using LinearAlgebra
using BSON
using Plots

include("./dftk_setup.jl")
include("./util.jl")
using .Util

using .DFTKSetup

maxiter = 20_000
tol1 = 1e-3
tol = 1e-8;

# path to BSON file containing an initial value ψ1.
# when left empty, a random initial guess will be used
path_psi = ""

path_imgs = "./figs/heatmaps_convergence"
path_figures = "./figs"
model_path = "./models/model_paper.pth"
unet = Util.TorchUNet(; model_path)

model, basis = DFTKSetup.gp2D_setup(Ecut = 100, a = 20, κ = 1000, ω = 1.4, v = (1.1, 1.0));



mutable struct PlotDensityCallback2
    default_callback
    indices
    iter
    function PlotDensityCallback2(;default_callback = DFTKSetup.get_default_callback(), indices = [5,15,60,150,500,1500,3500])
        new(default_callback, indices, 0)
    end
end

function (callback::PlotDensityCallback2)(info)
    callback.iter += 1
    idxs = findall(x->x==callback.iter, callback.indices)
    if  (length(idxs) == 1 || info.stage == :finalize )
        ρ = hasproperty(info, :ρout) ? info.ρout : info.ρ  
        id = length(idxs) == 1 ? idxs[1] : length(callback.indices) + 1
        save = joinpath(path_imgs, "hmp-enh-$(id).png")
        title = "iteration $(callback.iter)"
        Util.plot_density(ρ ;title, save)
    end
    callback.default_callback(info)    
end

if (path_psi == "")
    ψ1 = Util.random_ψ(basis)
else
    BSON.@load path_psi ψ1
end
ρ1 = guess_density(basis)

step_callback = PlotDensityCallback2(; default_callback = Util.default_callback)
scfres_rcg5 = DFTKSetup.enhanced_earcg(basis, tol, maxiter, unet; 
    ψ0 = ψ1, ρ0 = ρ1, 
    callback = step_callback,
    );

iters = [step_callback.indices..., scfres_rcg5[1].n_iter + scfres_rcg5[2].n_iter]


ψ_neural, norm_nn = unet(basis, scfres_rcg5[1].ψ, scfres_rcg5[1].grad)

ρ_neural = Util.compute_ρ(basis, ψ_neural)

Util.plot_density(ρ_neural; title = "Post NN", save = "$path_imgs/hmp-enh-post.png")
n_iter1 = scfres_rcg5[1].n_iter
Util.plot_density(scfres_rcg5[1].ρ; title = "Pre NN ($n_iter1)", save = "$path_imgs/hmp-enh-pre.png")

function create_figure(iters)
    st = """
    \\begin{figure}[H]
        \\begin{tabular}{cccc}
            \\includegraphics[width = 0.22\\textwidth, trim={3.8cm 0.2cm 3.8cm 0.7cm},clip]{$path_imgs/hmp-enh-1.png} & 
            \\includegraphics[width = 0.22\\textwidth, trim={3.8cm 0.2cm 3.8cm 0.7cm},clip]{$path_imgs/hmp-enh-2.png} & 
            \\includegraphics[width = 0.22\\textwidth, trim={3.8cm 0.2cm 3.8cm 0.7cm},clip]{$path_imgs/hmp-enh-3.png} & 
            \\includegraphics[width = 0.22\\textwidth, trim={3.8cm 0.2cm 3.8cm 0.7cm},clip]{$path_imgs/hmp-enh-pre.png} \\\\
            Iteration $(iters[1]) & Iteration $(iters[2]) & Iteration $(iters[3]) & Pre NN (Iter. $n_iter1)  \\\\[0.2cm]
            \\includegraphics[width = 0.22\\textwidth, trim={3.8cm 0.2cm 3.8cm 0.7cm},clip]{$path_imgs/hmp-enh-post.png} &
            \\includegraphics[width = 0.22\\textwidth, trim={3.8cm 0.2cm 3.8cm 0.7cm},clip]{$path_imgs/hmp-enh-5.png} &
            \\includegraphics[width = 0.22\\textwidth, trim={3.8cm 0.2cm 3.8cm 0.7cm},clip]{$path_imgs/hmp-enh-7.png} &
            \\includegraphics[width = 0.22\\textwidth, trim={3.8cm 0.2cm 3.8cm 0.7cm},clip]{$path_imgs/hmp-enh-8.png} \\\\
            Post NN (Iter. $n_iter1)  & Iteration $(iters[5]) & Iteration $(iters[7]) & Iteration $(iters[8])
        \\end{tabular}
        \\caption{Density plots from the neural network enhanced EARCG iteration for parameters \$h = 20, v = (1.1, 1), \\omega = 1.4\$ and \$\\kappa = 1000\$. The final vortex configuration is reached after around \$$(iters[5])\$ iterations, the correct orientation is only reached after \$$(iters[end])\$ iterations.}
        \\label{fig:enhanced-convergence}
    \\end{figure}
    """
    fname_full = joinpath(path_figures, "heatmaps_convergence_enhanced.tex")
    open(fname_full, "w") do file
        write(file, st)
    end
end

create_figure(iters)