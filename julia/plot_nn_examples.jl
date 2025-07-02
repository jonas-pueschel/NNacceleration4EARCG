module PlotExample

include("./dftk_setup.jl")
include("./util.jl")
using .DFTKSetup
using .Util
using CSV
using DFTK
using DataFrames
using Printf

export generate_example_fig

path_figures = "./figs"
model_path = "./models/model_paper.pth"

unet = Util.TorchUNet(; model_path)

maxiter = 30_000
tol = 1e-8;

scfres_rcg1 = nothing
scfres_rcg2 = nothing
scfres_rcg3 = nothing

#discretization & earcg params
a = 20 
inner_it = 25

function generate_pngs(path, fig_name; plot_classical = false)
    global scfres_rcg1, scfres_rcg2, scfres_rcg3
    model, basis, ψ1 = DFTKSetup.load_gp2D(path; a)

    scfres_rcg1, scfres_rcg2, ψ_neural_enh, ρ_neural_enh, ~ = DFTKSetup.enhanced_earcg(basis, tol, maxiter, unet; 
        ψ0 = ψ1, inner_it);
    
    plot_density(scfres_rcg1.ρ; title = "$fig_name pre NN", save = "$path_figures/nn_examples/$fig_name-pre-NN.png")
    plot_density(ρ_neural_enh; title = "$fig_name post NN", save = "$path_figures/nn_examples/$fig_name-post-NN.png")
    plot_density(scfres_rcg2.ρ; title = "$fig_name converged", save = "$path_figures/nn_examples/$fig_name-converged.png")
    if (plot_classical)
        scfres_rcg3 = DFTKSetup.earcg(basis, tol, maxiter; ψ0 = ψ1, inner_it);
        plot_density(scfres_rcg3.ρ; title = "$fig_name classical", save = "$path_figures/nn_examples/$fig_name-classical.png")
    
        println("energy nn: $(scfres_rcg2.energies.total)")
        println("energy_cl: $(scfres_rcg3.energies.total)")
    end


end


function generate_example_fig(path, fig_name; plot_classical = nothing, gen_pngs = true)
    csv_path = joinpath(path, "statistics.csv")
    df = DataFrame(CSV.File(csv_path))

    steps_neural, steps_classical, impr_ρ = load_gp2D_performance(path; enhanced = true)
    κ, ω, v = load_gp2D_params(path)

    energy_neural = df.energy₃_enhanced[1]
    energy_classical = df.energy_classical[1]

    energy_string = ""
    same_e = true
    if (abs(energy_classical - energy_neural) > 1e-8)
        energy_string = "The energy of the enhanced iteration was " * (energy_neural < energy_classical ? "lower" : "higher") * " than the classical energy."
        same_e = false
    end
    plot_classical = isnothing(plot_classical) ? !same_e : plot_classical 

    perc = abs(steps_classical - steps_neural)/steps_classical * 100
    perc_st = @sprintf "%.2f" perc

    verb = steps_neural < steps_classical ? "reduced" : "increased"

    v1_str = @sprintf "%.3f" v[1]
    v2_str = "1" #TODO?
    ω_str = @sprintf "%.3f" ω
    ρ_str = @sprintf "%.1f" abs(impr_ρ * 100)

    verb_ρ = impr_ρ > 0 ? "improved" : "deteriorated"
    verb_steps = steps_neural < steps_classical ? "reduced" : "increased"
    
    density_string = steps_classical == 30000 ? "Since the classical algorithm did not converge, no estimation of the density improvement can be given." : "The density $verb_ρ by $ρ_str\\%."

    classical_steps_string = steps_classical == 30000 ? "did not converge within 30000 steps" : "took $steps_classical steps"
    enhanced_steps = steps_neural == 30000 ? "no convergence within " : ""

    caption = "Neural network enhanced convergence of energy-adaptive RCG for parameters \$v \\approx ($v1_str, $v2_str)\$, \$\\omega \\approx $ω_str\$,
    \$\\kappa = $κ\$. Enhancement resulted in $enhanced_steps$steps_neural total steps,
    while the classical algorithm $classical_steps_string. Thus, the amount of steps was $verb_steps by $perc_st\\%. 
    $density_string $energy_string"

    width = plot_classical ? "0.22" : "0.3"
    table_format = plot_classical ? "cccc"  : "ccc"
    classic_png = plot_classical ? "& \\includegraphics[width = $width\\textwidth, trim={3.8cm 0.2cm 3.8cm 0.8cm},clip]{$path_figures/nn_examples/$fig_name-classical.png}" : ""

    fig = """
    \\begin{figure}[H]
        \\centering
        \\begin{tabular}{$table_format}
            \\includegraphics[width = $width\\textwidth, trim={3.8cm 0.2cm 3.8cm 0.8cm},clip]{$path_figures/nn_examples/$fig_name-pre-NN.png} & 
            \\includegraphics[width = $width\\textwidth, trim={3.8cm 0.2cm 3.8cm 0.8cm},clip]{$path_figures/nn_examples/$fig_name-post-NN.png} & 
            \\includegraphics[width = $width\\textwidth, trim={3.8cm 0.2cm 3.8cm 0.8cm},clip]{$path_figures/nn_examples/$fig_name-converged.png} 
            $classic_png \\\\
            NN in & NN out & converged $(plot_classical ? "& classical" : "")
        \\end{tabular}
        \\caption{$caption}
        \\label{fig:nn-$fig_name}
    \\end{figure}
    """

    fname_full = joinpath(path_figures, "nn-$fig_name.tex")

    open(fname_full, "w") do file
        write(file, fig)
    end
    if (gen_pngs)
        generate_pngs(path, fig_name; plot_classical)
    end
end

end
