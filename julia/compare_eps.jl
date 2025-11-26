include("compare_algorithms.jl")

load_dir = "./comparisons/comp-paper"
interval = 5

for i = 0:8
    nn_tol = 10^(-1 - i * 0.25)
    result_dir = "./comparisons/comp-nntol-$i-test"
    compare_algorithms(interval, nn_tol, result_dir, load_dir)
end