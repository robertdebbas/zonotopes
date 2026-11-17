# Quantization error bounds for a small synthetic network.
#
# Run from the repository root:
#     julia --project=. examples/small_example.jl

include(joinpath(@__DIR__, "..", "src", "QuantizedZonotopeVerification.jl"))
using .QuantizedZonotopeVerification
using LazySets, LinearAlgebra

# 2 inputs, one hidden layer of 3 ReLU neurons, 1 output
net = Network([
    Layer([1.0 0.5; -0.3 0.8; 0.7 -0.6], [0.1, -0.2, 0.3], ReLU()),
    Layer([0.4 -0.7 1.2], [0.5], Id()),
])

# (signedness tag, total bits, fractional bits)
quant_config = Dict(
    :input       => ("pm", 8, 8),
    :weights     => ("pm", 6, 4),
    :biases      => ("pm", 6, 4),
    :activations => ("p",  6, 4),
)

# Input box: center ± 0.05 in each dimension
input_zono = Zonotope([0.1, -0.2], 0.05 * Matrix(1.0I, 2, 2))

# Plain zonotope: returns the error zonotope
error_zono = quantization_error_zonotope(net, quant_config, input_zono)
box = overapproximate(error_zono, Hyperrectangle)
println("zonotope:    ", low(box), " ", high(box))

# Constrained zonotope (light variant): returns lower and upper bound vectors
lb, ub = quantization_error_zonotope(net, quant_config, input_zono;
             constrained=true, full_constraints=false, diagonal_constraints=false)
println("constrained: ", lb, " ", ub)
