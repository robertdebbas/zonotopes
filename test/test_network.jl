using Test
using LazySets
using LinearAlgebra
using Random

@testset "Quantization Error Zonotope" begin

    # 2 inputs, one hidden layer of 3 ReLU neurons, 1 output
    net = Network([
        Layer([ 1.0   0.5;
               -0.3   0.8;
                0.7  -0.6],
              [0.1, -0.2, 0.3],
              ReLU()),
        Layer([0.4  -0.7  1.2], [0.5], Id())
    ])

    Q = 6
    quant_config = Dict(
        :input       => ("pm", 8, 8),
        :weights     => ("pm", Q, Q - 2),
        :biases      => ("pm", Q, Q - 2),
        :activations => ("p",  Q, Q - 2)
    )

    center = [0.1, -0.2]
    radius = 0.05
    input_zono = Zonotope(center, radius * Matrix(1.0I, 2, 2))

    # Plain zonotope
    error_zono = quantization_error_zonotope(net, quant_config, input_zono)
    @test error_zono isa Zonotope
    @test length(error_zono.center) == 1

    box  = overapproximate(error_zono, Hyperrectangle)
    lb_z = box.center .- box.radius
    ub_z = box.center .+ box.radius

    # Constrained variants: light (box constraints) and complete (convex hull)
    lb_l, ub_l = quantization_error_zonotope(net, quant_config, input_zono;
                     constrained=true, full_constraints=false, diagonal_constraints=false)
    lb_c, ub_c = quantization_error_zonotope(net, quant_config, input_zono; constrained=true)

    # More constraints can only tighten the bounds
    tol = 1e-6
    @test all(lb_z .- tol .<= lb_l) && all(ub_l .<= ub_z .+ tol)
    @test all(lb_l .- tol .<= lb_c) && all(ub_c .<= ub_l .+ tol)

    # Soundness: every sampled concrete error lies inside all three bounds
    Random.seed!(0)
    sound = true
    for _ in 1:2000
        x = center .+ radius .* (2 .* rand(2) .- 1)
        err = propagate_quantized(net, quant_config, x) .- propagate(net, x)
        sound &= all(lb_z .- tol .<= err .<= ub_z .+ tol)
        sound &= all(lb_l .- tol .<= err .<= ub_l .+ tol)
        sound &= all(lb_c .- tol .<= err .<= ub_c .+ tol)
    end
    @test sound

end
