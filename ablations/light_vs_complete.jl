"""
Ablation: Constrained Zonotope Light vs Complete.

Light:    box constraints only (y ≥ 0, y ≤ C^ub per neuron) — no oblique hull edges.
Complete: full convex-hull constraints including oblique edges and slope-1 rounding bounds.

Both tighten the plain Mink zonotope via constrained LP.  This ablation quantifies
how much the extra constraints in Complete help over Light.

Run from the project root:
    julia --project=. ablations/light_vs_complete.jl
"""

include("../src/QuantizedZonotopeVerification.jl")
using .QuantizedZonotopeVerification
using LazySets, LinearAlgebra, Statistics, Printf

NETWORKS = [(1,1),(1,5),(1,9),(2,2),(2,5),(2,9),(3,3),(3,5),(4,1),(4,5)]
Q_VALUES  = [6, 8, 10]
RADII     = [3, 6, 13, 19, 26]
X0        = [-0.2, -0.25, -0.5, -0.3, -0.44]

const IN_MIN  = [0.0,      -3.141593, -3.141593, 100.0,   0.0]
const IN_MAX  = [60760.0,   3.141593,  3.141593, 1200.0, 1200.0]
const IN_MEAN = [1.9791091e4, 0.0,     0.0,       650.0,  600.0]
const IN_VAR  = [60261.0,  6.28318530718, 6.28318530718, 1100.0, 1200.0]
const IN_LB   = [(IN_MIN[k] - IN_MEAN[k]) / IN_VAR[k] for k in 1:5]
const IN_UB   = [(IN_MAX[k] - IN_MEAN[k]) / IN_VAR[k] for k in 1:5]

function make_input_zonotope(X0, eps)
    x_lb = clamp.(X0 .- eps, IN_LB, IN_UB)
    x_ub = clamp.(X0 .+ eps, IN_LB, IN_UB)
    Zonotope((x_lb .+ x_ub) ./ 2, diagm((x_ub .- x_lb) ./ 2))
end

println("Warming up...")
let _net = load_acasxu_network_from_json("networks/acasxu_1_1.json"),
    _z   = make_input_zonotope(X0, 3/256.0),
    _qc  = Dict(:input=>("pm",8,8),:weights=>("pm",6,4),:biases=>("pm",6,4),:activations=>("p",6,4))
    quantization_error_zonotope(_net, _qc, _z)
    quantization_error_constrained_zonotope(_net, _qc, _z; full_constraints=false, diagonal_constraints=false)
    quantization_error_constrained_zonotope(_net, _qc, _z; full_constraints=true,  diagonal_constraints=true)
end
println("Warmup done.\n")

println(@sprintf("%-14s %4s %4s | %10s %10s %10s | %8s %8s | %8s %8s",
    "Net", "Q", "r",
    "OD_zono", "OD_light", "OD_complete",
    "t_light", "t_compl",
    "Δ(zono-lt)", "Δ(lt-compl)"))
println("-" ^ 108)

let sum_zono = 0.0, sum_light = 0.0, sum_compl = 0.0,
    sum_t_light = 0.0, sum_t_compl = 0.0, n = 0

    for (i, j) in NETWORKS
        net = load_acasxu_network_from_json("networks/acasxu_$(i)_$(j).json")
        for Q in Q_VALUES
            qc = Dict(:input=>("pm",8,8), :weights=>("pm",Q,Q-2),
                      :biases=>("pm",Q,Q-2), :activations=>("p",Q,Q-2))
            for r in RADII
                z = make_input_zonotope(X0, r/256.0)

                out_z = quantization_error_zonotope(net, qc, z)
                box   = overapproximate(out_z, Hyperrectangle)
                od_z  = mean(2 .* box.radius)

                t_l = @elapsed begin
                    lb_l, ub_l = quantization_error_constrained_zonotope(net, qc, z;
                        full_constraints=false, diagonal_constraints=false)
                    od_l = mean(ub_l .- lb_l)
                end

                t_c = @elapsed begin
                    lb_c, ub_c = quantization_error_constrained_zonotope(net, qc, z;
                        full_constraints=true, diagonal_constraints=true)
                    od_c = mean(ub_c .- lb_c)
                end

                sum_zono += od_z; sum_light += od_l; sum_compl += od_c
                sum_t_light += t_l; sum_t_compl += t_c
                n += 1

                println(@sprintf("(%d,%d) Q=%2d r=%2d | %10.4f %10.4f %10.4f | %8.3fs %8.3fs | %+8.4f %+8.4f",
                    i, j, Q, r, od_z, od_l, od_c, t_l, t_c,
                    od_z - od_l, od_l - od_c))
            end
        end
        println()
    end

    println("=" ^ 108)
    @printf("AVERAGE over %d experiments:\n\n", n)
    @printf("  avg O_Diff Zonotope (Mink, unconstrained): %.4f\n", sum_zono  / n)
    @printf("  avg O_Diff Constrained Light:              %.4f  (%.2f%% of Zono)\n",
        sum_light / n, 100.0 * sum_light / sum_zono)
    @printf("  avg O_Diff Constrained Complete:           %.4f  (%.2f%% of Zono)\n",
        sum_compl / n, 100.0 * sum_compl / sum_zono)
    @printf("\n  avg gain Zono → Light:     %+.4f  (%.2f%% tighter)\n",
        (sum_zono - sum_light) / n, 100.0 * (sum_zono - sum_light) / sum_zono)
    @printf("  avg gain Light → Complete: %+.4f  (%.2f%% tighter)\n",
        (sum_light - sum_compl) / n, 100.0 * (sum_light - sum_compl) / sum_zono)
    @printf("\n  avg time Light:    %.4f s\n", sum_t_light / n)
    @printf("  avg time Complete: %.4f s\n", sum_t_compl / n)
    @printf("  Complete overhead vs Light: %.2f×\n", sum_t_compl / sum_t_light)
end
