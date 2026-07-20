include("../src/QuantizedZonotopeVerification.jl")
using .QuantizedZonotopeVerification
using LazySets, LinearAlgebra, Statistics, Printf

NETWORKS = [(1,1),(1,5),(1,9),(2,2),(2,5),(2,9),(3,3),(3,5),(4,1),(4,5)]
Q_VALUES  = [6, 8, 10]
RADII     = [3, 6, 13, 19, 26]

# Property 4 from QEBVerif (fixed input center for all experiments)
X0 = [-0.2, -0.25, -0.5, -0.3, -0.44]

# Physical input bounds from QEBVerif (normalised domain), used to clip the input box
const IN_MIN  = [0.0,      -3.141593, -3.141593, 100.0,   0.0]
const IN_MAX  = [60760.0,   3.141593,  3.141593, 1200.0, 1200.0]
const IN_MEAN = [1.9791091e4, 0.0,     0.0,       650.0,  600.0]
const IN_VAR  = [60261.0,  6.28318530718, 6.28318530718, 1100.0, 1200.0]
const IN_LB   = [(IN_MIN[k] - IN_MEAN[k]) / IN_VAR[k] for k in 1:5]
const IN_UB   = [(IN_MAX[k] - IN_MEAN[k]) / IN_VAR[k] for k in 1:5]

function make_input_zonotope(X0, eps)
    # Clip to physical bounds (matches QEBVerif's np.clip before integer conversion)
    x_lb = clamp.(X0 .- eps, IN_LB, IN_UB)
    x_ub = clamp.(X0 .+ eps, IN_LB, IN_UB)
    center = (x_lb .+ x_ub) ./ 2
    radii  = (x_ub .- x_lb) ./ 2
    return Zonotope(center, diagm(radii))
end

# Warmup: JIT-compile every function used inside the timed blocks
println("Warming up JIT...")
let _net = load_acasxu_network_from_json("networks/acasxu_1_1.json"),
    _z   = make_input_zonotope(X0, 3/256.0)
    for _Q in [6, 8, 10], _ in 1:2
        _qc = Dict(:input => ("pm",8,8), :weights => ("pm",_Q,_Q-2),
                   :biases => ("pm",_Q,_Q-2), :activations => ("p",_Q,_Q-2))
        out_z = quantization_error_zonotope(_net, _qc, _z)
        _box  = overapproximate(out_z, Hyperrectangle)
        _     = mean((_box.center .+ _box.radius) .- (_box.center .- _box.radius))
        lb_c, ub_c = quantization_error_constrained_zonotope(_net, _qc, _z; full_constraints=true)
        _     = mean(ub_c .- lb_c)
        lb_l, ub_l = quantization_error_constrained_zonotope(_net, _qc, _z; full_constraints=false)
        _     = mean(ub_l .- lb_l)
    end
end
GC.gc(); GC.gc()
println("Warmup complete.")

csv_path = "results/data/benchmark_results.csv"
open(csv_path, "w") do f
    println(f, "net_i,net_j,Q,r,eps,odiff_zono,time_zono,odiff_con,time_con,odiff_con_light,time_con_light")
end

total   = length(NETWORKS) * length(Q_VALUES) * length(RADII)
n_done  = 0

for (i, j) in NETWORKS
    println("Loading acasxu_$(i)_$(j) ...")
    net = load_acasxu_network_from_json("networks/acasxu_$(i)_$(j).json")
    GC.gc()   # collect JSON-parsing garbage once per network, outside any timing

    for Q in Q_VALUES
        qc = Dict(
            :input       => ("pm", 8, 8),
            :weights     => ("pm", Q, Q - 2),
            :biases      => ("pm", Q, Q - 2),
            :activations => ("p",  Q, Q - 2),
        )
        GC.gc()   # collect between Q values; keeps per-experiment GC pressure low

        for r in RADII
            eps        = r / 256.0
            input_zono = make_input_zonotope(X0, eps)

            # Zonotope method
            od_z = NaN; t_z = NaN
            try
                t_z = @elapsed begin
                    out_z = quantization_error_zonotope(net, qc, input_zono)
                    box   = overapproximate(out_z, Hyperrectangle)
                    od_z  = mean((box.center .+ box.radius) .- (box.center .- box.radius))
                end
            catch e
                @warn "Zonotope failed ($i,$j) Q=$Q r=$r: $e"
            end

            # Constrained Zonotope (full — all oblique constraints)
            od_c = NaN; t_c = NaN
            try
                t_c = @elapsed begin
                    lb_c, ub_c = quantization_error_constrained_zonotope(net, qc, input_zono; full_constraints=true)
                    od_c = mean(ub_c .- lb_c)
                end
            catch e
                @warn "Constrained failed ($i,$j) Q=$Q r=$r: $e"
            end

            # Constrained Zonotope (light — box constraints only, no oblique)
            od_l = NaN; t_l = NaN
            try
                t_l = @elapsed begin
                    lb_l, ub_l = quantization_error_constrained_zonotope(net, qc, input_zono; full_constraints=false)
                    od_l = mean(ub_l .- lb_l)
                end
            catch e
                @warn "Constrained light failed ($i,$j) Q=$Q r=$r: $e"
            end

            global n_done += 1
            @printf("[%d/%d] (%d,%d) Q=%d r=%d: Zono=%.3f (%.3fs) | Con=%.3f (%.3fs) | Light=%.3f (%.3fs)\n",
                    n_done, total, i, j, Q, r, od_z, t_z, od_c, t_c, od_l, t_l)

            open(csv_path, "a") do f
                println(f, "$i,$j,$Q,$r,$eps,$od_z,$t_z,$od_c,$t_c,$od_l,$t_l")
            end
        end
    end
end

println("\nDone. Results saved to $csv_path")
