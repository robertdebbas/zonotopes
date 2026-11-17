"""
Ablation: Mink-sum vs Shared-ε^(0) error zonotope.

Mink (main method): G_tilda = [G_qnn | -G_dnn]  — fully independent symbol spaces.
Shared:             G_tilda = [G_qnn[:,1:d] - G_dnn[:,1:d] | G_qnn[:,d+1:end] | -G_dnn[:,d+1:end]]
                    Uses abstract_round_clamp for QNN input so ε^(0) is genuinely shared.

Run from the project root:
    julia --project=. ablations/mink_vs_shared.jl
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

odiff(G) = mean(2 .* sum(abs.(G), dims=2))

# Propagate QNN with abstract_round_clamp input (shared ε^(0) variant).
function propagate_qnn_shared(net, quant_config, input_zonotope)
    layers = net.layers; N = length(layers)
    _, bits_input, scaling_input       = quant_config[:input]
    _, bits_weights, scaling_weights   = quant_config[:weights]
    _, bits_biases, scaling_biases     = quant_config[:biases]
    _, bits_activations, scaling_activations = quant_config[:activations]
    Cub = 2^bits_activations - 1; factor = 2^scaling_activations

    s_in   = float(2^scaling_input)
    off_in = float(2^(bits_input - 1))
    Cub_in = 2^bits_input - 1

    Z_scaled = Zonotope(s_in .* input_zonotope.center .+ off_in,
                        s_in .* genmat(input_zonotope))
    lambda_in, mu_in, E_in = abstract_round_clamp(Z_scaled, Cub_in; output=:triplet)
    Z = Zonotope(
        (lambda_in * Z_scaled.center + mu_in .- off_in) ./ s_in,
        hcat(lambda_in * genmat(Z_scaled), E_in) ./ s_in
    )

    for i in 1:N
        layer = layers[i]
        W_hat = (i == 1 ? factor : 1) * quantize_tensor(layer.weights, scaling_weights, bits_weights)
        b_hat = factor * quantize_tensor(layer.bias, scaling_biases, bits_biases)
        Z = Zonotope(W_hat * Z.center + b_hat, W_hat * genmat(Z))
        if isa(layer.activation, ReLU)
            lambda_hat, mu_hat, E_hat = abstract_round_clamp(Z, Cub; output=:triplet)
            Z = Zonotope(lambda_hat * Z.center + mu_hat, hcat(lambda_hat * genmat(Z), E_hat))
        end
    end
    return Zonotope(Z.center / factor, genmat(Z) / factor)
end

println("Warming up...")
let _net = load_acasxu_network_from_json("networks/acasxu_1_1.json"),
    _z   = make_input_zonotope(X0, 3/256.0),
    _qc  = Dict(:input=>("pm",8,8),:weights=>("pm",6,4),:biases=>("pm",6,4),:activations=>("p",6,4))
    propagate_dnn(_net, _qc, _z)
    propagate_qnn(_net, _qc, _z)
    propagate_qnn_shared(_net, _qc, _z)
end
println("Warmup done.\n")

println(@sprintf("%-14s %4s %4s | %10s %10s %10s %10s | %10s %10s | %+10s",
    "Net", "Q", "r", "|cancel|", "|ε_q|", "|qnn_act|", "|dnn_act|",
    "OD_mink", "OD_shared", "Δ(mink-shr)"))
println("-" ^ 110)

let sum_cancel    = 0.0, sum_qnn_input = 0.0, sum_dnn_input = 0.0,
    sum_eps_q     = 0.0, sum_qnn_act   = 0.0, sum_dnn_act   = 0.0,
    sum_od_mink   = 0.0, sum_od_shared = 0.0,
    sum_t_mink    = 0.0, sum_t_shared  = 0.0, n = 0

    for (i, j) in NETWORKS
        net = load_acasxu_network_from_json("networks/acasxu_$(i)_$(j).json")
        for Q in Q_VALUES
            qc = Dict(:input=>("pm",8,8), :weights=>("pm",Q,Q-2),
                      :biases=>("pm",Q,Q-2), :activations=>("p",Q,Q-2))
            for r in RADII
                z  = make_input_zonotope(X0, r/256.0)
                d  = size(z.generators, 2)

                Z_dnn, _, _ = propagate_dnn(net, qc, z)
                t_mink   = @elapsed Z_qnn_mink, _, _ = propagate_qnn(net, qc, z)
                t_shared = @elapsed Z_qnn_shr = propagate_qnn_shared(net, qc, z)

                G_dnn      = genmat(Z_dnn)
                G_qnn_mink = genmat(Z_qnn_mink)
                G_qnn_shr  = genmat(Z_qnn_shr)   # has 2d input columns

                # Mink: [G_qnn_mink | -G_dnn]  (both M columns)
                G_mink   = hcat(G_qnn_mink, -G_dnn)
                # Shared: cancel first d input columns, keep ε_q and act columns separate
                G_shared = hcat(G_qnn_shr[:, 1:d] - G_dnn[:, 1:d],
                                G_qnn_shr[:, d+1:end],
                                -G_dnn[:, d+1:end])

                od_mink   = odiff(G_mink)
                od_shared = odiff(G_shared)

                n_cancel  = sum(abs.(G_qnn_shr[:, 1:d] .- G_dnn[:, 1:d]))
                n_eps_q   = sum(abs.(G_qnn_shr[:, d+1:2d]))
                n_qnn_act = sum(abs.(G_qnn_shr[:, 2d+1:end]))
                n_dnn_act = sum(abs.(G_dnn[:, d+1:end]))

                sum_cancel    += n_cancel;    sum_eps_q   += n_eps_q
                sum_qnn_act   += n_qnn_act;   sum_dnn_act += n_dnn_act
                sum_qnn_input += sum(abs.(G_qnn_mink[:, 1:d]))
                sum_dnn_input += sum(abs.(G_dnn[:, 1:d]))
                sum_od_mink   += od_mink;     sum_od_shared += od_shared
                sum_t_mink    += t_mink;      sum_t_shared  += t_shared
                n += 1

                println(@sprintf("(%d,%d) Q=%2d r=%2d | %10.4f %10.4f %10.4f %10.4f | %10.4f %10.4f | %+10.5f",
                    i, j, Q, r,
                    n_cancel, n_eps_q, n_qnn_act, n_dnn_act,
                    od_mink, od_shared, od_mink - od_shared))
            end
        end
        println()
    end

    println("=" ^ 110)
    @printf("AVERAGE over %d experiments:\n\n", n)
    @printf("  === Mink decomposition (independent ε^(0)) ===\n")
    @printf("    [qnn_input]  avg |G_qnn[:,1:d]|                  = %.6f\n", sum_qnn_input / n)
    @printf("    [dnn_input]  avg |G_dnn[:,1:d]|                  = %.6f\n", sum_dnn_input / n)
    @printf("    [qnn_act]    avg |G_qnn[:,d+1:end]|  (Mink prop) = %.4f\n", sum_qnn_act   / n)
    @printf("    [dnn_act]    avg |G_dnn[:,d+1:end]|              = %.4f\n", sum_dnn_act   / n)
    @printf("\n  === Shared ε^(0) decomposition ===\n")
    @printf("    [cancel]     avg |G_qnn[:,1:d] - G_dnn[:,1:d]|  = %.6f\n", sum_cancel  / n)
    @printf("    [ε_q]        avg |G_qnn[:,d+1:2d]|              = %.6f\n", sum_eps_q   / n)
    @printf("    [qnn_act]    avg |G_qnn[:,2d+1:end]|            = %.4f\n", sum_qnn_act / n)
    @printf("    [dnn_act]    avg |G_dnn[:,d+1:end]|             = %.4f\n", sum_dnn_act / n)
    @printf("\n  avg O_Diff Mink:                %.6f\n", sum_od_mink   / n)
    @printf("  avg O_Diff Shared:              %.6f\n", sum_od_shared / n)
    @printf("  avg Δ(mink - shared):           %+.6f  (%.5f%%)\n",
        (sum_od_mink - sum_od_shared) / n,
        100.0 * (sum_od_mink - sum_od_shared) / sum_od_mink)
    @printf("\n  Conclusion: shared ε^(0) saves %.5f on average (%.4f%% of Mink).\n",
        (sum_od_mink - sum_od_shared) / n,
        100.0 * (sum_od_mink - sum_od_shared) / sum_od_mink)
    @printf("  ε_q contribution is %.1f× smaller than the cancellation benefit.\n",
        sum_cancel / max(sum_eps_q, 1e-12))
    @printf("\n  === Timing (QNN propagation only) ===\n")
    @printf("  avg time Mink   (quantize_zonotope input):        %.5f s\n", sum_t_mink   / n)
    @printf("  avg time Shared (abstract_round_clamp input):     %.5f s\n", sum_t_shared / n)
    @printf("  Shared overhead: %.2f×\n", sum_t_shared / sum_t_mink)
end
