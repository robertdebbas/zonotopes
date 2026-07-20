using JuMP
using HiGHS
using ModelVerification
import ModelVerification: ReLU, Id, Network, Layer
using LazySets
using LinearAlgebra
using SparseArrays

# ---------------------------------------------------------------------------
# Internal helpers: build linear constraint rows in the full ε-space
# ---------------------------------------------------------------------------

# For each DNN neuron with ReLU activation, adds triangle-relaxation constraints
# (Case 4 only: l ≤ 0 < u → y_i ≥ 0 AND y_i ≥ x_i).
function _relu_constraint_rows(
    Z::Zonotope, lambda, mu, E,
    m_curr::Int, M::Int
)
    c = Z.center
    G = Z.generators
    n = length(c)

    A = zeros(Float64, 2 * n, M)
    b = Vector{Float64}(undef, 2 * n)
    nrow = 0

    for i in 1:n
        sum_abs = sum(abs.(G[i, :]))
        l = c[i] - sum_abs
        u = c[i] + sum_abs

        # Dead (u≤0) or exact (l≥0) neurons need no relaxation constraint
        u <= 0.0 || l >= 0.0 && continue

        λi = lambda[i, i]
        μi = mu[i]
        ei = E[i, i]

        # y_i ≥ 0
        nrow += 1
        A[nrow, 1:m_curr] .= λi .* G[i, :]
        A[nrow, m_curr + i] = ei
        b[nrow] = λi * c[i] + μi

        # y_i ≥ x_i  (lower facet of triangle relaxation)
        nrow += 1
        A[nrow, 1:m_curr] .= (λi - 1.0) .* G[i, :]
        A[nrow, m_curr + i] = ei
        b[nrow] = (λi - 1.0) * c[i] + μi
    end

    nrow == 0 && return zeros(Float64, 0, M), Float64[]
    return A[1:nrow, :], b[1:nrow]
end

# For each neuron in the quantized path (round-clamp), adds constraint rows
# encoding the complete convex-hull overapproximation per the paper spec.
# Boundaries match abstract_round_clamp: -0.5 and Cub+0.5.
#
# Case 3b: 4 constraints (y≥rl, y≤ru, two oblique hull edges)
# Case 4:  3 constraints (y≥0, y≤C, one oblique hull edge)
# Case 5:  4 constraints (y≥0, y≤ru, y≥x-½, oblique lower at upper edge)
# Case 6:  4 constraints (y≥rl, y≤C, y≤x+½, oblique upper at lower edge)
function _rc_constraint_rows(
    Z_hat::Zonotope, lambda_hat, mu_hat, E_hat,
    m_curr::Int, M::Int, Cub::Int; oblique::Bool=true
)
    c  = Z_hat.center
    G  = Z_hat.generators   # n × m_curr
    n  = length(c)
    Cf = float(Cub)

    A = zeros(Float64, 4 * n, M)
    b = Vector{Float64}(undef, 4 * n)
    nrow = 0

    for i in 1:n
        sum_abs = sum(abs.(G[i, :]))
        l  = c[i] - sum_abs
        u  = c[i] + sum_abs
        λi = lambda_hat[i, i]
        μi = mu_hat[i]
        ei = E_hat[i, i]
        vi = c[i]

        if u <= -0.5 || l >= Cf + 0.5
            # Cases 1, 2: constant — no constraints
            continue

        elseif l >= -0.5 && u <= Cf + 0.5
            # Case 3
            rl = round(l); ru = round(u)
            rl == ru && continue   # Case 3a: constant

            # Case 3b: λ̂=1, μ̂=0, ê=0.5 — 4 constraints
            nrow += 1
            A[nrow, 1:m_curr] .=  G[i, :]; A[nrow, m_curr+i] =  0.5
            b[nrow] = vi - rl                                    # y_hat ≥ rl

            nrow += 1
            A[nrow, 1:m_curr] .= -G[i, :]; A[nrow, m_curr+i] = -0.5
            b[nrow] = -vi + ru                                   # y_hat ≤ ru

            if oblique
                d1 = rl + 0.5 - l                               # upper hull at left edge
                if d1 > 0.05
                    lm1 = 1.0 / d1; mm1 = rl - lm1 * l
                    nrow += 1
                    A[nrow, 1:m_curr] .= (lm1 - 1.0) .* G[i, :]; A[nrow, m_curr+i] = -0.5
                    b[nrow] = (lm1 - 1.0) * vi + mm1
                end

                d2 = u + 0.5 - ru                               # lower hull at right edge
                if d2 > 0.05
                    lm2 = 1.0 / d2; mm2 = ru - lm2 * u
                    nrow += 1
                    A[nrow, 1:m_curr] .= (1.0 - lm2) .* G[i, :]; A[nrow, m_curr+i] = 0.5
                    b[nrow] = (1.0 - lm2) * vi - mm2
                end
            end

        elseif l <= -0.5 && u >= Cf + 0.5
            # Case 4: full range — 3 constraints
            lam1 = Cf / (u - 0.5)
            lam2 = Cf / (Cf - l - 0.5)

            nrow += 1
            A[nrow, 1:m_curr] .=  λi .* G[i, :]; A[nrow, m_curr+i] =  ei
            b[nrow] = λi * vi + μi                               # y_hat ≥ 0

            nrow += 1
            A[nrow, 1:m_curr] .= -λi .* G[i, :]; A[nrow, m_curr+i] = -ei
            b[nrow] = Cf - λi * vi - μi                          # y_hat ≤ C

            if oblique
                if Cf - l <= u   # λ̂ = λ̂₁ — add upper edge y ≤ λ̂₂*(x−l)
                    nrow += 1
                    A[nrow, 1:m_curr] .= (lam2 - λi) .* G[i, :]; A[nrow, m_curr+i] = -ei
                    b[nrow] = (lam2 - λi) * vi - μi - lam2 * l
                else             # λ̂ = λ̂₂ — add lower edge y ≥ λ̂₁*(x−½)
                    nrow += 1
                    A[nrow, 1:m_curr] .= (λi - lam1) .* G[i, :]; A[nrow, m_curr+i] = ei
                    b[nrow] = (λi - lam1) * vi + μi + lam1 / 2
                end
            end

        elseif l <= -0.5
            # Case 5: crosses lower threshold only — 3 or 4 constraints
            ru = round(u)

            nrow += 1
            A[nrow, 1:m_curr] .=  λi .* G[i, :]; A[nrow, m_curr+i] =  ei
            b[nrow] = λi * vi + μi                               # y_hat ≥ 0

            nrow += 1
            A[nrow, 1:m_curr] .= -λi .* G[i, :]; A[nrow, m_curr+i] = -ei
            b[nrow] = ru - λi * vi - μi                          # y_hat ≤ ru

            nrow += 1
            A[nrow, 1:m_curr] .= (λi - 1.0) .* G[i, :]; A[nrow, m_curr+i] = ei
            b[nrow] = (λi - 1.0) * vi + μi + 0.5                # y_hat ≥ x−½

            if oblique
                d_star = u - ru + 0.5
                if d_star > 0.05
                    lam_star = 1.0 / d_star
                    mu_star  = ru - lam_star * u
                    nrow += 1
                    A[nrow, 1:m_curr] .= (λi - lam_star) .* G[i, :]; A[nrow, m_curr+i] = ei
                    b[nrow] = (μi - mu_star) + (λi - lam_star) * vi
                end
            end

        else
            # Case 6: crosses upper threshold only — 3 or 4 constraints
            rl = round(l)

            nrow += 1
            A[nrow, 1:m_curr] .=  λi .* G[i, :]; A[nrow, m_curr+i] =  ei
            b[nrow] = λi * vi + μi - rl                          # y_hat ≥ rl

            nrow += 1
            A[nrow, 1:m_curr] .= -λi .* G[i, :]; A[nrow, m_curr+i] = -ei
            b[nrow] = Cf - λi * vi - μi                          # y_hat ≤ C

            nrow += 1
            A[nrow, 1:m_curr] .= (1.0 - λi) .* G[i, :]; A[nrow, m_curr+i] = -ei
            b[nrow] = (1.0 - λi) * vi - μi + 0.5                # y_hat ≤ x+½

            if oblique
                d_star = rl - l + 0.5
                if d_star > 0.05
                    lam_star = 1.0 / d_star
                    mu_star  = rl - lam_star * l
                    nrow += 1
                    A[nrow, 1:m_curr] .= (lam_star - λi) .* G[i, :]; A[nrow, m_curr+i] = -ei
                    b[nrow] = (mu_star - μi) + (lam_star - λi) * vi
                end
            end
        end
    end

    nrow == 0 && return zeros(Float64, 0, M), Float64[]
    return A[1:nrow, :], b[1:nrow]
end

# ---------------------------------------------------------------------------
# Concretization
# ---------------------------------------------------------------------------

"""
    concretize_error_zonotope(Z_tilda, A_con, b_con) -> (lower, upper)

Solve one LP per output dimension (min and max) to obtain tight bounds on the
quantization error zonotope `Z_tilda`, subject to linear constraints
`A_con * ε + b_con ≥ 0` and `ε ∈ [-1, 1]^M`.
"""
function concretize_error_zonotope(
    Z_tilda::Zonotope,
    A_con::AbstractMatrix{Float64},
    b_con::AbstractVector{Float64}
)
    c = Z_tilda.center
    G = Z_tilda.generators
    d = length(c)
    M = size(G, 2)
    num_con = size(A_con, 1)

    lower = zeros(Float64, d)
    upper = zeros(Float64, d)

    for k in 1:d
        g = vec(G[k, :])
        interval_rad = sum(abs.(g))

        for (minimize, store) in ((true, lower), (false, upper))
            model = Model(HiGHS.Optimizer)
            set_silent(model)
            set_time_limit_sec(model, 2.0)
            @variable(model, -1.0 <= ε[1:M] <= 1.0)
            if num_con > 0
                @constraint(model, A_con * ε .+ b_con .>= 0.0)
            end
            if minimize
                @objective(model, Min, g' * ε)
            else
                @objective(model, Max, g' * ε)
            end
            try
                optimize!(model)
                stat = termination_status(model)
                if stat == MOI.OPTIMAL || stat == MOI.LOCALLY_SOLVED
                    store[k] = c[k] + objective_value(model)
                else
                    store[k] = minimize ? c[k] - interval_rad : c[k] + interval_rad
                end
            catch
                store[k] = minimize ? c[k] - interval_rad : c[k] + interval_rad
            end
        end
    end

    return lower, upper
end

# ---------------------------------------------------------------------------
# Network propagation
# ---------------------------------------------------------------------------

"""
    propagate_dnn(net, quant_config, input_zonotope; constrained=false)

Propagate the real (unquantized) network using the abstract ReLU transformer.

Returns `(Z, A, b)`:
- `Z`:  output zonotope (descaled to real domain)
- `A`, `b`: constraint rows `A * ε + b ≥ 0` from ReLU triangle relaxations;
  empty matrices/vectors when `constrained=false`.
"""
function propagate_dnn(
    net::Network,
    quant_config,
    input_zonotope::Zonotope;
    constrained::Bool = false
)
    layers = net.layers
    N = length(layers)

    _, _, scaling_activations = quant_config[:activations]
    factor = 2^scaling_activations

    m0 = size(input_zonotope.generators, 2)
    M  = m0 + sum(
        length(layer.bias)
        for layer in layers if isa(layer.activation, ReLU);
        init = 0
    )

    Z = input_zonotope
    A_chunks = Vector{Matrix{Float64}}()
    b_chunks = Vector{Vector{Float64}}()
    m_curr = m0

    for i in 1:N
        first_factor = i == 1 ? factor : 1
        layer = layers[i]
        W = first_factor * layer.weights
        b = factor * layer.bias

        Z = Zonotope(W * Z.center + b, W * genmat(Z))

        if isa(layer.activation, ReLU)
            n_layer = length(layer.bias)
            lambda, mu, E = abstract_relu(Z; output=:triplet)

            if constrained
                A_relu, b_relu = _relu_constraint_rows(Z, lambda, mu, E, m_curr, M)
                if length(b_relu) > 0
                    push!(A_chunks, A_relu)
                    push!(b_chunks, b_relu)
                end
            end

            Z = Zonotope(lambda * Z.center + mu, hcat(lambda * genmat(Z), E))
            m_curr += n_layer
        end
    end

    Z = Zonotope(Z.center / factor, genmat(Z) / factor)
    A = isempty(A_chunks) ? zeros(Float64, 0, M) : vcat(A_chunks...)
    b = isempty(b_chunks) ? Float64[]             : vcat(b_chunks...)
    return Z, A, b
end

"""
    propagate_qnn(net, quant_config, input_zonotope; constrained=false, full_constraints=true)

Propagate the quantized network using the abstract round-clamp transformer.

Returns `(Z, A, b)`:
- `Z`:  output zonotope (descaled to real domain)
- `A`, `b`: constraint rows `A * ε + b ≥ 0` from the round-clamp convex-hull;
  empty matrices/vectors when `constrained=false`.
  `full_constraints=false` omits the oblique hull edges (box constraints only).
"""
function propagate_qnn(
    net::Network,
    quant_config,
    input_zonotope::Zonotope;
    constrained::Bool      = false,
    full_constraints::Bool = true
)
    layers = net.layers
    N = length(layers)

    _, bits_input,       scaling_input       = quant_config[:input]
    _, bits_weights,     scaling_weights     = quant_config[:weights]
    _, bits_biases,      scaling_biases      = quant_config[:biases]
    _, bits_activations, scaling_activations = quant_config[:activations]
    Cub    = 2^bits_activations - 1
    factor = 2^scaling_activations

    m0 = size(input_zonotope.generators, 2)
    M  = m0 + sum(
        length(layer.bias)
        for layer in layers if isa(layer.activation, ReLU);
        init = 0
    )

    Z = quantize_zonotope(input_zonotope, scaling_input, bits_input)
    A_chunks = Vector{Matrix{Float64}}()
    b_chunks = Vector{Vector{Float64}}()
    m_curr = m0

    for i in 1:N
        first_factor = i == 1 ? factor : 1
        layer = layers[i]
        W_hat = first_factor * quantize_tensor(layer.weights, scaling_weights, bits_weights)
        b_hat = factor * quantize_tensor(layer.bias, scaling_biases, bits_biases)

        Z = Zonotope(W_hat * Z.center + b_hat, W_hat * genmat(Z))

        if isa(layer.activation, ReLU)
            n_layer = length(layer.bias)
            lambda_hat, mu_hat, E_hat = abstract_round_clamp(Z, Cub; output=:triplet)

            if constrained
                A_rc, b_rc = _rc_constraint_rows(
                    Z, lambda_hat, mu_hat, E_hat, m_curr, M, Cub;
                    oblique = full_constraints
                )
                if length(b_rc) > 0
                    push!(A_chunks, A_rc)
                    push!(b_chunks, b_rc)
                end
            end

            Z = Zonotope(lambda_hat * Z.center + mu_hat, hcat(lambda_hat * genmat(Z), E_hat))
            m_curr += n_layer
        end
    end

    Z = Zonotope(Z.center / factor, genmat(Z) / factor)
    A = isempty(A_chunks) ? zeros(Float64, 0, M) : vcat(A_chunks...)
    b = isempty(b_chunks) ? Float64[]             : vcat(b_chunks...)
    return Z, A, b
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    quantization_error_zonotope(net, quant_config, input_zonotope;
                                constrained=false, full_constraints=true)

Compute the quantization error between the QNN and DNN by propagating each
network separately and subtracting their output zonotopes.

- `constrained=false`: returns the error `Zonotope` (same as the previous
  unconstrained formulation).
- `constrained=true`: additionally accumulates ReLU and round-clamp linear
  constraints during propagation, then solves one LP per output dimension to
  obtain tight `(lower, upper)` bound vectors.
- `full_constraints=true` (only used when `constrained=true`): includes
  oblique hull edges for the round-clamp; `false` uses box constraints only.
"""
function quantization_error_zonotope(
    net::Network,
    quant_config,
    input_zonotope::Zonotope;
    constrained::Bool      = false,
    full_constraints::Bool = true
)
    Z_dnn, A_dnn, b_dnn = propagate_dnn(net, quant_config, input_zonotope;
                                         constrained)
    Z_qnn, A_qnn, b_qnn = propagate_qnn(net, quant_config, input_zonotope;
                                         constrained, full_constraints)

    Z_tilda = Zonotope(Z_qnn.center - Z_dnn.center, genmat(Z_qnn) - genmat(Z_dnn))

    constrained || return Z_tilda

    A_con = sparse(vcat(A_dnn, A_qnn))
    b_con = vcat(b_dnn, b_qnn)
    return concretize_error_zonotope(Z_tilda, A_con, b_con)
end

# Backward-compatibility alias
quantization_error_constrained_zonotope(net, qc, z; full_constraints=true, method=:lp) =
    quantization_error_zonotope(net, qc, z; constrained=true, full_constraints)
