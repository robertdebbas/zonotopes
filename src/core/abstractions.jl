using LazySets
using LinearAlgebra

"""
    abstract_relu(Z::Zonotope; output=:zonotope) -> Zonotope | (lambda, mu, E)

Abstract ReLU transformer over a zonotope using the standard triangle relaxation.

- `output=:zonotope` — returns a Zonotope with n new error generators appended
- `output=:triplet`  — returns (lambda, mu, E): diagonal n×n matrices and n-vector
"""
function abstract_relu(Z::Zonotope; output::Symbol=:zonotope)
    c = Z.center
    G = Z.generators
    n = length(c)

    lambda = zeros(Float64, n, n)
    mu     = zeros(Float64, n)
    E      = zeros(Float64, n, n)

    for i in 1:n
        sum_abs = sum(abs.(G[i, :]))
        l = c[i] - sum_abs
        u = c[i] + sum_abs

        if u <= 0.0
            # Dead neuron — lambda=0, mu=0, E=0 (already)
        elseif l >= 0.0
            lambda[i, i] = 1.0
        else
            a = u / (u - l)
            b = -u * l / (u - l)
            lambda[i, i] = a
            mu[i]        = b / 2
            E[i, i]      = b / 2
        end
    end

    output == :triplet && return lambda, mu, E
    return Zonotope(lambda * c + mu, hcat(lambda * G, E))
end

"""
    abstract_round_clamp(Z::Zonotope, Cub::Int; output=:zonotope) -> Zonotope | (lambda, mu, E)

Abstract transformer for rc(x) = clamp(round(x), 0, Cub) over a zonotope.
Case boundaries match the paper spec: -0.5 and Cub+0.5.

Cases (based on interval [l, u] of each neuron):
  1. u ≤ -0.5          → constant 0
  2. l ≥ Cub+0.5       → constant Cub
  3. -0.5 ≤ l, u ≤ Cub+0.5  → identity ± 0.5 rounding error (3a: constant if round(l)=round(u))
  4. l ≤ -0.5, u ≥ Cub+0.5  → full-range affine relaxation
  5. l ≤ -0.5, u ≤ Cub+0.5  → crosses lower threshold only
  6. l ≥ -0.5, u ≥ Cub+0.5  → crosses upper threshold only

- `output=:zonotope` — returns a Zonotope with n new error generators appended
- `output=:triplet`  — returns (lambda, mu, E): diagonal n×n matrices and n-vector
"""
function abstract_round_clamp(Z::Zonotope{Float64}, Cub::Int64; output::Symbol=:zonotope)
    c  = Z.center
    G  = Z.generators
    n  = length(c)
    Cf = float(Cub)

    lambda = zeros(Float64, n, n)
    mu     = zeros(Float64, n)
    E      = zeros(Float64, n, n)

    for i in 1:n
        sum_abs = sum(abs.(G[i, :]))
        l = c[i] - sum_abs
        u = c[i] + sum_abs

        if u <= -0.5
            # Case 1: fully below — output = 0 (lambda=0, mu=0 by default)

        elseif l >= Cf + 0.5
            # Case 2: fully above — output = Cub
            mu[i] = Cf

        elseif l >= -0.5 && u <= Cf + 0.5
            # Case 3: within clamp bounds — abstract rounding only
            rl = round(l)
            ru = round(u)
            if rl == ru
                # Case 3a: constant — no rounding threshold crossed
                mu[i] = rl
            else
                # Case 3b: identity with ±0.5 rounding error
                lambda[i, i] = 1.0
                E[i, i]      = 0.5
            end

        elseif l <= -0.5 && u >= Cf + 0.5
            # Case 4: full range — both lower and upper thresholds crossed
            lam1 = Cf / (u - 0.5)
            lam2 = Cf / (Cf - l - 0.5)
            a    = Cf - l <= u ? lam1 : lam2
            b1   = max(-a * l, Cf - a * (Cf - 0.5))
            b2   = min(-0.5 * a, Cf - a * u)
            lambda[i, i] = a
            mu[i]        = (b1 + b2) / 2
            E[i, i]      = (b1 - b2) / 2

        elseif l <= -0.5
            # Case 5: crosses lower threshold only (u < Cub+0.5 by elimination)
            ru = round(u)
            a  = ru / (ru - l - 0.5)
            b1 = -a * l          # = -λ̂ l
            b2 = -a / 2          # = -λ̂/2
            lambda[i, i] = a
            mu[i]        = (b1 + b2) / 2
            E[i, i]      = (b1 - b2) / 2

        else
            # Case 6: crosses upper threshold only (l >= -0.5 by elimination)
            rl = round(l)
            a  = (Cf - rl) / (u - rl - 0.5)
            b1 = Cf - a * (Cf - 0.5)
            b2 = Cf - a * u
            lambda[i, i] = a
            mu[i]        = (b1 + b2) / 2
            E[i, i]      = (b1 - b2) / 2
        end
    end

    output == :triplet && return lambda, mu, E
    return Zonotope(lambda * c + mu, hcat(lambda * G, E))
end
