using Test
using LazySets

@testset "Zonotope Abstractions" begin

    @testset "abstract_relu" begin
        Z = Zonotope([1.0, -1.0], [1.0 0.0; 0.0 1.0])
        Z_relu = abstract_relu(Z)

        # Check basic properties
        @test isa(Z_relu, Zonotope)
        @test length(Z_relu.center) == 2
        @test size(Z_relu.generators, 2) == 4  # m + n = 2 + 2

        # Example test: upper bound should be >= 0
        abs_G = sum(abs.(Z_relu.generators), dims=2)
        upper_bound = Z_relu.center + abs_G
        @test all(upper_bound .>= 0)
    end

    @testset "abstract_round_clamp" begin
        Z = Zonotope([0.5, 1.5], [0.3 0.1; 0.2 0.4])
        Z_clamped = abstract_round_clamp(Z, 2)

        # Basic shape tests
        @test isa(Z_clamped, Zonotope)
        @test length(Z_clamped.center) == 2
        @test size(Z_clamped.generators, 2) == 4  # m + n = 2 + 2

        # Soundness check: for each interval corner of Z, f(x) must lie within [lower, upper].
        # Note: the affine relaxation can exceed Cub at the interval level (e.g. Case 6),
        # so we check pointwise containment rather than upper_bound <= Cub.
        abs_G = sum(abs.(Z_clamped.generators), dims=2)
        lower_bound = Z_clamped.center .- abs_G
        upper_bound = Z_clamped.center .+ abs_G
        c_in = Z.center
        r_in = sum(abs.(Z.generators), dims=2)
        x_lo = vec(c_in .- r_in)
        x_hi = vec(c_in .+ r_in)
        f_lo = clamp.(round.(x_lo), 0.0, 2.0)
        f_hi = clamp.(round.(x_hi), 0.0, 2.0)
        @test all(vec(lower_bound) .<= f_lo .+ 1e-6)
        @test all(f_hi .<= vec(upper_bound) .+ 1e-6)
    end

    # Case 3 boundary u = C + 0.5: round(u) = C + 1 for odd C under ties-to-even,
    # but the clamped output never exceeds C, so r_u must be min(round(u), C).
    @testset "abstract_round_clamp boundary u = C + 0.5" begin
        rc(x, C) = clamp(round(x), 0.0, float(C))
        rc_rows  = QuantizedZonotopeVerification._rc_constraint_rows
        tol      = 1e-9

        # 1-D zonotope whose interval is [l, u]
        interval(l, u) = Zonotope([(l + u) / 2], reshape([(u - l) / 2], 1, 1))

        # Sample points: endpoints, a uniform grid, and both sides of every half-integer.
        function sample_points(l, u)
            xs = collect(range(l, u; length=401))
            for k in floor(l):ceil(u), d in (-1e-9, 0.0, 1e-9)
                x = k + 0.5 + d
                l <= x <= u && push!(xs, x)
            end
            return xs
        end

        # Soundness of the affine strip and of every constraint row on the concrete graph.
        function check_sound(Z, C; kwargs...)
            c = Z.center[1]; r = Z.generators[1, 1]
            lam, mu, E = abstract_round_clamp(Z, C; output=:triplet)
            λ = lam[1, 1]; μ = mu[1]; e = E[1, 1]
            A, b = rc_rows(Z, lam, mu, E, 1, 2, C; kwargs...)
            ok = true
            for x in sample_points(c - r, c + r)
                y  = rc(x, C)
                ok &= λ * x + μ - e - tol <= y <= λ * x + μ + e + tol
                ε1 = r > 0 ? (x - c) / r : 0.0
                ε2 = e > 0 ? (y - λ * x - μ) / e : 0.0
                ok &= all(A * [ε1, ε2] .+ b .>= -tol)
            end
            return ok
        end

        # Largest abstract output y over the constrained polygon (grid over ε ∈ [-1,1]²).
        function max_feasible_output(Z, C; kwargs...)
            c = Z.center[1]; r = Z.generators[1, 1]
            lam, mu, E = abstract_round_clamp(Z, C; output=:triplet)
            λ = lam[1, 1]; μ = mu[1]; e = E[1, 1]
            A, b = rc_rows(Z, lam, mu, E, 1, 2, C; kwargs...)
            best = -Inf
            for ε1 in range(-1, 1; length=201), ε2 in range(-1, 1; length=201)
                all(A * [ε1, ε2] .+ b .>= -tol) || continue
                best = max(best, λ * (c + r * ε1) + μ + e * ε2)
            end
            return best
        end

        variants = (
            (oblique=true,  diagonal=true),    # complete
            (oblique=false, diagonal=false),   # light
        )

        for C in (3, 63)   # odd C, as produced by C = 2^Q - 1
            Cf = float(C)
            @test round(Cf + 0.5) == Cf + 1   # the tie that triggers the issue

            # (a) Last plateau only: (C-0.5, C+0.5] ⊇ [C, C+0.5]. f ≡ C, so the
            #     optimal abstraction is the constant C with no error term and no rows.
            Z = interval(Cf, Cf + 0.5)
            lam, mu, E = abstract_round_clamp(Z, C; output=:triplet)
            @test lam[1, 1] == 0.0
            @test mu[1] == Cf
            @test E[1, 1] == 0.0
            Zout = abstract_round_clamp(Z, C)
            @test Zout.center[1] == Cf
            @test all(iszero, Zout.generators)
            for v in variants
                A, b = rc_rows(Z, lam, mu, E, 1, 2, C; v...)
                @test size(A, 1) == 0
                @test check_sound(Z, C; v...)
            end

            # (b) Several plateaus ending exactly at u = C + 0.5 (Case 3b).
            #     Strip is identity ± 0.5; the upper cut must be y ≤ C, not y ≤ C + 1.
            Z = interval(Cf - 2.0, Cf + 0.5)
            lam, mu, E = abstract_round_clamp(Z, C; output=:triplet)
            @test lam[1, 1] == 1.0
            @test mu[1] == 0.0
            @test E[1, 1] == 0.5
            for v in variants
                @test check_sound(Z, C; v...)
                A, b = rc_rows(Z, lam, mu, E, 1, 2, C; v...)
                # (x, y) = (u, C + 1) lies in the strip (ε = (1, 1)) but is infeasible
                @test minimum(A * [1.0, 1.0] .+ b) < -tol
                # (x, y) = (u, C) is on the graph (ε = (1, -1)) and stays feasible
                @test all(A * [1.0, -1.0] .+ b .>= -tol)
                @test max_feasible_output(Z, C; v...) <= Cf + 1e-6
            end

            # (c) Left endpoint on a rounding tie: l = C - 0.5 rounds to C - 1 (even).
            Z = interval(Cf - 0.5, Cf + 0.5)
            for v in variants
                @test check_sound(Z, C; v...)
                @test max_feasible_output(Z, C; v...) <= Cf + 1e-6
            end

            # (d) Corner (l, u) = (-0.5, C + 0.5): resolved as Case 3, must match
            #     the Case 4 box constraints 0 ≤ y ≤ C.
            Z = interval(-0.5, Cf + 0.5)
            lam, mu, E = abstract_round_clamp(Z, C; output=:triplet)
            @test lam[1, 1] == 1.0
            @test mu[1] == 0.0
            @test E[1, 1] == 0.5
            for v in variants
                @test check_sound(Z, C; v...)
                @test max_feasible_output(Z, C; v...) <= Cf + 1e-6
            end

            # (e) Precedence: l = u = C + 0.5 is Case 2 (constant C), never C + 1.
            Z = interval(Cf + 0.5, Cf + 0.5)
            lam, mu, E = abstract_round_clamp(Z, C; output=:triplet)
            @test lam[1, 1] == 0.0 && mu[1] == Cf && E[1, 1] == 0.0
        end

        # (f) Away from the boundary nothing changes: r_u = round(u) < C.
        Z = interval(0.75, 2.25)
        lam, mu, E = abstract_round_clamp(Z, 3; output=:triplet)
        @test lam[1, 1] == 1.0 && mu[1] == 0.0 && E[1, 1] == 0.5
        for v in variants
            @test check_sound(Z, 3; v...)
            @test max_feasible_output(Z, 3; v...) <= 2.0 + 1e-6
            A, b = rc_rows(Z, lam, mu, E, 1, 2, 3; v...)
            @test all(A * [1.0, -0.5] .+ b .>= -tol)   # (u, 2) = (2.25, 2) on the graph
        end
    end

    # Complete constrained formulation: the polygon cut out by the strip, the rows of
    # _rc_constraint_rows and l ≤ x ≤ u must equal the convex hull of the closed graph
    # of rc on [l, u], including when an oblique hull edge is (nearly) vertical, i.e.
    # when its run d = rl + 0.5 - l or d = u + 0.5 - ru is zero or tiny.
    @testset "complete round-clamp constraints: convex hull" begin
        rc(x, C) = clamp(round(x), 0.0, float(C))
        rc_rows  = QuantizedZonotopeVerification._rc_constraint_rows
        interval(l, u) = Zonotope([(l + u) / 2], reshape([(u - l) / 2], 1, 1))

        # Values of the closed graph of rc at x, restricted to [l, u]: the value itself
        # plus the one-sided limits that exist inside the interval.
        function closed_values(x, l, u, C)
            vals = [rc(x, C)]
            x > l && push!(vals, clamp(ceil(x - 0.5),  0.0, float(C)))   # left limit
            x < u && push!(vals, clamp(floor(x + 0.5), 0.0, float(C)))   # right limit
            return vals
        end

        function on_closed_graph(x, y, l, u, C; tol=1e-6)
            xs = [clamp(x, l, u)]
            h  = round(x - 0.5) + 0.5                       # nearest half-integer
            abs(h - x) <= tol && l <= h <= u && push!(xs, h)
            return any(abs(y - v) <= tol for xc in xs for v in closed_values(xc, l, u, C))
        end

        # Strip parameters and all halfspaces a1·ε1 + a2·ε2 + a0 ≥ 0 (rows + box on ε).
        function polygon(Z, C; kwargs...)
            c = Z.center[1]; r = Z.generators[1, 1]
            lam, mu, E = abstract_round_clamp(Z, C; output=:triplet)
            A, b = rc_rows(Z, lam, mu, E, 1, 2, C; kwargs...)
            H = vcat(hcat(A, b), [1.0 0.0 1.0; -1.0 0.0 1.0; 0.0 1.0 1.0; 0.0 -1.0 1.0])
            return (c=c, r=r, λ=lam[1, 1], μ=mu[1], e=E[1, 1], H=H, A=A, b=b)
        end

        # Every point of the closed graph satisfies the strip and every row.
        function is_sound(Z, C; tol=1e-9, kwargs...)
            P = polygon(Z, C; kwargs...)
            l = P.c - P.r; u = P.c + P.r
            xs = collect(range(l, u; length=201))
            for k in floor(l):ceil(u)
                l <= k + 0.5 <= u && push!(xs, k + 0.5)
            end
            for x in xs, y in closed_values(x, l, u, C)
                abs(y - P.λ * x - P.μ) <= P.e + tol || return false
                ε1 = P.r > 0 ? (x - P.c) / P.r : 0.0
                ε2 = P.e > 0 ? (y - P.λ * x - P.μ) / P.e : 0.0
                all(P.A * [ε1, ε2] .+ P.b .>= -tol) || return false
            end
            return true
        end

        # Vertices of the polygon in the (x, y) plane, from pairwise intersections of
        # constraint boundaries (rows normalised; near-parallel pairs are skipped).
        function polygon_vertices(Z, C; tol=1e-9, kwargs...)
            P  = polygon(Z, C; kwargs...)
            nr = sqrt.(P.H[:, 1] .^ 2 .+ P.H[:, 2] .^ 2)
            @assert all(P.H[nr .<= 1e-12, 3] .>= -tol)     # constant rows must hold
            H  = P.H[nr .> 1e-12, :] ./ nr[nr .> 1e-12]
            V  = Tuple{Float64,Float64}[]
            for i in 1:size(H, 1), j in i+1:size(H, 1)
                det = H[i, 1] * H[j, 2] - H[i, 2] * H[j, 1]
                abs(det) < 1e-9 && continue
                ε1 = (-H[i, 3] * H[j, 2] + H[i, 2] * H[j, 3]) / det
                ε2 = (-H[i, 1] * H[j, 3] + H[i, 3] * H[j, 1]) / det
                all(H * [ε1, ε2, 1.0] .>= -tol) || continue
                x = P.c + P.r * ε1
                push!(V, (x, P.λ * x + P.μ + P.e * ε2))
            end
            return V
        end

        # Lemma-2 criterion: sound, and every polygon vertex lies on the closed graph.
        function is_convex_hull(Z, C)
            l = Z.center[1] - Z.generators[1, 1]; u = Z.center[1] + Z.generators[1, 1]
            is_sound(Z, C; oblique=true, diagonal=true) || return false
            V = polygon_vertices(Z, C; oblique=true, diagonal=true)
            return !isempty(V) && all(on_closed_graph(x, y, l, u, C) for (x, y) in V)
        end

        C  = 7
        ds = (0.0, 1e-12, 1e-9, 1e-6, 1e-3, 0.01, 0.05, 0.050001, 0.25, 1.0)

        # d = 0 needs an endpoint on a half-integer that rounds away from the interval:
        # 2.5 rounds down to 2 (left edge), 3.5 rounds up to 4 (right edge).
        @test round(2.5) == 2.0 && round(3.5) == 4.0

        @testset "oblique run d = $d" for d in ds
            @test is_convex_hull(interval(2.5 - d, 4.25), C)        # Case 3b, left edge
            @test is_convex_hull(interval(1.25, 3.5 + d), C)        # Case 3b, right edge
            @test is_convex_hull(interval(2.5 - d, 3.5 + d), C)     # Case 3b, both edges
            @test is_convex_hull(interval(-2.0, 3.5 + d), C)        # Case 5
            @test is_convex_hull(interval(2.5 - d, 9.0), C)         # Case 6
        end

        # No row may contain a non-finite coefficient, whatever the run.
        for d in ds, Z in (interval(2.5 - d, 3.5 + d), interval(-2.0, 3.5 + d), interval(2.5 - d, 9.0))
            P = polygon(Z, C; oblique=true, diagonal=true)
            @test all(isfinite, P.A) && all(isfinite, P.b)
        end

        # Exhaustive sweep on a dyadic grid (hits every case and every boundary/tie
        # exactly): soundness of light and complete, convex hull for complete.
        let C = 3, n_bad_sound = 0, n_bad_hull = 0, n = 0
            for l in -2.0:0.125:5.0, u in (l + 0.125):0.125:5.5
                Z = interval(l, u)
                n += 1
                is_sound(Z, C; oblique=false, diagonal=false) || (n_bad_sound += 1)
                is_sound(Z, C; oblique=true,  diagonal=true)  || (n_bad_sound += 1)
                is_convex_hull(Z, C) || (n_bad_hull += 1)
            end
            @test n > 1000
            @test n_bad_sound == 0
            @test n_bad_hull == 0
        end
    end

end
