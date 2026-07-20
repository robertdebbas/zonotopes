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

end
