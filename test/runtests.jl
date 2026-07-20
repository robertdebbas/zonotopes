using Test
using LazySets
using LinearAlgebra
using ModelVerification
include("../src/QuantizedZonotopeVerification.jl")
using .QuantizedZonotopeVerification

@testset "QuantizedZonotopeVerification Tests" begin

    @testset "Abstraction Functions" begin
        include("test_abstract_round_clamp.jl")
    end

    @testset "Network Quantization" begin
        include("test_network.jl")
    end

end
