module QuantizedZonotopeVerification

# Core algorithm files
include("utils/conversions.jl")
include("core/quantization.jl")
include("core/abstractions.jl")
include("core/propagation.jl")
include("core/sampling.jl")

# Utilities
include("utils/network_io.jl")

# Export main functions
export quantize_zonotope,
       propagate_dnn,
       propagate_qnn,
       quantization_error_zonotope,
       quantization_error_constrained_zonotope,
       concretize_error_zonotope,
       box_to_zonotope,
       quantize_tensor,
       zonotope_to_box,
       vertices_to_box,
       box_to_vertices,
       Zonotope,
       Layer,
       Network,
       ReLU,
       Id,
       abstract_relu,
       abstract_round_clamp,
       sample_error_bounds,
       propagate_quantized,
       propagate,
       load_acasxu_network_from_json

end
