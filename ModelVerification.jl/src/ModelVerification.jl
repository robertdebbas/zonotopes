module ModelVerification

struct ReLU end
struct Id end

struct Layer
    weights::Matrix{Float64}
    bias::Vector{Float64}
    activation
end

struct Network
    layers::Vector{Layer}
end

export ReLU, Id, Layer, Network

end
