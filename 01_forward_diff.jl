

# Define dual struct
struct Dual{T<:Number}
    v::T
    δ::T
end

Base.:+(a::Dual, b::Dual) = Dual(a.v + b.v, a.δ + b.δ)
Base.:-(a::Dual, b::Dual) = Dual(a.v - b.v, a.δ - b.δ)

Base.:*(a::Dual, b::Dual) = Dual(a.v * b.v, a.δ * b.v + a.v * b.δ)
Base.:/(a::Dual, b::Dual) = Dual(a.v / b.v, (a.δ*b.v - a.v*b.δ) / b.v^2)

# Dual(1,0) + Dual(4.0,2.1)  # Dual{Float64}(5.0, 2.1)


# convert(Dual{Float64}, 1.0) => Dual{Float64}(1.0, 0.0)
# convert(Dual, 1) => Dual{Int64}(1, 0)
Base.convert(::Type{Dual{T}}, x::T) where {T<:Number} = Dual(x,zero(x))

Base.promote_rule(::Type{Dual{T}}, ::Type{T}) where {T<:Number} = Dual{T}

Base.promote_rule(::Type{Dual}, ::Type{T}) where {T<:Number} = Dual{T}