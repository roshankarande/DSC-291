struct Dual{T<:Number} <: Number
    x::T
    d::T
end

Base.:+(a::Dual, b::Dual)   = Dual(a.x+b.x, a.d+b.d)
Base.:-(a::Dual, b::Dual)   = Dual(a.x-b.x, a.d-b.d)
Base.:/(a::Dual, b::Dual)   = Dual(a.x/b.x, (a.d*b.x - a.x*b.d)/b.x^2) # recall  (a/b) =  a/b + (a'b - ab')/b^2 ϵ
Base.:*(a::Dual, b::Dual)   = Dual(a.x*b.x, a.d*b.x + a.x*b.d)

# Let's define some promotion rules
Dual(x::S, d::T) where {S<:Number, T<:Number} = Dual{promote_type(S, T)}(x, d)
Dual(x::Number) = Dual(x, zero(typeof(x)))
Dual{T}(x::Number) where {T} = Dual(T(x), zero(T))
Base.promote_rule(::Type{Dual{T}}, ::Type{S}) where {T<:Number,S<:Number} = Dual{promote_type(T,S)}
Base.promote_rule(::Type{Dual{T}}, ::Type{Dual{S}}) where {T<:Number,S<:Number} = Dual{promote_type(T,S)}

# and define api for forward differentionation
forward_diff(f::Function, x::Real) = _dual(f(Dual(x,1.0)))
_dual(x::Dual) = x.d
_dual(x::Vector) = _dual.(x)

g(x) = 3x^2
forward_diff(g, 1)


g(Dual(1+1im, 1 + 0im))
