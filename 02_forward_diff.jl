import Base: +, -, *, /

struct Dual{T<:Real} <: Real
    x::T
    ϵ::T
end

Dual(x::S, d::T) where {S<:Real, T<:Real} = Dual{promote_type(S, T)}(x, d)
Dual(x::Real) = Dual(x, zero(x))
Dual{T}(x::Real) where {T} = Dual(T(x), zero(T))

# Dual(1, 2)

Base.show(io::IO, d::Dual) = print(io, d.x, " + ", d.ϵ, "ϵ")
# Dual(1, 2)


a::Dual + b::Dual = Dual(a.x + b.x, a.ϵ + b.ϵ)
a::Dual - b::Dual = Dual(a.x - b.x, a.ϵ - b.ϵ)
a::Dual * b::Dual = Dual(a.x * b.x, b.x * a.ϵ + a.x * b.ϵ)
a::Dual / b::Dual = Dual(a.x * b.x, (a.ϵ*b.x - a.x*b.ϵ) / b.x^2)

# Base.sin(d::Dual) = Dual(sin(d.x), d.ϵ * cos(d.x))
# Base.cos(d::Dual) = Dual(cos(d.x), - d.ϵ * sin(d.x))

Base.convert(::Type{Dual{T}}, x::Dual) where T = Dual(convert(T, x.x), convert(T, x.ϵ))
Base.convert(::Type{Dual{T}}, x::Real) where T = Dual(convert(T, x), zero(T))
Base.promote_rule(::Type{Dual{T}}, ::Type{R}) where {T,R} = Dual{promote_type(T,R)}
Base.promote_rule(::Type{Dual{T}}, ::Type{Dual{R}}) where {T<:Real, R<:Real} = Dual{promote_type(T,R)}


h(x) = sin(x) 

Dual(1, 2) * 3

Dual(1,2)  + 4

g(x) = 3x^2

f(x) = x / (1 + x*x)

f(x) = x^2 + 5x

f(Dual(5., 1.))


g(Dual(1,1))

h(x,y) = x + 2y

h(1,1)

h(Dual(1,0),Dual(1,1))