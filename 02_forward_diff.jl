import Base: +, -, *, /

struct Dual{T<:Real} <: Real
    x::T
    ϵ::T
end

Dual(x::S, d::T) where {S<:Real, T<:Real} = Dual{promote_type(S, T)}(x, d)
Dual(x::Real) = Dual(x, zero(x))
Dual{T}(x::Real) where {T} = Dual(T(x), zero(T))

# Dual(1, 2)

function Base.show(io::IO, d::Dual)
    if signbit(d.ϵ)
        print(io, d.x, " - ", -d.ϵ, "ϵ")
    else
        print(io, d.x, " + ", d.ϵ, "ϵ")
    end
end
# Dual(1, -2)

a::Dual + b::Dual = Dual(a.x + b.x, a.ϵ + b.ϵ)
a::Dual - b::Dual = Dual(a.x - b.x, a.ϵ - b.ϵ)
a::Dual * b::Dual = Dual(a.x * b.x, b.x * a.ϵ + a.x * b.ϵ)
a::Dual / b::Dual = Dual(a.x / b.x, (a.ϵ*b.x - a.x*b.ϵ) / b.x^2)


# Let us define a few primitives
Base.sin(d::Dual) = Dual(sin(d.x), d.ϵ * cos(d.x))
Base.cos(d::Dual) = Dual(cos(d.x), - d.ϵ * sin(d.x))

Base.convert(::Type{Dual{T}}, d::Dual) where T = Dual(convert(T, d.x), convert(T, d.ϵ))
Base.convert(::Type{Dual{T}}, d::Real) where T = Dual(convert(T, d), zero(T))
Base.promote_rule(::Type{Dual{T}}, ::Type{R}) where {T,R} = Dual{promote_type(T,R)}
Base.promote_rule(::Type{Dual{T}}, ::Type{Dual{R}}) where {T<:Real, R<:Real} = Dual{promote_type(T,R)}

# This will give an error
# +sin(Dual(1.0,1.0))


## Let us create a utility to differentiate any function

f(x) = x / (1 + x*x)
f(5.)
f(Dual(5., 1.))

D(f, x) = f(Dual(x, one(x))).ϵ


@code_typed f(1.0)
@code_typed D(f, 1.0)

# h(x) = sin(x) 
# Dual(1, 2) * 3
# Dual(1,2)  + 4
# g(x) = 3x^2
# f(x) = x / (1 + x*x)
# f(x) = x^2 + 5x
# f(Dual(5., 1.))
# g(Dual(1,1))
# h(x,y) = x + 2y
# h(1,1)
# h(Dual(1,0),Dual(1,1))