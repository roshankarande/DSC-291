import Base: +, -, *, /

# Dual Struct
struct Dual{T<:Real} <: Real
    x::T
    ϵ::T
end

# Basic Dual Operations
a::Dual + b::Dual = Dual(a.x + b.x, a.ϵ + b.ϵ)
a::Dual - b::Dual = Dual(a.x - b.x, a.ϵ - b.ϵ)
a::Dual * b::Dual = Dual(a.x * b.x, b.x * a.ϵ + a.x * b.ϵ)
a::Dual / b::Dual = Dual(a.x / b.x, (a.ϵ*b.x - a.x*b.ϵ) / b.x^2)

# Higher order primitives
Base.sin(d::Dual) = Dual(sin(d.x), d.ϵ * cos(d.x))
Base.cos(d::Dual) = Dual(cos(d.x), - d.ϵ * sin(d.x))
Base.log(d::Dual) = Dual(log(d.x), d.ϵ/d.x)

function Base.max(a::Dual, b::Dual) 
    x = max(a.x, b.x)
    ϵ = a.x > b.x ? a.ϵ : b.ϵ
    return Dual(x,ϵ)
end

# Syntactic Sugar
Dual(x::S, d::T) where {S<:Real, T<:Real} = Dual{promote_type(S, T)}(x, d)
Dual(x::Real) = Dual(x, zero(x))
Dual{T}(x::Real) where {T} = Dual(T(x), zero(T))

Base.convert(::Type{Dual{T}}, d::Dual) where T = Dual(convert(T, d.x), convert(T, d.ϵ))
Base.convert(::Type{Dual{T}}, d::Real) where T = Dual(convert(T, d), zero(T))
Base.promote_rule(::Type{Dual{T}}, ::Type{R}) where {T,R} = Dual{promote_type(T,R)}
Base.promote_rule(::Type{Dual{T}}, ::Type{Dual{R}}) where {T<:Real, R<:Real} = Dual{promote_type(T,R)}

# Displaying in a good way
function Base.show(io::IO, d::Dual)
    if signbit(d.ϵ)
        print(io, d.x, " - ", -d.ϵ, "ϵ")
    else
        print(io, d.x, " + ", d.ϵ, "ϵ")
    end
end

# Let us consider our example
f(a,b) = log(a*b + max(a,2))

a, b = 3,2
f(a,b)  # this will give us the primal value

# Let us find the primal and partial w.r.t a
a = Dual(3,1)
b = Dual(2,0)

f(a,b)

# Let us find the primal and partial w.r.t b``
a = Dual(3,0)
b = Dual(2,1)

f(a,b)

## Another example
g(x) = x / (1 + x*x)
g(5.)
g(Dual(5., 1.))

ϵ = rand()*1e-13
(g(5.0+ϵ)-g(5.0))/ϵ

ϵ = rand()*1e-14
(g(5.0+ϵ)-g(5.0))/ϵ

ϵ = rand()*1e-15
(g(5.0+ϵ)-g(5.0))/ϵ


@code_typed g(5.0)
@code_typed  g(Dual(5., 1.))





@code_lowered g(5.0)
@code_lowered  g(Dual(5., 1.))


## Another example
h(x) = x^2
h(5.)
h(Dual(5., 1.))

@code_typed h(5.0)
@code_typed  h(Dual(5., 1.))

@code_lowered h(5.0)
@code_lowered  h(Dual(5., 1.))


# D(f, x) = f(Dual(x, one(x))).ϵ


# @code_typed D(f, 1.0)

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