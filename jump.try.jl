using MathOptInterface

model = Model(Ipopt.Optimizer)
@variable(model, x[i=1:2])
A = [1 2; 3 4]
b = [5, 6]
@constraint(model, con, A * x .== b)
# You can access the constraints using con, con[1] for first constraint and so on

# model = Model()
# @variable(model, x[1:3])

# # c1, c2 tag names are optional
# @constraint(model, c1, sum(x) <= 1)
# @constraint(model, c2, x[1] + 2 * x[3] >= 2)
# @constraint(model, c3, sum(i * x[i] for i in 1:3) == 3)
# @constraint(model, c4, 4 <= 2 * x[2] <= 5)
# print(model)


# @variable(model, t >= 0)
# @constraint(model, my_q, x[1]^2 + x[2]^2 <= t^2)

# @constraint(model, c[i=1:3], x[i] <= i^2)

# constraint(model, c[i=1:3, j=i:3], x[i] <= j)

# you can filter elements in the sets using the ; syntax
# @constraint(model, c[i=1:9; mod(i, 3) == 0], x[i] <= i)

# @constraint(model, con, [2x + 3x, 4x] in MOI.Nonnegatives(2))

A = [4 5; 6 7]
b =  [9,13]
model = Model(Ipopt.Optimizer)
@variable(model, x[1:2]);
@variable(model, t >= 0)
@constraint(model, sum((A*x-b).^2) <= t^2)

# Norm would not work because x is vector of affine expressions
# @constraint(model, norm(A*x-b) <= t^2)

@objective(model, Min, t )
optimize!(model)



t = 10
model = Model(Ipopt.Optimizer)
@variable(model, 0 <= v[1:t] <= 1) # grayscale only
# `LinearAlgebra.norm` isn't accepted so make our own L2 norm
@NLobjective(model, Min, sqrt(sum((v[i] - y[i])^2 for i in 1:t)))
solve!(model)