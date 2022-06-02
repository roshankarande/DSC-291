# very simple example of L2 norm optimization using JuMP
using Images, TestImages, JuMP, Ipopt

# crop a test image to 50x50 and then flatten it
original = Gray.(testimage("lighthouse"))[250:299, 150:199]
w = size(original, 1)
h = size(original, 2)
t = w * h
y = reshape(float.(original), t)

# model the problem
model = Model(solver=IpoptSolver(tol=1e-6))
@variable(model, 0 <= v[1:t] <= 1) # grayscale only
# `LinearAlgebra.norm` isn't accepted so make our own L2 norm
@NLobjective(model, Min, sqrt(sum((v[i] - y[i])^2 for i in 1:t)))
solve(model)
solution = Gray.(reshape(getvalue(v), w, h))

# side by side result + diff
d = (float.(solution) - float.(original))^2
hcat(original, solution, Gray.(scaleminmax(minimum(d), maximum(d)).(d)))