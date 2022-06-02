```python
model = ConcreteModel()

model.x = Var([1,2], domain=pyo.NonNegativeReals)



model.objective = Objective(expr = 12*model.x[1] + 20*model.x[2])

model.constraint = Constraint(expr = 0 <= model.x[2] <= 3)
model.constraint = Constraint(expr = 6*model.x[1] + 8*model.x[2] >= 100)
model.constraint = Constraint(expr = 7*model.x[1] + 12*model.x[2] >= 120)



results = SolverFactory('glpk').solve(model)
results.write()
if results.solver.status:
    model.pprint()

```

```julia
The following constrains \|(x-1, x-2)\|_2 \le t∥(x−1,x−2)∥ 
2
​
 ≤t and t \ge 0t≥0:

julia> model = Model();

julia> @variable(model, x)
x

julia> @variable(model, t)
t

julia> @constraint(model, [t, x-1, x-2] in SecondOrderCone())
[t, x - 1, x - 2] ∈ MathOptInterface.SecondOrderCone(3)

```