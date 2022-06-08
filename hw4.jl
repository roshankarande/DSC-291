using JuMP,Ipopt

model = Model(Ipopt.Optimizer)
@variable(model, x)  
@variable(model, y)
@variable(model, z)

@objective(model, Min, -x^2-y^2-z^2 + 2x+2y+2z)

@constraint(model, c1, x^2 + y^2 + z^2 <= 1)

optimize!(model)

print(model)
@show objective_value(model)
@show value(x), value(y)

@show termination_status(model)
@show primal_status(model)
@show dual_status(model)


solution_summary(model, verbose=true)