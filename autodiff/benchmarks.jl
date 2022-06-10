using BenchmarkTools
using ForwardDiff, ReverseDiff, Zygote, Enzyme
using Plots

f(x) = log(sum(exp.(x)))

E, K = 1, 1

function alloc_timer(n; E, K)

    x = rand(Float32, n)
    t0 = @belapsed f($x) evals = E samples = K
    t1 = @belapsed ForwardDiff.gradient(f, $x) evals = E samples = K
    t2 = @belapsed ReverseDiff.gradient(f, $x) evals = E samples = K
    t3 = @belapsed Zygote.gradient(f, $x) evals = E samples = K
    t4 = @belapsed Enzyme.gradient(Enzyme.Forward, f, $x) evals = E samples = K
    t5 = @belapsed Enzyme.gradient(Enzyme.Reverse, f, $x) evals = E samples = K
    t1, t2, t3, t4, t5, t0
end

ns = 2 .^ (2:12)

x = rand()
res = [alloc_timer(n; E=E, K=K) for n in ns]
fwd_diff = [t[1] for t in res]
rev_diff = [t[2] for t in res]
zygote_diff = [t[3] for t in res]
enzyme_fwd_diff = [t[4] for t in res]
enzyme_rev_diff = [t[5] for t in res]
func_eval = [t[6] for t in res]

plot(ns, fwd_diff, label="Forward Diff", xscale=:log10, yscale=:log10, legend=:topright, title="log_sum_exp(x)", ylabel="time (s)", xlabel="size(x)",)
plot!(ns, rev_diff, label="Reverse Diff")
plot!(ns, zygote_diff, label="Zygote Diff")
plot!(ns, enzyme_fwd_diff, label="Enzyme Fwd")
plot!(ns, enzyme_rev_diff, label="Enzyme Rev", linecolor="grey")
plot!(ns, func_eval, label="Func Eval", linecolor="red")

print("benchmarks computed")
savefig("julia_benchmarks.png")