using LinearAlgebra, BenchmarkTools, CUDA, LoopVectorization, Plots

function mygemmturbo!(C, A, B)
    @tturbo for m ∈ axes(A, 1), n ∈ axes(B, 2)
        Cmn = zero(eltype(C))
        for k ∈ axes(A, 2)
            Cmn += A[m, k] * B[k, n]
        end
        C[m, n] = Cmn
    end
end

function alloc_timer(n)
    A = rand(Float32,n,n)
    B = rand(Float32,n,n)
    C = rand(Float32,n,n)
    t1 = @belapsed $A * $B
    t2 = @belapsed (mul!($C,$A,$B))
    t3 = @belapsed (mygemmturbo!($C,$A,$B))
    A,B,C = (cu(A), cu(B), cu(C))
    t4 = @belapsed CUDA.@sync($A * $B)
    t5 = @belapsed CUDA.@sync(mul!($C,$A,$B))
    t1,t2,t3,t4,t5
end
ns = 2 .^ (2:11)
res = [alloc_timer(n) for n in ns]
alloc      = [t[1] for t in res]
noalloc    = [t[2] for t in res]
noalloclv  = [t[3] for t in res]
allocgpu   = [t[4] for t in res]
noallocgpu = [t[5] for t in res]


plot(ns, alloc, label="*", xscale=:log10, yscale=:log10, legend=:bottomright,
    title="Which Micro-optimizations matter for BLAS3?",
    yticks=10.0 .^ (-8:0.5:2),
    ylabel="Time (s)", xlabel="N",)
plot!(ns,noalloc,label="mul! (OpenBLAS)")
plot!(ns,noalloclv,label="mygemmturbo!")
plot!(ns,allocgpu,label="* gpu")
plot!(ns,noallocgpu,label="mul! gpu")
savefig("microopts_blas3.png")