using PyCall
using Flux
using BenchmarkTools

torch = pyimport("torch")

NN = torch.nn.Sequential(
    torch.nn.Linear(8, 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 32),
    torch.nn.Tanh(),
    torch.nn.Linear(32, 2),
    torch.nn.Tanh()
)

torch_nn(in) = NN(in)

Flux_nn = Chain(Dense(8,64,tanh),
                Dense(64,32,tanh),
                Dense(32,2,tanh))


for i in [1, 10, 100, 1000]
    println("Batch size: $i")
    torch_in = torch.rand(i,8)
    flux_in = rand(Float32,8,i)
    print("pytorch:")
    @btime torch_nn($torch_in)
    print("flux   :")
    @btime Flux_nn($flux_in)
end