using PyCall,Flux, BenchmarkTools

torch = pyimport("torch")

NN = torch.nn.Sequential(
    torch.nn.Linear(8, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 2),
    torch.nn.ReLU(),
)

torch_nn(in) = NN(in)

Flux_nn = Chain(Dense(8,64,relu),
                Dense(64,32,relu),
                Dense(32,2,relu))

for i in [1, 10, 100, 1000]
    println("Batch size: $i")
    torch_in = torch.rand(i,8)
    flux_in = rand(Float32,8,i)
    print("pytorch:")
    @btime torch_nn($torch_in)
    print("flux   :")
    @btime Flux_nn($flux_in)
end

