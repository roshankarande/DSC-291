using Flux
using Flux.Losses: mse

model = Chain(
    Dense(4,32,tanh),
    Dense(32,16,tanh),
    Dense(16,4)
)

