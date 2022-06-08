```julia
model = Chain(
    Dense(784, 32, relu),
    Dense(32, 10), softmax
)
```

``julia
lenet = SimpleChain(
  (static(7), static(28), static(1)),
  SimpleChains.Conv(SimpleChains.relu, (5, 5), 6),
  SimpleChains.MaxPool(2, 2),
  SimpleChains.Conv(SimpleChains.relu, (5, 5), 16),
  SimpleChains.MaxPool(2, 2),
  Flatten(3),
  TurboDense(SimpleChains.relu, 120),
  TurboDense(SimpleChains.relu, 84),
  TurboDense(identity, 10),
)
```



```julia
Chain(
    Flux.Conv((5, 5), imgsize[end] => 6, Flux.relu),
    Flux.MaxPool((2, 2)),
    Flux.Conv((5, 5), 6 => 16, Flux.relu),
    Flux.MaxPool((2, 2)),
    Flux.flatten,
    Flux.Dense(prod(out_conv_size), 120, Flux.relu),
    Flux.Dense(120, 84, Flux.relu),
    Flux.Dense(84, nclasses),
  ) |> device
```

```julia
lenet = SimpleChain(
  (static(28), static(28), static(1)),
  SimpleChains.Conv(SimpleChains.relu, (5, 5), 6),
  SimpleChains.MaxPool(2, 2),
  SimpleChains.Conv(SimpleChains.relu, (5, 5), 16),
  SimpleChains.MaxPool(2, 2),
  Flatten(3),
  TurboDense(SimpleChains.relu, 120),
  TurboDense(SimpleChains.relu, 84),
  TurboDense(identity, 10),
)

```julia