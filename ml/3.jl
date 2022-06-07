using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated
using CUDA

imgs, labels = get

imgs = MNIST.images()
X = hcat(float.(reshape.(imgs, :))...) |> gpu
labels = MNIST.labels()
Y = onehotbatch(labels, 0:9) |> gpu

m = Chain(
         Dense(28^2, 32, relu),
         Dense(32, 10),
         softmax
         ) |> gpu

loss(x, y) = crossentropy(m(x), y)

accuracy1(x, y) = mean(onecold(m(x)) .== onecold(y))

dataset = repeated((X, Y), 200)

evalcb = () -> @show(loss(X, Y))
#3 (generic function with 1 method)

opt = ADAM()
ADAM(0.001, (0.9, 0.999), IdDict{Any,Any}())

Flux.train!(loss, params(m), dataset, opt, cb = throttle(evalcb, 10))
loss(X, Y) = 2.2738786f0

accuracy(X, Y)
