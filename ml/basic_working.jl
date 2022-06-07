using Flux, MLDatasets, CUDA
using Flux: train!, onehotbatch

CUDA.allowscalar(false)

x_train, y_train = MLDatasets.MNIST.traindata(Float32);
x_test, y_test = MLDatasets.MNIST.testdata(Float32);

y_train = onehotbatch(y_train, 0:9);

model = Chain(
    Dense(784, 32, relu),
    Dense(32, 10), softmax
) |>gpu

loss(x, y) = Flux.Losses.logitcrossentropy(model(x), y)
optimizer = ADAM(0.001)

parameters = params(model)
# flatten() function converts array 28x28x60000 into 784x60000 (28*28x60000)
train_data = [(Flux.flatten(x_train) |> gpu, Flux.flatten(y_train) |> gpu)]
# Range in loop can be used smaller
for i in 1:50
    Flux.train!(loss, parameters, train_data, optimizer)

    if i%5 == 0
        println(i)
    end   

end

test_data = [(Flux.flatten(x_test), y_test)];
accuracy = 0
for i in 1:length(y_test)
    if findmax(model(test_data[1][1][:, i]))[2] - 1  == y_test[i]
        accuracy += 1
    end
end
println(accuracy / length(y_test))