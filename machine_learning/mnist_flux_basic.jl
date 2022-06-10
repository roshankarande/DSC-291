using Flux, MLDatasets, CUDA
using Flux: train!, onehotbatch, flatten

x_train, y_train = MLDatasets.MNIST.traindata(Float32);
x_test, y_test = MLDatasets.MNIST.testdata(Float32);

y_train = onehotbatch(y_train, 0:9);

model = Chain(
    Dense(784, 32, relu),
    Dense(32, 10), softmax
)

loss(x, y) = Flux.Losses.logitcrossentropy(model(x), y)
optimizer = ADAM(0.001)

parameters = params(model)

train_data = [(Flux.flatten(x_train), Flux.flatten(y_train))]

for i in 1:50
    Flux.train!(loss, parameters, train_data, optimizer)

    if i % 5 == 0
        println("epochs $(i)")
    end

end


test_data = [(Flux.flatten(x_test), y_test)]
let accuracy = 0
    for i in 1:length(y_test)
        if findmax(model(test_data[1][1][:, i]))[2] - 1 == y_test[i]
            accuracy += 1
        end
    end
    println("Accuracy ", accuracy / length(y_test))
end
