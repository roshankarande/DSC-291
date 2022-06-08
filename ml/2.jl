using MLDatasets
using Plots
using ImageInspector
using Flux
using Flux: onehotbatch, onecold
using Flux: crossentropy
using BSON
using Statistics

T = Float32
X_train, y_train = MLDatasets.MNIST.traindata(T)
X_test, y_test = MLDatasets.MNIST.testdata(T)

function reshape_data(X::AbstractArray{<:Real, 3})
    s = size(X)
    return reshape(X, s[1], s[2], 1, s[3])
end


function load_data(dataset; T=Float32, onehot=false, classes=0:9)
    X_train, y_train = dataset.traindata(T)
    X_test, y_test = dataset.testdata(T)

    X_train = reshape_data(X_train)
    X_test = reshape_data(X_test)

    y_train = onehotbatch(y_train, classes)
    y_test = onehotbatch(y_test, classes)

    # if onehot
    #     y_train = onehotbatch(y_train, classes)
    #     y_test = onehotbatch(y_test, classes)
    # end

    return X_train, y_train, X_test, y_test
end

X_train, y_train, X_test, y_test = load_data(MLDatasets.MNIST; T=T, onehot=true)

m = Chain(
    Conv((2,2), 1=>16, relu),
    MaxPool((2,2)),
    Conv((2,2), 16=>8, relu),
    MaxPool((2,2)),
    flatten,
    Dense(288, size(y_train,1)),
    softmax,
)

L(X, y) = crossentropy(m(X), y)


function train_model!(m, L, X, y; opt = Descent(0.1),  batchsize = 128,n_epochs = 10, file_name = "")

    batches = DataLoader((X, y); batchsize, shuffle = true)

    for _ in 1:n_epochs
        Flux.train!(L, params(m), batches, opt)
    end

    !isempty(file_name) && BSON.bson(file_name, m=m)

    return
end

file_name = "mnist_simple.bson"
train_model!(m, L, X_train, y_train; n_epochs=10, file_name=file_name)


acc(x, y) = mean(onecold(m(x)) .== onecold(y))
"Test accuracy = " * string(acc(X_test, y_test))