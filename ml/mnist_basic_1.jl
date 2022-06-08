using Flux, MLDatasets, CUDA
using Flux: train!, onehotbatch, onecold
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser,WeightDecay

import Flux.Optimise.train!

function get_data()
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)
    (
        (reshape(xtrain, 28^2, :), UInt32.(ytrain .+ 1)),
        (reshape(xtest, 28^2, :), UInt32.(ytest .+ 1)),
      )
end

function loaders(xtrain, ytrain, xtest, ytest, args)
    ytrain, ytest = onehotbatch(ytrain, 1:10), onehotbatch(ytest, 1:10)

    train_loader = DataLoader((device(xtrain), device(ytrain)), batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader((device(xtest), device(ytest)), batchsize=args.batchsize)

    return train_loader, test_loader
end

function loaders(args)
    (xtrain, ytrain), (xtest, ytest) = get_data()
    loaders(xtrain, ytrain, xtest, ytest, args)
end

function eval_loss_accuracy(loader, model, device)
    l = 0.0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
        x, y = x |> device, y |> device
        ŷ = model(x)
        l += loss(ŷ, y) * size(x)[end]
        acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))
        ntot += size(x)[end]
    end
    return (acc=acc / ntot * 100 |> round4, loss=l / ntot |> round4)
end

Base.@kwdef struct Args
    η::Float64 = 3e-4             # learning rate
    λ::Float64 = 0                # L2 regularizer param, implemented as weight decay
    batchsize::Int = 128      # batch size
    epochs::Int = 10          # number of epochs
    seed::Int = 0             # set seed > 0 for reproducibility
end

args = Args();
device = gpu

(xtrain, ytrain), (xtest, ytest) = get_data();

model = Chain(
    Dense(784, 32, relu),
    Dense(32, 10), softmax
) |> device

train_loader, test_loader = get_data()

loss(x, y) = Flux.Losses.logitcrossentropy(model(x), y)
optimizer = ADAM(0.001)
# opt = ADAM(args.η)
train!(model,train_loader, args, optimizer)


# function train(; kws...)
#     args = Args(; kws...)
#     args.seed > 0 && Random.seed!(args.seed)

#     ## DATA
#     train_loader, test_loader = get_data(args)
#     @info "Dataset MNIST: $(train_loader.nobs) train and $(test_loader.nobs) test examples"

#     ## MODEL AND OPTIMIZER
#     model = LeNet5() |> device
#     @info "LeNet5 model: $(num_params(model)) trainable params"


#     opt = ADAM(args.η)
#     train!(model, args, opt)

# end

function train!(model, train_loader, args=Args(), opt=ADAM(0.001))
    ps = Flux.params(model)

        for (x, y) in train_loader
            x = device(x)
            y = device(y)
            gs = Flux.gradient(ps) do
                ŷ = model(x)
                loss(ŷ, y)
            end
            # Perform an update step of the parameters ps (or the single parameter p) according to optimizer opt and the gradients gs (the gradient g).
            Flux.Optimise.update!(opt, ps, gs)
        end
end



# parameters = params(model)


# for i in 1:50
#     Flux.train!(loss, parameters, train_data, optimizer)

#     if i % 5 == 0
#         println(i)
#     end

# end


# accuracy = 0
# for i in 1:length(y_test)
#     if findmax(model(test_data[1][1][:, i]))[2] - 1 == y_test[i]
#         accuracy += 1
#     end
# end
# println(accuracy / length(y_test))