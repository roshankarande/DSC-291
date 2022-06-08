using SimpleChains

mlpd = SimpleChain(
  static(4),
  TurboDense(tanh, 32),
  TurboDense(tanh, 16),
  TurboDense(identity, 4)
)

function f(x)
    N = Base.isqrt(length(x))
    A = reshape(view(x, 1:N*N), (N,N))
    expA = exp(A)
    vec(expA)
  end
  
  T = Float32;
  X = randn(T, 2*2, 10_000);
  Y = reduce(hcat, map(f, eachcol(X)));
  Xtest = randn(T, 2*2, 10_000);
  Ytest = reduce(hcat, map(f, eachcol(Xtest)));

@time p = SimpleChains.init_params(mlpd);
G = SimpleChains.alloc_threaded_grad(mlpd);

mlpdloss = SimpleChains.add_loss(mlpd, SquaredLoss(Y));
mlpdtest = SimpleChains.add_loss(mlpd, SquaredLoss(Ytest));

report = let mtrain = mlpdloss, X=X, Xtest=Xtest, mtest = mlpdtest
  p -> begin
    let train = mlpdloss(X, p), test = mlpdtest(Xtest, p)
      @info "Loss:" train test
    end
  end
end

report(p)
for _ in 1:3
  @time SimpleChains.train_unbatched!(
    G, p, mlpdloss, X, SimpleChains.ADAM(), 10_000
  );
  report(p)
end