function nextfib(n)
    a, b = one(n), one(n)
    while b <= n
        a, b = b, a + b
    end
    return b
end

nextfib(5)

@code_lowered nextfib(123)

@code_typed nextfib(123)

@code_llvm nextfib(123)

@code_native nextfib(123)