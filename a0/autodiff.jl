# Autodiff
m = 5
n = 10
x = randn(m, 1)
y = randn(m, 1)
A = randn(m, n)
B = randn(m, m)

using Test
@testset "Data size" begin
    @test size(x) == (m, 1)
    @test size(y) == (m, 1)
    @test size(A) == (m, n)
    @test size(B) == (m, m)
end

using Zygote: gradient
f1(x, y) = (x' * y)[1]
df1dx = gradient(f1, x, y)[1]

f2(x) = (x' * x)[1]
df2dx = gradient(f2, x)[1]

f3(x) = x' * A          # gradient not well defined since output's dim != 1
function jacobian(f, x) # calls gradient n=length(y) times
    y = f(x)
    n = length(y)
    m = length(x)
    T = eltype(y)
    j = Array{T, 2}(undef, n, m)
    for i in 1:n
        j[i, :] = gradient(x -> f(x)[i], x)[1]
    end
    return j
end

df3dx = jacobian(f3, x)

f4(x) = (x' * B * x)[1]
df4dx = gradient(f4, x)[1]

# Test that AD result == hand-derived gradients
@testset "AD == hand derived" begin
    @test df1dx == y
    @test df2dx == 2 .* x
    @test df3dx == A'
    @test df4dx ≈ (B .+ B')*x
end
