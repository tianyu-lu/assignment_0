function gaussian_pdf(x; mean=0., variance=0.01)
    return exp(-1/2*((x - mean)/sqrt(variance))^2) / (sqrt(2 * pi * variance))
end

gaussian_pdf(0)

using Test
using Distributions: pdf, Normal

@testset "Gaussian pdf" begin
    x = randn()
    @test gaussian_pdf(x) ≈ pdf.(Normal(0., sqrt(0.01)), x)
end;

function sample_gaussian(n; mean=0., variance=0.01)
    x = randn(n)
    z = mean.+(sqrt(variance).*x)
    return z
end;

sample_gaussian(5)

using Statistics: mean, var

@testset "Testing Gaussian sample statistics" begin
    samples = sample_gaussian(100000, mean=5., variance=1.5)
    @test isapprox(mean(samples), 5., atol=1e-2)
    @test isapprox(var(samples), 1.5, atol=1e-2)
end;

using Plots

samples = sample_gaussian(10000, mean=10., variance=2.)
histogram(samples, normalize=true, label="Gaussian samples, mean 10 var 2", title="Gaussian")
xs = 0:0.01:20
ys = gaussian_pdf.(xs, mean=10., variance=2.)
plot!(xs, ys,
     label="pdf")

savefig("gaussian samples")
