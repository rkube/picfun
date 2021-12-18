#Encoding: UTF-8 -*-

using picfun
using Test
using Statistics
using LinearAlgebra

@testset "solver_tests" begin

    Lz = 2π
    Nz_list = [64 128 256 512 1024]
    dist_list = zeros(size(Nz_list))

    ctr = 1
    for Nz ∈ Nz_list
        zgrid = init_grid(Lz, Nz)
        # Expand colocation points. 
        zvals = (0:Nz-1) * zgrid.Δz     

        # Calculate a Gaussian and its second derivative
        σ = Lz / 20.0
        # f(z) is a Gaussian of width σ
        u = exp.(-0.5 * (zvals .- 0.5 * Lz).^ 2.0 / σ / σ)
        # The second derivative is given analytically
        d2u = u .* ((zvals .- 0.5 * Lz).^2.0 .- σ * σ) / σ^4.0;

        # Invert the second derivative to retrieve numerical approximation to u
        #u_num = ∇⁻²(d2u, zgrid)
        u_num = invert_laplace(d2u, zgrid)
        # Subtract the mean from the profile before comparing
    #     dist_list[ctr] = norm((u .- mean(u)) - u_num)
        # @show norm(u - u_num)
        dist_list[ctr] = norm(u - u_num)
        ctr += 1
    end
    @show dist_list
    # Do a least squares fit..
    x = log.(Nz_list[1:end])
    y = log.(dist_list[1:end])
    @test β = (x' * x) \ (x' * y) < -1.0
end

# End of file runtests.jl
