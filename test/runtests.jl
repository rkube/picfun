#Encoding: UTF-8 -*-

using picfun
using Test
using Statistics
using LinearAlgebra

@testset "solver_tests" begin

    Lz = 2π
    Nz_list = [16; 32; 64; 128; 256]
    dist_solver1 = []
    dist_solver2 = []

    ctr = 1
    for Nz ∈ Nz_list
        zgrid = init_grid(Lz, Nz)
        # Expand colocation points. 
        zvals = (0:Nz-1) * zgrid.Δz     

        # Calculate a Gaussian and its second derivative
        σ = Lz / 10.0
        # f(z) is a Gaussian of width σ
        u = exp.(-0.5 * (zvals .- 0.5 * Lz).^ 2.0 / σ / σ)
        # The second derivative is given analytically
        d2u = u .* ((zvals .- 0.5 * Lz).^2.0 .- σ * σ) / σ^4.0;

        # Invert the second derivative to retrieve numerical approximation to u
		u_num1 = invert_laplace(d2u, zgrid)
		u_num2 = inv_laplace2(d2u, zgrid)
	
	# Subtract the mean from the profile before comparing
#     dist_list[ctr] = norm((u .- mean(u)) - u_num)
	# @show norm(u - u_num)
		append!(dist_solver1, sqrt(Δz * sum((u - u_num1).^2)))
		append!(dist_solver2, sqrt(Δz * sum(((u .- mean(u)) - u_num2[1:Nz]).^2)))
        
        # u_num = inv_laplace2(d2u, zgrid)
        # dist_solver2[ctr] = norm(u - u_num)
        # ctr += 1
    end

    β1 = log10.(Lz ./ Nz_list) \ log10.(dist_solver1)
    β2 = log10.(Lz ./ Nz_list) \ log10.(dist_solver2)
    @show β1, β2
    @test β1 > 2
    @test β2 > 2

    # y = log.(dist_solver2[1:end])
    # β2 = x \ y
    # @show β2
    # @test β2 < -1.0


end

# End of file runtests.jl
