# -*- Encoding: UFTF-8 -*-


"""Test the elliptic solver."""


push!(LOAD_PATH, "/Users/ralph/source/repos/picfun/implicit")

using grids: grid_1d, init_grid
using solvers: ∇⁻²
using Statistics
using LinearAlgebra

Lz = 2π
Nz = 32

zgrid = init_grid(Lz, Nz)
zvals = (0:Nz - 1) * zgrid.Δz 

# Calculate a Gaussian and its second derivative

sigma = Lz / 10.0
u = exp.(-0.5 * (zvals .- 0.5 * Lz).^ 2.0/ sigma / sigma)
d2u = u .* ((zvals .- 0.5 * Lz).^2.0 .- sigma * sigma) / sigma^4.0;

# Analyitcally invert second derivative
u_num = ∇⁻²(d2u, zgrid)

print(norm((u .- mean(u)) - u_num))

