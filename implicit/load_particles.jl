
#
# Implementation of particle loading schemes
#

module load_particles

export load_pert_x

using NLsolve
using Distributions
using Random
using particles: particle

#""" See Birdsall, Langdon, chapter 16"""

@doc """
load_pert_x

Loads particle position and velocity according to a perturbed Maxwellian function

    f(x, v) = [1 + ϵ cos(kx)] f0(v)

Where f0(v) is a Maxwellian:
fM(v) = √(m / 2 / kB / T) exp(-m v^2 / 2 / k / T)

""" ->
function load_pert_x(num_ptl, L, ϵ, k, vth)

    Random.seed!(1)
    println("Initializing perturbation")
    # Load the particle velocities from a Maxwellian Distribution
    ptl_v = rand(Normal(0.0, 1.0), num_ptl)
    # Load the particle positions by sampling from 1 + \eps cos(kx) for 0 < x < L
    r = rand(Uniform(1e-6, L-1e-6), num_ptl)

    # g(x) = 1 + ϵ cos (kx)
    # Define CDF:
    # G(x) = int_{0}^{x} g(y) dy
    # and assume r ~ Uniform[0;1]. Then:
    # r = int_{0}^{x} g(y) dy = x + ϵ sin(kx) / k
    # Now we need to solve the equation above for x
    # See Langdon, Chapter 16.2

    ptl_z = zeros(num_ptl)
    # Contains random numbers uniformly sampled from [0:1]
    vals = rand(num_ptl)

    # The zeros of f! give the inverted CDF for the distribution
    # f(x) = 1 + ϵ cos(kx)
    # on the invertval [0:L]
    f! = function(res, x, ϵ, k, L, val)
        res[1] = x[1]./L .+ ϵ .* sin.(k .* x[1]) ./ k ./ L .+ val
    end

    # For each particle in [0:1], invert the CDF
    for idx in 1:num_ptl
        val = vals[idx]
        f2!(res, x) = f!(res, x, ϵ, k, L, val)
        sol = nlsolve(f2!, [0.1])
        #println("$(val), $(sol.zero)")
        ptl_z[idx] = sol.zero[1]
    end

    # function f!(dx, x, r, ϵ, k)
    #     dx .= x .+ ϵ * sin.(k .* x) ./ k .- r
    # end
    # dx = zeros(num_ptl)
    # sol = nlsolve((dx, x) -> f!(dx, x, r, ϵ, k), zeros(num_ptl))
    # ptl_z = copy(sol.zero)
    # sort!(ptl_z)

    # Generate a vector of particles
    ptl_vec = Array{particle}(undef, num_ptl)
    for idx ∈ 1:num_ptl
        ptl_vec[idx] = particle(abs(ptl_z[idx]), ptl_v[idx])
    end

    return(ptl_vec)
end
#
# using Plots
# # Below is some code to test particle generation
# ϵ = 0.1
# k = 3.0
# Lz = 2π
# Nz = 32
# Δz = Lz / Nz
# zrg = 0:Δz:Lz
#
# f! = function(res, x, ϵ, k, L, val)
#     res[1] = x[1]./L .+ ϵ .* sin.(k .* x[1]) ./ k ./ L .+ val
# end
#
# num_ptl = 250_000
# sol_vec = zeros(num_ptl)
# vals = rand(num_ptl)
# for idx in 1:num_ptl
#     val = vals[idx]
#     f2!(res, x) = f!(res, x, ϵ, k, Lz, val)
#     sol = nlsolve(f2!, [0.1])
#     #println("$(val), $(sol.zero)")
#     sol_vec[idx] = sol.zero[1]
# end
#
# histogram(sol_vec, normalize=true, bins=200)
# plot!(zrg .- Lz, (1. .+ ϵ * cos.(k .* (zrg .- Lz))) ./ Lz)

end
