#
# Implementation of particle loading schemes
#

module load_particles

export load_pert_x

using NLsolve
using Distributions
using particles: particle

@doc """
load_pert_x

Loads particle position and velocity according to a perturbed Maxwellian function

    f(x, v) = [1 + ϵ cos(kx)] f0(v)

Where f0(v) is a Maxwellian:
fM(v) = √(m / 2 / kB / T) exp(-m v^2 / 2 / k / T)

""" ->
function load_pert_x(num_ptl, L, ϵ, k, vth)
    print("Initializing perturbation")
    # Load the particle velocities from a Maxwellian Distribution
    ptl_v = rand(Normal(0.0, 1.0), num_ptl)

    # Load the particle positions by sampling from 1 + \eps cos(kx) for 0 < x < L
    r = rand(Uniform(1e-6, L-1e-6), num_ptl)

    # g(x) = 1 + eps cos (kx)
    # Define CDF:
    # G(x) = int_{0}^{x} g(y) dy
    # and assume r ~ Uniform[0;1]. Then:
    # r = int_{0}^{x} g(y) dy = x + eps sin(kx) / k
    # Now we need to solve the equation above for x

    function f!(dx, x, r, ϵ, k)
        dx .= x .+ ϵ * sin.(k .* x) ./ k .- r
    end
    dx = zeros(num_ptl)
    sol = nlsolve((dx, x) -> f!(dx, x, r, ϵ, k), zeros(num_ptl))

    ptl_z = copy(sol.zero)
    sort!(ptl_z)


    ptl_vec = Array{particle}(undef, num_ptl)
    # Generate a vector of particle
    idx = 0
    for ptl in zip(ptl_z, ptl_v)
        ptl_vec[idx] = particle(ptl[0], ptl[1])
        idx += 1
    end

    return(ptl_vec)
end


end
