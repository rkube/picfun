#  Implementation of an implicit scheme, advised by Ben.


using Interpolations
using Plots
using Distributions
using StatsFuns
using Random

push!(LOAD_PATH, pwd())

using units: qe, qi, ϵ0, mi, me
using grids: grid_1d, init_grid
using pic_utils: smooth, deposit
using load_particles: load_pert_x
using particle_push: push_v1!
using particles: particle, fix_position!


# Time-stepping parameters
Δt = 1.0
Nt = 1
Nν = 1
Δτ = Δt / Nν

# Relative and absolute tolerance for convergence of Picard iteration
rel_tol = 1e-6
abs_tol = 1e-10

# Domain parameters
Lz = 2. * π
Nz = 32

# Initialize the grid
zgrid = init_grid(Lz, Nz)
zrg = range(0, Nz) * zgrid.Δz

# Set random seed
Random.seed!(1)

# Initial number of particles per cell
particle_per_cell = 16
num_ptl = Nz * particle_per_cell
println("Nz = $Nz, L = $(zgrid.Lz), num_ptl = $num_ptl")

# Initialize  electron and ion population
ptl_e = load_pert_x(num_ptl, zgrid.Lz, 0.1, 20.0, 1.0)
ptl_i = Array{particle}(undef, num_ptl)
ptl_i_z = rand(Uniform(1e-6, zgrid.Lz - 1e-6), num_ptl)

# Initialize ions with zero velocity
sort!(ptl_i_z)
for idx ∈ 1:num_ptl
    ptl_i[idx] = particle(ptl_i_z[idx], 0.0)
end

ptl_e0 = copy(ptl_e)
ptl_i0 = copy(ptl_i)

E_initial = zeros(Nz)
_E_per = copy(E_initial)
push!(_E_per, _E_per[1])
itp = interpolate(_E_per, BSpline(Linear()))
itp2 = Interpolations.scale(itp, zrg)
E_ip_old = extrapolate(itp2, Periodic())


# Define flags for iterations
E_converged = false
ptl_converged = false

# Outer loop: Guess Etilde
E_guess = 0.01 .+ E_initial
_E_per = copy(E_guess)
push!(_E_per, _E_per[1])
itp = interpolate(_E_per, BSpline(Linear()))
itp2 = Interpolations.scale(itp, zrg)
E_ip_new = extrapolate(itp2, Periodic())

while(E_converged == false)
    # Iterate over particles and make their position and velocity consistent
    # with the current guess of the electric field
    for ele ∈ range(1, stop=num_ptl)
        print("Iterating particle $(ele) / $(num_ptl)")
        # Guess a new position for the particle
        xp_new_guess = ptl_e[ele].pos + 1e-4
        vp_new_guess = ptl_e[ele].vel

        num_it_ptl = 0
        # Then: iterate the particle's coordinates until convergence
        while(ptl_converged == false)
            # Guess a new position for the electron
            # Calculate x_p^{n+1/2}
            xp_n12 = 0.5 * (ptl_e[ele].pos + xp_new_guess)
            # Calculate v_p^{n+1}
            vp_new = ptl_e[ele].vel + Δt * qe * (E_ip_new(xp_n12) + E_ip_old(xp_n12)) / me /2.
            # Calculate v_p^{n+1/2}
            vp_n12 = 0.5 * (ptl_e[ele].vel + vp_new)
            # Calculate x_p^{n+1}
            xp_new = ptl_e[ele].pos + Δt * vp_n12

            # Check convergence
            if ((abs(xp_new - xp_new_guess) ≤ rel_tol * abs(xp_new) + abs_tol)
                & (abs(vp_new - vp_new_guess) ≤ rel_tol * abs(vp_new) + abs_tol))
                ptl_converged = true
            end
            xp_new_guess = xp_new
            vp_new_guess = vp_new
            num_it_ptl += 1

            if(num_it_ptl > 100)
                print("Iterations exceeded $(num_it_ptl), terminating")
                break
            end
        end
    end
end



    #while not converged:

end # particle iteration
end
