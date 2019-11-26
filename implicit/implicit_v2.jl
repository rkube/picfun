#  Implementation of an implicit scheme, advised by Ben.


using Interpolations
using Plots
using Distributions
using StatsFuns
using Random
using LinearAlgebra: norm

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

# Calculate initial j_avg
j_avg_0 = deposit(ptl_e, zgrid, p -> p.vel * qe)

ptl_e0 = copy(ptl_e)
ptl_i0 = copy(ptl_i)

E_initial = zeros(Nz)
_E_per = copy(E_initial)
push!(_E_per, _E_per[1])
itp = interpolate(_E_per, BSpline(Linear()))
itp2 = Interpolations.scale(itp, zrg)
E_ip_old = extrapolate(itp2, Periodic())

# Outer loop: Guess Etilde
E_new = 0.01 .+ E_initial
_E_per = copy(E_new)
push!(_E_per, _E_per[1])
itp = interpolate(_E_per, BSpline(Linear()))
itp2 = Interpolations.scale(itp, zrg)
E_ip_new = extrapolate(itp2, Periodic())

# Define flags for iterations
E_converged = false
num_it_E = 0
while(E_converged == false)
    # Iterate over particles and make their position and velocity consistent
    # with the current guess of the electric field
    for ele ∈ 1:10
        println("Iterating particle $(ele) / $(num_ptl)")
        # Initial guess of new position and velocity before we start iterations.
        xp_new_guess = 0.1#ptl_e[ele].pos + Δt * ptl_e[ele].vel
        vp_new_guess = 0.02#ptl_e[ele].vel
        num_it_ptl = 0
        E_test = E_ip_new(0.01)
        ptl_converged=false
        # Then: iterate the particle's coordinates until convergence
        while(ptl_converged == false)
            # Guess a new position for the electron
            println("*** ptl $(ele): xp_new_guess=$(xp_new_guess), vp_new_guess=$(vp_new_guess)")
            # Calculate x_p^{n+1/2}
            xp_n12 = 0.5 * (ptl_e[ele].pos + xp_new_guess)
            # Calculate v_p^{n+1}
            vp_new = ptl_e[ele].vel + Δt * qe * 0.5 * (E_ip_new(xp_n12) + E_ip_old(xp_n12)) / me
    #         E_old = E_ip_old(xp_n12)
            # vp_new = ptl_e[ele].vel + Δt * qe * 0.5 * (E_new + E_old) / me
            # Calculate v_p^{n+1/2}
            vp_n12 = 0.5 * (ptl_e[ele].vel + vp_new)
            # Calculate x_p^{n+1}
            xp_new = ptl_e[ele].pos + Δt * vp_n12
            println("*** it $(num_it_ptl): xp_n12=$(xp_n12), xp_new=$(xp_new), vp_n12=$(vp_n12), vp_new=$(vp_new)")

            # Check convergence
            if ((abs(xp_new - xp_new_guess) ≤ rel_tol * abs(xp_new) + abs_tol)
                & (abs(vp_new - vp_new_guess) ≤ rel_tol * abs(vp_new) + abs_tol))
                println("*** $(xp_new), $(xp_new_guess), $(rel_tol), $(xp_new), $(abs_tol)")
                println("*** Converged: |x - x_new| = $(abs(xp_new - xp_new_guess))")
                println("***            |v - v_new| = $(abs(vp_new - vp_new_guess))")
                ptl_converged = true
                break
            end
            # Let xp_new and vp_new be the new guesses.
            xp_new_guess = xp_new
            vp_new_guess = vp_new
            num_it_ptl += 1
            println(" ")
            if(num_it_ptl > 10)
                println("Iterations exceeded $(num_it_ptl), terminating")
                ptl_converged = true
                break
            end
        end # while_ptl_converged==false
        # Check convergence of #
    end # for ptl in ptl_e

    j_avg_new = deposit(ptl_e, zgrid, p -> p.vel * qe)

    E_new = Δt * (j_avg_new - j_avg_0) / ϵ0 + E_initial

    if((norm(E_new - E_initial) ≤ rel_tol * norm(E_initial) + abs_tol))
        println("Iteration: $(num_it_E)")
        println("||E_new - E_old|| = $(norm(E_new-E_initial)) ≤ $(rel_tol) * norm(E_new) + $(abs_tol)")
        E_converged = true
        break
    end

    if(num_it_E > 10)
        println("Iterations exceed 10, terminating")
        E_converged = true
        break

    end
    global num_it_E += 1


end
