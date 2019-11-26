#  Implementation of an implicit scheme, advised by Ben.


using Interpolations
using Plots
using Distributions
using StatsFuns
using Random
using LinearAlgebra: norm

push!(LOAD_PATH, pwd())

using units: qₑ, qᵢ, ϵ₀, mᵢ, mₑ
using grids: grid_1d, init_grid
using pic_utils: smooth, deposit
using load_particles: load_pert_x
using particles: particle, fix_position!
using solvers: ∇⁻²

# Time-stepping parameters
Δt = 1e-3
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
particle_per_cell = 32
num_ptl = Nz * particle_per_cell
println("Nz = $Nz, L = $(zgrid.Lz), num_ptl = $num_ptl")

# Initialize  electron and ion population
ptl_e = load_pert_x(num_ptl, zgrid.Lz, 0.1, 20.0, 1.0)
ptl_i = Array{particle}(undef, num_ptl)
ptl_i_z = rand(Uniform(0, zgrid.Lz), num_ptl)

# Initialize ions with zero velocity
sort!(ptl_i_z)
for idx ∈ 1:num_ptl
    ptl_i[idx] = particle(ptl_i_z[idx], 0.0)
end

# deposit particle density on the grid
nₑ = deposit(ptl_e, zgrid, p -> 1.)
nᵢ = deposit(ptl_i, zgrid, p -> 1.)
ρ₀ = (nᵢ - nₑ) / ϵ₀
ϕ₀ = ∇⁻²(-ρ₀, zgrid)
# Calculate initial electric field with centered difference stencil
E₀ = zeros(Nz)
E₀[1] = (ϕ₀[2] - ϕ₀[end]) * 0.5 / zgrid.Δz
E₀[2:end-1] = (ϕ₀[1:end-2] - ϕ₀[3:end]) * 0.5 / zgrid.Δz
E₀[end] = (ϕ₀[end-1] - ϕ₀[1]) * 0.5 / zgrid.Δz

plot(ϕ₀)
plot!(E₀)


ptlₑ₀ = copy(ptl_e)
ptlᵢ₀ = copy(ptl_i)
ptl_e12 = Array{particle}(undef, num_ptl)

# Calculate initial j_avg
j_avg_0 = sum(deposit(ptl_e, zgrid, p -> p.vel * qₑ / zgrid.Δz)) / zgrid.Lz

# Construct a periodic interpolator for E₀
_E_per = copy(E₀)
push!(_E_per, _E_per[1])
itp = interpolate(_E_per, BSpline(Linear()))
itp2 = Interpolations.scale(itp, zrg)
E_ip_old = extrapolate(itp2, Periodic())

# Construct a periodic interpolator for new E-field
E_new = E₀ + rand(-1e-2:1e-4:1e-2, Nz)
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
    for ele ∈ 1:num_ptl
        #println("Iterating particle $(ele) / $(num_ptl)")
        # Initial guess of new position and velocity before we start iterations.
        xp_guess = ptl_e[ele].pos + 0.1
        num_it_ptl = 0
        ptl_converged=false
        # Then: iterate the particle's coordinates until convergence
        while(ptl_converged == false)
            # Calculate x_p^{n+1/2}
            xₚⁿ⁺½ =  0.5 * (ptlₑ₀[ele].pos + xp_guess)
            # Calculate v_p^{n+1}
            vp_new = ptlₑ₀[ele].vel + Δt * qₑ * 0.5 * (E_ip_new(xₚⁿ⁺½) + E_ip_old(xₚⁿ⁺½)) / mₑ
            # Calculate v_p^{n+1/2}
            vₚⁿ⁺½ = 0.5 * (ptlₑ₀[ele].vel + vp_new)
            #vp_n12 = 0.5 * (ptl_e0[ele].vel + vp_new)
            # Calculate x_p^{n+1}
            xp_new = ptlₑ₀[ele].pos + Δt * vₚⁿ⁺½
            #println("*** it $(num_it_ptl): xp_n12=$(xp_n12), xp_new=$(xp_new), vp_n12=$(vp_n12), vp_new=$(vp_new)")

            # Check convergence
            if ((abs(xp_new - xp_guess) ≤ rel_tol * abs(xp_new) + abs_tol))
                #println("*** Converged: |x - x_new| = $(abs(xp_new - xp_guess))")
                #println("***            |v - v_new| = $(abs(vp_new - ptl_e[ele].vel))")
                #println("***            with xp_new=$(xp_new), vp_new=$(vp_new)")
                ptl_converged = true
                ptl_e[ele].pos = xp_new
                ptl_e[ele].vel = vp_new

                #println("New values $(ptl_e[ele].pos), $(ptl_e[ele].vel)")
                break
            end
            # Let xp_new and vp_new be the new guesses.
            xp_guess = xp_new
            num_it_ptl += 1
            if(num_it_ptl > 100)
                println("Iterations exceeded $(num_it_ptl), terminating")
                ptl_converged = true
                break
            end
        end #while_ptl_converged==false
        fix_position!(ptl_e[ele], zgrid.Lz)
        # Update ptl_e12 for the current particle
        ptl_e12[ele] = particle(0.5 * (ptl_e[ele].pos + ptlₑ₀[ele].pos),
                                0.5 * (ptl_e[ele].vel + ptlₑ₀[ele].vel))
        fix_position!(ptl_e12[ele], zgrid.Lz)
    end #for ptl in ptl_e

    # Calculate j_i^{n+1/2}
    j_new = deposit(ptl_e12, zgrid, p -> p.vel * qₑ / zgrid.Δz)
    global E_new = Δt * (j_avg_0 .- j_new) / ϵ₀ + E₀

    # Update interpolator for E_new
    _E_per = copy(E_new)
    push!(_E_per, _E_per[1])
    itp = interpolate(_E_per, BSpline(Linear()))
    itp2 = Interpolations.scale(itp, zrg)
    global E_ip_new = extrapolate(itp2, Periodic())

    if(num_it_E % 100 == 0)
        p = plot!(1:Nz, E_new, label="Iteration $(num_it_E)")
        display(p)
    end

    println("Iteration: $(num_it_E): norm = $(norm(E_new - E₀))")

    if((norm(E_new - E₀) ≤ 1e-2 * norm(E₀) + abs_tol))
        println("||E_new - E_old|| = $(norm(E_new-E₀)) ≤ $(rel_tol * norm(E₀) + abs_tol)")
        global E_converged = true
        break
    end
    #Take current E to be E₀
    E₀[:] = E_new[:]
    _E_per = copy(E₀)
    push!(_E_per, _E_per[1])
    itp = interpolate(_E_per, BSpline(Linear()))
    itp2 = Interpolations.scale(itp, zrg)
    global E_ip_old = extrapolate(itp2, Periodic())

    if(num_it_E > 500)
        println("Iteration for E: Iterations exceed 10, terminating")
        global E_converged = true
        break
    end
    global num_it_E += 1
end
