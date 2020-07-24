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
using particle_push: push_v2!
using solvers: ∇⁻²

# Time-stepping parameters
Δt = 1e-3
Nt = 1
Nν = 1
Δτ = Δt / Nν

# Relative and absolute tolerance for convergence of Picard iteration
ϵᵣ = 1e-4
ϵₐ = 1e-8

# Domain parameters
Lz = 2π
Nz = 32

# Initialize the grid
zgrid = init_grid(Lz, Nz)
zrg = (0:Nz) * zgrid.Δz

# Set random seed
Random.seed!(1)

# Initial number of particles per cell
particle_per_cell = 16
num_ptl = Nz * particle_per_cell
println("Nz = $Nz, L = $Lz, num_ptl = $num_ptl")

# Initialize  electron and ion population
ptlₑ = load_pert_x(num_ptl, zgrid.Lz, 0.1, 2.0, 1.0)
ptlᵢ = Array{particle}(undef, num_ptl)
ptl_iz = rand(Uniform(0, zgrid.Lz), num_ptl)

# Initialize ions with zero velocity
sort!(ptl_iz)
for idx ∈ 1:num_ptl
    ptlᵢ[idx] = particle(ptl_iz[idx], 0.0)
end
# Calculate initial j_avg
j_avg_0 = sum(deposit(ptlₑ, zgrid, p -> p.vel * qₑ / zgrid.Δz)) / zgrid.Lz

# deposit particle density on the grid
nₑ = deposit(ptlₑ, zgrid, p -> 1.)
nᵢ = deposit(ptlᵢ, zgrid, p -> 1.)
ρⁿ = (nᵢ - nₑ) / ϵ₀
ϕⁿ = ∇⁻²(-ρⁿ, zgrid)
# Calculate initial electric field with centered difference stencil
Eⁿ = zeros(Nz)
Eⁿ[1] = (ϕⁿ[2] - ϕⁿ[end]) * 0.5 / zgrid.Δz
Eⁿ[2:end-1] = (ϕⁿ[1:end-2] - ϕⁿ[3:end]) * 0.5 / zgrid.Δz
Eⁿ[end] = (ϕⁿ[end-1] - ϕⁿ[1]) * 0.5 / zgrid.Δz

ptlₑ₀ = copy(ptlₑ)
ptlᵢ₀ = copy(ptlᵢ)
ptlₑ½ = Array{particle}(undef, num_ptl)

# Construct a periodic interpolator for E₀
_E_per = copy(Eⁿ)
push!(_E_per, _E_per[1])
itp = interpolate(_E_per, BSpline(Linear()))
itp2 = Interpolations.scale(itp, zrg)
ip_Eⁿ = extrapolate(itp2, Periodic())

# Guess the next electric field
Ẽ = Eⁿ + rand(-1e-1:1e-3:1e-1, Nz)
_E_per = copy(Ẽ)
push!(_E_per, _E_per[1])
itp = interpolate(_E_per, BSpline(Linear()))
itp2 = Interpolations.scale(itp, zrg)
ip_Ẽ = extrapolate(itp2, Periodic())

plot(Eⁿ, label="Eⁿ")
plot!(Ẽ, label="initial Ẽ")

# Define flags for iterations
E_converged = false
num_it_E = 0
while(E_converged == false)
    # Iterate over particles and make their position and velocity consistent
    # with the current guess of the electric field
    push_v2!(ptlₑ, ptlₑ₀, ptlₑ½, ϵᵣ, ϵₐ, zgrid, Δt, ip_Ẽ, ip_Eⁿ)
    # Calculate j_i^{n+1/2}
    j_new = deposit(ptlₑ½, zgrid, p -> p.vel * qₑ / zgrid.Δz)
    # Calculate electric field resulting from particle push
    global E_new = Δt * (j_avg_0 .- j_new) / ϵ₀ + Eⁿ

    if(num_it_E % 100 == 0)
        p = plot!(1:Nz, E_new, label="Iteration $(num_it_E)")
        display(p)
    end

    println("Iteration: $(num_it_E): ‖E_new - Ẽ‖ = $(norm(E_new - Ẽ))")

    if((norm(Ẽ - E_new) ≤ ϵᵣ* norm(E_new) + ϵₐ))
        println("‖Ẽ - Eⁿ‖ = $(norm(Ẽ - Eⁿ)) ≤ $(ϵᵣ * norm(Eⁿ) + ϵₐ)")
        global E_converged = true
        break
    end

    # Update E_new to be Ẽ
    global Ẽ[:] = E_new[:]
    _E_per = copy(E_new)
    push!(_E_per, _E_per[1])
    itp = interpolate(_E_per, BSpline(Linear()))
    itp2 = Interpolations.scale(itp, zrg)
    global ip_Ẽ = extrapolate(itp2, Periodic())

    if(num_it_E > 500)
        println("Iteration for E: Iterations exceed 10, terminating")
        global E_converged = true
        break
    end
    global num_it_E += 1
end
