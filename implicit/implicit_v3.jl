#  Implementation of the implicit scheme presented in 
# G. Chen et al. Journ., Comp. Phys 230 7018 (2011).
#
#
# This builds on v2, but uses NLSolve to find the electric field update


using Interpolations
using Plots
using Distributions
using StatsFuns
using Random
using LinearAlgebra: norm
using NLsolve

push!(LOAD_PATH, pwd())

using units: qₑ, qᵢ, ϵ₀, mᵢ, mₑ
using grids: grid_1d, init_grid
using pic_utils: smooth, deposit
#using load_particles: load_pert_x
using particles: particle, fix_position!
using particle_push: push_v3!
using solvers: ∇⁻²

# Time-stepping parameters
Δt = 1e-6
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
particle_per_cell = 64
num_ptl = Nz * particle_per_cell
println("Nz = $Nz, L = $Lz, num_ptl = $num_ptl")

# Initialize  electron and ion population
ptlᵢ = Array{particle}(undef, num_ptl)
ptlₑ = Array{particle}(undef, num_ptl)

# Initial position for the ions
ptl_pos = rand(Uniform(0, zgrid.Lz), num_ptl)
sort!(ptl_pos)
# Initial position for the electrons
ptl_perturbation = rand(Uniform(-1e-2, 1e-2), num_ptl)


# Initialize stationary electrons and ions.
# The electron positions are perturbed slightly around the ions
for idx ∈ 1:num_ptl
    ptlᵢ[idx] = particle(ptl_pos[idx], 0.0)
    ptlₑ[idx] = particle(ptl_pos[idx] + ptl_perturbation[idx], 0.0)
    fix_position!(ptlₑ[idx], zgrid.Lz)
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

plot(map(p -> p.pos, ptlₑ), seriestype=:scatter)
plot!(map(p -> p.pos, ptlᵢ), seriestype=:scatter)


# Define a 
function residuals!(res, E_new, E, ptlₑ, ptlᵢ, zgrid)
# Calculates the residuals, given a guess for the electric
# E_new: New electric field
# E: electric field from current time step 
# ptlₑ : Electrons at current time step
# ptlᵢ: Ions at current time step
# zgrid: Simulation Domain
# j_avg_0: 
    

    # Construct a periodic interpolator for E
    _E_per = copy(E)
    push!(_E_per, _E_per[1])
    itp = interpolate(_E_per, BSpline(Linear()))
    itp2 = Interpolations.scale(itp, zrg)
    ip_E = extrapolate(itp2, Periodic())

    # Construct a periodic interpolator for Ẽ
    _E_per = copy(E_new)
    push!(_E_per, _E_per[1])
    itp = interpolate(_E_per, BSpline(Linear()))
    itp2 = Interpolations.scale(itp, zrg)
    ip_Enew = extrapolate(itp2, Periodic())

    # Allocate new vector electrons and ions at half time-step
    ptlₑ½ = Array{particle}(undef, num_ptl)
    ptlᵢ½ = Array{particle}(undef, num_ptl)

    # Particle enslavement: Push particles into a consistent state
    push_v3!(ptlₑ, ptlₑ₀, ptlₑ½, qₑ, mₑ, ϵᵣ, ϵₐ, zgrid, Δt, ip_Enew, ip_E)
    push_v3!(ptlᵢ, ptlᵢ₀, ptlᵢ½, qᵢ, mᵢ, ϵᵣ, ϵₐ, zgrid, Δt, ip_Enew, ip_E)

    # Calculate j_i^{n+1/2}
    j_n12_e = deposit(ptlₑ½, zgrid, p -> p.vel * qₑ / zgrid.Δz)
    j_n12_i = deposit(ptlᵢ½, zgrid, p -> p.vel * qᵢ / zgrid.Δz)
    j_n12 = j_n12_e + j_n12_i
    j_n12_avg = sum(j_n12) / zgrid.Lz

    # Calculate the residual of Eq. (23)
    res_new = ϵ₀ .* (E_new - E) ./ Δt .+ (j_n12 .- j_n12_avg)
    # This is a mutating function. Update the entries of res one-by-one
    for ii ∈ 1:length(res)
        res[ii] = res_new[ii]
    end
    println("Residual norm: $(norm(res))")
end

res_func!(res_vec, E_initial) = residuals!(res_vec, E_initial, Eⁿ, ptlₑ, ptlᵢ, zgrid)

max_E = max(abs.(Eⁿ)...)

E_new = Eⁿ + rand(Uniform(-0.01 * max_E, 0.01 * max_E), length(Eⁿ))

plot(Eⁿ, seriestype=:scatter)
plot!(E_new, seriestype=:scatter)

res_new = zero(Eⁿ)
#res_func!(res_new, E_new)
#residuals!(res_new, E_new, Eⁿ, ptlₑ, ptlᵢ, zgrid)
#println("New residual norm: $(norm(res_new))")

result = nlsolve(res_func!, E_new; xtol=1e-4)
