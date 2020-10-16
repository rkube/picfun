#  Implementation of the implicit scheme presented in
# G. Chen et al. Journ., Comp. Phys 230 7018 (2011).
#
#
# This builds on v2, but uses NLSolve to find the electric field update


using Interpolations
using Plots
using Distributions
#using StatsFuns
using Statistics
using Random
using LinearAlgebra: norm
using NLsolve
using Printf

push!(LOAD_PATH, pwd())

using units: qₑ, qᵢ, ϵ₀, mᵢ, mₑ
using grids: grid_1d, init_grid
using pic_utils: smooth, deposit
#using load_particles: load_pert_x
using particles: particle, fix_position!
using particle_push: push_v3!
using solvers: ∇⁻²
using diagnostics: diag_ptl, diag_energy

# Time-stepping parameters
Δt = 1e-3
Nt = 100
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
particle_per_cell = 128
num_ptl = Nz * particle_per_cell
println("Nz = $Nz, L = $Lz, num_ptl = $num_ptl")

# Initialize  electron and ion population
ptlᵢ₀ = Array{particle}(undef, num_ptl)
ptlₑ₀ = Array{particle}(undef, num_ptl)

# Initial position for the ions
ptl_pos = rand(Uniform(0, zgrid.Lz), num_ptl)
sort!(ptl_pos)
# Initial position for the electrons
ptl_perturbation = rand(Uniform(-1e-3, 1e-3), num_ptl)

# Initialize stationary electrons and ions.
# The electron positions are perturbed slightly around the ions
for idx ∈ 1:num_ptl
    ptlᵢ₀[idx] = particle(ptl_pos[idx], 0.0)
    ptlₑ₀[idx] = particle(ptl_pos[idx] + ptl_perturbation[idx], 0.0)
    fix_position!(ptlₑ₀[idx], zgrid.Lz)
end
# Calculate initial j_avg
j_avg_0 = sum(deposit(ptlₑ₀, zgrid, p -> p.vel * qₑ / zgrid.Δz)) / zgrid.Lz

# deposit particle density on the grid
nₑ = deposit(ptlₑ₀, zgrid, p -> 1.)
nᵢ = deposit(ptlᵢ₀, zgrid, p -> 1.)
ρⁿ = (nᵢ - nₑ) / ϵ₀
ϕⁿ = ∇⁻²(-ρⁿ, zgrid)
# Calculate initial electric field with centered difference stencil
Eⁿ = zeros(Nz)
Eⁿ[1] = -1. * (ϕⁿ[2] - ϕⁿ[end]) / 2. / zgrid.Δz
Eⁿ[2:end-1] = -1. * (ϕⁿ[1:end-2] - ϕⁿ[3:end]) / 2. / zgrid.Δz
Eⁿ[end] = -1. * (ϕⁿ[end-1] - ϕⁿ[1]) / 2. / zgrid.Δz
smEⁿ = smooth(Eⁿ)


ptlₑ = copy(ptlₑ₀)
ptlᵢ = copy(ptlᵢ₀)


plotly()


plot(map(p -> p.pos, ptlₑ))
plot!(map(p -> p.pos, ptlᵢ))

plot(collect(0:zgrid.Δz:zgrid.Lz-0.001), ϕⁿ)
plot!(collect(0:zgrid.Δz:zgrid.Lz-0.001), Eⁿ)
plot!(collect(0:zgrid.Δz:zgrid.Lz-0.001), smEⁿ)

# Define a
function residuals!(res, E_new, E, ptlₑ₀, ptlᵢ₀, ptlₑ, ptlᵢ, zgrid)
# Ca5lculates the residuals, given a guess for the electric
# E_new: New electric field
# E: electric field from current time step
# ptlₑ₀ : Electrons at current time step
# ptlᵢ₀ : Ions at current time step
# zgrid: Simulation Domain
# j_avg_0:

    #println("Function residuals. E_new[1] = $(E_new[1]), E[1]=$(E[1])")

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

    # Allocate vector for particle position half time-step
    num_ptl = length(ptlₑ₀)
    #ptlₑ½ = Array{particle}(undef, num_ptl)
    #ptlᵢ½ = Array{particle}(undef, num_ptl)
    ptlₑ½ = copy(ptlₑ)
    ptlᵢ½ = copy(ptlₑ)

    #println("residuals!: ptlₑ[1] = $(ptlₑ[1])")
    #println("residuals!: ptlₑ½[1] = $(ptlₑ½[1])")
    #println("residuals!: ptl₀[1] = $(ptl₀[1])")

    # Particle enslavement: Push particles into a consistent state
    # ptlₑ and ptlᵢ will be updated.
    push_v3!(ptlₑ, ptlₑ₀, ptlₑ½, qₑ, mₑ, ϵᵣ, ϵₐ, zgrid, Δt, ip_Enew, ip_E)
    push_v3!(ptlᵢ, ptlᵢ₀, ptlᵢ½, qᵢ, mᵢ, ϵᵣ, ϵₐ, zgrid, Δt, ip_Enew, ip_E)

    # Calculate j_i^{n+1/2}
    j_n12_e = deposit(ptlₑ½, zgrid, p -> p.vel * qₑ / zgrid.Δz)
    j_n12_i = deposit(ptlᵢ½, zgrid, p -> p.vel * qᵢ / zgrid.Δz)
    j_n12 = j_n12_e + j_n12_i
    j_n12_avg = mean(j_n12)

    # Calculate the residual of Eq. (23)
    res_new = ϵ₀ / Δt .* (E_new - E) .+ (smooth(j_n12) .- j_n12_avg)
    # This is a mutating function. Update the entries of res one-by-one
    for ii ∈ 1:length(res)
        res[ii] = res_new[ii]
    end

    #println("----------------- Residuals: $(norm(res))")
end
#plot(smEⁿ)


for nn in 1:Nt
    println("======================== $(nn)/$(Nt)===========================")

    res_func!(res_vec, E_guess) = residuals!(res_vec, E_guess, smEⁿ, ptlₑ₀, ptlᵢ₀, ptlₑ, ptlᵢ, zgrid)
    global ptlₑ₀ = copy(ptlₑ)
    global ptlᵢ₀ = copy(ptlᵢ)
    delta_E = mean(abs.(smEⁿ))
    println(smEⁿ)
    E_new = smooth(smEⁿ + rand(Uniform(-1e-3 * delta_E, 1e-3 * delta_E), length(smEⁿ)))
    result = nlsolve(res_func!, E_new; xtol=1e-3, iterations=10000)
    global smEⁿ[:] = smooth(result.zero[:])

    #plot!(smEⁿ)

    println(result)
    #println(result.iterations)
    if mod(nn, 1) == 0
        diag_ptl(ptlₑ, ptlᵢ, nn)
    end
    diag_energy(ptlₑ, ptlᵢ, smEⁿ, nn)

end
