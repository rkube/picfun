#  Implementation of the implicit scheme presented in
# G. Chen et al. Journ., Comp. Phys 230 7018 (2011).
#
#
# This builds on v3, but uses Picard iteration to solve the electric field update.


using Interpolations
using Plots
using Distributions
using Statistics
using Random
using LinearAlgebra: norm
using NLsolve
using Printf

push!(LOAD_PATH, pwd())

using units: qₑ, qᵢ, mᵢ, mₑ, n₀
using grids: grid_1d, init_grid
using pic_utils: smooth, deposit
using particles: particle, fix_position!
using particle_push: push_v3!
using solvers: ∇⁻², invert_laplace
using diagnostics: diag_ptl, diag_energy, diag_fields


# Time-stepping parameters
# Time is in units of ωpe
Δt = 1e-1
Nt = 100
Nν = 1
Δτ = Δt / Nν

# Domain parameters
# Length is in units of λde
Lz = 2π
Nz = 64
num_ptl = 8192

# Relative and absolute tolerance for convergence of Picard iteration
ϵᵣ = 1e-3
ϵₐ = 1e-8

# Initialize the grid
zgrid = init_grid(Lz, Nz)
zrg = (0:Nz) * zgrid.Δz

# Set random seed
Random.seed!(1)

# Initial number of particles per cell
#particle_per_cell = Int(n₀ ÷ Nz)
#num_ptl = Nz * particle_per_cell
println("Nz = $Nz, L = $Lz, num_ptl = $num_ptl")

# Initialize  electron and ion population
ptlᵢ₀ = Array{particle}(undef, num_ptl)
ptlₑ₀ = Array{particle}(undef, num_ptl)

# Initial position for electrons and ions
ptl_pos = range(0.0, step=zgrid.Lz / num_ptl, length=num_ptl)

# Initialize stationary electrons and ions.
# The electron positions are perturbed slightly around the ions
for idx ∈ 1:num_ptl
    x0 = ptl_pos[idx]
    ptlᵢ₀[idx] = particle(x0, 0.0)
    ptlₑ₀[idx] = particle(x0 + 1e-2 .* cos(x0), 0.0) 
    fix_position!(ptlₑ₀[idx], zgrid.Lz)
end
# Calculate initial j_avg
j_avg_0 = sum(deposit(ptlₑ₀, zgrid, p -> p.vel * qₑ)) / zgrid.Lz / n₀

# deposit particle density on the grid
nₑ = deposit(ptlₑ₀, zgrid, p -> 1. / n₀)
nᵢ = deposit(ptlᵢ₀, zgrid, p -> 1. / n₀)
ρⁿ = (nᵢ - nₑ)
#ϕⁿ = ∇⁻²(-ρⁿ, zgrid)
ϕⁿ = invert_laplace(-ρⁿ, zgrid)
# Calculate initial electric field with centered difference stencil
Eⁿ = zeros(Nz)
Eⁿ[1] = -1. * (ϕⁿ[end] - ϕⁿ[2]) / 2. / zgrid.Δz
Eⁿ[2:end-1] = -1. * (ϕⁿ[1:end-2] - ϕⁿ[3:end]) / 2. / zgrid.Δz
Eⁿ[end] = -1. * (ϕⁿ[end-1] - ϕⁿ[1]) / 2. / zgrid.Δz
smEⁿ = smooth(Eⁿ)

ptlₑ = deepcopy(ptlₑ₀)
ptlᵢ = deepcopy(ptlᵢ₀)

diag_fields(ptlₑ, ptlᵢ, zgrid, 0)

# Define a
function G(E_new, E, ptlₑ₀, ptlᵢ₀, ptlₑ, ptlᵢ, zgrid)
# Calculates the residuals, given a guess for the electric
# E_new: New electric field
# E: electric field from current time step
# ptlₑ₀ : Electrons at current time step
# ptlᵢ₀ : Ions at current time step
# ptlₑ  : Electrons consistent with E_new
# ptlᵢ  : Ions consistent with E_new
# zgrid: Simulation Domain

    #println("Function residuals. E_new[1] = $(E_new[1]), E[1]=$(E[1])")

    # Construct a periodic interpolator for E
    _E_per = deepcopy(E)
    push!(_E_per, _E_per[1])
    itp = interpolate(_E_per, BSpline(Linear()))
    itp2 = Interpolations.scale(itp, zrg)
    ip_E = extrapolate(itp2, Periodic())

    # Construct a periodic interpolator for Ẽ
    _E_per = deepcopy(E_new)
    push!(_E_per, _E_per[1])
    itp = interpolate(_E_per, BSpline(Linear()))
    itp2 = Interpolations.scale(itp, zrg)
    ip_Enew = extrapolate(itp2, Periodic())

    # Allocate vector for particle position half time-step
    ptlₑ½ = deepcopy(ptlₑ)
    ptlᵢ½ = deepcopy(ptlₑ)

    # Particle enslavement: Push particles into a consistent state
    # ptlₑ and ptlᵢ will be updated.
    push_v3!(ptlₑ, ptlₑ₀, ptlₑ½, qₑ, 1.0, ϵᵣ, ϵₐ, zgrid, Δt, ip_Enew, ip_E)
    push_v3!(ptlᵢ, ptlᵢ₀, ptlᵢ½, qᵢ, mᵢ / mₑ, ϵᵣ, ϵₐ, zgrid, Δt, ip_Enew, ip_E)

    # Calculate j_i^{n+1/2}
    j_n12_e = deposit(ptlₑ½, zgrid, p -> p.vel * qₑ) ./ n₀ ./ zgrid.Δz
    j_n12_i = deposit(ptlᵢ½, zgrid, p -> p.vel * qᵢ) ./ n₀ ./ zgrid.Δz
    j_n12 = j_n12_e + j_n12_i
    j_n12_avg = mean(j_n12)

    # Calculate the residual of Eq. (23)
    residuals = (E_new - E) ./ Δt .+ (smooth(j_n12) .- j_n12_avg)

    return residuals
end


for nn in 1:Nt
    println("======================== $(nn)/$(Nt)===========================")
  
    # Picard iteration to get electric field
    E_converged = false
    num_it = 0

    global ptlₑ₀ = deepcopy(ptlₑ)
    global ptlᵢ₀ = deepcopy(ptlᵢ)

    # Guess a new E field.
    delta_E = std(abs.(smEⁿ))
    E_guess = smooth(smEⁿ + rand(Uniform(-0.1 * delta_E, 0.1 * delta_E), length(smEⁿ)))

    while (E_converged == false)
        # Updates residuals
        res_vec = G(E_guess, smEⁿ, ptlₑ₀, ptlᵢ₀, ptlₑ, ptlᵢ, zgrid)
        println("           it $(num_it)/100: Residual = $(norm(res_vec)).")

        # Break if residuals are stationary
        if (norm(res_vec) ≤ ϵᵣ * norm(smEⁿ) + ϵₐ)
            E_converged = true
            println("              -> converged.")
        end

        # Update guessed E-field
        E_guess -= smooth(Δt .* res_vec)

        # Break if too many iterations
        num_it += 1
        if (num_it > 100)
            E_converged = true
            
        end
    end
    # Update smEⁿ with new electric field
    global smEⁿ[:] = E_guess
    println("New norm(smEⁿ) = $(norm(smEⁿ))")    

    if mod(nn, 1) == 0
        diag_ptl(ptlₑ, ptlᵢ, nn)
    end
    diag_energy(ptlₑ, ptlᵢ, smEⁿ, nn, zgrid)
    diag_fields(ptlₑ, ptlᵢ, zgrid, nn)
end