#  Implementation of the implicit scheme presented in
# G. Chen et al. Journ., Comp. Phys 230 7018 (2011).
#
#
# This builds on v3, but uses Picard iteration to solve the electric field update.


using Interpolations
using Distributions
using Statistics
using Random
using LinearAlgebra: norm
using Printf

push!(LOAD_PATH, pwd())

using units: qₑ, qᵢ, mₑ, mᵢ
using grids: grid_1d, init_grid
using pic_utils: smooth, deposit
using particles: particle, fix_position!
using particle_push: push_v3!
using solvers: ∇⁻², invert_laplace
using diagnostics: diag_ptl, diag_energy, diag_fields

using Plots
#
const n₀ = 1.0
const num_ptl = 32768

# Time-stepping parameters
# Time is in units of ωpe
const Δt = 1e-1
const Nt = 1

# Domain parameters
# Length is in units of λde
const Lz = 2π
const Nz = 32

# Relative and absolute tolerance for convergence of Picard iteration
const ϵᵣ = 1e-6
const ϵₐ = 1e-8
const max_iter_E = 10000

const ptl_per_cell = num_ptl ÷ Nz
const ptl_wt = n₀ / ptl_per_cell

# Initialize the grid
zgrid = init_grid(Lz, Nz)

# Set random seed
Random.seed!(1)

# Initial number of particles per cell
println("Nz = $(Nz), L = $(Lz), ptl_wt = $(ptl_wt)")

# Initialize  electron and ion population
ptlᵢ₀ = Array{particle}(undef, num_ptl)
ptlₑ₀ = Array{particle}(undef, num_ptl)
#
ptlᵢ = Array{particle}(undef, num_ptl)
ptlₑ = Array{particle}(undef, num_ptl)

# Initial position for electrons and ions
ptl_pos = range(0.0, step=zgrid.Lz / num_ptl, length=num_ptl)

# Initialize stationary electrons and ions.
# The electron positions are perturbed slightly around the ions
for idx ∈ 1:num_ptl
    x0 = ptl_pos[idx]
    ptlᵢ₀[idx] = particle(x0, 0.0)
    ptlₑ₀[idx] = particle(x0 + 1e-2 .* cos(x0), 0.0) 
    fix_position!(ptlₑ₀[idx], zgrid.Lz - zgrid.Δz)
end
# Calculate initial j_avg
j_avg_0_e = deposit(ptlₑ₀, zgrid, p -> p.vel * qₑ * ptl_wt)
j_avg_0_i = deposit(ptlₑ₀, zgrid, p -> p.vel * qₑ * ptl_wt)
j_avg_0 =  sum(j_avg_0_e + j_avg_0_i) * zgrid.Δz / zgrid.Lz

# deposit particle density on the grid
nₑ = deposit(ptlₑ₀, zgrid, p -> ptl_wt)
nᵢ = deposit(ptlᵢ₀, zgrid, p -> ptl_wt)
ρ = (qᵢ*nᵢ + qₑ*nₑ)
#ϕ = ∇⁻²(-ρ, zgrid)
ϕ = invert_laplace(-ρ, zgrid)
# Calculate initial electric field with centered difference stencil
E = zeros(zgrid.Nz)
E[1] = -0.5 * (ϕ[2] - ϕ[end]) / zgrid.Δz
E[2:end-1] = -0.5 * (ϕ[3:end] - ϕ[1:end-2]) / zgrid.Δz
E[end] = -0.5 * (ϕ[1] - ϕ[end-1]) / zgrid.Δz
smE = smooth(E)

diag_energy(ptlₑ₀, ptlᵢ₀, smE, 0, zgrid)
diag_fields(ptlₑ₀, ptlᵢ₀, zgrid, 0, ptl_wt)

# Use copy to create new particle objects in ptlₑ, ptlᵢ
for pidx ∈ 1:num_ptl
    ptlₑ[pidx] = particle(0.0, 0.0)
    ptlᵢ[pidx] = particle(0.0, 0.0)
end

function G!(E_new, E, ptlₑ₀, ptlᵢ₀, ptlₑ, ptlᵢ, zgrid, num_it, nn)
# Calculates the residuals, given a guess for the electric
# E_new: New electric field
# E: electric field from current time step
# ptlₑ₀ : Electrons at current time step
# ptlᵢ₀ : Ions at current time step
# ptlₑ  : Electrons consistent with E_new
# ptlᵢ  : Ions consistent with E_new
# zgrid: Simulation Domain

    # Construct a periodic interpolator for E
    zrg = (0:Nz) * zgrid.Δz

    # Interpolator for E at n+1/2
    itp = interpolate([0.5 * (E + E_new); 0.5 * (E[1] + E_new[1])], BSpline(Linear()))
    itp2 = Interpolations.scale(itp, zrg)
    ip_E12 = extrapolate(itp2, Periodic())

    # Allocate vector for particle position half time-step
    ptlₑ½ = Array{particle}(undef, num_ptl)
    ptlᵢ½ = Array{particle}(undef, num_ptl)

    # Particle enslavement: Push particles into a consistent state
    # ptlₑ and ptlᵢ will be updated.
    push_v3!(ptlₑ, ptlₑ₀, ptlₑ½, qₑ, 1.0, 1e-10, 1e-12, zgrid, Δt, ip_E12)
    push_v3!(ptlᵢ, ptlᵢ₀, ptlᵢ½, qᵢ, mₑ / mᵢ, 1e-10, 1e-12, zgrid, Δt, ip_E12)
 
    # Calculate j_i^{n+1/2}
    j_n12_e = deposit(ptlₑ½, zgrid, p -> p.vel * qₑ * ptl_wt)
    j_n12_i = deposit(ptlᵢ½, zgrid, p -> p.vel * qᵢ * ptl_wt)
    j_n12 = j_n12_e + j_n12_i
    j_n12_avg = sum(j_n12) * zgrid.Δz / zgrid.Lz

    # Calculate the residual of Eq. (23)
    residuals = (E_new - E) / Δt + (smooth(j_n12) .- j_n12_avg)
    #p = plot((E_new - E) / Δt, label="(E_new - E) / Δt", title="norm((E_new-E)/Δt + SM(j_n12))=$(norm(residuals))")
    #plot!(p, smooth(j_n12), label="smooth_jn12")
    # fname = @sprintf "plot_convergence_nn_%03d_iter_%04d.png" nn num_it
    # savefig(p, fname)

    return residuals
end



for nn in 1:Nt
    println("======================== $(nn)/$(Nt)===========================")
  
    # Picard iteration to get electric field
    E_converged = false
    num_it = 0

    # Guess a new E field.
    delta_E = std(abs.(smE))
    E_guess = smooth(smE + rand(Uniform(-0.1 * delta_E, 0.1 * delta_E), length(smE)))

    fname_conv = @sprintf "convergence_%03d.txt" nn

    t = @elapsed begin
        while (E_converged == false)
            # Updates residuals
            res_vec = G!(E_guess, smE, ptlₑ₀, ptlᵢ₀, ptlₑ, ptlᵢ, zgrid, num_it, nn)
            println("           it $(num_it)/$(max_iter_E): Residual = $(norm(res_vec)).")

            # Break if residuals are stationary
            if (norm(res_vec) ≤ ϵᵣ * norm(smE) + ϵₐ)
                E_converged = true
                println("              -> converged.")
            end

            # Update guessed E-field
            E_guess -= smooth(Δt .* res_vec)
            # Break if too many iterations
            num_it += 1
            if (num_it > max_iter_E)
                E_converged = true
                println("             -> not converged. Hit iteration limit.")
            end
            open(fname_conv, "a") do io
                write(io, "$(num_it)\t$(norm(res_vec))\n")
            end
        end
    end
    # Update smE with new electric field
    smE[:] = E_guess
    # Copy new positions into slot for previous time step
    for pidx ∈ 1:num_ptl
        ptlₑ₀[pidx] = copy(ptlₑ[pidx])
        ptlᵢ₀[pidx] = copy(ptlᵢ[pidx])
    end

    # Write diagnostics
    if mod(nn, 1) == 0
        diag_ptl(ptlₑ, ptlᵢ, nn)
    end
    diag_energy(ptlₑ, ptlᵢ, smE, nn, zgrid)
    diag_fields(ptlₑ, ptlᵢ, zgrid, nn, ptl_wt)

    println("Iteration took $(t) seconds.")  

        

end