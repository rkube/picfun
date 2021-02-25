#  Implementation of the implicit scheme presented in
# G. Chen et al. Journ., Comp. Phys 230 7018 (2011).
#
#
# This builds on v3, but uses Picard iteration to solve the electric field update.

using IterativeSolvers
using LinearMaps
using Interpolations
using Distributions
using Statistics
using Random
using LinearAlgebra
using Printf
using JSON

push!(LOAD_PATH, pwd())

using units: qₑ, qᵢ, mₑ, mᵢ
using grids: grid_1d, init_grid
using pic_utils: smooth, deposit
using particles: particle, fix_position!
using particle_push: push_v3!
using solvers: ∇⁻², invert_laplace
using diagnostics: diag_ptl, diag_energy, diag_fields

#using Plots
#

stringdata = join(readlines("simulation.json"))
config = JSON.parse(stringdata)

const n₀ = config["n0"]
const num_ptl = config["num_ptl"]

# Time-stepping parameters
# Time is in units of ωpe
const Δt = config["deltat"]
const Nt = config["Nt"]

# Domain parameters
# Length is in units of λde
const Lz = config["Lz"]
const Nz = config["Nz"]

# Relative and absolute tolerance for convergence of Picard iteration
const ϵᵣ = config["epsr"]
const ϵₐ = config["epsa"] * √(num_ptl)
const max_iter_E = config["max_iter_E"]

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
    fix_position!(ptlₑ₀[idx], zgrid.Lz)
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

diag_ptl(ptlₑ₀, ptlᵢ₀, 0)
diag_energy(ptlₑ₀, ptlᵢ₀, E, 0, zgrid)
diag_fields(ptlₑ₀, ptlᵢ₀, zgrid, 0, ptl_wt)

# Use copy to create new particle objects in ptlₑ, ptlᵢ
for pidx ∈ 1:num_ptl
    ptlₑ[pidx] = particle(0.0, 0.0)
    ptlᵢ[pidx] = particle(0.0, 0.0)
end

function G!(E_new, E, ptlₑ₀, ptlᵢ₀, ptlₑ, ptlᵢ, zgrid)
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
    E_smooth = 0.5 * smooth(E + E_new)
    itp = interpolate([E_smooth; E_smooth[1]], BSpline(Linear()))
    itp2 = Interpolations.scale(itp, zrg)
    ip_E12 = extrapolate(itp2, Periodic())

    # Allocate vector for particle position half time-step
    ptlₑ½ = Array{particle}(undef, num_ptl)
    ptlᵢ½ = Array{particle}(undef, num_ptl)

    # Particle enslavement: Push particles into a consistent state
    # ptlₑ and ptlᵢ will be updated.
    push_v3!(ptlₑ, ptlₑ₀, ptlₑ½, qₑ, 1.0, 1e-6, 1e-10, zgrid, Δt, ip_E12)
    push_v3!(ptlᵢ, ptlᵢ₀, ptlᵢ½, qᵢ, mₑ / mᵢ, 1e-6, 1e-10, zgrid, Δt, ip_E12)

    # for pidx ∈ 1:num_ptl
    #     @assert(ptlₑ₀[pidx].pos < zgrid.Lz)
    #     @assert(ptlₑ½[pidx].pos < zgrid.Lz)
    #     @assert(ptlₑ[pidx].pos < zgrid.Lz)
    #     @assert(ptlᵢ₀[pidx].pos < zgrid.Lz)
    #     @assert(ptlᵢ½[pidx].pos < zgrid.Lz)
    #     @assert(ptlᵢ[pidx].pos < zgrid.Lz)
    # end
 
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


struct MyRes{T} <: LinearMaps.LinearMap{T}
    E0::AbstractVector{T}   # This is xᵏ
    size::Dims{2}           # This is (grid.Nz, grid.Nz)
    G::Function             # This evaluates the residual of the PIC system
    function MyRes(E0::AbstractVector{T}, G::Function) where {T}
        promote_type(T, eltype(E0)) == T || throw(InexactError())
        return new{T}(E0, (length(E0), length(E0)), G)
    end
end
Base.size(A::MyRes) = A.size


function LinearAlgebra.mul!(y::AbstractVecOrMat, A::MyRes, x::AbstractVector)
    # A x = y
    # Matrix-Vector multiplication implements the Gateaux-derivative:
    # ∂Gᵏ/∂x * v = lim(ϵ → 0) [G(xᵏ * ϵv) - G(xᵏ)]/ϵ
    LinearMaps.check_dim_mul(y, A, x)
	
	G0 = A.G(A.E0)
	ϵ = 1e-7
	# Re-scale like in Kelley's book.
	xs = (A.E0' * x) / norm(x)
	if abs(xs) > 0.0
		ϵ = ϵ * max(abs(xs), 1.0) * sign(xs)
	end
	ϵ = ϵ / norm(xs)
	
	G1 = A.G(A.E0 + ϵ * x)
	result = (G1 - G0) / ϵ
	y[:] = result[:]
end



for nn in 1:Nt
    println("======================== $(nn)/$(Nt)===========================")

    # Use Newton's method to solve the system G(E)=0
    E_converged = false
    num_it = 0

    # Guess a new E field. This is our Eᵏ
    delta_E = std(abs.(E))
    Eᵏ = E + rand(Uniform(-0.1 * delta_E, 0.1 * delta_E), length(E))
    #Eᵏ[1] = 0.0
    #Eᵏ[end] = 0.0
    # Define a new closure for for matrix-vector multiplications
    # Here the reference vector is smE, the converged solution of the
    # previous time step.
    G_res(Enew) = G!(Enew, E, ptlₑ₀, ptlᵢ₀, ptlₑ, ptlᵢ, zgrid)
    initial_norm = norm(G_res(Eᵏ))

    # Solve the system ∂G/∂E|ᵏ δEᵏ = - G(Eᵏ)
    # Keep track of the convergence history:
    newton_α = 1.5
    newton_γ = 0.9
    newton_ζmax = 0.8
    newton_ζA = [newton_γ]
    newton_ζB = []
    newton_ζk = []
    newton_ϵt = ϵₐ + ϵᵣ * initial_norm

    t = @elapsed begin
        while (E_converged == false)

            # A_iter implements matrix-vector multiplication for A = ∂G/∂E|ᵏ
            A_iter = MyRes(Eᵏ, G_res)
            # Calculate the convergence tolerance 

            # Calculate the current residual -G(Eᵏ)
            δEᵏ = IterativeSolvers.gmres(A_iter, -G_res(Eᵏ), reltol=1e-10, verbose=true)
            Eᵏ[:] += δEᵏ[:]
            num_it += 1

            current_norm = norm(G_res(Eᵏ))
            # Updates residuals
            println("           it $(num_it)/$(max_iter_E): Residual = $(current_norm).")

            # Break if residuals are stationary
            if (current_norm ≤ newton_ϵt) || num_it > max_iter_E
                E_converged = true
                println("              -> converged /hit iteration limit")
            end
        end # while
    end # @elapsed
    # The implicit system is now in a consistent state. We now have to save 
    # the electric field that was updated in the iteration as well as the
    # particle state that support this electric field
    E[:] = Eᵏ[:]
    for pidx ∈ 1:num_ptl
        ptlₑ₀[pidx] = copy(ptlₑ[pidx])
        ptlᵢ₀[pidx] = copy(ptlᵢ[pidx])
    end

    # Write diagnostics
    if mod(nn, 100) == 0
        diag_ptl(ptlₑ, ptlᵢ, nn)
    end
    diag_energy(ptlₑ, ptlᵢ, E, nn, zgrid)
    diag_fields(ptlₑ, ptlᵢ, zgrid, nn, ptl_wt)

    println("Iteration took $(t) seconds.")  

        

end