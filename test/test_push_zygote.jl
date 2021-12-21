# Test particle push in reverse mode diff


using Interpolations
using Distributions
using DelimitedFiles
using Statistics
using Random
using LinearAlgebra
using Printf


#using FiniteDiff
#using ForwardDiff
using Zygote
import Base: +, -

push!(LOAD_PATH, "/home/rkube/repos/picfun")

using picfun: qₑ, qᵢ, mₑ, mᵢ
using picfun: grid_1d, init_grid
#using particles: particle, fix_position!
using picfun: invert_laplace
using picfun: b1


# Particles have a position and velocity
mutable struct particle
    pos
    vel
end

# Copy method creates a new instance.
Base.copy(p::particle) = particle(p.pos, p.vel)

pos(p::particle) = p.pos
Zygote.@adjoint function pos(p::particle)
    #@show p
    #println("@adjoint x(p)")
    (p.pos, x̄ -> (particle(x̄, 0), ))
end


vel(p::particle) = p.vel
Zygote.@adjoint function vel(p::particle)
    #@show p
    #println("@adjoint vel(p)")
    (p.vel, v̄ -> (particle(0, v̄), ))
end

# Define arithmetic rules for particles
# In practice they act like a vector from R²
# These should never be used directly, but serve only to define 
# rules for derivatives.
a::particle + b::particle = particle(pos(a) + pos(b), vel(a) + vel(b))
a::particle - b::particle = particle(pos(a) - pos(b), vel(a) - vel(b))

Zygote.@adjoint particle(x, v) = particle(x, v), p̄ -> (p̄.pos, p̄.vel)


function fix_position_here(pos, L)
    # If a particle position is outside the domain, move it
    # by one domain length so that it is within the domain
    if pos < 0
        return pos + L - 10 * eps(typeof(pos))
    elseif pos ≥ L
        return pos - L + 10 * eps(typeof(pos))
    end
    return pos
end



# Modified to use Zygote.Buffer instead of an array.
function smooth_here(Q)
	Nz = size(Q)[1]
	Q_sm = Zygote.Buffer(Q)
	Q_sm[1] = (Q[Nz] + 2 * Q[1] + Q[2]) / 4.0
	Q_sm[2:Nz-1] = (Q[1:Nz-2] + 2 * Q[2:Nz-1] + Q[3:Nz]) / 4.0
	Q_sm[Nz] = (Q[Nz-1] + 2 * Q[Nz] + Q[1]) / 4.0
	return copy(Q_sm)
end


# Modified to use Zygote.Buffer for accumulation instead of an array.
function deposit_here(ptl_vec, zgrid::grid_1d, fun::Function)
    # Get the z-coordinates of the grid, plus one point at the upper boundary.
    zz = (0:1:zgrid.Nz) * zgrid.Δz
    # S contains the sum over all particles of f(p) * b1(z_i, z_p, Δz)
	S_mut = Zygote.Buffer(zeros(zgrid.Nz), eltype(pos(ptl_vec[1])))
    for i ∈ 1:Nz
        S_mut[i] = 0.0
    end
    # Find the last grid index i where zz[i] < ptl_vec.
    # Add 1 since we have 1-based indexing
    last_idx = map(p -> 1 + Int(floor(pos(p) / zgrid.Δz)), ptl_vec)

    # Don't parallelize just yet. This leads to faulty results. May have to use atomics
    # somewhere in here?
    for idx ∈ 1:length(ptl_vec)
        gidx0 = last_idx[idx]
        if gidx0 > zgrid.Nz
            println("idx=$(idx) pos=$(pos(ptl_vec[idx])), gidx=$(gidx0)")
        end
        # When wrapping at Nz, add one
        gidx1 = gidx0 == zgrid.Nz ? 1 : gidx0 + 1
        wt_left = b1((gidx0 - 1) * zgrid.Δz, pos(ptl_vec[idx]), zgrid.Δz)
        # Use gidx0 to calculate right_val instead of gidx1.
        # This captures the case, where gidx1 is 0, at the right side of the domain.
		wt_right = 1.0 - wt_left
        #right_val = b1(gidx0 * zgrid.Δz, ptl_vec[idx].pos, zgrid.Δz)
        #println("idx = $(idx), gidx0 = $(gidx0), gidx1 = $(gidx1), left_val=$(left_val), right_val=$(right_val)")
        
		left_val = wt_left * fun(ptl_vec[idx])
		right_val = wt_right * fun(ptl_vec[idx])
		S_mut[gidx0] += left_val 
        S_mut[gidx1] += right_val 
    end
    return copy(S_mut)
end

function my_interpolate(ys, xip, Δx, Lx)
    # Linear interpolation on a regular 1d grid: 0:Δx:Lx-Δx
    # Periodic extrapolation
    #1. Wrap xip into the domain 0:Lx-Δx
    while xip < 0.0
        xip += Lx
    end
    while xip ≥ Lx
        xip -= Lx
    end

    # 2. Find the left grid node. idx_right can be wrapped if out-of-bounds. 
    N = length(ys)
    idx_left = Int(floor(xip / Δx)) + 1
    idx_right = idx_left == N ? 1 : idx_left + 1

    # 3. Apply linear interpolation formula
    y_left = ys[idx_left]
    y_right = ys[idx_right]
    # idx_right will be wrapped if we are at the boundary. so 
    x_left = (idx_left - 1) * Δx
    #x_right = x_left + Δx
    y_ip = y_left + (xip - x_left) * (y_right - y_left) / Δx
    return y_ip
end


# Modified particle push. Instead of pre-allocating x̃, it is defined within the particle loop
function push_single(ptl₀, E, q, mₑ_over_m, ϵᵣ, ϵₐ, Δt, zgrid)
    xⁿ⁺¹ = 0.0
    vⁿ⁺¹ = 0.0
    x̃ = pos(ptl₀) + Δt * vel(ptl₀)
    num_it_ptl = 0
    ptl_converged=false

    # Picard iteration to move each particle into a state consistent with
    # proposed electric field
    while(ptl_converged == false)
        xⁿ⁺½ =  0.5 * (pos(ptl₀) + x̃)
        vⁿ⁺¹= vel(ptl₀) + Δt * q * mₑ_over_m * my_interpolate(E, xⁿ⁺½, zgrid.Δz, zgrid.Lz)
        vⁿ⁺½ = 0.5 * (vel(ptl₀) + vⁿ⁺¹)
        xⁿ⁺¹ = pos(ptl₀) + Δt * vⁿ⁺½

        # Check convergence
        if ((abs(xⁿ⁺¹ - x̃) ≤ ϵᵣ * abs(x̃) + ϵₐ) || num_it_ptl > 200)
            ptl_converged = true
            break
        end

        # Let xⁿ⁺¹ be the new guess.
        x̃ = xⁿ⁺¹
        num_it_ptl += 1
    end #while_ptl_converged==false

    return particle(xⁿ⁺¹, vⁿ⁺¹)
end


const n₀ = 1.0
const num_ptl = 32768

# Time-stepping parameters
# Time is in units of ωpe
const Δt = 0.1
const Nt = 1

# Domain parameters
# Length is in units of λde
const Lz = 2π
const Nz = 32

# Relative and absolute tolerance for convergence of Picard iteration
const ϵᵣ = 1e-8
const ϵₐ = 1e-10
const max_iter_E = 100

const ptl_per_cell = num_ptl ÷ Nz
const ptl_wt = n₀ / ptl_per_cell

# Initialize the grid
zgrid = init_grid(Lz, Nz)

# Set random seed
Random.seed!(1)

# Initial number of particles per cell
println("Nz = $(Nz), L = $(Lz), ptl_wt = $(ptl_wt)")

# Allocate vectors for particles at old time-step
ptlᵢ₀ = Array{particle}(undef, num_ptl)
ptlₑ₀ = Array{particle}(undef, num_ptl)
# Allocate vector for particles at half time-step
ptlₑ½ = Array{particle}(undef, num_ptl)
ptlᵢ½ = Array{particle}(undef, num_ptl)
# Allocate vector for particles at new time-step
ptlᵢ = Array{particle}(undef, num_ptl)
ptlₑ = Array{particle}(undef, num_ptl)

# Initial position for electrons and ions
ptl_pos = range(0.0, step=zgrid.Lz / num_ptl, length=num_ptl)

# Initialize stationary electrons and ions.
# The electron positions are perturbed slightly around the ions
for pidx ∈ 1:num_ptl
	x0 = ptl_pos[pidx]
	ptlᵢ₀[pidx] = particle(x0, 0.0)
	ptlₑ₀[pidx] = particle(fix_position_here(x0 + 1e-2 .* cos(x0), zgrid.Lz), 0.0)

    # Particles at half and new time-step need to be zero
    #ptlₑ½[pidx] = particle(0.0, 0.0)
    #ptlᵢ½[pidx] = particle(0.0, 0.0)
	ptlₑ[pidx] = particle(0.0, 0.0)
	ptlᵢ[pidx] = particle(0.0, 0.0)
end
# Calculate initial j_avg
j_avg_0_e = deposit_here(ptlₑ₀, zgrid, p -> p.vel * qₑ * ptl_wt)
j_avg_0_i = deposit_here(ptlₑ₀, zgrid, p -> p.vel * qₑ * ptl_wt)
j_avg_0 =  sum(j_avg_0_e + j_avg_0_i) * zgrid.Δz / zgrid.Lz

# deposit particle density on the grid
nₑ = deposit_here(ptlₑ₀, zgrid, p -> ptl_wt)
nᵢ = deposit_here(ptlᵢ₀, zgrid, p -> ptl_wt)
ρ = (qᵢ*nᵢ + qₑ*nₑ)
#ϕ = ∇⁻²(-ρ, zgrid)
ϕ = invert_laplace(-ρ, zgrid)
# Calculate initial electric field with centered difference stencil
E = zeros(zgrid.Nz)
E[1] = -0.5 * (ϕ[2] - ϕ[end]) / zgrid.Δz
E[2:end-1] = -0.5 * (ϕ[3:end] - ϕ[1:end-2]) / zgrid.Δz
E[end] = -0.5 * (ϕ[1] - ϕ[end-1]) / zgrid.Δz

# We have an initial electric field. Now we guess a new one.
E_guess = rand(Uniform(-1e-3, 1e-3), length(E)) + E
# E_smooth = 0.5 * smooth_here(E + E_guess)
# itp = interpolate([E_smooth; E_smooth[1]], BSpline(Linear()))
# itp2 = Interpolations.scale(itp, zrg)
# ip_E12 = extrapolate(itp2, Periodic())
# ip_E(x) = Zygote.forwarddiff(x -> ip_E12[x], x)
zrg = (0:Nz) * zgrid.Δz

# function build_ip(E_guess)
#     E_smooth = 0.5 * smooth_here(E + E_guess)
#     itp = interpolate([E_smooth; E_smooth[1]], BSpline(Linear()))
#     itp2 = Interpolations.scale(itp, zrg)
#     ip_E12 = extrapolate(itp2, Periodic())
#     #@show ip_E12
#     return ip_E12 
# end


function push_capture(E_guess)
    #ip_E = Zygote.forwarddiff(build_ip, E_guess)

    # @show ip_E12
    # @show E_guess
    # Create local buffers for the particles 
    ptl_e_buf = Zygote.Buffer(ptlₑ)
    ptl_e12_buf = Zygote.Buffer(ptlₑ)

    for idx ∈ 1:num_ptl
        ptl_e_buf[idx] = push_single(ptlₑ₀[idx], E_guess, 1.0, 1.0, 1e-6, 1e-10, 0.1, zgrid)
        ptl_e12_buf[idx] = particle(0.5 * (pos(ptlₑ₀[idx]) + pos(ptl_e_buf[idx])), 0.5 * (vel(ptlₑ₀[idx]) + vel(ptl_e_buf[idx])))
    end

    j_n12_e = deposit_here(copy(ptl_e12_buf), zgrid, p -> vel(p) * -1.0 * ptl_wt)

    return j_n12_e
end

# g, back = Zygote.pullback(push_capture, smooth_here(E_guess + E))

J_zg = zeros(Float64, Nz, Nz)
e1 = zeros(Float64, Nz)

@time pb, back = Zygote.pullback(push_capture, smooth_here(E_guess + E))
# # For large number of particles (65536) this becomes unusably slow
for ridx in 1:32
    println("$(ridx)")
    e1[:] .= 0.0
    e1[ridx] = 1.0 # E_guess[ridx]
    @time J_zg[ridx, :] .= back(e1)[1]
end



