### A Pluto.jl notebook ###
# v0.12.16

using Markdown
using InteractiveUtils

# ╔═╡ 13e3cf52-3948-11eb-34dc-c3e09bf2100b
#  Implementation of the implicit scheme presented in
# G. Chen et al. Journ., Comp. Phys 230 7018 (2011).
#
#
# This builds on v2, but uses NLSolve to find the electric field update

begin
	using Interpolations
	using Plots
	using Distributions
	using Statistics
	using Random
	using LinearAlgebra: norm, dot
	using NLsolve
	using Printf
	using BifurcationKit: newton
	using IterativeSolvers: gmres, gmres!
	using LinearAlgebra
	using LinearMaps
	
	push!(LOAD_PATH, "/Users/ralph/source/repos/picfun/implicit")

	using units: qₑ, qᵢ, n₀, mᵢ, mₑ
	using grids: grid_1d, init_grid
	using pic_utils: smooth, deposit
	using particles: particle, fix_position!
	using particle_push: push_v3!
end

# ╔═╡ eac4c3aa-3948-11eb-3d69-85b083c591c8
using solvers: ∇⁻²

# ╔═╡ e2f3dd00-3948-11eb-21f2-77d6afee4ac0
using diagnostics: diag_ptl, diag_energy, diag_fields

# ╔═╡ 6f2498cc-3948-11eb-14a7-19afd79ddfd7
md"# Head"

# ╔═╡ 3bc0b238-3948-11eb-1fae-5f4ced4bd206
begin
	# Time-stepping parameters
	Δt = 1e-4
	Nt = 10
	Nν = 1
	Δτ = Δt / Nν

	# Relative and absolute tolerance for convergence of Picard iteration
	ϵᵣ = 1e-6
	ϵₐ = 1e-10

	# Domain parameters
	Lz = 2π
	Nz = 32

	# Initialize the grid
	zgrid = init_grid(Lz, Nz)
	zrg = (0:Nz) * zgrid.Δz
end


# ╔═╡ 4c0388e6-3948-11eb-081d-db1b34a31635
begin
	# Set random seed
	Random.seed!(1)

	# Initial number of particles per cell
	particle_per_cell = 128
	num_ptl = Nz * particle_per_cell
	println("Nz = $Nz, L = $Lz, num_ptl = $num_ptl")

# 	# Initialize  electron and ion population
	ptlᵢ₀ = Array{particle}(undef, num_ptl)
	ptlₑ₀ = Array{particle}(undef, num_ptl)

	# Initial position for the ions
# 	# Initial position for the electrons
	ptl_pos = range(0.0, step=zgrid.Lz / num_ptl, length=num_ptl)

	# Initialize stationary electrons and ions.
	# The electron positions are perturbed slightly around the ions
	for idx ∈ 1:num_ptl
		x0 = ptl_pos[idx]
		ptlᵢ₀[idx] = particle(x0, 0.0)
		ptlₑ₀[idx] = particle(x0 + 1e-4 .* cos(x0), 0.0) #ptl_perturbation[idx], 0.0)
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
	Eⁿ[1] = -1. * (ϕⁿ[end] - ϕⁿ[2]) / 2. / zgrid.Δz
	Eⁿ[2:end-1] = -1. * (ϕⁿ[1:end-2] - ϕⁿ[3:end]) / 2. / zgrid.Δz
	Eⁿ[end] = -1. * (ϕⁿ[end-1] - ϕⁿ[1]) / 2. / zgrid.Δz
	smEⁿ = smooth(Eⁿ)

end

# ╔═╡ 9e659afa-394a-11eb-0ee7-5be526e851a5
begin
	plot(range(0.0, step=zgrid.Δz, length=zgrid.Nz), nₑ, label=:"nₑ")
	plot!(range(0.0, step=zgrid.Δz, length=zgrid.Nz), nᵢ, label=:"nᵢ")
end

# ╔═╡ d82733a4-3955-11eb-28f4-fd6799dd3c88
begin
	plot(range(0.0, step=zgrid.Δz, length=zgrid.Nz), smEⁿ, label=:"smEⁿ")
	plot!(range(0.0, step=zgrid.Δz, length=zgrid.Nz), ϕⁿ, label=:"ϕⁿ")
	plot!(range(0.0, step=zgrid.Δz, length=zgrid.Nz), ρⁿ, label=:"ρⁿ")
end

# ╔═╡ dfd8ec0c-3955-11eb-3703-d7f11654cd4f


# ╔═╡ 5a94cdd4-3948-11eb-21c8-37a771ab1146
begin
	ptlₑ = deepcopy(ptlₑ₀)
	ptlᵢ = deepcopy(ptlᵢ₀)
end

# ╔═╡ 4beaf792-3948-11eb-3f7b-b1664cace320

function G(E_new, E₀, ptlₑ₀, ptlᵢ₀, ptlₑ, ptlᵢ, zgrid)
# Calculates residuals, given a guess for the electric
# E_new: New (guessed) electric field
# E: electric field from current time step
# ptlₑ₀ : Electrons at current time step
# ptlᵢ₀ : Ions at current time step
# zgrid: Simulation Domain
# j_avg_0:

    #println("Function residuals. E_new[1] = $(E_new[1]), E[1]=$(E[1])")

    # Construct a periodic interpolator for E
    _E_per = copy(E₀)
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
    ptlₑ½ = deepcopy(ptlₑ)
    ptlᵢ½ = deepcopy(ptlₑ)

    # Particle enslavement: Push particles into a consistent state
    # ptlₑ and ptlᵢ will be updated.
    push_v3!(ptlₑ, ptlₑ₀, ptlₑ½, qₑ, mₑ, ϵᵣ, ϵₐ, zgrid, Δt, ip_Enew, ip_E)
    push_v3!(ptlᵢ, ptlᵢ₀, ptlᵢ½, qᵢ, mᵢ, ϵᵣ, ϵₐ, zgrid, Δt, ip_Enew, ip_E)
	
    # Calculate j_i^{n+1/2}
    j_n½_e = deposit(ptlₑ½, zgrid, p -> p.vel * qₑ / zgrid.Δz)
    j_n½_i = deposit(ptlᵢ½, zgrid, p -> p.vel * qᵢ / zgrid.Δz)
    j_n½ = j_n½_e + j_n½_i
    j_n½_avg = mean(j_n½)

    # Calculate the residual of Eq. (23)
    residuals = ϵ₀ / Δt .* (E_new - E₀) .+ (smooth(j_n½) .- j_n½_avg)
	return residuals
end

# ╔═╡ 488d6ba6-395b-11eb-3678-3958332cd82d
function directional_deriv(G, x0, w, G0, ϵ)
	# Defines how the Jacobian acts on a vector, see Eq. 28
	# Implementation taken from CT Kelley -
	#                           Iterative Methods for Linear and Nonlinear Equations
	# G: Callable
	# x0: point where G was last evaluated at
	# w: Direction in which to calculate the derivative
	# G0: G(x0)
	# ϵ: Small but finite value

	# Return zero if the direction is too small
	if (norm(w) < 1e-16)
	 	result = zeros(length(w))
	 	return result 
	end

	# Scale the difference increment. This is taken from Kelleys book.
	xs = dot(x0, w) / norm(w)
	println("xs = $(xs) norm(w) = $(norm(w))")
	if (abs.(xs) > 0.0)
		ϵ = ϵ * max(abs.(xs), 1.0) * sign(xs)
	end
	ϵ = ϵ / norm(w)
	println("ϵ = $(ϵ)")
	
	G1 = G(x0 .+ ϵ .* w)
	
	J = (G1 .- G0) ./ ϵ
	return J
end

# ╔═╡ 08d40470-396f-11eb-1588-29fd28b2688c
G_closure(E_guess) = G(E_guess, smEⁿ, ptlₑ₀, ptlᵢ₀, ptlₑ, ptlᵢ, zgrid)

# ╔═╡ 19633c40-396f-11eb-298d-bf3eb3e6aee5
# Try a custom LinearMap for use in GMRES
# https://jutho.github.io/LinearMaps.jl/dev/generated/custom/
#
# This implements matrix-free jacobian-vector product.
# We do this by calculating the directional derivative in calls to
# mul!(y, MyPICResidual, w)
#
# Here w is the direction along which we take the derivative

struct MyPICResidual{T} <: LinearMaps.LinearMap{T}
	
	x0 :: AbstractVector
	G0 :: AbstractVector
	ϵ :: T
	size::Dims{2}
	
	function MyPICResidual(x0::AbstractVector, G0 :: AbstractVector, ϵ::T, dims::Dims{2}) where {T}
		all(≥(0), dims) || throw(ArgumentError("dims of MyPICResidual must be non-negative"))
		promote_type(T, typeof(ϵ)) == T || throw(InexactError())
		return new{T}(x0, G0, ϵ, dims)
	end
end

# ╔═╡ 5ca4168a-3995-11eb-20d5-2137a87c4e40
Base.size(A::MyPICResidual) = A.size

# ╔═╡ bf204d54-3994-11eb-33a0-2bd82e28a164
function LinearAlgebra.mul!(y::AbstractVecOrMat, A::MyPICResidual, w::AbstractVector)
	LinearMaps.check_dim_mul(y, A, w)
	# Implements matrix-vector multiplication for A of type MyPICResidual
	#
	# w: Direction along which we take the derivative
	# Calculates the directional derivative
	#
	#
		# Return zero if the direction is too small
	if (norm(w) < 1e-16)
	 	result = zeros(length(w))
	 	return result 
	end

	# Scale the difference increment. This is taken from Kelleys book.
	xs = dot(A.x0, w) / norm(w)
	println("xs = $(xs) norm(w) = $(norm(w))")
	ϵnew = A.ϵ
	if (abs.(xs) > 0.0)
		ϵnew = ϵnew * max(abs.(xs), 1.0) * sign(xs)
	end
	ϵnew = ϵnew / norm(w)
	println("ϵ = $(ϵnew)")
	
	G1 = G_closure(A.x0 .+ ϵnew .* w)
	println("G1 = $(G1)")
	println("G0 = $(A.G0)")
	
	J = (G1 .- A.G0) ./ ϵnew
	println("J = $(J)")
	return J
end

# ╔═╡ 732f2048-395b-11eb-092b-b9174490400e
begin
	# Set up MyPICResidal
	x0 = smEⁿ
	G0 = G_closure(x0)

	A = MyPICResidual(smEⁿ, G0, 1e-7, (32, 32))
	w = rand(length(x0))
	update = similar(x0)
end

# ╔═╡ cc1de848-3a18-11eb-0455-e9af931b1d5b
# Matrix multiplication should return a vector
typeof(A*x0), size(A*x0)

# ╔═╡ cbeab360-3a18-11eb-3a43-4bd2f8384331
y = similar(x0); mul!(y, A, x0), size(y), typeof(y)

# ╔═╡ cbcfed82-3a18-11eb-1343-272df7524771
eltype(A)

# ╔═╡ cbb7ec1e-3a18-11eb-2c6a-770961a859bc
size(A, 1), size(A, 2)

# ╔═╡ 3b67ec0a-396d-11eb-2938-fd763c4adcec
gmres!(update, A, x0, verbose=true)

# ╔═╡ 4bd5a0de-3948-11eb-1d66-47b3dc283c58

# for nn in 1:Nt
#     println("======================== $(nn)/$(Nt)===========================")

#     res_func!(res_vec, E_guess) = residuals!(res_vec, E_guess, smEⁿ, ptlₑ₀, ptlᵢ₀, ptlₑ, ptlᵢ, zgrid)
#     global ptlₑ₀ = copy(ptlₑ)
#     global ptlᵢ₀ = copy(ptlᵢ)
#     delta_E = std(abs.(smEⁿ))
#     println(smEⁿ)
#     E_new = smooth(smEⁿ + rand(Uniform(-0.1 * delta_E, 0.1 * delta_E), length(smEⁿ)))
#     result = nlsolve(res_func!, E_new; xtol=1e-3, iterations=25000)
#     global smEⁿ[:] = smooth(result.zero[:])

#     #plot!(smEⁿ)

#     println(result.zero .- E_new)
#     #println(result.iterations)
#     if mod(nn, 1) == 0
#         diag_ptl(ptlₑ, ptlᵢ, nn)
#     end
#     diag_energy(ptlₑ, ptlᵢ, smEⁿ, nn)
#     diag_fields(ptlₑ, ptlᵢ, zgrid, nn)
# end

# ╔═╡ 17fce976-3fb4-11eb-0356-d17aa521a00d
12 ÷ 4.0

# ╔═╡ 17e3a524-3fb4-11eb-1304-c54159ddf6e7


# ╔═╡ 17cd448c-3fb4-11eb-04c4-998c4806323c


# ╔═╡ 17a2173a-3fb4-11eb-0900-59f11f7b4860


# ╔═╡ Cell order:
# ╠═6f2498cc-3948-11eb-14a7-19afd79ddfd7
# ╠═13e3cf52-3948-11eb-34dc-c3e09bf2100b
# ╠═eac4c3aa-3948-11eb-3d69-85b083c591c8
# ╠═e2f3dd00-3948-11eb-21f2-77d6afee4ac0
# ╠═3bc0b238-3948-11eb-1fae-5f4ced4bd206
# ╠═4c0388e6-3948-11eb-081d-db1b34a31635
# ╠═9e659afa-394a-11eb-0ee7-5be526e851a5
# ╠═d82733a4-3955-11eb-28f4-fd6799dd3c88
# ╠═dfd8ec0c-3955-11eb-3703-d7f11654cd4f
# ╠═5a94cdd4-3948-11eb-21c8-37a771ab1146
# ╠═4beaf792-3948-11eb-3f7b-b1664cace320
# ╠═488d6ba6-395b-11eb-3678-3958332cd82d
# ╠═08d40470-396f-11eb-1588-29fd28b2688c
# ╠═19633c40-396f-11eb-298d-bf3eb3e6aee5
# ╠═5ca4168a-3995-11eb-20d5-2137a87c4e40
# ╠═bf204d54-3994-11eb-33a0-2bd82e28a164
# ╠═732f2048-395b-11eb-092b-b9174490400e
# ╠═cc1de848-3a18-11eb-0455-e9af931b1d5b
# ╠═cbeab360-3a18-11eb-3a43-4bd2f8384331
# ╠═cbcfed82-3a18-11eb-1343-272df7524771
# ╠═cbb7ec1e-3a18-11eb-2c6a-770961a859bc
# ╠═3b67ec0a-396d-11eb-2938-fd763c4adcec
# ╠═4bd5a0de-3948-11eb-1d66-47b3dc283c58
# ╠═17fce976-3fb4-11eb-0356-d17aa521a00d
# ╠═17e3a524-3fb4-11eb-1304-c54159ddf6e7
# ╠═17cd448c-3fb4-11eb-04c4-998c4806323c
# ╠═17a2173a-3fb4-11eb-0900-59f11f7b4860
