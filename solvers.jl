# Encodin: UTF-8 -*-

module solvers

using Zygote
using FFTW: fft, ifft
using LinearAlgebra: norm
using grids: grid_1d

export ∇⁻², invert_laplace


@doc """Given a charge distribution on a grid, solves for the electrostatic
potential.
""" ->
function ∇⁻²(y::Array{Float64}, zgrid::grid_1d)
	# Calculate wave number vector
	Nz, Lz = zgrid.Nz, zgrid.Lz

	k = zeros(Nz)
	k[2:Nz ÷ 2  + 1] = collect(1:Nz ÷ 2) .* 2π / Lz
	k[Nz ÷ 2 + 2:Nz] = -collect(Nz ÷ 2 - 1:-1:1) .* 2π / Lz
	# Zero out the zero-mode 
	y_ft = -fft(y) ./ k ./ k
	y_ft[1] = 0.0
	d2y = real(ifft(y_ft))
	return d2y
end


function invert_laplace(y,  zgrid::grid_1d)
    # Solve Laplace equation
    # ρ: RHS function
    # Nz: Number of grid points
	# Lz: Length of the domain
	Nz, Lz = zgrid.Nz, zgrid.Lz
    Δz = Lz / Nz
    #zrg = range(0.0, step=Δz, length=Nz)
    #zrg = 0.0:Δz:Lz-Δz
    # Squeeze a minus in here.
    invΔz² = 1.0 / Δz / Δz

    # There should be an Eigenvalue around zero, i.e. the matrix is not invertible.
    # Periodic boundary conditions imply that the mean of the solution to is not fixed.
    # solutions can differ by a constant.
    # We can get around this by fixing phi(x0) = c:
    # http://www-m16.ma.tum.de/foswiki/pub/M16/Allgemeines/StefanPossanner/Poisson1D_FD.html#19
    # So the first row is just [1, 0, 0, ...0]

    A0 = Zygote.Buffer(zeros(Nz, Nz))
    for n ∈ 1:Nz
        for m ∈ 1:Nz
            A0[n,m] = 0.0
        end
    end
    for n ∈ 2:Nz
        A0[n, n] = -2.0 * invΔz²
        A0[n, n-1] = invΔz²
        A0[n-1, n] = invΔz²
    end
    A0[1,1] = 1.0
    A0[1, 2] = 0.0
    A0[Nz,1] = invΔz²

    A = copy(A0)
    yvec = vcat([0.0], y[2:Nz])

    ϕ_num = A \ yvec

    return(ϕ_num)
end






# """Calculcates the directional derivative of a vector."""
# function dirder(G, v, x0, G0, ϵ)
# 	# G: Callable
# 	# v: direction
# 	# x0: point where G was last evaluated at
# 	# G0: G(x0)
# 	# ϵ: Small but finite

# 	# If norm(v) < 1e-16: return zero vector
# 	if (norm(v) < 1e-16)
# 		result = zeros(length(v))
# 		return result 
# 	end

# 	# Re-scale ϵ
# 	G1 = G(x0 .+ ϵ .* v)
# 	result = (G1 .- G0) ./ ϵ
# 	return result
# end

end # module
# End of file solvers.jl
