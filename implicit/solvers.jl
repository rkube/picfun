# Encodin: UTF-8 -*-

module solvers

using FFTW: fft, ifft
using LinearAlgebra: norm
using grids: grid_1d
export ∇⁻²


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
