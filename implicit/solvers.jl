# Encodin: UTF-8 -*-

module solvers

using FFTW: fft, ifft
using grids: grid_1d
export ∇⁻²


@doc """Given a charge distribution on a grid, solves for the electrostatic
potential.
""" ->
function ∇⁻²(rho::Array{Float64}, zgrid::grid_1d)
	# Calculate wave number vector
	Nz, Lz = zgrid.Nz, zgrid.Lz

	k = zeros(Nz)
	k[1] = Inf
	k[2:Nz ÷ 2  + 1] = collect(1:Nz ÷ 2) * 2π / Lz
	k[Nz ÷ 2 + 2:Nz] = -collect(Nz ÷ 2 - 1:-1:1) * 2π / Lz
	#print(k)
	phi = real(ifft(fft(rho)))# .* (k.^(-2.0))))
	return phi
end


end # module
# End of file solvers.jl
