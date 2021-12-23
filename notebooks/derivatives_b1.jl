### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 6805184e-3a58-4a75-9c25-2b6befffccfe
begin
	import Pkg
	Pkg.activate("/home/rkube/repos/picfun")
	Pkg.instantiate()
	# Pkg.add("PlutoUI")

	using PlutoUI
	using Plots
	using Printf
	using Zygote
	using picfun
end

# ╔═╡ 8215f730-6272-11ec-024d-159c7cc505b7
md"# Test differentiability of shape function

Shape functions are used to distribute particle properties, such as mass and charge,
to nearby grid nodes. This notebook tests and illustrates the differentiability of shape functions."

# ╔═╡ fcb3e3be-511d-481c-a12f-68094a0c36e0
md"## Setup

We set up a grid with few nodes and place one particle into the domain. Later we plot the shape function on a finer grid. In addition we calculate the derivative of the shape function using automatic differentiation on the fine grid. 

The particle itself will move from one grid node to the next one. For each of the particle positions we update the shape function and its derivative on the fine grid. We also update the values of the shape function b1 evaluated on the grid points that are neighboring the particle."

# ╔═╡ 56db95f0-d375-4a3f-9eb3-941469c2d3c0
begin
	Nz = 4
    Lz = 1.0
    zgrid = init_grid(Lz, Nz)
    zvals = (0:Nz-1) * zgrid.Δz

    # Place the particle between the third and fourth grid point
    j = 2    # OBS! Setting this offset to 2 means particle is at 3rd grid-point due to zero-based indexing of the grid points.
    qc = 1.0
end

# ╔═╡ c5480433-944c-432d-a965-14159847c73f
begin
	@bind dd Slider(0.0:0.01:1.0)
end

# ╔═╡ 53289c1b-fafc-4da7-ad04-9ad4ca0eef21
begin
	zp = (j+dd) * zgrid.Δz
	ptl = particle(zp, 0.0)
end

# ╔═╡ 33c60bec-cf70-4cba-bf49-a26bbe64e199
begin
	# Fine grid for plotting b1 on all of zgrid
	Δz_plot = 0.1* zgrid.Δz
	Nz_plot = Int(Lz / Δz_plot)
	zvals_plot = 0:Δz_plot:Lz-0.001
	b1_zp(z) = b1(z, x(ptl), zgrid.Δz)

	# Semi-coarse grid for evaluating the derivative
	#Δz_deriv = 0.1 * zgrid.Δz
	
	deriv_b1= zeros(Nz_plot)
	for n ∈ 1:Nz_plot
		res = gradient(b1_zp, n * Δz_plot)
		deriv_b1[n] = res[1] == nothing ? 0.0 : res[1]
	end
	

	plot(zvals_plot, b1_zp.(zvals_plot), w=3, label="B1")
	plot!(zvals_plot, deriv_b1, line=(3, :dashdot), label="∂(B1)/∂z")
	plot!([zp], [1.0], seriestype=:scatter, ms=6, label="ptl.pos")
	plot!(zvals, zeros(size(zvals)), seriestype=:scatter, color=:black, ms=6, label="Grid points")
	plot!([zvals[j+1], zvals[j+1]], [0.0, b1_zp(zvals[j+1])], color=:black, lw=3, label=nothing)
	plot!([zvals[j+2], zvals[j+2]], [0.0, b1_zp(zvals[j+2])], color=:black, lw=3, label=nothing)
end

# ╔═╡ 7ad7957c-d776-471d-932c-4df64fb83600
begin
num_pos = 10

	anim = @animate for i ∈ 1:num_pos
	       zp = (j + (i/num_pos)) * zgrid.Δz
	       ptl = particle(zp, 0.0)
		   b1_zp(z) = b1(z, x(ptl), zgrid.Δz)
	
		   # Semi-coarse grid for evaluating the derivative
		   #Δz_deriv = 0.1 * zgrid.Δz
		   
		   deriv_b1= zeros(Nz_plot)
		   for n ∈ 1:Nz_plot
				   res = gradient(b1_zp, n * Δz_plot)
				   deriv_b1[n] = res[1] == nothing ? 0.0 : res[1]
		   end
		   
	
		   plot(zvals_plot, b1_zp.(zvals_plot), w=3, label="B1")
		   plot!(zvals_plot, deriv_b1, line=(3, :dashdot), label="∂(B1)/∂z")
		   plot!([zp], [1.0], seriestype=:scatter, ms=6, label="ptl.pos")
		   plot!(zvals, zeros(size(zvals)), seriestype=:scatter, color=:black, ms=6, label="Grid points")
		   plot!([zvals[j+1], zvals[j+1]], [0.0, b1_zp(zvals[j+1])], color=:black, lw=3, label=nothing)
		   plot!([zvals[j+2], zvals[j+2]], [0.0, b1_zp(zvals[j+2])], color=:black, lw=3, label=nothing)
	end
	fname = @sprintf "b1_Nz%02d.gif" Nz
	gif(anim , fname, fps=5)

end

# ╔═╡ Cell order:
# ╠═8215f730-6272-11ec-024d-159c7cc505b7
# ╠═6805184e-3a58-4a75-9c25-2b6befffccfe
# ╠═fcb3e3be-511d-481c-a12f-68094a0c36e0
# ╠═56db95f0-d375-4a3f-9eb3-941469c2d3c0
# ╠═c5480433-944c-432d-a965-14159847c73f
# ╠═53289c1b-fafc-4da7-ad04-9ad4ca0eef21
# ╠═33c60bec-cf70-4cba-bf49-a26bbe64e199
# ╠═7ad7957c-d776-471d-932c-4df64fb83600
