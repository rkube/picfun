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

# ╔═╡ 517b8c3c-2822-4b04-9766-bdb1099d9995
begin
	import Pkg
	Pkg.activate("/home/rkube/repos/picfun")
	Pkg.instantiate()
	# Pkg.add("PlutoUI")

	using PlutoUI
	using Plots
	using Printf
	using Zygote
	using NLsolve
	using Distributions
	using picfun
	using StatsPlots
end

# ╔═╡ fe36442e-63de-11ec-2a28-e31dd54121e8
md"# Demonstrates particle noise effects

Full-f PIC methods evolve the entire distribution function f. To connect f with particles, we sample from f. This is akin to drawing samples from say, a Normal or a Uniform distribution. 

Grid quantities, such as a density, are calculated by depositing particle quantities on the grid. Depending on the number of particles, density values at the grid points may be noise. That is even though f itself is smooth.

This notebook demonstrates this noise effect."

# ╔═╡ bfc88d04-40e6-4987-b508-d304fdafc07b
md"# Particle sampling.
We start out by specifing a domain and number of particles."

# ╔═╡ 346a250a-5238-4cd9-a402-cc87cb4397b0
begin
	Lz = 2π
	Nz = 16
	zgrid = init_grid(Lz, Nz)
	zvals = 0:zgrid.Δz:(Lz-eps(Lz))
end

# ╔═╡ 50d78ed1-0aa0-4a5c-a64c-19d72ed37a7a
md"As a next step, we specify an initial density profile.  Using that the integral of distribution function over velocity space yields the particle density at a given point, ∫f(z, v) dv = n(z), we specify an initial density profile like

	n(z) = 1.0 + sin(k * z)

To sample from this density we follow [this tutorial](https://web.mit.edu/urban_or_book/www/book/chapter7/7.1.3.html)


	n(z) = n₀(1.0 + A * sin(k * z))

	∫n(z) = N(z) = n₀z - A n₀ cos(k * z) / k + 1

	with N(0) = 0, N(2π) = n₀(2 A (sin²(π k) / k + 2π)

The idea is to connect samples from a Uniform distribution to the CDF N. This figure illustrates the idea

![figure](https://web.mit.edu/urban_or_book/www/book/chapter7/images7/fig7.2.gif)

Mathematically we sample r ∼ Uniform(0.0, 2π), and then solve r = N(z) for z. For the given sinusoidal form of n(z) we need to solve r=N(z) numerically. NLsolve allows us to do this.
"

# ╔═╡ dcd7b2e2-7a57-4db2-ab64-f9bbff890626
begin
	k = 1.0
	n₀ = 1.0
	A = 0.2
	n(z) = n₀ .* (1.0 .+ A .* sin.(k * z))
	N(z) = n₀ * z .- n₀ .* A .* cos.(k .* z) / k .+ n₀

	plot(zvals, n(zvals), label="n(z)", xlabel="z")
	plot!(zvals, N(zvals), label="N(z)")
end

# ╔═╡ fb2955ac-80ff-4cd3-9918-e2928c443577
@bind num_ptl Slider(128:128:16384, show_value=true)

# ╔═╡ c090bb1e-7d4d-4b89-a687-6f0299da535b
begin
	vals = rand(Uniform(0.0, N(Lz)), num_ptl)
	N2(z) = vals .- N(z)
	sol = nlsolve(N2 , π * ones(num_ptl), autodiff=:forward)
end

# ╔═╡ 6bd71c4d-5fa3-44e1-961c-6c637b344342
begin
	plot(sol.zero, zeros(num_ptl) .+ rand(Uniform(-0.05, 0.05), num_ptl), seriestype=:scatter, ms=1)
	density!(sol.zero)
	plot!(zvals, n.(zvals))
end

# ╔═╡ e496c155-d191-4c7a-b095-7404d89fc3b9
md"The cells above let us sample particles with the derivative technique. The mapping is solved numerically. As we see, multiple particles have a coordinate below 0. This comes from the numerical solver, which does not constrain the solution to the interval [0:2π]. Looking at a density estimate from the particles we see that it approximately follows the specified density function.

To make a more accurate comparison we need to deposit the density of the particles to the grid. This is done using so-called shape functions"

# ╔═╡ Cell order:
# ╠═fe36442e-63de-11ec-2a28-e31dd54121e8
# ╠═517b8c3c-2822-4b04-9766-bdb1099d9995
# ╠═bfc88d04-40e6-4987-b508-d304fdafc07b
# ╠═346a250a-5238-4cd9-a402-cc87cb4397b0
# ╟─50d78ed1-0aa0-4a5c-a64c-19d72ed37a7a
# ╠═dcd7b2e2-7a57-4db2-ab64-f9bbff890626
# ╠═fb2955ac-80ff-4cd3-9918-e2928c443577
# ╠═c090bb1e-7d4d-4b89-a687-6f0299da535b
# ╠═6bd71c4d-5fa3-44e1-961c-6c637b344342
# ╟─e496c155-d191-4c7a-b095-7404d89fc3b9
