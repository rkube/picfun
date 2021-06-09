### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 930c960c-7858-11eb-2d72-f7cb7bf44c49
begin
	using Plots
	using DelimitedFiles
	using Printf
	using PlutoUI
	using FFTW
	using Statistics: mean
	using LinearAlgebra: norm
	using Distributions
	using StatsBase
	using Interpolations
end

# ╔═╡ aa0db0ca-7858-11eb-3a9a-5b8b6065ec21
begin
	datadir = "/Users/ralph/source/repos/picfun"
end

# ╔═╡ a9f4ec02-7858-11eb-2fe3-1737367034e5
begin
	vrg = -5:0.1:5.0
	tidx_rg = 0:10:100
	nbins=size(vrg)[1]
	
	
	v_hist_all = zeros(11, nbins-1)
	
	for tt in tidx_rg
		fname = @sprintf "%s/particles_%04d.txt" datadir tt + 0
		ptl_kin = readdlm(fname, '\t', Float64, header=true)[1]
#		plot(ptl_kin[1:skip_ptl:end, 1], ptl_kin[1:skip_ptl:end, 2], label="electrons", seriestype=:scatter, xlabel="Position / λDe", ylabel="Velocity / vₜₕ")
		#plot!(ptl_kin[1:skip_ptl:end,3], ptl_kin[1:skip_ptl:end,4], label="ions", seriestype=:scatter)

		h = fit(Histogram, ptl_kin[:, 2], vrg)
		v_hist_all[(tt÷10)+1, :] = h.weights[:]
	
		# h = histogram(ptl_kin[:, 2], normalize=true)
		# plot!(vrg, 1/√(2π) * exp.(-0.5 * vrg.^2))
	end
end

# ╔═╡ d67be7e4-7858-11eb-2297-553f1420454b
contourf(vrg[1:end-1], tidx_rg * 0.1, log.(v_hist_all .+ 1e-3))

# ╔═╡ 94c0e0c4-7859-11eb-34a3-d562e8be4a96
(size(vrg), size(tidx_rg), size(v_hist_all))

# ╔═╡ Cell order:
# ╠═930c960c-7858-11eb-2d72-f7cb7bf44c49
# ╠═aa0db0ca-7858-11eb-3a9a-5b8b6065ec21
# ╠═a9f4ec02-7858-11eb-2fe3-1737367034e5
# ╠═d67be7e4-7858-11eb-2297-553f1420454b
# ╠═94c0e0c4-7859-11eb-34a3-d562e8be4a96
