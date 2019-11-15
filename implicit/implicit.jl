# Implmentation of the implicit PIC method described in 
#"An energy- and charge-conserving, implicit, electrostatic particle-in-cell algorithm
# G. Chen, L. Chacon, D.C. Barnes, J. Comput. Phys. 230 7018-7036 (2011)

using Interpolations
using Plots
using Distributions
using Sobol
using StatsFuns

push!(LOAD_PATH, pwd())
#include("halton.jl")

#using Halton: HaltonSeq!, HaltonDraws!

using pic_utils: S_vec, SM
using load_particles: load_pert_x

# Description of the domain
# Number of grid points
Nz = 32
# Length of the domain
L = 2 * π
# Distance between grid points
Δz = ones(Nz) * L / Nz
# Z-coordinate of the grid points
zgrid = collect(0:Nz - 1) .* Δz

qe = -1.0
qi = 1.0
ϵ0 = 1.0
me = 1.0 
mi = 100.0
Δt = 1e-3
Nt = 1
Nν = 10
Δτ = Δt / Nν
# Initial number of particles per cell
particle_per_cell = 16
num_ptl = Nz * particle_per_cell
println("Nz = $Nz, L = $L")

ptl_e_z, ptl_e_v = load_pert_x(num_ptl, L, 0.1, 20.0, 1.0)
ptl_i_z = rand(Uniform(1e-6, L-1e-6), num_ptl)
sort!(ptl_i_z)
ptl_i_v = zeros(num_ptl)

ptl_e_z0 = copy(ptl_e_z)
ptl_e_v0 = copy(ptl_e_v)
ptl_i_z0 = copy(ptl_i_z)
ptl_i_v0 = copy(ptl_i_v)

# Initialize the electric field
E_field = zeros(Nz)
current = zeros(Nz)
E_smooth = SM(E_field)

# Create copies which include the first element for itnerpolation purposes
zgrid_per = copy(zgrid)
E_smooth_per = copy(E_smooth)
push!(zgrid_per, zgrid_per[end] + Δz[end])
push!(E_smooth_per, E_smooth_per[1])
E_ip = LinearInterpolation(zgrid_per, E_smooth_per)

# Orbit averaged current density
j_avg = zeros(Nz)

p = plot(ptl_e_z, seriestype=:scatter)
# Push particles explicitly
for n in 1:Nt
  println("Subcycle $n / $Nt")

  #### Electrons
  # Get electric field at the particle positions
  E_particles = E_ip(ptl_e_z)
  # Forward Euler for particle position and velocity
  #@. ptl_z = ptl_z + ptl_v * Δτ
  global ptl_e_z = map((z, v) -> z + v * Δt, ptl_e_z, ptl_e_v)
  # Wrap particle positions to periodic boundary conditions
  @. ptl_e_z = rem(ptl_e_z + L, L)
  # Forward Euler for particle velocity
  global ptl_e_v = map((v, Ep) -> v + qe * Ep * Δt / me, ptl_e_v, E_particles)
  #@. ptl_v = ptl_v + (q/m) * E_particles * Δτ
  # Calculate current density

  ### Ions
  E_particles = E_ip(ptl_e_z)
  global ptl_i_z = map((z, v) -> z + v * Δt, ptl_i_z, ptl_i_v)
  @. ptl_i_z = rem(ptl_i_z + L, L)
  global ptl_i_v = map((v, Ep) -> v + qi * Ep * Δt / mi, ptl_i_v, E_particles)
  
  j_avg .= j_avg + sum(map((z, v) -> S_vec(z, zgrid, Δz) * v * qe, ptl_e_z, ptl_e_v))  
  j_avg .= j_avg + sum(map((z, v) -> S_vec(z, zgrid, Δz) * v * qi, ptl_i_z, ptl_i_v))
end

plot!(ptl_e_z, seriestype=:scatter)

# Normalize j_avg
j_avg = map((j, Δzi) -> j / Δt / Δzi, j_avg, Δz)
# Calculate electric field from sub-
E_new = E_field - Δt * j_avg / ϵ0

# Update average e_field
E_avg = 0.5 * (E_new + E_field)

display(p)
println("End")