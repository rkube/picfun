# Implmentation of the implicit PIC method described in
#"An energy- and charge-conserving, implicit, electrostatic particle-in-cell algorithm
# G. Chen, L. Chacon, D.C. Barnes, J. Comput. Phys. 230 7018-7036 (2011)

using Interpolations
using Plots
using Distributions
using Sobol
using StatsFuns

push!(LOAD_PATH, pwd())

using units: qe, qi, ϵ0, mi, me
using pic_utils: S_vec, SM
using load_particles: load_pert_x
using particle_push: push_v0


# Description of the domain
# Number of grid points
Nz = 32
# Length of the domain
L = 2 * π
# Distance between grid points
Δz = ones(Nz) * L / Nz
# Z-coordinate of the grid points
zgrid = collect(0:Nz - 1) .* Δz

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
# The output after one time-step
E_initial = [3.5735791004149227, -5.317250081092494, 17.565761759877663, 28.06105971652772, 11.971455915053918, 7.590527908414766, -22.114023493277212, -32.607057144960265, -23.496295783798136, -6.382908887804389, 13.384948464784504, -14.957084619913218, 6.864587272637304, 12.679840963982537, 8.698683358381297, 6.414613285733398, -8.63535497318402, 11.984446740068847, -9.697309670018369, -2.7353688969451957, 16.59126493614053, 21.259407009932087, 27.413061622342518, -23.763608877170288, -5.386046408358324, -2.1000174457746987, -35.5461125662315, -2.459060225853311, 12.662883280937624, 22.986280866174923, -9.703274909437598, -24.499831980789445]

current = zeros(Nz)
E_i_sm = SM(E_initial)

# Create copies which include the first element for interpolation purposes
zgrid_per = copy(zgrid)
E_i_sm_per = copy(E_i_sm)
push!(zgrid_per, zgrid_per[end] + Δz[end])
push!(E_i_sm_per, E_i_sm_per[1])
E_ip = LinearInterpolation(zgrid_per, E_i_sm_per)

# Orbit averaged current density
j_avg = zeros(Nz)

#p = plot(ptl_e_z, seriestype=:scatter)
# Push particles explicitly

for n in 1:Nt
  println("Subcycle $n / $Nt")

  particle_push.push_v0!(ptl_e_z, ptl_e_v, ptl_i_z, ptl_i_v, L, Δt, E_ip, E_initial, S_vec, zgrid, Δz, j_avg)
  #
  # #### Electrons
  # # Get electric field at the particle positions
  # E_particles = E_ip(ptl_e_z)
  # # Forward Euler for particle position and velocity
  # #@. ptl_z = ptl_z + ptl_v * Δτ
  # global ptl_e_z = map((z, v) -> z + v * Δt, ptl_e_z, ptl_e_v)
  # # Wrap particle positions to periodic boundary conditions
  # @. ptl_e_z = rem(ptl_e_z + L, L)
  # # Forward Euler for particle velocity
  # global ptl_e_v = map((v, Ep) -> v + qe * Ep * Δt / me, ptl_e_v, E_particles)
  # #@. ptl_v = ptl_v + (q/m) * E_particles * Δτ
  # # Calculate current density
  #
  # ### Ions
  # E_particles = E_ip(ptl_e_z)
  # global ptl_i_z = map((z, v) -> z + v * Δt, ptl_i_z, ptl_i_v)
  # @. ptl_i_z = rem(ptl_i_z + L, L)
  # global ptl_i_v = map((v, Ep) -> v + qi * Ep * Δt / mi, ptl_i_v, E_particles)
  #
  # j_avg .= j_avg + sum(map((z, v) -> S_vec(z, zgrid, Δz) * v * qe, ptl_e_z, ptl_e_v))
  # j_avg .= j_avg + sum(map((z, v) -> S_vec(z, zgrid, Δz) * v * qi, ptl_i_z, ptl_i_v))
end

#plot!(ptl_e_z, seriestype=:scatter)

# Normalize j_avg
j_avg = map((j, Δzi) -> j / Δt / Δzi, j_avg, Δz)
# Calculate electric field from sub-cycle
E_new = E_initial - Δt * j_avg / ϵ0

plot(zgrid, E_initial, reuse=false)
plot!(zgrid, E_new)

# Update average e_field
E_avg = 0.5 * (E_new + E_initial)

#display(p)
println("End")
