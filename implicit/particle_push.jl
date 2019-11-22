#
# Implementations of simple particle pushing schemes
#

module particle_push

using units: qe, qi, ϵ0, mi, me


export push_v0!

@doc """
Pushes the electrons for one sub-cycle
"""->
function push_v0!(ptl_e, ptl_i, L, Δt, E_ip, E_initial, S_vec, zgrid, Δz, j_avg)
   #### Electrons
   # Get electric field at the particle positions
   E_particles = E_ip(map(p -> p.pos, ptl_e)))
   # Forward Euler for particle position and velocity
   #@. ptl_z = ptl_z + ptl_v * Δτ
   #ptl_e_z = map((z, v) -> z + v * Δt, ptl_e_z, ptl_e_v)

   idx = 0
   for ptl ∈ ptl_e
      ptl.pos = rem(ptl.pos + ptl_v * Δt, L)
      ptl.vel = ptl_vel + qe * E_particles[idx] * Δt / me

   # Wrap particle positions to periodic boundary conditions
   #@. ptl_e_z = rem(ptl_e_z + L, L)

   # Forward Euler for particle velocity
   #ptl_e_v = map((v, Ep) -> v + qe * Ep * Δt / me, ptl_e_v, E_particles)

   #@. ptl_v = ptl_v + (q/m) * E_particles * Δτ
   # Calculate current density

   ### Ions
   E_particles = E_ip(map(p -> p.pos, ptl_e))
   
   ptl_i_z = map((z, v) -> z + v * Δt, ptl_i_z, ptl_i_v)
   @. ptl_i_z = rem(ptl_i_z + L, L)
   ptl_i_v = map((v, Ep) -> v + qi * Ep * Δt / mi, ptl_i_v, E_particles)

   j_avg .= j_avg + sum(map((z, v) -> S_vec(z, zgrid, Δz) * v * qe, ptl_e_z, ptl_e_v))
   j_avg .= j_avg + sum(map((z, v) -> S_vec(z, zgrid, Δz) * v * qi, ptl_i_z, ptl_i_v))

   return

end


end
