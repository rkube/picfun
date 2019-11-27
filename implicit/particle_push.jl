#
# Implementations of simple particle pushing schemes
#

module particle_push

using units: qₑ, qᵢ, mₑ, mᵢ, ϵ₀
using grids: grid_1d
using particles: particle, fix_position!

export push_v0!, push_v1!

@doc """
Pushes the electrons and ions for one sub-cycle
"""->
function push_v0!(ptl_e_z, ptl_e_v, ptl_i_z, ptl_i_v, L, Δt, E_ip, E_initial, S_vec, zgrid, Δz, j_avg)
   #### Electrons
   # Get electric field at the particle positions
   E_particles = E_ip(ptl_e_z)
   # Forward Euler for particle position and velocity
   #@. ptl_z = ptl_z + ptl_v * Δτ
   ptl_e_z = map((z, v) -> z + v * Δt, ptl_e_z, ptl_e_v)
   # Wrap particle positions to periodic boundary conditions
   @. ptl_e_z = rem(ptl_e_z + L, L)
   # Forward Euler for particle velocity
   ptl_e_v = map((v, Ep) -> v + qₑ * Ep * Δt / mₑ, ptl_e_v, E_particles)
   #@. ptl_v = ptl_v + (q/m) * E_particles * Δτ
   # Calculate current density

   ### Ions
   E_particles = E_ip(ptl_e_z)
   ptl_i_z = map((z, v) -> z + v * Δt, ptl_i_z, ptl_i_v)
   @. ptl_i_z = rem(ptl_i_z + L, L)
   ptl_i_v = map((v, Ep) -> v + qᵢ * Ep * Δt / mᵢ, ptl_i_v, E_particles)

   j_avg .= j_avg + sum(map((z, v) -> S_vec(z, zgrid, Δz) * v * qₑ, ptl_e_z, ptl_e_v))
   j_avg .= j_avg + sum(map((z, v) -> S_vec(z, zgrid, Δz) * v * qᵢ, ptl_i_z, ptl_i_v))

   return
end #push_v0!

@doc """
Next version of the push-kernel: Only integrate equations of motion with
Euler in time. Do not update j_avg.
"""->
function push_v1!(ptl_vec::Array{particle}, grid::grid_1d, Δt, E_ip)
   #### Electrons
   # Get electric field at the particle positions
   E_ptl = E_ip(map(p -> p.pos, ptl_vec))
   num_ptl = length(ptl_vec)

   idx = 0
   for idx ∈ 1:num_ptl
      ptl_vec[idx].pos = rem(ptl_vec[idx].pos + ptl_vec[idx].vel * Δt, grid.Lz)
      ptl_vec[idx].vel = ptl_vec[idx].vel + qₑ * E_ptl[idx] * Δt / me
   end
end #push_v1!


@doc """
Iteratively determines new positions given an Electric field
"""->
function push_v2!(ptlₑ::Array{particle},
                  ptlₑ₀::Array{particle},
                  ptlₑ½::Array{particle},
                  ϵᵣ, ϵₐ, zgrid, Δt, ip_Ẽ, ip_Eⁿ)
   for ele ∈ 1:length(ptlₑ)
      #println("Iterating particle $(ele) / $(num_ptl)")
      # Initial guess of new position and velocity before we start iterations.
      x̃= ptlₑ[ele].pos + 0.1
      num_it_ptl = 0
      ptl_converged=false
      # Then: iterate the particle's coordinates until convergence
      while(ptl_converged == false)
           # Calculate x_p^{n+1/2}
           xₚⁿ⁺½ =  0.5 * (ptlₑ₀[ele].pos + x̃)
           # Calculate v_p^{n+1}
           vₚⁿ⁺¹= ptlₑ₀[ele].vel + Δt * qₑ * 0.5 * (ip_Ẽ(xₚⁿ⁺½) + ip_Eⁿ(xₚⁿ⁺½)) / mₑ
           # Calculate v_p^{n+1/2}
           vₚⁿ⁺½ = 0.5 * (ptlₑ₀[ele].vel + vₚⁿ⁺¹)
           # Calculate x_p^{n+1}
           xₚⁿ⁺¹ = ptlₑ₀[ele].pos + Δt * vₚⁿ⁺½
           #println("*** it $(num_it_ptl): xp_n12=$(xp_n12), xp_new=$(xp_new), vp_n12=$(vp_n12), vp_new=$(vp_new)")

           # Check convergence
           if ((abs(xₚⁿ⁺¹ - x̃) ≤ ϵᵣ * abs(xₚⁿ⁺¹) + ϵₐ))
               #println("*** Converged: x̃ - xₚⁿ⁺¹| = $(abs(xₚⁿ⁺¹ - x̃))")
               ptl_converged = true
               ptlₑ[ele].pos = xₚⁿ⁺¹
               ptlₑ[ele].vel = vₚⁿ⁺¹
               break
           end
           # Let xₚⁿ⁺¹ be the new guess.
           x̃ = xₚⁿ⁺¹
           num_it_ptl += 1
           if(num_it_ptl > 100)
               println("Iterations exceeded $(num_it_ptl), terminating")
               ptl_converged = true
               break
           end
      end #while_ptl_converged==false
      fix_position!(ptlₑ[ele], zgrid.Lz)
      # Update ptl_e12 for the current particle
      ptlₑ½[ele] = particle(0.5 * (ptlₑ[ele].pos + ptlₑ₀[ele].pos),
                            0.5 * (ptlₑ[ele].vel + ptlₑ₀[ele].vel))
      fix_position!(ptlₑ½[ele], zgrid.Lz)
   end #for ptl in ptl_e
end

end #module
