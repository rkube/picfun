#
# Implementations of simple particle pushing schemes
#

module particle_push

using units: qₑ, qᵢ, mₑ, mᵢ, ϵ₀
using grids: grid_1d
using particles: particle, fix_position!
using Distributions

export push_v0!, push_v1!, push_v2!, push_v3!

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
   # Parallelize this across threads
   Threads.@threads for ele ∈ 1:length(ptlₑ)
      # Initial guess of new position and velocity before we start iterations.
      x̃ = ptlₑ[ele].pos + rand(Uniform(-0.1, 0.1), 1)[1]
      #println("Orignial position: $(ptlₑ[ele].pos), Starting guess: $(x̃). Original velocity: $(ptlₑ[ele].vel)")
      num_it_ptl = 0
      ptl_converged=false
      # Then: iterate the particle's coordinates until convergence
      # Note: We don't need to worry moving the particle out of bounds here since the
      # electric field interpolator is periodic. And distance measuring between particle
      # positions is fine too. We only need to fix their position after the Picard
      # iteration has converged.
      while(ptl_converged == false)
           # Calculate x_p^{n+1/2}
           x_pⁿ⁺½ =  0.5 * (ptlₑ₀[ele].pos + x̃)
           # Calculate v_p^{n+1}
           v_pⁿ⁺¹= ptlₑ₀[ele].vel + Δt * qₑ * 0.5 * (ip_Ẽ(x_pⁿ⁺½) + ip_Eⁿ(x_pⁿ⁺½)) / mₑ
           # Calculate v_p^{n+1/2}
           v_pⁿ⁺½ = 0.5 * (ptlₑ₀[ele].vel + v_pⁿ⁺¹)
           # Calculate x_p^{n+1}
           x_pⁿ⁺¹ = ptlₑ₀[ele].pos + Δt * v_pⁿ⁺½
           #println("*** it $(num_it_ptl): x_pⁿ⁺¹=$(x_pⁿ⁺¹), v_pⁿ⁺¹=$(v_pⁿ⁺¹)")

           # Check convergence
           if ((abs(x_pⁿ⁺¹ - x̃) ≤ ϵᵣ * abs(x_pⁿ⁺¹) + ϵₐ))
               #println("*** Starting point: $(ptlₑ[ele].pos), Guess: $(x̃), Converged to: $(x_pⁿ⁺¹) Converged: |x̃ - x_pⁿ⁺¹| = $(abs(x_pⁿ⁺¹ - x̃)), $(num_it_ptl) iterations")
               ptl_converged = true
               ptlₑ[ele].pos = x_pⁿ⁺¹
               ptlₑ[ele].vel = v_pⁿ⁺¹
               break
           end
           # Let x_pⁿ⁺¹ be the new guess.
           x̃ = x_pⁿ⁺¹
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



@doc """
Iteratively determines new positions given an Electric field.
Basically the same as v2, but works for both ions and electrons
"""->
function push_v3!(ptl::Array{particle},
                  ptl₀::Array{particle},
                  ptl½::Array{particle},
                  q, m,
                  ϵᵣ, ϵₐ, zgrid, Δt, ip_Ẽ, ip_Eⁿ)

  # Initial guess of new position before we start iterations.
  x̃ = map(p -> p.pos, ptl₀) + rand(Uniform(-0.1, 0.1), length(ptl₀))
  for ele ∈ 1:length(ptl)
     #x̃ = ptl₀[ele].pos + rand(Uniform(-0.1, 0.1), 1)[1]
     num_it_ptl = 0
     ptl_converged=false
      # Then: iterate the particle's coordinates until convergence
      # Note: We don't need to worry moving the particle out of bounds here since the
      # electric field interpolator is periodic. And distance measuring between particle
      # positions is fine too. We only need to fix their position after the Picard
      # iteration has converged.
      while(ptl_converged == false)
           # Calculate x_p^{n+1/2}
           x_pⁿ⁺½ =  0.5 * (ptl₀[ele].pos + x̃[ele])
           # Calculate v_p^{n+1}
           v_pⁿ⁺¹= ptl₀[ele].vel + Δt * q * 0.5 * (ip_Ẽ(x_pⁿ⁺½) + ip_Eⁿ(x_pⁿ⁺½)) / m
           # Calculate v_p^{n+1/2}
           v_pⁿ⁺½ = 0.5 * (ptl₀[ele].vel + v_pⁿ⁺¹)
           # Calculate x_p^{n+1}
           x_pⁿ⁺¹ = ptl₀[ele].pos + Δt * v_pⁿ⁺½
           #println("*** it $(num_it_ptl): x_pⁿ⁺¹=$(x_pⁿ⁺¹), v_pⁿ⁺¹=$(v_pⁿ⁺¹)")

           # Check convergence
           if ((abs(x_pⁿ⁺¹ - x̃[ele]) ≤ ϵᵣ * abs(x_pⁿ⁺¹) + ϵₐ))
               #println("*** Starting point: $(ptl[ele].pos), Guess: $(x̃), Converged to: $(x_pⁿ⁺¹) Converged: |x̃ - x_pⁿ⁺¹| = $(abs(x_pⁿ⁺¹ - x̃)), $(num_it_ptl) iterations")
               ptl_converged = true
               ptl[ele].pos = x_pⁿ⁺¹
               ptl[ele].vel = v_pⁿ⁺¹
               break
           end
           # Let x_pⁿ⁺¹ be the new guess.
           x̃[ele] = x_pⁿ⁺¹
           num_it_ptl += 1
           if(num_it_ptl > 1000)
               #println("Picard: Iterations exceeded $(num_it_ptl), terminating")
               ptl_converged = true
               break
           end
      end #while_ptl_converged==false
      fix_position!(ptl[ele], zgrid.Lz)
      # Update ptl_e12 for the current particle
      ptl½[ele] = particle(0.5 * (ptl[ele].pos + ptl₀[ele].pos),
                           0.5 * (ptl[ele].vel + ptl₀[ele].vel))
      fix_position!(ptl½[ele], zgrid.Lz)
   end #for ptl i
end



end #module
