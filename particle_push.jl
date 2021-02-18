#
# Implementations of simple particle pushing schemes
#

module particle_push

using units: qₑ, qᵢ, mₑ, mᵢ
using grids: grid_1d
using particles: particle, fix_position!
using Distributions

export push_v3!


"""

    push_v3!(ptl, ptl₀, ptl½, q, m_over_mₑ, ϵᵣ, ϵₐ, zgrid, Δt, ip_Ẽ, ip_Eⁿ)

Iteratively determine new particle position and velocity given a guess Ẽ.

# Arguments
- `ptl::Array{particle, 1}`: Array of updated particle states. Will be updated.
- `ptlₑ::Array{particle, 1}`: Array of initial particle states
- `ptl½::Array{particle, 1}`: Array of particle states at half time-step. Will be updated
- `q::Float64`: Particle charge
- `m_over_mₑ::Float64`: Particle mass relative to electron mass
- `ϵᵣ::Float64`: Relative tolerance threshold
- `ϵₐ::Float64`: Absolute tolerance threshold
- `Δt::Float64`: Time step size
- `ip_Ẽ::Function(Float)`: interpolator for electric field at t=(n+1)*Δt
- `ip_Eⁿ::Function(Float)`: interpolator for electric field at t=n*Δn on


"""
function push_v3!(ptl::Array{particle},
                  ptl₀::Array{particle},
                  ptl½::Array{particle},
                  q, mₑ_over_m,
                  ϵᵣ, ϵₐ, zgrid, Δt, ip_E12)
    # Initial guess of new position before we start Picard iteration.
    # No modulo since we take the average later: xn+1/2 = (x + x̃) / 2.
    x̃ = map(p -> p.pos + p.vel * Δt, ptl₀)
  
    Threads.@threads for ele ∈ 1:length(ptl)
        num_it_ptl = 0
        ptl_converged=false
        # Then: iterate the particle's coordinates until convergence
        # Note: We don't need to worry moving the particle out of bounds here since the
        # electric field interpolator is periodic. To ensure that the stop condition for the
        # iteration is not affected by periodicity, we restrict the particle position ot the
        # domain only after iteration has converged.
        while(ptl_converged == false)
            # Calculate x_p^{n+1/2}
            x_pⁿ⁺½ =  0.5 * (ptl₀[ele].pos + x̃[ele])
            # Calculate v_p^{n+1}
            v_pⁿ⁺¹= ptl₀[ele].vel + Δt * q * mₑ_over_m * ip_E12(x_pⁿ⁺½)
            # Calculate v_p^{n+1/2}
            v_pⁿ⁺½ = 0.5 * (ptl₀[ele].vel + v_pⁿ⁺¹)
            # Calculate x_p^{n+1}
            x_pⁿ⁺¹ = ptl₀[ele].pos + Δt * v_pⁿ⁺½
    
            # Check convergence
            if ((abs(x_pⁿ⁺¹ - x̃[ele]) ≤ ϵᵣ * abs(x̃[ele]) + ϵₐ))
                ptl_converged = true
                ptl[ele].pos = x_pⁿ⁺¹
                ptl[ele].vel = v_pⁿ⁺¹
                break
            end
    
            if (num_it_ptl > 100)
                println("*** Failed to converge: Starting point: $(ptl[ele].pos), Guess: $(x̃[ele]), Converged to: $(x_pⁿ⁺¹) Converged: |x̃[ele] - x_pⁿ⁺¹| = $(abs(x_pⁿ⁺¹ - x̃[ele])), $(num_it_ptl) iterations")
                ptl_converged = true
                break
            end
    
            # Let x_pⁿ⁺¹ be the new guess.
            x̃[ele] = x_pⁿ⁺¹
            num_it_ptl += 1
        end #while_ptl_converged==false
        # Update ptl_e12 for the current particle
        # Do this before calling fix_position!
        ptl½[ele] = particle(0.5 * (ptl[ele].pos + ptl₀[ele].pos),  0.5 * (ptl[ele].vel + ptl₀[ele].vel))
        fix_position!(ptl[ele], zgrid.Lz)
        fix_position!(ptl½[ele], zgrid.Lz)
    end #for ptl i
end

end #module
