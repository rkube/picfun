#
#
#


module diagnostics

using Printf
using particles: particle
using units: mₑ, mᵢ, ϵ₀
using grids: grid_1d

function diag_ptl(ptlₑ::Array{particle, 1}, ptlᵢ::Array{particle, 1}, tidx)
    # Write particle kinetics to file

    fname = @sprintf "particles_%04d.txt" tidx
    open(fname, "w") do io
        write(io, "x(ele)\tv(ele)\tx(ion)\tv(ion)")
        for i in 1:length(ptlₑ) 
           write(io, "$(ptlₑ[i].pos)\t$(ptlₑ[i].vel)\t$(ptlᵢ[i].pos)\t$(ptlᵢ[i].vel)  \n")
        end
    end
end

function diag_energy(ptlₑ::Array{particle, 1}, ptlᵢ::Array{particle, 1}, E::AbstractArray, 
                     tidx)

    # Calculate kinetic energy of ions and electrons
    ekin_ele = sum(map(p -> p.vel * p.vel * mₑ * 0.5, ptlₑ))
    ekin_ion = sum(map(p -> p.vel * p.vel * mₑ * 0.5, ptlᵢ))
    # Energy in the electric field
    enrg_elc = 0.5 * ϵ₀ * sum(E .* E)

    fname = @sprintf "energy.txt" tidx 
    open(fname, "a") do io
        write(io, "$(tidx)\t$(ekin_ele)\t$(ekin_ion)\t$(enrg_elc)\n")
    end
end



end # module diagnostics