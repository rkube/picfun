#
#
#
module diagnostics

using Printf
using particles: particle
using units: mₑ, mᵢ, n₀
using grids: grid_1d
using solvers: ∇⁻²
using pic_utils: deposit, smooth

function diag_ptl(ptlₑ::Array{particle, 1}, ptlᵢ::Array{particle, 1}, tidx)
    # Write particle kinetics to file

    fname = @sprintf "particles_%04d.txt" tidx
    open(fname, "w") do io
        write(io, "x(ele)\tv(ele)\tx(ion)\tv(ion)\n")
        for i in 1:length(ptlₑ)
           write(io, "$(ptlₑ[i].pos)\t$(ptlₑ[i].vel)\t$(ptlᵢ[i].pos)\t$(ptlᵢ[i].vel)  \n")
        end
    end
end

function diag_energy(ptlₑ::Array{particle, 1}, ptlᵢ::Array{particle, 1}, E::AbstractArray,
                     tidx, zgrid)

    # Calculate kinetic energy of ions and electrons
    ekin_ele = sum(map(p -> p.vel * p.vel * mₑ * 0.5 / n₀, ptlₑ))
    ekin_ion = sum(map(p -> p.vel * p.vel * mᵢ * 0.5 / n₀, ptlᵢ))
    # Energy in the electric field
    enrg_elc = n₀ * 0.5  * sum(E .* E) * zgrid.Δz

    open("Efield.txt", "a") do io
        write(io, "$(tidx)\t")
        for i ∈ 1:length(E)
            write(io, "$(E[i]) ")
        end
        write(io, "\n")
    end

    open("energy.txt", "a") do io
        write(io, "$(tidx)\t$(ekin_ele)\t$(ekin_ion)\t$(enrg_elc)\n")
    end
end

function diag_fields(ptlₑ:: Array{particle, 1}, ptlᵢ::Array{particle, 1}, zgrid, tidx)
    """Calculate fields and write them to file"""

    nₑ = deposit(ptlₑ, zgrid, p -> 1. / n₀)
    nᵢ = deposit(ptlᵢ, zgrid, p -> 1. / n₀)
    ρⁿ = (nᵢ - nₑ)
    ϕⁿ = ∇⁻²(-ρⁿ, zgrid)
    # Calculate initial electric field with centered difference stencil
    Eⁿ = zeros(zgrid.Nz)
    Eⁿ[1] = -1. * (ϕⁿ[2] - ϕⁿ[end]) / 2. / zgrid.Δz
    Eⁿ[2:end-1] = -1. * (ϕⁿ[1:end-2] - ϕⁿ[3:end]) / 2. / zgrid.Δz
    Eⁿ[end] = -1. * (ϕⁿ[end-1] - ϕⁿ[1]) / 2. / zgrid.Δz
    smEⁿ = smooth(Eⁿ)

    open("ni.txt", "a") do io
        write(io, "$(tidx)\t")
        for i ∈ 1:length(nᵢ)
            write(io, "$(nᵢ[i]) ")
        end
        write(io, "\n")
    end

    open("ne.txt", "a") do io
        write(io, "$(tidx)\t")
        for i ∈ 1:length(nₑ)
            write(io, "$(nₑ[i]) ")
        end
        write(io, "\n")
    end

    open("rho.txt", "a") do io
        write(io, "$(tidx)\t")
        for i ∈ 1:length(ρⁿ)
            write(io, "$(ρⁿ[i]) ")
        end
        write(io, "\n")
    end

    open("phi.txt", "a") do io
        write(io, "$(tidx)\t")
        for i ∈ 1:length(ϕⁿ)
            write(io, "$(ϕⁿ[i]) ")
        end
        write(io, "\n")
    end

    open("E.txt", "a") do io
        write(io, "$(tidx)\t")
        for i ∈ 1:length(smEⁿ)
            write(io, "$(smEⁿ[i]) ")
        end
        write(io, "\n")
    end
end #function diag_fields

end # module diagnostics
