#
#
#
module diagnostics

using Printf
using particles: particle
using units: qₑ, qᵢ, mₑ, mᵢ
using grids: grid_1d
using solvers: ∇⁻², invert_laplace
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
    ekin_ele = sum(map(p -> 0.5 * mₑ * p.vel * p.vel, ptlₑ))
    ekin_ion = sum(map(p -> 0.5 * mᵢ * p.vel * p.vel, ptlᵢ))
    # Energy in the electric field
    enrg_elc = 0.5 * sum(E .* E) * zgrid.Δz

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

function diag_fields(ptlₑ:: Array{particle, 1}, ptlᵢ::Array{particle, 1}, zgrid, tidx, ptl_wt)
    """Calculate fields and write them to file"""

    nₑ = deposit(ptlₑ, zgrid, p -> ptl_wt)
    nᵢ = deposit(ptlᵢ, zgrid, p -> ptl_wt)
    ρ = (qᵢ*nᵢ + qₑ*nₑ)
    #ϕⁿ = ∇⁻²(-ρⁿ, zgrid)
    ϕ = invert_laplace(-ρ, zgrid)
    # Calculate initial electric field with centered difference stencil
    E = zeros(zgrid.Nz)
    E[1] = -0.5 * (ϕ[2] - ϕ[end]) / zgrid.Δz
    E[2:end-1] = -0.5 * (ϕ[3:end] - ϕ[1:end-2]) / zgrid.Δz
    E[end] = -0.5 * (ϕ[1] - ϕ[end-1]) / zgrid.Δz
    smE = smooth(E)

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
        for i ∈ 1:length(ρ)
            write(io, "$(ρ[i]) ")
        end
        write(io, "\n")
    end

    open("phi.txt", "a") do io
        write(io, "$(tidx)\t")
        for i ∈ 1:length(ϕ)
            write(io, "$(ϕ[i]) ")
        end
        write(io, "\n")
    end

    open("E.txt", "a") do io
        write(io, "$(tidx)\t")
        for i ∈ 1:length(E)
            write(io, "$(E[i]) ")
        end
        write(io, "\n")
    end

    open("smE.txt", "a") do io
        write(io, "$(tidx)\t")
        for i ∈ 1:length(smE)
            write(io, "$(smE[i]) ")
        end
        write(io, "\n")
    end
end #function diag_fields

end # module diagnostics
