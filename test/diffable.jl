using Zygote
using picfun


# Define approximate equal for particle type
# This is true if both particles have about the same position and velocity
function Base.isapprox(p1::particle, p2::particle; atol::Real=0, rtol::Real=atol>0 ? 0 : sqrt(eps(Float64)))
    return isapprox([p1.pos, p1.vel], [p2.pos, p2.vel]; atol=atol, rtol=rtol)
end

@testset "diffable_1" begin
    # Test gradients for simple particle deposition
    # Charge cloud particle
    # Grid points Zj = j * Δz, Z_{j+1} = (j+1) * Δz
    # Particle cloud position of center: zp
    # Cloud charge: qc
    ################################ Test ##########################
    
    # Choose Zj < zp < Zj.
    Nz = 4
    Lz = 1.0
    zgrid = init_grid(Lz, Nz)
    zvals = (0:Nz-1) * zgrid.Δz

    # Place the particle between the third and fourth grid point
    j = 2    # OBS! Setting this offset to 2 means particle is at 3rd grid-point due to zero-based indexing of the grid points.
    zp = (j+0.22) * zgrid.Δz
    qc = 1.0

    ptl = particle(zp, 0.0)
    # Charge deposited on grid point zvals[j+1], to the left of the particle:
    qi_left(qc, Zj, ptl, Δz) = qc * (Zj - x(ptl)) / Δz
    # ∂(qj) / ∂(zp) = -qc / Δz
    res = gradient(qi_left, qc, zvals[j+1], ptl, zgrid.Δz)
    @test res[1] ≈ -0.22
    @test res[3] ≈ particle(-qc / zgrid.Δz, 0.0)


    # Charge deposited on grid point zvals[j+1], to the right of the particle:
    qi_right(qc, Zj, ptl, Δz) = qc * (x(ptl) - Zj) / Δz
    # ∂(qj) / ∂(zp) = -qc / Δz
    res = gradient(qi_right, qc, zvals[j+2], ptl, zgrid.Δz)
    @test res[1] ≈ -0.78
    @test res[3] ≈ particle(qc / zgrid.Δz, 0.0)

    

    # Now repeat the tests with the b1 shape function
    @test b1(zvals[j+1], x(ptl), zgrid.Δz) ≈ 0.78
    @test gradient(b1, zvals[j+1], x(ptl), zgrid.Δz)[2] ≈ -1.0 / zgrid.Δz

    @test b1(zvals[j+2], x(ptl), zgrid.Δz) ≈ 0.22
    @test gradient(b1, zvals[j+2], x(ptl), zgrid.Δz)[2] ≈ 1.0 / zgrid.Δz

end # testset diffable_1
