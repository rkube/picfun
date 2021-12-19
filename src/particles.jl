#
# Define particle struct
#

using Zygote
import Base: +, -


export particle, fix_position!

# Particles have a position and velocity
mutable struct particle
    pos
    vel
end

"""

    copy(p::particle)

Returns a new partilce instance with the same position and velocity as the original particle.
"""
Base.copy(p::particle) = particle(p.pos, p.vel)


x(p::particle) = p.pos
Zygote.@adjoint function x(p::particle)
    @show p
    println("@adjoint x(p)")
    (p.pos, x̄ -> (particle(x̄, 0), ))
end

v(p::particle) = p.v
Zygote.@adjoint function v(p::particle)
    @show p 
    println("@adjoint v(p)")
    (p.vel, v̄ -> (particle(0, v̄), ))
end

# Define arithmetic rules for particles
# In practice they act like a vector from R²
# These should never be used directly, but serve only to define 
# rules for derivatives.

"""

    +(a::particle, b::particle)

Add position and velocity for particle structs.
"""
(+)(a::particle, b::particle) = particle(a.pos + b.pos, a.vel + b.vel)

"""

    -(a::particle, b::particle)

Subtract position and velocity for particle structs.
"""
(-)(a::particle, b::particle) = particle(a.pos - b.pos, a.vel - b.vel)


Zygote.@adjoint particle(x, v) = particle(x, v), p̄ -> (p̄.x, p̄.v)


"""

    fix_position!(p, L)

Moves a particle's position back into the domain [0:L]
"""
function fix_position!(ptl::particle, L)
    ptl.pos = mod(ptl.pos, L)
end

# End of file particles.jl