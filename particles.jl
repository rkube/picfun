#
# Define particle struct
#
module particles

using Zygote
import Base: +, -


export particle, fix_position!

# Particles have a position and velocity
mutable struct particle
    pos
    vel
end

# Copy method creates a new instance.
Base.copy(p::particle) = particle(p.pos, p.vel)

x(p::particle) = p.x
Zygote.@adjoint function x(p::particle)
    @show p
    println("@adjoint x(p)")
    (p.x, x̄ -> (particle(x̄, 0), ))
end

v(p::particle) = p.v
Zygote.@adjoint function v(p::particle)
    @show p 
    println("@adjoint v(p)")
    (p.v, v̄ -> (particle(0, v̄), ))
end

# Define arithmetic rules for particles
# In practice they act like a vector from R²
# These should never be used directly, but serve only to define 
# rules for derivatives.
a::particle + b::particle = particle(x(a) + x(b), v(a) + v(b))
a::particle - b::particle = particle(x(a) - x(b), v(a) - v(b))

# Adjoint constructor
Zygote.@adjoint particle(x, v) = particle(x, v), p̄ -> (p̄.x, p̄.v)

function fix_position!(particle, L)
    # If a particle position is outside the domain, move it
    # by one domain length so that it is within the domain
    if particle.pos < 0
        particle.pos += L - 10 * eps(typeof(particle.pos))
    elseif particle.pos ≥ L
        particle.pos -= L + 10 * eps(typeof(particle.pos))
    end
    #particle.pos = mod(particle.pos, L)
end


end
