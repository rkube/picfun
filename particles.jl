#
# Define particle struct
#

module particles

export particle, fix_position!

# Particles have a position and velocity
mutable struct particle{T}
    pos::T
    vel::T
end

# Copy method creates a new instance.
Base.copy(p::particle) = particle(p.pos, p.vel)

function fix_position!(particle, L)
    particle.pos = mod(particle.pos, L)
end


end
