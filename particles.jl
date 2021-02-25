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
