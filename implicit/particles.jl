#
# Define particle struct
#

module particles

export particle, fix_position!

mutable struct particle{T<:AbstractFloat}
    pos::T
    vel::T
end

function fix_position!(particle, L)
    particle.pos = mod(particle.pos, L)
end


end
