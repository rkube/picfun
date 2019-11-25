#
# Define particle struct
#

module particles

export particle, fix_position!

mutable struct particle
    pos::Float64
    vel::Float64
end

function fix_position!(particle, L)
    particle.pos = mod(particle.pos, L)
end


end
