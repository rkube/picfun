#
# Define the simulation grid
#

module grids

export grid_1d

struct grid_1d
  Lz::Float64
  Nz::Int64
  Δz::Float64
end

function init_grid(Lz::Float64, Nz::Int64) :: grid_1d
  return grid_1d(Lz, Nz, Lz/Nz)
end

end
