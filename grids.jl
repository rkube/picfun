#
# Define the simulation grid
#

module grids

export grid_1d, init_grid

struct grid_1d
  Lz
  Nz
  Δz
end

function init_grid(Lz, Nz) :: grid_1d
  return grid_1d(Lz, Nz, Lz/Nz)
end

end
