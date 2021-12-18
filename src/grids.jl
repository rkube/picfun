# Define the simulation grid

export grid_1d, init_grid

struct grid_1d
  Lz
  Nz
  Δz
end

# Convenience constructor for a 1d grid
init_grid(Lz, Nz) = grid_1d(Lz, Nz, Lz / Nz)

# End of file grids.jl