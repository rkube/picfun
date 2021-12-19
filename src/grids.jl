# Define the simulation grid

export grid_1d, init_grid

"""
    struct grid_1d

Representation of the simulation Domain
"""
struct grid_1d
  Lz
  Nz
  Δz
end

"""
  init_grid(Lz, Nz)


Convenience constructor for a 1d grid.
"""
init_grid(Lz, Nz) = grid_1d(Lz, Nz, Lz / Nz)

# End of file grids.jl