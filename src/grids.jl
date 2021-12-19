# Define the simulation grid

export grid_1d, init_grid

"""
`grid_1d` represents a periodic grid on ``[0:L]``.
Field values are disretized on the grid points ``x_n = n \\times \\Delta z``,
where ``n=0,1,...N_z-1``.

# Fields
- Lz: Total length of the domain
- Nz: Number of grid points
- Δz: Distance between neighbouring grid points.
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