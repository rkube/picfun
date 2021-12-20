# Utility functions
These functions are used internally.


## grids
Utility functions related to the particle grid.

```@docs
init_grid(Lz, Nz)
```

## particles
Utility functions related to the particles.


```@docs
particle
x(::particle)
v(::particle)
copy(::particle)
fix_position!(ptl::particle, L)
+(::particle, ::particle)
-(::particle, ::particle)
```

## pic_utils
Utility functions related to PIC algorithm
```@docs
b1(z, zp, Δz)
deposit(::Array{particle}, zgrid::grid_1d, fun::Function)
```
##