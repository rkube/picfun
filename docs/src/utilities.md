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
copy(p::particle)
fix_position!(ptl::particle, L)
+(::particle, ::particle)
-(::particle, ::particle)
```
##