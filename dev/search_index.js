var documenterSearchIndex = {"docs":
[{"location":"diffable/#Differentiability","page":"Differentiability","title":"Differentiability","text":"","category":"section"},{"location":"diffable/","page":"Differentiability","title":"Differentiability","text":"Notes on differentiability of various code components","category":"page"},{"location":"diffable/#Shape-function","page":"Differentiability","title":"Shape function","text":"","category":"section"},{"location":"diffable/","page":"Differentiability","title":"Differentiability","text":"In a hurry, so only the basics here. In the notebooks folder, the notebook derivatives_b1.jl shows the details for differentiablity of the shape functions. The graphics below visualize how the derivative of the basic shape function b1 looks like","category":"page"},{"location":"diffable/","page":"Differentiability","title":"Differentiability","text":"For 4 grid points (Image: \"Nz=4\") For 8 gridpoints (Image: \"Nz=8\")","category":"page"},{"location":"simsetup/#Setting-up-a-simulation","page":"Simulation setup","title":"Setting up a simulation","text":"","category":"section"},{"location":"simsetup/","page":"Simulation setup","title":"Simulation setup","text":"How to set up a simulation","category":"page"},{"location":"utilities/#Utility-functions","page":"Code Reference","title":"Utility functions","text":"","category":"section"},{"location":"utilities/","page":"Code Reference","title":"Code Reference","text":"These functions are used internally.","category":"page"},{"location":"utilities/#grids","page":"Code Reference","title":"grids","text":"","category":"section"},{"location":"utilities/","page":"Code Reference","title":"Code Reference","text":"Utility functions related to the particle grid.","category":"page"},{"location":"utilities/","page":"Code Reference","title":"Code Reference","text":"init_grid(Lz, Nz)","category":"page"},{"location":"utilities/#picfun.init_grid-Tuple{Any, Any}","page":"Code Reference","title":"picfun.init_grid","text":"init_grid(Lz, Nz)\n\nConvenience constructor for a 1d grid.\n\n\n\n\n\n","category":"method"},{"location":"utilities/#particles","page":"Code Reference","title":"particles","text":"","category":"section"},{"location":"utilities/","page":"Code Reference","title":"Code Reference","text":"Utility functions related to the particles.","category":"page"},{"location":"utilities/","page":"Code Reference","title":"Code Reference","text":"particle\nx(::particle)\nv(::particle)\ncopy(::particle)\nfix_position!(ptl::particle, L)\n+(::particle, ::particle)\n-(::particle, ::particle)","category":"page"},{"location":"utilities/#picfun.particle","page":"Code Reference","title":"picfun.particle","text":"particle represents the phase-space coordinates of a particle.\n\nFields\n\npos: Position in space\nvel: Velocity\n\n\n\n\n\n","category":"type"},{"location":"utilities/#picfun.x-Tuple{particle}","page":"Code Reference","title":"picfun.x","text":"x(p::particle)\n\nAccesses the particle's position. This is wrapped in a function so that we can define the adjoint as\n\nbarx  ightarrow  mathrmparticleleft( barx 0 right).\n\n\n\n\n\n","category":"method"},{"location":"utilities/#picfun.v-Tuple{particle}","page":"Code Reference","title":"picfun.v","text":"v(p::particle)\n\nAccesses the particle's velocity. This is wrapped in a function so that we can define the adjoint\n\nbarv  ightarrow  mathrmparticleleft( 0 barv right).\n\n\n\n\n\n","category":"method"},{"location":"utilities/#Base.copy-Tuple{particle}","page":"Code Reference","title":"Base.copy","text":"copy(p::particle)\n\nReturns a new partilce instance with the same position and velocity as the original particle.\n\n\n\n\n\n","category":"method"},{"location":"utilities/#picfun.fix_position!-Tuple{particle, Any}","page":"Code Reference","title":"picfun.fix_position!","text":"fix_position!(p, L)\n\nMoves a particle's position back into the domain [0:L]\n\n\n\n\n\n","category":"method"},{"location":"utilities/#Base.:+-Tuple{particle, particle}","page":"Code Reference","title":"Base.:+","text":"+(a::particle, b::particle)\n\nAdd position and velocity for particle structs.\n\n\n\n\n\n","category":"method"},{"location":"utilities/#Base.:--Tuple{particle, particle}","page":"Code Reference","title":"Base.:-","text":"-(a::particle, b::particle)\n\nSubtract position and velocity for particle structs.\n\n\n\n\n\n","category":"method"},{"location":"utilities/#pic_utils","page":"Code Reference","title":"pic_utils","text":"","category":"section"},{"location":"utilities/","page":"Code Reference","title":"Code Reference","text":"Utility functions related to PIC algorithm","category":"page"},{"location":"utilities/","page":"Code Reference","title":"Code Reference","text":"b1(z, zp, Δz)\ndeposit(::Array{particle}, zgrid::grid_1d, fun::Function)","category":"page"},{"location":"utilities/#picfun.b1-Tuple{Any, Any, Any}","page":"Code Reference","title":"picfun.b1","text":"b1(z, zp, Δz)\n\nImplements a triangular interpolation. This function constructs x = (z - zp) / Δz from its arguments and calculates:\n\n      0     for       |x| ≥ 1\n\nb1(x) = { x+1   for   -1 < x < 0           1-x   for    0 ≤ x < 1\n\n\n\n\n\n","category":"method"},{"location":"utilities/#picfun.deposit-Tuple{Array{particle, N} where N, grid_1d, Function}","page":"Code Reference","title":"picfun.deposit","text":"deposit(ptl_vec::Array{particle}, zgrid::grid_1d, fun::Function)\n\nEvaluate the expression ∑{particle} f(p) * b1(zi, z_particle, Δz), where i indices the grid, for all grid points.\n\n\n\n\n\n","category":"method"},{"location":"utilities/#","page":"Code Reference","title":"","text":"","category":"section"},{"location":"#picfun.jl-Documentation","page":"Home","title":"picfun.jl Documentation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This repository implements the particle-in-cell (PIC) algorithm described by Chen et al.. The goal of this project is to study  how and where machine learning can be used in PIC codes to accelerate the simulation. Most parts of the code should be differentiable.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This code has been used for arXiv:2110.12444","category":"page"},{"location":"","page":"Home","title":"Home","text":"The documentation is still work in progress. Please check back later or contact the author.","category":"page"}]
}