# PICfun

This repository implements the PIC algorithm described by [Chen et al.](http://dx.doi.org/10.1016/j.jcp.2011.05.031)
With this project I'm testing how to accelerate iterative linear solvers in simple PIC algorithms with ML.

This code has been used for [arXiv:2110.12444](https://arxiv.org/abs/2110.12444)
## PIC equations

The code solves the equations

![equation](https://latex.codecogs.com/gif.latex?\frac{x_\mathrm{p}^{n&plus;1}&space;-&space;x_\mathrm{p}^{n}}{\triangle&space;t}&space;=&space;v_\mathrm{p}^{n&plus;1/2})

![equation](https://latex.codecogs.com/gif.latex?\frac{v_\mathrm{p}^{n&plus;1}&space;-&space;v_\mathrm{p}^{n}}{\triangle&space;t}&space;=&space;\frac{q_\mathrm{p}}{m_\mathrm{p}}&space;\mathrm{SM}&space;\left[&space;E^{n&plus;1/2}&space;\right]&space;\left(&space;x_\mathrm{p}^{n&plus;1/2}&space;\right))

on a 1-dimensional grid. The new electric field is determined by
 
![equation](https://latex.codecogs.com/gif.latex?\epsilon_0&space;\frac{E^{n&plus;1}_{i}&space;-&space;E^{n}_{i}}{\triangle&space;t}&space;&plus;&space;\mathrm{SM}&space;\left[&space;\bar{j}_{i}^{n&plus;1/2}&space;\right]&space;=&space;\langle&space;\bar{j}&space;\rangle^{n&plus;1/2})

. Here 
```math
SM[Q] = (Q_{i-1} + 2Q_i + Q_{i+1}) / 4
```
is a binomial smoothing operator and angular brackets denote spatial averaging.


A good point to start the simulation is by running
```
$ julia implicit_gmres.jl
```

