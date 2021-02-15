# PICfun

This repository implements the PIC algorithm described by Chen et al. http://dx.doi.org/10.1016/j.jcp.2011.05.031


## PIC equations

The code solves the equations

![equation](https://latex.codecogs.com/gif.latex?\frac{x_\mathrm{p}^{n&plus;1}&space;-&space;x_\mathrm{p}^{n}}{\triangle&space;t}&space;=&space;v_\mathrm{p}^{n&plus;1/2})
![equation](https://latex.codecogs.com/gif.latex?\frac{v_\mathrm{p}^{n&plus;1}&space;-&space;v_\mathrm{p}^{n}}{\triangle&space;t}&space;=&space;\frac{q_\mathrm{p}}{m_\mathrm{p}}\mathrm{SM}\left[&space;E^{n&plus;1/2}&space;\right&space;]\left(x_\mathrm{p}^{n&plus;1/2}&space;\right&space;)

on a 1-dimensional grid. The new electric field is determined by
 
![equation](https://latex.codecogs.com/gif.latex?\epsilon_0&space;\frac{E^{n&plus;1}_{i}&space;-&space;E^{n}_{i}}{\triangle&space;t}&space;&plus;&space;\mathrm{SM}&space;\left[&space;\bar{j}_{i}^{n&plus;1/2}&space;\right]&space;=&space;\langle&space;\bar{j}&space;\rangle^{n&plus;1/2})

. Here $SM[Q] = (Q_{i-1} + 2Q_i + Q_{i+1}) / 4$ is a binomial smoothing operator and angular brackets denote spatial averaging.
