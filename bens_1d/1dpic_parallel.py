                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     #-*- Coding: UTF-8 -*-
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from pic_utils import weights_lr, init_velocity


# Total timestep delta_t = num_subcycles * dt
num_subcycles = 80
dt = 0.00125
delta_t = dt * num_subcycles

# Initialize mesh #Number of grid elements
Nz = 16
# Define spacing between grid elements
#dz = np.ones(Nz) / Nz
#dz = np.array([0.01, 0.015, 0.022, 0.03, 0.04, 0.052, 0.07, 0.09, 0.15, 0.22, 0.301])
#Nz = dz.shape[0]

# Get a nice and irregular grid
#Nz = 200
dz = np.random.uniform(1e-5, 1e0, Nz)
# Round dz to next one-thousands and sort
dz = np.floor(dz * 1e6) * 1e-6
dz.sort()

# Define grid points
zarr = np.concatenate([np.zeros(1), np.cumsum(dz)])
# Define grid length
Lz = zarr[-1]

# Particles per cell
particles_per_cell = 1024
# Total number of particles
num_ptl = Nz * particles_per_cell

# Below we initialize the variables pertaining to each particle
#1.) z-position of the particlese
ptl_z = np.linspace(0, Lz * (1. - 1. / num_ptl), num_ptl)
#2.) Velocity of the particles. Randomly initialized using Halton set
ptl_v = init_velocity(num_ptl)
#3.) Particle weights
ptl_w = np.zeros(num_ptl)
#4.) Index of the left grid vertex
ptl_vtx = np.searchsorted(zarr, ptl_z, side='right') - 1
#5.) Z-coordinate of the grid vertex to the left of the particle
ptl_zvtx = zarr[ptl_vtx]
#6.) distance from the grid vertex to the left to grid vertex to the right of the particle
ptl_dz = dz[ptl_vtx]

# Sanity check for particles:
# All particles should be to the right of their grid vertex
assert(np.all(ptl_z >= ptl_zvtx))

# Store initial particle states
ptl_z0 = np.copy(ptl_z)
ptl_v0 = np.copy(ptl_v)
ptl_w0 = np.copy(ptl_w)
ptl_vtx0 = np.copy(ptl_vtx)
ptl_zvtx0 = np.copy(ptl_zvtx)
ptl_dz0 = np.copy(ptl_dz)

# Electric field
E_field = np.zeros(Nz + 1, dtype=np.float)
# Particle density
dens = np.zeros(Nz + 1, dtype=np.float)
# Current density j = q * n * u, considering only electrons and q = 1
curr = np.zeros(Nz + 1, dtype=np.float)

kappa_n = np.zeros([Nz, Nz])
kappa_j = np.zeros([Nz, Nz])

for fld in range(Nz):
    print(fld)
    # Restore particles from stored state
    ptl_z[:] = ptl_z0[:]
    ptl_v[:] = ptl_v0[:]
    ptl_w[:] = ptl_w0[:]
    ptl_vtx[:] = ptl_vtx0[:]
    ptl_zvtx[:] = ptl_zvtx0[:]
    ptl_dz[:] = ptl_dz0[:]

    E_field[:] = 0.0
    E_field[fld] = 0.1
    E_field[-1] = E_field[0]

    # Set particle and flux density to zero
    dens[:] = 0.0
    curr[:] = 0.0

    # Deposit the time step zero current onto the grid
    ptl_wr = (ptl_z - ptl_zvtx) / ptl_dz
    ptl_wl = 1.0 - ptl_wr
    # Use ptl_vtx to index the current array since 0 < ptl_el < 32 and shape(curr) = (33,)
    np.add.at(curr, ptl_vtx, ptl_wl * ptl_w * ptl_v / ptl_dz)
    np.add.at(curr, ptl_vtx + 1, ptl_wr * ptl_w * ptl_v / ptl_dz)

    for p in range(num_ptl):
        # Get the element of the current particle
        j = int(ptl_vtx[p])
        wl, wr = weights_lr(ptl_z[p], zarr[j], dz[j])
        curr[j] = curr[j] + wl / dz[j] * ptl_w[p] * ptl_v[p]
        curr[j + 1] = curr[j + 1] + wr / dz[j] * ptl_w[p] * ptl_v[p]

    # Account for periodic boundary conditions for current deposit
    curr0 = curr[0]
    curr[0] = curr[0] + curr[-1]
    curr[-1] = curr[-1] + curr0
    curr = curr / num_ptl

    currcum = curr

    # Time integration subcycling
    for k in range(num_subcycles):        
        # Interpolate electric field at particle position
        ptl_wr = (ptl_z - ptl_zvtx) / ptl_dz
        ptl_wl = 1. - ptl_wr
        E_ip = ptl_wl * E_field[ptl_vtx] + ptl_wr * E_field[ptl_vtx + 1]
        # Update particle weights
        ptl_w = ptl_w + dt * ptl_v * E_ip

        # Update particle position
        ptl_z = ptl_z + dt * ptl_v
        # Apply periodic boundary conditions to particle position
        ptl_z = np.mod(ptl_z, Lz * np.ones_like(ptl_z))

        # sort particle vectors
        ptl_sort_idx = np.argsort(ptl_z)
        ptl_z = ptl_z[ptl_sort_idx]
        ptl_v = ptl_v[ptl_sort_idx]
        ptl_w = ptl_w[ptl_sort_idx]
        
        # Update particle vertices, zgrid values and dz values
        ptl_vtx = np.searchsorted(zarr, ptl_z, side='right') - 1
        ptl_zvtx = zarr[ptl_vtx]
        ptl_dz = dz[ptl_vtx]

        ## Deposit current at the grid points
        ptl_wr = (ptl_z - ptl_zvtx) / ptl_dz
        ptl_wl = 1.0 - ptl_wr
        #np.add.at is used to accumulate for each index specified in ptl_vtx
        np.add.at(curr, ptl_vtx, ptl_wl * ptl_w * ptl_v / ptl_dz)
        np.add.at(curr, ptl_vtx + 1, ptl_wr * ptl_w * ptl_v / ptl_dz)

        # Account for periodic boundary conditions for current deposit
        curr0 = curr[0]
        curr[0] = curr[0] + curr[-1]
        curr[-1] = curr[-1] + curr0
        curr = curr / num_ptl

        currcum = currcum + curr


    plt.figure() 
    plt.title("fld = " + str(fld))
    #plt.subplot(111)
    plt.plot(ptl_z0, ptl_w0, '.')
    plt.plot(ptl_z, ptl_w, '.')
    plt.xlabel("z")
    plt.ylabel("w")

    # current is the average over all sub-cycle time steps
    currcum = currcum / num_subcycles

    # deposit density
    ptl_wr = (ptl_z - ptl_zvtx) / ptl_dz
    ptl_wl = 1. - ptl_wr
    #np.add.at is used to accumulate for each index specified in ptl_vtx
    np.add.at(dens, ptl_vtx, ptl_wl * ptl_w / ptl_dz)
    np.add.at(dens, ptl_vtx + 1, ptl_wr * ptl_w / ptl_dz)

    # Account for periodic boundary conditions for density deposit
    den0 = dens[0]
    dens[0] = dens[0] + dens[-1]
    dens[-1] = dens[-1] + den0
    dens = dens / num_ptl

    kappa_n[fld, :] = dens[:-1]
    kappa_j[fld, :] = currcum[:-1]  


nrg = np.arange(Nz)

plt.figure()
plt.subplot(121)
for j in range(16): 
    plt.plot(nrg, np.hstack([kappa_j[j, j:], kappa_j[j, :j]]), '.-') 

plt.xlabel("N")
plt.ylabel("kappa_j")

plt.subplot(122)
for j in range(16): 
    plt.plot(nrg, np.hstack([kappa_n[j, j:], kappa_n[j, :j]]), '.-') 

plt.xlabel("N")
plt.ylabel("kappa_n")

xx, zz = np.meshgrid(nrg, dz)
fig_den = plt.figure()
ax_den = fig_den.add_subplot(111, projection='3d')
ax_den.plot_surface(xx, zz, kappa_n, cmap=plt.get_cmap("PuOr"))
ax_den.set_title("Density")

fig_curr = plt.figure()
ax_curr = fig_curr.add_subplot(111, projection='3d')
ax_curr.plot_surface(xx, zz, kappa_j, cmap=plt.get_cmap("RdYlBu"))
ax_curr.set_title("Flux")

plt.show()
