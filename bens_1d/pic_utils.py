#-*- Coding: UTF-8 -*-

def weights_lr(ptl_z, grid_z, dz):
    """Calculate distance to left and right grid cell in units of dz

    Input:
    ======
    ptl_z, float: Z-coordinate of the particle
    grid_z, float: Z-coordinate of the left-most grid cell
    dz, float: spacing between left and right grid cell


    Returns:
    ========
    w_l: Weight w.r.t. left grid cell
    w_r:  Weight w.r.t. right grid cell
    """

    wr = (ptl_z - grid_z) / dz
    wl = 1.0 - wr

    return(wl, wr)


def init_velocity(num_ptl):
    """Initializes the particle velocity using the Halton series
    
    Input:
    ======
    num_ptl - int: the number of particles

    Returns:
    ========
    ptl_z: ndarray (float) - the particle velocities
    """
    import numpy as np
    from scipy.stats import norm
    from ghalton import Halton

    Sequencer = Halton(1)
    ptl_v = np.squeeze(np.array(Sequencer.get(num_ptl)))
    np.random.shuffle(ptl_v)
    # norm.ppf does the same as norminv, 
    # See https://stackoverflow.com/questions/29369776/what-is-the-sci-currmpython-equivalent-to-matlabs-norminv-normal-inverse-cumu
    ptl_v = norm.ppf(np.array(ptl_v))

    return(ptl_v)

# End of file pic_utils.py