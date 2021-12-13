import numpy as np
from scipy import integrate, special
from numba import jit

@jit(cache=True)
def gaussian(x, y, *p):
    """[summary]
    
    Arguments:
        xy_mesh {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    # Parameters
    A, x0, sigmax, y0, sigmay, C = p

    xc = x - x0
    yc = y - y0

    return A * np.exp(-xc**2/sigmax**2 - yc**2/sigmay**2) + C

@jit(cache=True)
def gaussian_iso(x, y, *p):
    """[summary]
    
    Arguments:
        xy_mesh {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    # Parameters
    A, x0, y0, sigma, C = p

    xc = x - x0
    yc = y - y0
    r2 = xc**2 + yc**2

    return A * np.exp(-r2/sigma**2) + C

# @jit(cache=True)
@jit
def psf(x, y, *p):
    """[summary]
    
    Arguments:
        xy_mesh {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """    
    # Parameters
    A, x0, y0, alpha, C = p

    xc = x - x0
    yc = y - y0

    r = np.sqrt(xc**2 + yc**2) * alpha
    z = np.zeros(r.shape)

    (size1, size2) = r.shape
    r = np.ravel(r)
    z = np.ravel(z)

    for n in range(size1*size2):
        if r[n] > 0:
            z[n] = A * (2 * special.j1(r[n]) / r[n])**2
            # continue
        else:
            z[n] = A
    # tmp = (r == 0)
    # z[tmp] = A

    # tmp = not tmp
    # z[tmp] = A * (2 * special.j1(r[r!=0]) / r[r!=0])**2

    # z = z.reshape([size1, size2])
    # return z.reshape([size1, size2]) + C
    return z + C