import numpy as np
import scipy as sp
from scipy import integrate, special

from .function_fast import function_1d, function_2d

## One-dimensional functions
def linear(x, *p):
    """[summary]
    
    Arguments:
        x {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    return p[0] * x + p[1]

def quad_1d(x, *p):
    """[summary]
    
    Arguments:
        x {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    A, x0, C = p
    xc = x - x0

    return A * xc**2 + C

def gaussian_1d(x, *p):
    """[summary]
    
    Arguments:
        x {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """

    return function_1d.gaussian(x, *p)

def ThomasFermi_1d(x, *p):
    """[summary]
    
    Arguments:
        x {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """    
    A, x0, r0, C = p
    xc = x - x0
    
    y = 1 - (xc / r0)**2
    y[y<0] = 0

    return A * y + C

def bimodal_1d(x, *p):
    """[summary]
    
    Arguments:
        x {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """    
    ATF, xTF, rTF, Ag, x0, sigma, C = p
    
    # Thomas Fermi distribution
    pTF = [ATF, xTF, rTF, 0]
    yTF = ThomasFermi_1d(x, *pTF)
    
    # Gaussian distribution
    pg = [Ag, x0, sigma, 0]
    yg = gaussian_1d(x, *pg)

    return yTF + yg + C

def double_gauss_1d(x, *p):
    """[summary]
    
    Arguments:
        x {[type]} -- [description]
    # Parameters
    # 0: amplitude - 1, 1: center - 1, 2: 1/e width - 1,
    # 3: amplitude - 2, 4: center - 2, 5: 1/e width - 2,
    # 6: offset
    
    Returns:
        [type] -- [description]
    """
    A1, x01, sigma1, A2, x02, sigma2, C = p

    # Gaussian - 1
    xc1 = x - x01
    y1 = A1 * np.exp(-xc1**2 / sigma1**2)
    
    # Gaussian - 2
    xc2 = x - x02
    y2 = A2 * np.exp(-xc2**2 / sigma2**2)
    
    return y1 + y2 + C

def lorentzian_1d(x, *p):
    """[summary]
    
    Arguments:
        x {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """    
    A, x0, sigma, C = p
    
    x_shift = x - x0

    return A * sigma**2 / (x_shift**2 + sigma**2) + C
    
def psf_1d(x, *p):
    """[summary]
    
    Arguments:
        x {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """    
    A, x0, alpha, C = p

    r = (x - x0) * alpha
    y = np.zeros(r.shape)

    y[r!=0] = A * (2 * special.j1(r[r!=0]) / r[r!=0])**2
    y[r==0] = A

    return y + C

## Two-dimensional functions
def gaussian_2d(xy_mesh, *p):
    """[summary]
    
    Arguments:
        xy_mesh {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """    
    # unpack 1D list into 2D x and y coords
    (x, y) = xy_mesh

    return function_2d.gaussian(x, y, *p)

def gaussian_2d_tmp(xy_mesh, *p):
    """[summary]
    
    Arguments:
        xy_mesh {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """    
    # unpack 1D list into 2D x and y coords
    (x, y) = xy_mesh

    # # Parameters
    A, x0, sigmax, y0, sigmay, C = p

    xc = x - x0
    yc = y - y0

    return A * np.exp(-xc**2/sigmax**2 - yc**2/sigmay**2) + C

def gaussian_2dr(xy_mesh, *p):
    """[summary]
    
    Arguments:
        x {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """    
    # unpack 1D list into 2D x and y coords
    (x, y) = xy_mesh

    # Parameters
    A, x0, sigmax, y0, sigmay, theta, C = p

    xc = x - x0
    yc = y - y0

    xr = xc * np.cos(theta) - yc * np.sin(theta)
    yr = xc * np.sin(theta) + yc * np.cos(theta)

    return A * np.exp(-xr**2/sigmax**2 - yr**2/sigmay**2) + C

def psf_2d(xy_mesh, *p):
    """[summary]
    
    Arguments:
        xy_mesh {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """    
    # unpack 1D list into 2D x and y coords
    (x, y) = xy_mesh

    # Parameters
    A, x0, y0, alpha, C = p

    xc = x - x0
    yc = y - y0

    r = np.sqrt(xc**2 + yc**2) * alpha

    z = np.zeros(r.shape)
    
    z[r!=0] = A * (2 * special.j1(r[r!=0]) / r[r!=0])**2
    z[r==0] = A

    return z + C

def psf_2d_tmp(xy_mesh, *p):
    """[summary]
    
    Arguments:
        xy_mesh {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """    
    # unpack 1D list into 2D x and y coords
    (x, y) = xy_mesh

    z = function_2d.psf(x, y, *p)
    return z.reshape(x.shape)

def psf_2d_int(xy_mesh, *p):
    """[summary]
    
    Arguments:
        xy_mesh {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """    
    # unpack 1D list into 2D x and y coords
    (x, y) = xy_mesh

    A, x0, y0, alpha, C = p
    xc = x - x0
    yc = y - y0
    
    s_eff = x[1, 0] - x[0, 0]
    z = np.zeros(x0.shape)
    z_err = np.zeros(x0.shape)

    def f(x, y):
        if x == 0:
            val = 1
        else:
            r = alpha * np.sqrt(xc**2 + yc**2)
            val = (2 * special.j1(r) / r)**2
            
        return alpha**2 / (4 * np.pi) * val

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i, j], z_err[i, j] = integrate.dblquad(f, xc[i, j], xc[i, j] - s_eff,
                                                                            lambda x: yc[i, j] + s_eff, lambda x: yc[i, j])

    z = C + A * z
        
    return z

def psf_2d_gaussian(xy_mesh, *p):
    """[summary]
    
    Arguments:
        xy_mesh {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """    
    # unpack 1D list into 2D x and y coords
    (x, y) = xy_mesh

    # # Parameters
    # A, x0, y0, sigma, C = p

    # r2 = (x - x0)**2 + (y - y0)**2

    # return A * np.exp(-r2/sigma**2) + C

    return function_2d.gaussian_iso(x, y, *p)
