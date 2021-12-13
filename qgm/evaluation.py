import copy

import pandas as pd

import numpy as np
import scipy as sp

from numba import jit

import PIL
from PIL import Image

from . import image
from . import function

def psf_evaluation(image: image, coordinates):
    pass
    
    # return img_psf

def lattice_geometry_1d_define(Nsite=2*6+1):
    def lattice_geometory_1d(x, *p):
        """[summary]
        
        Arguments:
            x {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        A, sigmax, a1, x0, C = p
        
        n = np.array([[i - (Nsite-1)/2] for i in range(Nsite)])
        xs = n * a1 + x0
        
        y = np.zeros(x.shape)

        for n in range(xs.size):
            p_tmp = np.array([A, xs[n], sigmax, 0])
            y += function.gaussian_1d(x, *p_tmp)
        
        return y + C

    return lattice_geometory_1d

def lattice_geometory_1d_evaluation(xedges, Iy, range, step=0.1):
    pass

def lattice_geometry_2d_define(Nsite=2*6+1):
    def lattice_geometory_2d(xy_mesh, *p):
        """[summary]
        
        Arguments:
            xy_mesh {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """    
        # unpack 1D list into 2D x and y coords
        (x, y) = xy_mesh

        A, sigmax, sigmay, a1, a2, theta1, theta2, x0, y0, C = p
        
        n = [[i - (Nsite-1)/2] for i in range(Nsite)]
        n1, n2 = np.meshgrid(n, n)
        xs = n1 * a1 * np.cos(theta1) + n2 * a2 * np.cos(theta2) + x0
        ys = n1 * a1 * np.sin(theta1) + n2 * a2 * np.sin(theta2) + y0
        
        xs = np.ravel(xs)
        ys = np.ravel(ys)

        z = np.zeros(x.shape)

        for n in range(xs.size):
            p_tmp = np.array([A, xs[n], sigmax, ys[n], sigmay, 0])
            z += function.gaussian_2d(xy_mesh, *p_tmp)

        return z + C

    return lattice_geometory_2d

def lattice_geometry_2d_define_tmp(Nsite=2*6+1):
    def lattice_geometory_2d(xy_mesh, *p):
        """[summary]
        
        Arguments:
            xy_mesh {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """    
        # unpack 1D list into 2D x and y coords
        (x, y) = xy_mesh

        A, sigma, a1, a2, theta1, theta2 = p
        
        n = [[i - (Nsite-1)/2] for i in range(Nsite)]
        n1, n2 = np.meshgrid(n, n)
        xs = n1 * a1 * np.cos(theta1) + n2 * a2 * np.cos(theta2)
        ys = n1 * a1 * np.sin(theta1) + n2 * a2 * np.sin(theta2)
        
        xs = np.ravel(xs)
        ys = np.ravel(ys)

        z = np.zeros(x.shape)

        for n in range(xs.size):
            p_tmp = np.array([A, xs[n], sigma, ys[n], sigma, 0])
            z += function.gaussian_2d(xy_mesh, *p_tmp)

        return z

    return lattice_geometory_2d

def site_occupation_evaluation(image, threshold):
    factor_sites = np.matrix(np.copy(image.system.lattice['Lattice sites']['Amplitude'])).T
    factor_sites[factor_sites < threshold] = 0
    factor_sites[factor_sites >= threshold] = 1
    
    (im_width, im_height) = image.image_ROI.shape
    psfm = np.copy(image.psfm)

    PSFM_flat = np.matrix(np.reshape(psfm, [im_width * im_height, psfm.shape[2]]))
    img_sites = np.array(np.reshape(PSFM_flat * factor_sites, [im_width, im_height]))

    return img_sites

def fidelity_evaluation(image_1st, image_2nd, threshold):
    site1 = image_1st.system.lattice['Lattice sites']
    site2 = image_2nd.system.lattice['Lattice sites']

    x1s = np.array(site1['X Center'])
    y1s = np.array(site1['Y Center'])
    flag1s = np.array(site1['Amplitude'])
    flag1s[flag1s < threshold] = 0
    flag1s[flag1s >= threshold] = 1

    x2s_tmp = np.array(site2['X Center'])
    y2s_tmp = np.array(site2['Y Center'])
    flag2s_tmp = np.array(site2['Amplitude'])
    flag2s_tmp[flag2s_tmp < threshold] = 0
    flag2s_tmp[flag2s_tmp >= threshold] = 1

    x2s = []
    y2s = []
    flag2s = []

    diffs = []

    for n, (x1, y1, flag1) in enumerate(zip(x1s, y1s, flag1s)):
        dr = np.sqrt((x2s_tmp - x1)**2 + (y2s_tmp - y1)**2)
        
        id_site = np.argmin(dr)

        x2s += [x2s_tmp[id_site]]
        y2s += [y2s_tmp[id_site]]
        flag2s += [flag2s_tmp[id_site]]
        diffs += [flag1 - flag2s_tmp[id_site]]
    
    drs = np.sqrt((x1s-x2s)**2+(y1s-y2s)**2)

    a_lat = (image_1st.system.lattice['Lattice 1'].info['Constant (um)'] + image_1st.system.lattice['Lattice 2'].info['Constant (um)'])/2
    mask = drs < a_lat/3
    x1s = x1s[mask]; y1s = y1s[mask]
    x2s = np.array(x2s)[mask]
    y2s = np.array(y2s)[mask]
    flag1s = np.array(flag1s)[mask]
    flag2s = np.array(flag2s)[mask]
    diffs = np.array(diffs)
    diffs = diffs[mask]
    drs = drs[mask]

    dsite = pd.DataFrame({'x1': x1s, 'y1': y1s, 'x2': x2s, 'y2': y2s, 'dr': drs,
                          'flag1': flag1s, 'flag2': flag2s, 'difference': diffs})

    N1 = np.sum(flag1s)
    N2 = np.sum(flag2s)
    Nloss = N1 - N2
    Nhopping = np.sum(diffs<0)

    fidelity = pd.DataFrame({'N1': [N1], 'N2': [N2],
                             'Nloss': [Nloss],
                             'Nhopping': [Nhopping],
                             'Rloss': [Nloss/N1],
                             'Rhopping': [Nhopping/N1],
                             })
    
    return dsite, fidelity