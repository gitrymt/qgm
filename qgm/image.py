import numpy as np
import pandas as pd

from numba import jit, uint16, float64, prange

from PIL import Image
import cv2

import joblib

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.collections as mc

from . import function
from .parameter import system, psf, lattice
from .function_fast import function_1d, function_2d

class image():
    def __init__(self, image_path=''):
        self.system = system()
        
        if image_path is not '':
            try:
                self.image_path = image_path
                # print(self.image_path)
                self.load_image(self.image_path, remove_offset=False)
            except TypeError:
                self.image_path = None
                self.image = np.array([])
                self.image_loaded = False
        
        self.image_ROI = np.array([])
        
        self.bg_image = np.array([])
        self.bg_image_loaded = False
        self.offset_removed = False

        self.xymesh = None
        self.xymesh_ROI = None

        self.psfm = None
        self.ROI = {'Position (px)': pd.DataFrame([], columns=['x0 (px)', 'x1 (px)', 'y0 (px)', 'y1 (px)']),
                    'Position (um)': pd.DataFrame([], columns=['x0 (um)', 'x1 (um)', 'y0 (um)', 'y1 (um)']),
                    }
        
    def load_image(self, path, remove_offset=False):
        self.image = np.array(Image.open(path), dtype=float)
        self.offset_removed = False
        self.image_loaded = True

        if remove_offset:
            self.remove_offset()
        
    def load_bg_image(self, path, remove_offset=True):
        self.bg_image = joblib.load(path)
        
        if remove_offset:
            self.remove_offset()
            self.offset_removed = True

    def save_bg_image(self, path, compress_level=3):
        if self.bg_image_loaded:
            joblib.dump(self.bg_image, path, compress=compress_level)
        else:
            print('Error: Background image is not prepared.')

    def generate_bg_image(self, paths, remove_offset=True):
        if type(paths) == str:
            self.bg_image = np.array(Image.open(paths), dtype=float)
        else:
            for n, path in enumerate(paths):
                if n > 0:
                    self.bg_image += np.array(Image.open(path), dtype=float)
                else:
                    self.bg_image = np.array(Image.open(path), dtype=float)

            self.bg_image = self.bg_image / len(paths)

        if remove_offset:
            self.remove_offset()
            self.offset_removed = True
        
        self.bg_image_loaded = True

    def remove_offset(self):
        if not self.offset_removed:
            self.image -= self.bg_image
            self.offset_removed = True

    def generate_xymesh(self, x0=None, y0=None, unit='um'):
        if x0 is None:
            x0 = self.system.info['Cloud center (px)'][0]
        if y0 is None:
            y0 = self.system.info['Cloud center (px)'][1]

        if unit == 'um':
            spx = self.system.info['Effective Pixel size (um/px)']
            lx = (np.arange(0, self.image.shape[1]) - x0) * spx
            ly = (np.arange(0, self.image.shape[0]) - y0) * spx
        elif unit == 'px':
            lx = (np.arange(0, self.image.shape[1]) - x0)
            ly = (np.arange(0, self.image.shape[0]) - y0)
        
        self.xymesh = np.meshgrid(lx, ly)

        return self.xymesh

    def generate_lattice_sites(self, Nsite=2*10+1, rxlim=10, rylim=10, unit='um', limits='circle', origin='best'):
        n = [[i - (Nsite-1)/2] for i in range(Nsite)]
        n1, n2 = np.meshgrid(n, n)
        
        info_L1 = self.system.lattice['Lattice 1'].info
        info_L2 = self.system.lattice['Lattice 2'].info

        if unit == 'um':
            a1 = info_L1['Constant (um)']
            a2 = info_L2['Constant (um)']
            
            if origin == 'best':
                id_orig = np.argmax(self.system.lattice['Origins']['Goodness'])

                x0 = self.system.lattice['Origins']['X Center (um)'][id_orig]
                y0 = self.system.lattice['Origins']['Y Center (um)'][id_orig]
            else:
                if type(origin) == int:
                    x0 = self.system.lattice['Origins']['X Center (um)'][origin]
                    y0 = self.system.lattice['Origins']['Y Center (um)'][origin]
            
        elif unit == 'px':
            a1 = info_L1['Constant (px)']
            a2 = info_L2['Constant (px)']

            if origin == 'best':
                id_orig = np.argmax(self.system.lattice['Origins']['Goodness'])

                x0 = self.system.lattice['Origins']['X Center (px)'][id_orig]
                y0 = self.system.lattice['Origins']['Y Center (px)'][id_orig]
            else:
                if type(origin) == int:
                    x0 = self.system.lattice['Origins']['X Center (px)'][origin]
                    y0 = self.system.lattice['Origins']['Y Center (px)'][origin]

        # print(x0, y0)

        angle1 = info_L1['Angle (radian)']
        angle2 = info_L2['Angle (radian)']

        a1x = a1 * np.cos(angle1)
        a1y = a1 * np.sin(angle1)
        a2x = a2 * np.cos(angle2)
        a2y = a2 * np.sin(angle2)

        det = (a1x * a2y - a2x * a1y)
        n01 = (a2y * x0 - a2x * y0) / det
        n02 = (-a1y * x0 + a1x * y0) / det
        
        x0 = x0 - (n01 * a1x + n02 * a2x)
        y0 = y0 - (n01 * a1y + n02 * a2y)

        xs_all = n1 * a1 * np.cos(angle1) + n2 * a2 * np.cos(angle2) + x0
        ys_all = n1 * a1 * np.sin(angle1) + n2 * a2 * np.sin(angle2) + y0

        if limits is 'circle':
            xs = xs_all[(xs_all/rxlim)**2 + (ys_all/rylim)**2 < 1]
            ys = ys_all[(xs_all/rxlim)**2 + (ys_all/rylim)**2 < 1]
        elif limits is 'square':
            xs = xs_all[np.abs(xs_all) <= rxlim]
            ys = ys_all[np.abs(xs_all) <= rxlim]
            xs = xs[np.abs(ys) <= rylim]
            ys = ys[np.abs(ys) <= rylim]
        else:
            xs = xs_all
            ys = ys_all
        
        self.system.lattice['Lattice sites'] = pd.DataFrame([], columns=['X Center', 'Y Center', 'Amplitude'])
        self.system.lattice['Lattice sites']['X Center'] = xs
        self.system.lattice['Lattice sites']['Y Center'] = ys

        return xs, ys
    
    def generate_psfm(self, psf_model=None):
        x0s = np.array(self.system.lattice['Lattice sites']['X Center'], dtype=float)
        y0s = np.array(self.system.lattice['Lattice sites']['Y Center'], dtype=float)
        psfm_size = np.array((self.xymesh_ROI[0].shape[0], self.xymesh_ROI[0].shape[1], x0s.size), dtype=np.uint16)
        
        if psf_model == None:
            psf_model = self.system.psf.info['Model']

        if psf_model == 'psf':
            self.psfm = np.zeros([self.image_ROI.shape[0], self.image_ROI.shape[1], len(self.system.lattice['Lattice sites'])])

            alpha0 = 2 * np.pi / self.system.psf.info['Wavelength (um)'] * self.system.psf.info['Effective NA']
            
            # for ind_item, row in self.system.lattice['Lattice sites'].iterrows():
            for ind_item, (x0, y0) in enumerate(zip(x0s, y0s)):
                # x0 = row['X Center']
                # y0 = row['Y Center']

                p_psf = np.array([1, x0, y0, alpha0, 0])
                self.psfm[:, :, ind_item] = function.psf_2d(self.xymesh_ROI, *p_psf)

        elif psf_model == 'gaussian':
            sigmax = self.system.psf.info['HWHM width - x (um)'] / np.sqrt(np.log(2))
            sigmay = self.system.psf.info['HWHM width - y (um)'] / np.sqrt(np.log(2))
            
            self.psfm = generate_gauss_psfm(psfm_size, self.xymesh_ROI[0], self.xymesh_ROI[1],
                                            x0s, y0s, np.array([sigmax, sigmay], dtype=np.float64))

            # self.psfm = np.zeros([self.image_ROI.shape[0], self.image_ROI.shape[1], len(self.system.lattice['Lattice sites'])])
            
            # for ind_item, (x0, y0) in enumerate(zip(x0s, y0s)):
            #     # x0 = row['X Center']
            #     # y0 = row['Y Center']

            #     p_gaussian = np.array([1, x0, sigmax, y0, sigmay, 0])
            #     self.psfm[:, :, ind_item] = function.gaussian_2d(self.xymesh_ROI, *p_gaussian)

    def set_ROI(self, ROI=None, unit='um'):
        if ROI == None:
            ROI_px = {'x0 (px)': [0], 'x1 (px)': [self.image.shape[1]-1], 'y0 (px)': [0], 'y1 (px)': [self.image.shape[0]-1]}
            ROI_um = {'x0 (um)': [np.min(self.xymesh[0])], 'x1 (um)': [np.max(self.xymesh[0])],
                      'y0 (um)': [np.min(self.xymesh[1])], 'y1 (um)': [np.max(self.xymesh[1])]}

            self.ROI['Position (px)'] = pd.DataFrame(ROI_px, index=['ROI 1'])
            self.ROI['Position (um)'] = pd.DataFrame(ROI_um, index=['ROI 1'])

            # print(ROI_px)
            # self.ROI['Position (um)'] = pd.DataFrame([self.xymesh[0][0], self.xymesh[0][-1], self.xymesh[1][0], self.xymesh[1][-1]],
            #                                          columns=['x0 (um)', 'x1 (um)', 'y0 (um)', 'y1 (um)'])

            self.image_ROI = self.image
            self.xymesh_ROI = self.xymesh
        else:
            (x0, x1, y0, y1) = ROI
            (xx, yy) = self.xymesh

            id_x0 = np.argmin(np.abs(xx-x0))
            id_x1 = np.argmin(np.abs(xx-x1))
            id_y0 = np.argmin(np.abs(yy.T-y0))
            id_y1 = np.argmin(np.abs(yy.T-y1))

            ROI_px = {'x0 (px)': [id_x0], 'x1 (px)': [id_x1], 'y0 (px)': [id_y0], 'y1 (px)': [id_y1]}
            ROI_um = {'x0 (um)': [xx[0, id_x0]], 'x1 (um)': [xx[0, id_x1]],
                      'y0 (um)': [yy[id_y0, 0]], 'y1 (um)': [yy[id_y1, 0]]}

            self.ROI['Position (px)'] = pd.DataFrame(ROI_px, index=['ROI 1'])
            self.ROI['Position (um)'] = pd.DataFrame(ROI_um, index=['ROI 1'])

            self.image_ROI = self.image[id_y0:id_y1, id_x0:id_x1]
            self.xymesh_ROI = (xx[id_y0:id_y1, id_x0:id_x1], yy[id_y0:id_y1, id_x0:id_x1])

    def generate_lattice_separation(self, color='w', alpha=0.75, dx=0, dy=0):
        dpx = self.system.info['Effective Pixel size (um/px)']
        
        a1um = self.system.lattice['Lattice 1'].info['Constant (um)']
        a2um = self.system.lattice['Lattice 2'].info['Constant (um)']
        theta1 = self.system.lattice['Lattice 1'].info['Angle (radian)']
        theta2 = self.system.lattice['Lattice 2'].info['Angle (radian)']
        
        a1vec = np.array([a1um * np.cos(theta1), a1um * np.sin(theta1)])
        a2vec = np.array([a2um * np.cos(theta2), a2um * np.sin(theta2)])
        a3vec = a1vec - a2vec
        
        xs = np.array(self.system.lattice['Lattice sites']['X Center'])
        ys = np.array(self.system.lattice['Lattice sites']['Y Center'])
        
        lseps = []
        
        for k in range(xs.size):
            p0 = (xs[k]+dx, ys[k]+dy)

            lseps += [[p0 + (2*a2vec + a3vec)/3, p0 + (2*a1vec - a2vec)/3,
                    p0 + (2*a3vec - a1vec)/3, p0 + (-2*a2vec - a3vec)/3,
                    p0 + (-2*a1vec + a2vec)/3, p0 + (-2*a3vec + a1vec)/3,
                    p0 + (2*a2vec + a3vec)/3,
                    ]]
        
        sep_lines = mc.LineCollection(lseps, colors=color, alpha=alpha)

        return sep_lines

    def show_image(self):
        pass

@jit(float64[:,:,:](uint16[:], float64[:,:], float64[:,:], float64[:], float64[:], float64[:]))
def generate_gauss_psfm(size:np.ndarray, x: np.ndarray, y: np.ndarray,
                      x0s:np.ndarray, y0s:np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    psfm = np.zeros((size[0], size[1], size[2]), dtype=np.float64)
    
    for n in prange(size[2]):
        for k in range(size[0]):
            for l in range(size[1]):
                xs = x[k,l] - x0s[n]
                ys = y[k,l] - y0s[n]
                
                psfm[k,l,n] = np.exp(-xs**2/sigmas[0]**2-ys**2/sigmas[1]**2)
    
    return psfm

def generate_cmap(colors):
    """ Generate user defined colormap

    Parameters
    ----------
    colors : list
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    values = range(len(colors))
    vmax = np.ceil(np.max(values))
    color_list = []
    
    for v, c in zip(values, colors):
        color_list.append( ( v / vmax, c) )
    
    return LinearSegmentedColormap.from_list('custom', color_list)

def pixel_shift(image, dx, dy, interporation='Lanczos'):
    matrix = [[1, 0, dx], [0, 1, dy]]
    affine_matrix_fit = np.float32(matrix)

    image_shifted = cv2.warpAffine(image, affine_matrix_fit, image.shape,
                                   flags=cv2.INTER_LANCZOS4)
    
    return image_shifted

