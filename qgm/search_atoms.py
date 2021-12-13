import copy

import numpy as np
import scipy as sp
from scipy.signal import find_peaks

import pandas as pd

import cv2
from PIL import Image

from . import image, function, fitting
from . import filter

def select_interporation(interporation, library):
    if library == 'OpenCV':
        if interporation == 'Lanczos':
            interporation_method = Image.LANCZOS
        elif interporation == 'Nearest':
            interporation_method = Image.NEAREST
        elif interporation == 'Box':
            interporation_method = Image.BOX
        elif interporation == 'Bilinear':
            interporation_method = Image.BILINEAR
        elif interporation == 'Bicubic':
            interporation_method = cv2.INTER_CUBIC
        else:
            interporation_method = cv2.INTER_LANCZOS4
    elif library == 'Pillow':
        if interporation == 'Lanczos':
            interporation_method = Image.LANCZOS
        elif interporation == 'Nearest':
            interporation_method = Image.NEAREST
        elif interporation == 'Box':
            interporation_method = Image.BOX
        elif interporation == 'Bilinear':
            interporation_method = Image.BILINEAR
        elif interporation == 'Hamming':
            interporation_method = Image.HAMMING
        elif interporation == 'Bicubic':
            interporation_method = Image.BICUBIC
        else:
            interporation_method = Image.LANCZOS
    else:
        pass
    
    return interporation_method

def search_atoms(img, filter_factor=0.65,
                 psf_model='gaussian', psf_size=2*15+1,
                 fine_search=True, magnification_fine=11,
                 scaling=False):
    from astropy.stats import sigma_clipped_stats
    from astropy.table import Table

    # from photutils.datasets import make_100gaussians_image
    from photutils import find_peaks

    # from astropy.visualization import simple_norm
    # from astropy.visualization.mpl_normalize import ImageNormalize
    # from photutils import CircularAperture

    # from photutils.datasets import make_4gaussians_image
    # from photutils import centroid_com, centroid_1dg, centroid_2dg
    
    # from time import time

    # t1_lowpass = time()

    img_lowpass = copy.deepcopy(img)
    img_lowpass.image, img_fft, (fx, fy), (FX, FY) = filter.lowpass(img, factor=filter_factor)

    # t2_lowpass = time()
    # print('Elapsed time (filter): %f ms' % ((t2_lowpass - t1_lowpass)*1e3))

    # t1_find = time()

    img_th = copy.copy(img_lowpass.image)
    mean, median, std = sigma_clipped_stats(img_th, sigma=3.0)
    threshold = median + (5. * std)
    img_th[img_th < threshold] = 0
    tbl = find_peaks(img_th, threshold, box_size=psf_size)

    positions = np.transpose((tbl['x_peak'], tbl['y_peak']))
    # apertures = CircularAperture(positions, r=5.)

    # t2_find = time()
    # print('Elapsed time (peak search): %f ms' % ((t2_find - t1_find)*1e3))

    # t1_fit = time()

    system_info = img.system.info
    psf_info = img.system.psf.info

    if psf_model == 'psf':
        fit_func = function.psf_2d
        sigma_psf = 2 * np.pi / (psf_info['Wavelength (um)'] / system_info['Effective Pixel size (um/px)']) * psf_info['Effective NA']
    elif psf_model == 'gaussian':
        fit_func = function.psf_2d_gaussian
        sigma_psf = psf_info['HWHM width - iso (um)'] / system_info['Effective Pixel size (um/px)']
    else:
        print('Keyword error')


    dL = int((psf_size-1)/2)
    L = np.arange(-dL, dL+1, 1, dtype=float)
    interporation_method = Image.LANCZOS

    XX_sub, YY_sub = np.meshgrid(L, L)

    # psf_mean1 = np.zeros(XX_sub.shape)

    x0s = []
    y0s = []
    goodnesses = []
    psfs = []

    A0s = []
    sigma0s = []
    NAeffs = []

    n = 1
    count = 1

    if fine_search:
        size_fine = psf_size * magnification_fine
        
        XX_sub_fine = np.array(Image.fromarray(XX_sub).resize((size_fine, size_fine), resample=interporation_method))
        YY_sub_fine = np.array(Image.fromarray(YY_sub).resize((size_fine, size_fine), resample=interporation_method))

        psf_mean = np.zeros(XX_sub_fine.shape)

        for x0, y0 in (positions):
            img_sub = img_lowpass.image[y0-dL:y0+dL+1, x0-dL:x0+dL+1]
        #     img_sub = img.image[x0-15:x0+16, y0-15:y0+16]
            
            if img_sub.shape  == (psf_size, psf_size):
                p_ini = [np.max(img_sub) - np.min(img_sub), x0, y0, sigma_psf, np.min(img_sub)]
                XYmesh_sub = (XX_sub + x0, YY_sub + y0)
                
                p_fit, p_err, fit_goodness = fitting.fit_2d(fit_func, img_sub, XYmesh_sub, p_ini)
                
                if fit_goodness > 0.9:
                    if scaling:
                        img_pil = Image.fromarray(img_sub / magnification_fine**2)
                    else:
                        img_pil = Image.fromarray(img_sub)
                    img_psf_resize = img_pil.resize((size_fine, size_fine), resample=Image.LANCZOS)
                
                    XYmesh_sub_fine = (XX_sub_fine, YY_sub_fine)
                    
                    p_ini = [np.max(img_sub) - np.min(img_sub), 0, 0, sigma_psf, np.min(img_sub)]
                    p_fit, p_err, fit_goodness = fitting.fit_2d(fit_func, np.array(img_psf_resize), XYmesh_sub_fine, p_ini)
                    NAeff = p_fit[3] * (psf_info['Wavelength (um)'] / system_info['Effective Pixel size (um/px)']) / (2 * np.pi)

                    # print(count, n, fit_goodness, NAeff)
                    
                    x0s += [x0 + p_fit[1]]
                    y0s += [y0 + p_fit[2]]
                    goodnesses += [fit_goodness]

                    A0s += [p_fit[0]]
                    sigma0s += [p_fit[3]]
                    NAeffs += [NAeff]

                    img_sub = img.image[y0-dL:y0+dL+1, x0-dL:x0+dL+1]
                    if scaling:
                        img_pil = Image.fromarray(img_sub / magnification_fine**2)
                    else:
                        img_pil = Image.fromarray(img_sub)

                    img_psf_resize = img_pil.resize((size_fine, size_fine), resample=Image.LANCZOS)
                    
                    img_shift = image.pixel_shift(np.array(img_psf_resize), -p_fit[1], -p_fit[2])
                    psf_mean += img_shift
                    psfs += [img_shift]
                    # psf_mean += image.pixel_shift(np.array(img_psf_resize), -p_fit[1], -p_fit[2])
                    # psfs += [image.pixel_shift(np.array(img_psf_resize), -p_fit[1], -p_fit[2])]
                    
                    count += 1

            n += 1

        psf_mean = psf_mean / (count - 1)
    else:
        psf_mean = np.zeros(XX_sub.shape)

        for x0, y0 in (positions):
            img_sub = img_lowpass.image[y0-dL:y0+dL+1, x0-dL:x0+dL+1]
        #     img_sub = img.image[x0-15:x0+16, y0-15:y0+16]
            
            if img_sub.shape  == (psf_size, psf_size):
                p_ini = [np.max(img_sub) - np.min(img_sub), x0, y0, sigma_psf, np.min(img_sub)]
                XYmesh_sub = (XX_sub + x0, YY_sub + y0)
                
                p_fit, p_err, fit_goodness = fitting.fit_2d(fit_func, img_sub, XYmesh_sub, p_ini)
                
                if fit_goodness > 0.9:            
                    p_ini = [np.max(img_sub) - np.min(img_sub), 0, 0, sigma_psf, np.min(img_sub)]
                    p_fit, p_err, fit_goodness = fitting.fit_2d(fit_func, np.array(img_sub), XYmesh_sub, p_ini)
                    NAeff = p_fit[3] * (psf_info['Wavelength (um)'] / system_info['Effective Pixel size (um/px)']) / (2 * np.pi)

                    # print(count, n, fit_goodness, NAeff)
                    
                    x0s += [x0 + p_fit[1]]
                    y0s += [y0 + p_fit[2]]
                    # x0s += [x0 - p_fit[1]]
                    # y0s += [y0 - p_fit[2]]
                    goodnesses += [fit_goodness]

                    A0s += [p_fit[0]]
                    sigma0s += [p_fit[3]]
                    NAeffs += [NAeff]

                    img_sub = img.image[y0-dL:y0+dL+1, x0-dL:x0+dL+1]
                    psf_mean += image.pixel_shift(img_sub, -p_fit[1], -p_fit[2])
                    
                    count += 1

            n += 1

        psf_mean = psf_mean / (count - 1)

    # t2_fit = time()
    # print('Elapsed time (fit): %f ms' % ((t2_fit - t1_fit)*1e3))

    pos_org = pd.DataFrame([], columns=['X Center (um)', 'Y Center (um)',
                                        'X Center (px)', 'Y Center (px)',
                                        'Goodness'])

    pos_org['X Center (um)'] = (np.array(x0s) - img.system.info['Cloud center (px)'][0]) * img.system.info['Effective Pixel size (um/px)']
    pos_org['Y Center (um)'] = (np.array(y0s) - img.system.info['Cloud center (px)'][1]) * img.system.info['Effective Pixel size (um/px)']
    pos_org['X Center (px)'] = x0s
    pos_org['Y Center (px)'] = y0s
    pos_org['Goodness'] = goodnesses

    img.system.lattice['Origins'] = pos_org

    fit_results = pd.DataFrame({'Amplitude': A0s,
                                'sigma0': sigma0s,
                                'Effective NA': NAeffs})

    return pos_org, psfs, psf_mean, fit_results
