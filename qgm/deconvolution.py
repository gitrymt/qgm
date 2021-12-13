import copy

import numpy as np
import scipy as sp

from . import image

def deconvolution(image, Niter=100, method=''):
    # if (len(image.system.lattice['Lattice sites']['X Center']) == 0) or (len(image.system.lattice['Lattice sites']['Y Center']) == 0):
    #     print('Test')
    # else:
    #     print('A')

    img = copy.copy(image.image_ROI)
    psfm = copy.copy(image.psfm)

    (im_width, im_height) = img.shape
    PSFM_flat = np.matrix(np.reshape(psfm, [im_width * im_height, psfm.shape[2]]))
    PSFM_T = PSFM_flat.T

    img_comp = np.reshape(img, [im_width * im_height, 1])

    factor_sites = np.matrix(np.ones([psfm.shape[2], 1]))
    factor_evol = np.matrix(np.ones([psfm.shape[2], Niter+1]))

    # t1 = time()

    for n in range(Niter):
        img_est = PSFM_flat * factor_sites
        img_est_inv = 1 / img_est
        img_err = np.multiply(img_comp, img_est_inv)
        site_err = PSFM_T * img_err
        factor_sites = np.multiply(factor_sites, site_err)
        factor_sites[factor_sites < 0] = 0

        factor_evol[:, n+1] = factor_sites

    #     print(factor_sites[0])

    # t2 = time()

    # print(t2-t1)

    im_dec = np.array(np.reshape(PSFM_flat * factor_sites, [im_width, im_height]))
    image.system.lattice['Lattice sites']['Amplitude'] = factor_sites

    return im_dec, factor_evol

