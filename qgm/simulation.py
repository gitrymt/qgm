import numpy as np
from tqdm import tqdm

from . import function
from .image import image
from .parameter import system


def generate_simulation_image(setting: system,
                              img_size={'x': 100, 'y': 100, 'unit': 'px'}) -> image:
    img = image()
    img.system = setting

    if img_size['unit'] == 'px':
        size_x = img_size['x']
        size_y = img_size['y']
    else:
        size_x = int(img_size['x'] / setting.info['Effective Pixel size (um/px)'])
        size_y = int(img_size['y'] / setting.info['Effective Pixel size (um/px)'])

    img.image = np.zeros((size_y, size_x))
    system_info = {
                   'Cloud center (px)': (size_x/2, size_y/2),
                   'Cloud center (um)': (0, 0),
                   }
    img.system.set_info(**system_info)
    xy_mesh = img.generate_xymesh()
    (xs, ys) = img.generate_lattice_sites()

    dist = 'uniform'
    threshold = 0.5

    if dist == 'uniform':
        pos_bool = np.random.rand(len(xs)) > threshold
    elif dist == 'gauss':
        sigmax = 10
        sigmay = 10
        param_dist = [1, 0, sigmax, 0, sigmay, 0]
        p_dist = function.gaussian_2d((xs, ys), *param_dist)
        pos_bool = p_dist > np.random.rand(len(xs))
        pos_bool[np.random.rand(len(xs)) < threshold] = False
    else:
        pos_bool = np.random.rand(len(xs)) > threshold

    xs_ex = xs[pos_bool]
    ys_ex = ys[pos_bool]
    Nex = len(xs_ex)

    photon = 1000
    photon_noise = 100
    Nphs = np.random.poisson(photon, Nex)
    xmax = np.max(xs) * 1.25
    ymax = np.max(ys) * 1.25
    Nsample = 100000
    spx_eff = setting.info['Effective Pixel size (um/px)']

    alpha_psf = 2 * np.pi * setting.psf.info['Effective NA'] / setting.psf.info['Wavelength (um)']
    show_progress = True

    if show_progress:
        for i in tqdm(range(Nex)):
            N = 0
            p_psf = np.array([1, xs_ex[i], ys_ex[i], alpha_psf, 0])

            xs_rnd = np.array([])
            ys_rnd = np.array([])

            while N < Nphs[i]:
                x_rnd = 2 * xmax * (np.random.rand(Nsample) - 0.5)
                y_rnd = 2 * ymax * (np.random.rand(Nsample) - 0.5)
                z_rnd = np.random.rand(Nsample)
                s = function.psf_2d((x_rnd, y_rnd), *p_psf)

                xs_rnd = np.append(xs_rnd, x_rnd[s > z_rnd])
                ys_rnd = np.append(ys_rnd, y_rnd[s > z_rnd])

                N = xs_rnd.size

            if N > Nphs[i]:
                xs_rnd = xs_rnd[0:Nphs[i]]
                ys_rnd = ys_rnd[0:Nphs[i]]

                N = xs_rnd.size

            for j in range(len(xs_rnd)):
                x_tmp1 = xy_mesh[0] >= xs_rnd[j]
                x_tmp2 = xy_mesh[0] <= xs_rnd[j] + spx_eff
                x_tmp = x_tmp1 * x_tmp2
                y_tmp1 = xy_mesh[1] >= ys_rnd[j]
                y_tmp2 = xy_mesh[1] <= ys_rnd[j] + spx_eff
                y_tmp = y_tmp1 * y_tmp2

                img.image += x_tmp * y_tmp
    else:
        for i in range(Nex):
            N = 0
            p_psf = np.array([1, xs_ex[i], ys_ex[i], alpha_psf, 0])

            # print(i+1, Nex, Nphs[i], N)

            xs_rnd = np.array([])
            ys_rnd = np.array([])

            while N < Nphs[i]:
                x_rnd = 50 * (np.random.rand(Nsample) - 0.5)
                y_rnd = 50 * (np.random.rand(Nsample) - 0.5)
                z_rnd = np.random.rand(Nsample)
                s = function.psf_2d((x_rnd, y_rnd), *p_psf)

                xs_rnd = np.append(xs_rnd, x_rnd[s > z_rnd])
                ys_rnd = np.append(ys_rnd, y_rnd[s > z_rnd])

                N = xs_rnd.size

            if N > Nphs[i]:
                xs_rnd = xs_rnd[0:Nphs[i]]
                ys_rnd = ys_rnd[0:Nphs[i]]

                N = xs_rnd.size

            for j in range(len(xs_rnd)):
                x_tmp1 = xy_mesh[0] >= xs_rnd[j]
                x_tmp2 = xy_mesh[0] <= xs_rnd[j] + spx_eff
                x_tmp = x_tmp1 * x_tmp2
                y_tmp1 = xy_mesh[1] >= ys_rnd[j]
                y_tmp2 = xy_mesh[1] <= ys_rnd[j] + spx_eff
                y_tmp = y_tmp1 * y_tmp2

                img.image += x_tmp * y_tmp

            # print(i+1, Nex, Nphs[i], N)

    Nbgs = np.random.poisson(photon_noise, img.image.shape)
    img.image += Nbgs

    x_img = xy_mesh[0] - spx_eff/2
    y_img = xy_mesh[1] - spx_eff/2

    return img, x_img, y_img, xs_ex, ys_ex, Nphs, Nbgs
    # return img


def qgm_image_rnd(xs, ys,
                  Nsample=100000, threshold=0.5, dist='uniform', sigmax=10, sigmay=10, seed=None,
                  fast=False, show_progress=False):

    if seed is not None:
        np.random.seed(seed)

    # alpha_psf = 2 * np.pi * self._NA / self._wavelength

#     nxmin = int(np.min(xs) / self._spx_eff) - 1
#     nxmax = int(np.max(xs) / self._spx_eff) + 1
#     dx = np.linspace(nxmin, nxmax, nxmax - nxmin + 1) * self._spx_eff

#     nymin = int(np.min(ys) / self._spx_eff) - 1
#     nymax = int(np.max(ys) / self._spx_eff) + 1
#     dy = np.linspace(nymin, nymax, nymax - nymin + 1) * self._spx_eff

#     xy_mesh = np.meshgrid(dx, dy)
#     (x, y) = xy_mesh

#     img = np.zeros(x.shape)
#     xmax = np.max(xs) * 1.25
#     ymax = np.max(ys) * 1.25

#     if dist is 'uniform':
#         pos_bool = np.random.rand(len(xs)) > threshold
#     elif dist is 'gauss':
#         param_dist = np.array([1, 0, sigmax, 0, sigmay, 0])
#         p_dist = function.gaussian_2d((xs, ys), *param_dist)
#         pos_bool = p_dist > np.random.rand(len(xs))
#         pos_bool[np.random.rand(len(xs)) < threshold] = False
#     else:
#         pos_bool = np.random.rand(len(xs)) > threshold

#     xs_ex = xs[pos_bool]
#     ys_ex = ys[pos_bool]
#     Nex = len(xs_ex)

#     Nphs = np.random.poisson(self._photon, Nex)

#     if show_progress:
#         for i in tqdm(range(Nex)):
#             N = 0
#             p_psf = np.array([1, xs_ex[i], ys_ex[i], alpha_psf, 0])

#             xs_rnd = np.array([])
#             ys_rnd = np.array([])

#             while N < Nphs[i]:
#                 x_rnd = 2 * xmax * (np.random.rand(Nsample) - 0.5)
#                 y_rnd = 2 * ymax * (np.random.rand(Nsample) - 0.5)
#                 z_rnd = np.random.rand(Nsample)
#                 s = function.psf_2d((x_rnd, y_rnd), *p_psf)

#                 xs_rnd = np.append(xs_rnd, x_rnd[s > z_rnd])
#                 ys_rnd = np.append(ys_rnd, y_rnd[s > z_rnd])

#                 N = xs_rnd.size

#             if N > Nphs[i]:
#                 xs_rnd = xs_rnd[0:Nphs[i]]
#                 ys_rnd = ys_rnd[0:Nphs[i]]

#                 N = xs_rnd.size

#             for j in range(len(xs_rnd)):
#                 x_tmp1 = x >= xs_rnd[j]
#                 x_tmp2 = x <= xs_rnd[j] + self._spx_eff
#                 x_tmp = x_tmp1 * x_tmp2
#                 y_tmp1 = y >= ys_rnd[j]
#                 y_tmp2 = y <= ys_rnd[j] + self._spx_eff
#                 y_tmp = y_tmp1 * y_tmp2

#                 img += x_tmp * y_tmp
#     else:
#         for i in range(Nex):
#             N = 0
#             p_psf = np.array([1, xs_ex[i], ys_ex[i], alpha_psf, 0])

#             # print(i+1, Nex, Nphs[i], N)

#             xs_rnd = np.array([])
#             ys_rnd = np.array([])

#             while N < Nphs[i]:
#                 x_rnd = 50 * (np.random.rand(Nsample) - 0.5)
#                 y_rnd = 50 * (np.random.rand(Nsample) - 0.5)
#                 z_rnd = np.random.rand(Nsample)
#                 s = function.psf_2d((x_rnd, y_rnd), *p_psf)

#                 xs_rnd = np.append(xs_rnd, x_rnd[s > z_rnd])
#                 ys_rnd = np.append(ys_rnd, y_rnd[s > z_rnd])

#                 N = xs_rnd.size

#             if N > Nphs[i]:
#                 xs_rnd = xs_rnd[0:Nphs[i]]
#                 ys_rnd = ys_rnd[0:Nphs[i]]

#                 N = xs_rnd.size

#             for j in range(len(xs_rnd)):
#                 x_tmp1 = x >= xs_rnd[j]
#                 x_tmp2 = x <= xs_rnd[j] + self._spx_eff
#                 x_tmp = x_tmp1 * x_tmp2
#                 y_tmp1 = y >= ys_rnd[j]
#                 y_tmp2 = y <= ys_rnd[j] + self._spx_eff
#                 y_tmp = y_tmp1 * y_tmp2

#                 img += x_tmp * y_tmp

#             # print(i+1, Nex, Nphs[i], N)

#     Nbgs = np.random.poisson(self._photon_noise, img.shape)
#     img += Nbgs

#     x_img = x - self._spx_eff/2
#     y_img = y - self._spx_eff/2

#     return img, x_img, y_img, xs_ex, ys_ex, Nphs, Nbgs


#########################################################################
# def search_isolate_atoms(self, img,
#                             filter_method='gauss', sigma_filter=2,
#                             threshold_binary=70, parameter_hough=[1,30,75,10,2,6],
#                             size_iso=2*15, interpolation_method=cv2.INTER_CUBIC):

#     if filter_method == 'gauss':
#         img_filter = gaussian_filter(img, sigma=sigma_filter)
#     elif filter_method == 'median':
#         img_filter = median_filter(img, sigma_filter)
#     else:
#         img_filter = gaussian_filter(img, sigma=sigma_filter)

#     img_filter_binary = np.array(255 * (img_filter - np.min(img_filter)) / np.max(img_filter), dtype=np.uint8)
#     img_filter_binary[img_filter_binary < threshold_binary] = 0
#     img_filter_binary[img_filter_binary >= threshold_binary] = 255

#     dp_hough, mindist_hough, p1_hough, p2_hough, rmin_hough, rmax_hough = parameter_hough
#     circles = cv2.HoughCircles(img_filter_binary,
#                                 cv2.HOUGH_GRADIENT,
#                                 dp=dp_hough, minDist=mindist_hough,
#                                 param1=p1_hough, param2=p2_hough,
#                                 minRadius=rmin_hough, maxRadius=rmax_hough)
#     circles = np.uint16(np.around(circles))

#     size_fit = int(3 * size_iso / 2)
#     shape_fit = tuple(np.array([size_fit, size_fit]))
#     shape = tuple(np.array([size_iso, size_iso]))

#     imgs_iso = np.zeros([size_fit, size_fit, circles.shape[1]])
#     imgs_iso_fit = np.zeros([size_iso, size_iso, circles.shape[1]])

#     n_isolate = 0

#     x_iso = []
#     y_iso = []
#     r_iso = []


#     for (x, y, r) in circles[0]:
#         # try:
#         xmin_sub = x - size_fit / 2
#         xmax_sub = x + size_fit / 2
#         ymin_sub = y - size_fit / 2
#         ymax_sub = y + size_fit / 2

#         matrix = [[1, 0, -xmin_sub], [0, 1, -ymin_sub]]
#         affine_matrix_fit = np.float32(matrix)
#         img_iso = cv2.warpAffine(img, affine_matrix_fit, shape_fit,
#                                 flags=interpolation_method)
#         imgs_iso[:, :, n_isolate] = img_iso
#         img_iso_fit = gaussian_filter(img_iso, sigma=sigma_filter)

#         p_ini = [np.max(img_iso_fit)*0.9, size_fit/2, size_fit/2, 2 * np.pi * self._NA / (self._wavelength / self._spx_eff), np.min(img_iso_fit)]
#         # p_fit = p_ini
#         xy_fit = np.meshgrid(np.arange(size_fit), np.arange(size_fit))

#         try:
#             p_fit, p_err, fit_goodness = fitting.fit_2d(function.psf_2d, img_iso_fit, xy_fit, p_ini)

#             dx = p_fit[1] - size_iso / 2
#             dy = p_fit[2] - size_iso / 2
#             matrix = [[1, 0, -dx], [0, 1, -dy]]
#             affine_matrix = np.float32(matrix)
#             imgs_iso_fit[:, :, n_isolate] = cv2.warpAffine(img_iso, affine_matrix, shape,
#                                                     flags=interpolation_method)

#             x_iso += [x + p_fit[1]-size_fit/2]
#             y_iso += [y + p_fit[2]-size_fit/2]
#             # y_iso += [y]
#             r_iso += [r]
#         except:
#             # imgs_iso[:, :, n_isolate] = 0

#             x_iso += [x]
#             y_iso += [y]
#             r_iso += [r]
#         # except:
#         #     xmin_sub = x - size_iso / 2
#         #     xmax_sub = x + size_iso / 2
#         #     ymin_sub = y - size_iso / 2
#         #     ymax_sub = y + size_iso / 2

#         #     shape = tuple(np.array([size_iso, size_iso]))
#         #     matrix = [[1, 0, -xmin_sub], [0, 1, -ymin_sub]]
#         #     affine_matrix = np.float32(matrix)
#         #     imgs_iso[:, :, n_isolate] = cv2.warpAffine(img, affine_matrix, shape,
#         #                                                flags=interpolation_method)

#         #     x_iso += [x]
#         #     y_iso += [y]
#         #     r_iso += [r]

#         n_isolate += 1

#     # imgs_iso = imgs_iso[np.sum(imgs_iso, axis=2)>0]
#     return (x_iso, y_iso, r_iso), imgs_iso_fit, imgs_iso, img_filter, img_filter_binary
#     # return 0
