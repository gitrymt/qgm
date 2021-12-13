import numpy as np
import scipy as sp

from . import image

def lowpass(image: image, factor=1) -> []:
    # Sampling frequency
    fs = 1 / image.system.info['Effective Pixel size (um/px)']
    f_Abbe = 1 / image.system.psf.info['R abbe (um)']

    # Image size
    (im_height, im_width) = image.image.shape

    # Set frequency step size
    dfx = fs / im_width
    dfy = fs / im_height

    # Frequency axis
    if im_width % 2 == 0:
        fx = np.arange(-fs/2, fs/2, step=dfx)
    else:
        fx = np.arange(-fs/2+dfx/2, fs/2+dfx/2, step=dfx)

    if im_height % 2 == 0:
        fy = np.arange(-fs/2, fs/2, step=dfy)
    else:
        fy = np.arange(-fs/2+dfy/2, fs/2+dfy/2, step=dfy)

    # Meshgrid - Frequency space
    (FX, FY) = np.meshgrid(fx, fy)

    # Fourier transform
    img_fft = np.fft.fftshift(np.fft.fft2(image.image))

    # Lowpass filter
    img_fft[FX**2+FY**2 > (factor * f_Abbe)**2] = 0

    # Inverse Fourier transform
    img_lowpass = np.abs(np.fft.ifft2(img_fft))

    # Debug
    if False:
        print('Sampling frequnecy: %f (1/um)' % fs)
        print('Abbe frequnecy: %f (1/um)' % f_Abbe)
        print('Image size: %dx%d' % (im_width, im_height))
        print('Fx size: %dx%d' % (FX.shape[1], FX.shape[0]))
        print('Frequency step size: (dfx, dfy) = (%f, %f)' % (dfx, dfy))
        print(fx.shape, im_width)

    return img_lowpass, img_fft, (fx, fy), (FX, FY)
    
def gaussian():
    pass
