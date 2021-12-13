import numpy as np
import pandas as pd

class system():
    def __init__(self, **kwargs):
        self.set_defaults()

        if len(kwargs) > 0:
            self.set_info(**kwargs)

    def set_defaults(self):
        self.info = {'Camera model': 'Andor iXon Ultra DU-897U-CS0-EXF',
                     'Camera maker': 'Oxford Instruments',
                     'Pixel size (um/px)': 16,
                     'Effective Pixel size (um/px)': 16 / 85,
                     'Magnification': 85,
                     'EM gain': 1000,
                     'Pre amp.': 3,
                     'HS speed (MHz)': 17.0,
                     'VS speed (us/px)': 3.3,
                     'Temperatrue': -75,
                     'Objective NA': 0.65, 
                     'Cloud center (px)': (0, 0), 
                     'Cloud center (um)': (0, 0), 
                    #  'Lattice 1': {'Constant (px)': 2 * 1.064 / 3 / (16 / 85),
                    #                'Constant (um)': 2 * 1.064 / 3,
                    #                'Angle (degree)': 0,
                    #               },
                    #  'Lattice 2': {'Constant (px)': 2 * 1.064 / 3 / (16 / 85),
                    #                'Constant (um)': 2 * 1.064 / 3,
                    #                'Angle (degree)': 60,
                    #               },
                    }
        
        self.lattice = {'Lattice 1': lattice(),
                        'Lattice 2': lattice(),
                        'Origins': pd.DataFrame([],
                                                columns=['X Center (um)', 'Y Center (um)',
                                                         'X Center (px)', 'Y Center (px)',
                                                         'Goodness']),
                        'Lattice sites': pd.DataFrame([], columns=['X Center', 'Y Center', 'Amplitude'])
                        }
        
        self.psf = psf()

        self.conversion_rate = 1
        
    def set_info(self, **kwargs):
        for key in kwargs:
            if key in self.info:
                if key == 'Lattice 1' or key == 'Lattice 2':
                    lattice_info = self.info[key]
                    
                    kwargs2 = kwargs[key]
                    for key2 in kwargs2:
                        if key2 in lattice_info:
                            lattice_info[key2] = kwargs2[key2]
                        else:
                            print('KeyError: %s is not found in %s.' % (key2, key))
                    
                    self.info[key] = lattice_info
                else:
                    self.info[key] = kwargs[key]
            else:
                print('KeyError: %s is not found.' % key)
    
    def recalc_effective_pixel_size(self):
        self.info['Effective Pixel size (um/px)'] = self.info['Pixel size (um/px)'] / self.info['Magnification']

    # def recalculation(self, target):
    #     if target == 'Effective pixel size':
    #         self.info['Effective Pixel size (um/px)'] = self.info['Pixel size (um/px)'] / self.info['Magnification']
    #     elif target == 'Lattice constants (px)':
    #         system_info = {'Lattice 1': {'Constant (px)': 3,},
    #                        'Lattice 2': {'Constant (px)': 3.78842,},
    #                        }
    #         self.set_info(**system_info)
    #     elif target == 'Lattice constants (px)':
    #         system_info = {'Lattice 1': {'Constant (px)': 3,},
    #                        'Lattice 2': {'Constant (px)': 3.78842,},
    #                        }
    #         self.set_info(**system_info)
    #     else:
    #         pass

class lattice():
    def __init__(self, **kwargs):
        self.set_defaults()

        if len(kwargs) > 0:
            self.set_info(**kwargs)

    def set_defaults(self):
        self.info = {'Constant (px)': 2 * 1.064 / 3 / (16 / 85),
                     'Constant (um)': 2 * 1.064 / 3,
                     'Angle (degree)': 0,
                     'Angle (radian)': 0,
                     }

    def set_info(self, **kwargs):
        for key in kwargs:
            if key in self.info:
                self.info[key] = kwargs[key]
            else:
                print('KeyError: %s is not found.' % key)

class psf():
    def __init__(self, **kwargs):
        self.set_defaults()
        
        if len(kwargs) > 0:
            print(kwargs)
            self.set_info(**kwargs)

    def set_defaults(self):
        self.info = {'Model': 'psf',
                     'Effective NA': 0.65,
                     'Wavelength (um)': 0.78,
                     'Peak count': 1,
                     'HWHM width - iso (um)': 0.257248492490547012870142829691886372215752428677258 * 0.78 / 0.65, # Calculated by Mathematica online
                     'HWHM width - x (um)': 0.257248492490547012870142829691886372215752428677258 * 0.78 / 0.65, # Calculated by Mathematica online
                     'HWHM width - y (um)': 0.257248492490547012870142829691886372215752428677258 * 0.78 / 0.65, # Calculated by Mathematica online
                     'Offset count': 0,
                     'R abbe (um)': 0.78 / (2 * 0.65),
                     'R Rayleigh (um)': 0.609834945633252227463269423732627588940 * 0.78 / 0.65, # Calculated by Mathematica online 
                    }
        self.psf_image = np.array([])

    def set_info(self, **kwargs):
        for key in kwargs:
            if key in self.info:
                print(key)
                self.info[key] = kwargs[key]
            else:
                print('KeyError: %s is not found.' % key)
            
    def get_info(self):
        return self.info

    def recalc_resolution(self, base='NA'):
        if base == 'NA':
            tmp = self.info['Wavelength (um)'] / self.info['Effective NA']

            self.info['HWHM width - iso (um)'] = 0.257248492490547012870142829691886372215752428677258 * tmp # Calculated by Mathematica online
            self.info['HWHM width - x (um)'] = 0.257248492490547012870142829691886372215752428677258 * tmp # Calculated by Mathematica online
            self.info['HWHM width - y (um)'] = 0.257248492490547012870142829691886372215752428677258 * tmp # Calculated by Mathematica online
            self.info['R abbe (um)'] = tmp / 2
            self.info['R Rayleigh (um)'] = 0.609834945633252227463269423732627588940 * tmp # Calculated by Mathematica online
        elif base == 'R Rayleigh':
            print('Error: Unimplemented.')
        else:
            print('Error: Input calculation base is not defined.')

    def set_psf_image(self, image):
        self.psf_image = image

    def get_psf_image(self):
        return self.psf_image
