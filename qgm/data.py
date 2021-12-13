import numpy as np
import joblib

from . import parameter

class data():
    def __init__(self, **kwargs):
        for key in kwargs:
            if key == 'Number of images':
                self.Nimage = kwargs[key]
            else:
                print('KeyError: %s is not found.' % key)

def load(load_path):
    return joblib.load(load_path)

def save(data, save_path, compress_level=3):
    joblib.dump(data, save_path, compress=compress_level)

