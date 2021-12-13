import numpy as np
from numba import jit, uint16, float64

@jit(cache=True)
def gaussian(x:np.ndarray, *p:np.ndarray) -> np.ndarray:
    """[summary]
    
    Arguments:
        x {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    # Parameters
    A, x0, sigma, C = p
    xc = x - x0

    return A * np.exp(-xc**2/sigma**2) + C
