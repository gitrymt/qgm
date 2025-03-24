import numpy as np
from numba import float64, jit, uint16


@jit(cache=True)
def gaussian(x: np.ndarray, *p: np.ndarray) -> np.ndarray:
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
