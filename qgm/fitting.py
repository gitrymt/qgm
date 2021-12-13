from . import function
# from . import estparm

import numpy as np
import scipy as sp
from scipy import optimize

def fit_1d(func, y, x, *p_ini, yerr=None, param_est=False):
    """
    1D 関数に対するフィッティングをScipyのOptimize.curve_fitモジュールを利用して行う.

    Arguments:
        func {[callable]} -- [description]
        y {[ndarray]} -- [description]
        x {[ndarray]} -- [description]
    
    Keyword Arguments:
        yerr {[ndarray]} -- [description] (default: {None})
        param_est {bool} -- [description] (default: {False})
    
    Returns:
        [type] -- [description]
    """
    if (yerr is None):
        p_fit, cov_mat = optimize.curve_fit(func, x, y, p0=p_ini)
    else:
        p_fit, cov_mat = optimize.curve_fit(func, x, y, p0=p_ini, sigma=yerr)
    
    p_err = np.sqrt(np.diag(cov_mat))

    # manually calculate R-squared goodness of fit
    fit_residual = y - func(x, *p_fit)
    fit_Rsquared = 1 - np.var(fit_residual) / np.var(y)
    
    return p_fit, p_err, fit_Rsquared

def fit_2d(func, z, xy_mesh, p_ini, param_est=False):
    """
    2D 関数に対するフィッティングをScipyのOptimize.curve_fitモジュールを利用して行う.
    
    Arguments:
        func {[type]} -- [description]
        z {[type]} -- [description]
        xy_mesh {[type]} -- [description]
        p_ini {[type]} -- [description]

    Keyword Arguments:
        param_est {bool} -- [description] (default: {False})
    
    Returns:
        [type] -- [description]
    """

    def func_fit(xy_mesh, *p):
        return np.ravel(func(xy_mesh, *p))

    p_fit, cov_mat = optimize.curve_fit(func_fit, xy_mesh, np.ravel(z), p0=p_ini)
    p_err = np.sqrt(np.diag(cov_mat))

    # manually calculate R-squared goodness of fit
    fit_residual = z - func(xy_mesh, *p_fit)
    fit_Rsquared = 1 - np.var(fit_residual) / np.var(z)

    return p_fit, p_err, fit_Rsquared
