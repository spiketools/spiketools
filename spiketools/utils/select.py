"""Utility functions for selecting options of interest."""

import numpy as np
from scipy.stats import sem

###################################################################################################
###################################################################################################

def get_avg_func(avg_type):
    """Select a function to use for averaging.

    Parameters
    ----------
    avg_type : {'mean', 'median'}
        The type of averaging function to use.

    Returns
    -------
    avg_func : callable
        Requested averaging function.
    """

    if avg_type == 'mean':
        avg_func = np.mean
    elif avg_type == 'median':
        avg_func = np.median
    else:
        raise ValueError('Averaging method not understood.')

    return avg_func


def get_var_func(var_type):
    """Select a function to use for a variance-related measure.

    Parameters
    ----------
    var_type : {'var', 'std', 'sem'}
        The type of variance-related function to use.

    Returns
    -------
    var_func : callabel
        Requested variance related function.
    """

    if var_type == 'var':
        var_func = np.var
    elif var_type == 'std':
        var_func = np.std
    elif var_type == 'sem':
        var_func = sem
    else:
        raise ValueError('Variance method not understood.')

    return var_func
