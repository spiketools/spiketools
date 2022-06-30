"""Utility functions for selecting options of interest."""

import numpy as np
from scipy.stats import sem

from spiketools.utils.checks import check_param_options

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

    check_param_options(avg_type, 'avg_type', ['mean', 'median'])

    if avg_type == 'mean':
        avg_func = np.mean
    elif avg_type == 'median':
        avg_func = np.median

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

    check_param_options(var_type, 'var_type', ['var', 'std', 'sem'])

    if var_type == 'var':
        var_func = np.var
    elif var_type == 'std':
        var_func = np.std
    elif var_type == 'sem':
        var_func = sem

    return var_func
