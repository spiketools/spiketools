"""Utility functions for selecting options of interest."""

import numpy as np
import scipy.stats

from spiketools.utils.checks import check_param_options

###################################################################################################
###################################################################################################

def get_avg_func(avg_type):
    """Select a function to use for averaging.

    Parameters
    ----------
    avg_type : {'mean', 'median', 'nanmean', 'nanmedian'}
        Which averaging function to select.

    Returns
    -------
    avg_func : callable
        Requested averaging function.
    """

    check_param_options(avg_type, 'avg_type', ['mean', 'median', 'nanmean', 'nanmedian'])
    avg_func = getattr(np, avg_type)

    return avg_func


def get_var_func(var_type):
    """Select a function to use for a variance-related measure.

    Parameters
    ----------
    var_type : {'var', 'std', 'sem'}
        Which variance-related function to select.

    Returns
    -------
    var_func : callable
        Requested variance related function.
    """

    check_param_options(var_type, 'var_type', ['var', 'std', 'sem'])

    if var_type in ['var', 'std']:
        var_func = getattr(np, var_type)
    elif var_type in ['sem']:
        var_func = getattr(scipy.stats, var_type)

    return var_func


def get_comp_func(comp_type):
    """Select a function to use for comparison.

    Parameters
    ----------
    comp_type : {'greater', 'less', 'greater_equal', 'less_equal'}
        Which comparison function to select.

    Returns
    -------
    comp_func : callable
        Requested comparison function.
    """

    check_param_options(comp_type, 'comp_type', ['greater', 'less', 'greater_equal', 'less_equal'])
    comp_func = getattr(np, comp_type)

    return comp_func
