"""Utility functions for working with data arrays."""

from copy import deepcopy

import numpy as np

from scipy.ndimage import gaussian_filter

###################################################################################################
###################################################################################################

def compute_range(data):
    """Compute the range of an array of data.

    Parameters
    ----------
    data : array
        Array of numerical data.

    Returns
    -------
    min, max : float
        Minimum and maximum values of the data array.
    """

    return np.nanmin(data), np.nanmax(data)


def smooth_data(data, sigma):
    """Smooth an array of data, using a gaussian kernel.

    Parameters
    ----------
    data : ndarray
        Data to smooth.
    sigma : float
        Standard deviation of the gaussian kernel to apply for smoothing.

    Returns
    -------
    data : 2d array
        The smoothed data.

    Notes
    -----
    This function is applied on a copy of the data (to not change the original).
    Any NaN values will be set as 0 for smoothing purposes.
    """

    data = deepcopy(data)
    data[np.isnan(data)] = 0

    data = gaussian_filter(data, sigma=sigma)

    return data


def drop_nans(data):
    """Drop any NaNs values from an array.

    Parameters
    ----------
    data : 1d or 2d array
        Data array to check and drop NaNs from.

    Returns
    -------
    data : 1d or 2d array
        Data array with NaNs removed.

    Notes
    -----
    For 2d arrays, this function assumes the same columns to be NaN across all rows.
    """

    nans = np.isnan(data)

    if data.ndim == 1:
        data = data[np.where(~data_nans)]
    elif data.ndim == 2:
        data = data[~ans].reshape(nans.shape[0], sum(~nans[0, :]))
    else:
        raise ValueError('Only 1d or 2d arrays supported.')

    return data
