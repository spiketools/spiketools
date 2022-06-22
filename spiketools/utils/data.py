"""Utility functions for managing data."""

from copy import deepcopy

import numpy as np
from scipy.ndimage import gaussian_filter

###################################################################################################
###################################################################################################

def get_range(data):
    """Get the range of an array of data.

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


def restrict_range(data, min_value=None, max_value=None, reset=None):
    """Restrict a vector of data to a specified range.

    Parameters
    ----------
    data : 1d array
        Array of data.
    min_value, max_value : float, optional
        Mininum and/or maximum value to restrict input array to.
    reset : float, optional
        If provided, resets the values in the data array by the given reset value.

    Returns
    -------
    data : 1d array
        Data array, restricted to desired time range.

    Examples
    --------
    Select all values greater than a specific value:

    >>> data = np.array([5, 10, 15, 20, 25, 30])
    >>> restrict_range(data, min_value=10, max_value=None)
    array([10, 15, 20, 25, 30])

    Select all values less than a specific value:

    >>> data = np.array([5, 10, 15, 20, 25, 30])
    >>> restrict_range(data, min_value=None, max_value=25)
    array([ 5, 10, 15, 20, 25])

    Restrict a data array to a specific range:

    >>> data = np.array([5, 10, 15, 20, 25, 30])
    >>> restrict_range(data, min_value=10, max_value=20)
    array([10, 15, 20])
    """

    min_value = -np.inf if min_value is None else min_value
    max_value = np.inf if max_value is None else max_value

    data = data[(data >= min_value) & (data <= max_value)]

    if reset:
        data = data - reset

    return data


def get_value_by_time(times, values, time):
    """Get the value for a data array at a specific time point.

    Parameters
    ----------
    times : 1d array
        Time indices.
    values : ndarray
        Data values, corresponding to the times vector.
    time : float
        Time value to extract

    Returns
    -------
    out
        The value at the requested time point.
    """

    return values[:].take(indices=np.abs(times[:] - time).argmin(), axis=-1)


def get_value_by_time_range(times, values, t_min, t_max):
    """Extract data for a requested time range.

    Parameters
    ----------
    times : 1d array
        Time indices.
    values : ndarray
        Data values, corresponding to the times indices.
    t_min, t_max : float
        Time range to extract.

    Returns
    -------
    times : 1d array
        Selected time indices.
    out : ndarray
        Selected values.
    """

    select = np.logical_and(times[:] >= t_min, times[:] <= t_max)

    return times[select], values[:].take(indices=np.where(select)[0], axis=-1)


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
