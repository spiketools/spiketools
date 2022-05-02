"""Utility functions for managing data."""

import numpy as np

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


def restrict_range(values, min_value=None, max_value=None):
    """Restrict a vector of data to a specified range.

    Parameters
    ----------
    values : 1d array
        Array of data.
    min_value, max_value : float, optional, default: None
        Mininum and/or maximum value to restrict input array to.

    Returns
    -------
    1d array
        Data array, restricted to desired time range.

    Examples
    --------
    Select all values greater than a specific value:

    >>> values = np.array([5, 10, 15, 20, 25, 30])
    >>> restrict_range(values, min_value=10, max_value=None)
    array([10, 15, 20, 25, 30])

    Select all values less than a specific value:

    >>> values = np.array([5, 10, 15, 20, 25, 30])
    >>> restrict_range(values, min_value=None, max_value=25)
    array([ 5, 10, 15, 20, 25])

    Restrict a data array to a specific range:

    >>> values = np.array([5, 10, 15, 20, 25, 30])
    >>> restrict_range(values, min_value=10, max_value=20)
    array([10, 15, 20])
    """

    min_value = -np.inf if min_value is None else min_value
    max_value = np.inf if max_value is None else max_value

    return values[(values >= min_value) & (values <= max_value)]


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
