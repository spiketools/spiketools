"""Utility functions for extracting data segments of interest."""

import numpy as np

###################################################################################################
###################################################################################################

def get_range(data, min_value=None, max_value=None, reset=None):
    """Get a specified range from a vector of data.

    Parameters
    ----------
    data : 1d array
        Array of data.
    min_value, max_value : float, optional
        Mininum and/or maximum value to extract from the input array.
    reset : float, optional
        If provided, resets the values in the data array by the given reset value.

    Returns
    -------
    data : 1d array
        Data array, restricted to desired time range.

    Examples
    --------
    Get all values greater than a specific value:

    >>> data = np.array([5, 10, 15, 20, 25, 30])
    >>> get_range(data, min_value=10, max_value=None)
    array([10, 15, 20, 25, 30])

    Get all values less than a specific value:

    >>> data = np.array([5, 10, 15, 20, 25, 30])
    >>> get_range(data, min_value=None, max_value=25)
    array([ 5, 10, 15, 20, 25])

    Get a specified range from a data array:

    >>> data = np.array([5, 10, 15, 20, 25, 30])
    >>> get_range(data, min_value=10, max_value=20)
    array([10, 15, 20])
    """

    min_value = -np.inf if min_value is None else min_value
    max_value = np.inf if max_value is None else max_value

    data = data[(data >= min_value) & (data <= max_value)]

    if reset:
        data = data - reset

    return data


def get_value_by_time(times, values, timepoint, threshold=np.inf):
    """Get the value for a data array at a specific time point.

    Parameters
    ----------
    times : 1d array
        Time indices.
    values : ndarray
        Data values, corresponding to the times vector.
    timepoint : float
        Time value to extract.
    threshold : float
        The threshold that the closest time value must be within to be returned.
        If the temporal distance is greater than the threshold, output is NaN.

    Returns
    -------
    out : float of 1d array
        The value at the requested time point.
    """

    idx = np.abs(times[:] - timepoint).argmin()

    if np.abs(times[idx] - timepoint) < threshold:
        out = values[:].take(indices=idx, axis=-1)
    else:
        out = np.nan

    return out


def get_values_by_times(times, values, timepoints, threshold=np.inf):
    """Get values from a data array for a set of specified time points.

    Parameters
    ----------
    times : 1d array
        Time indices.
    values : ndarray
        Data values, corresponding to the times vector.
    extract : 1d array
        The time indices to extract corresponding values for.

    Returns
    -------
    outputs : 1d array
        The extracted vaues for the requested time points.
    """

    outputs = np.zeros(len(timepoints))
    for ind, timepoint in enumerate(timepoints):
        outputs[ind] = get_value_by_time(times, values, timepoint, threshold=threshold)

    return outputs


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
    out = values[:].take(indices=np.where(select)[0], axis=-1)

    return times[select], out
