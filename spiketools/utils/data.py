"""Utility functions for managing data."""

import numpy as np

###################################################################################################
###################################################################################################

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
