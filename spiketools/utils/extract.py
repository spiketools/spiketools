"""Utility functions for extracting data segments of interest."""

import numpy as np

from spiketools.utils.options import get_comp_func

###################################################################################################
###################################################################################################

def get_range(data, min_value=None, max_value=None, reset=None):
    """Get a specified range from a vector of data.

    Parameters
    ----------
    data : 1d array
        Array of data.
    min_value, max_value : float, optional
        Minimum and/or maximum value to extract from the input array.
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


def get_ind_by_time(times, timepoint, threshold=None):
    """Get the index for a data array for a specified timepoint.

    Parameters
    ----------
    times : 1d array
        Time indices.
    timepoint : float
        Time value to extract the index for.
    threshold : float, optional
        The threshold that the closest time value must be within to be returned.
        If the temporal distance is greater than the threshold, output is -1.

    Returns
    -------
    ind : int
        The index value for the requested timepoint, or -1 if out of threshold range.
    """

    ind = np.abs(times - timepoint).argmin()
    if threshold:
        if np.abs(times[ind] - timepoint) > threshold:
            ind = -1

    return ind


def get_inds_by_times(times, timepoints, threshold=None, drop_null=True):
    """Get indices for a data array for a set of specified time points.

    Parameters
    ----------
    times : 1d array
        Time indices.
    timepoints : 1d array
        The time values to extract indices for.
    threshold : float, optional
        The threshold that the closest time value must be within to be returned.
        If the temporal distance is greater than the threshold, output is NaN.
    drop_null : bool, optional, default: True
        Whether to drop any null indices from the outputs (outside threshold range).
        If False, indices for any null values are -1.

    Returns
    -------
    inds : 1d array
        Indices for all requested timepoints.
    """

    inds = np.zeros(len(timepoints), dtype=int)
    for ind, timepoint in enumerate(timepoints):
        inds[ind] = get_ind_by_time(times, timepoint, threshold=threshold)

    if drop_null:
        inds = inds[inds >= 0]

    return inds


def get_value_by_time(times, values, timepoint, threshold=None):
    """Get the value from a data array at a specific time point.

    Parameters
    ----------
    times : 1d array
        Time indices.
    values : 1d or 2d array
        Data values, corresponding to the times vector.
    timepoint : float
        Time value to extract.
    threshold : float, optional
        The threshold that the closest time value must be within to be returned.
        If the temporal distance is greater than the threshold, output is NaN.

    Returns
    -------
    out : float or 1d array
        The value(s) at the requested time point.
    """

    idx = get_ind_by_time(times, timepoint, threshold=threshold)
    out = values.take(indices=idx, axis=-1) if idx >= 0 else np.nan

    return out


def get_values_by_times(times, values, timepoints, threshold=None, drop_null=True):
    """Get values from a data array for a set of specified time points.

    Parameters
    ----------
    times : 1d array
        Time indices.
    values : 1d or 2d array
        Data values, corresponding to the times vector.
    timepoints : 1d array
        The time indices to extract corresponding values for.
    threshold : float, optional
        The threshold that the closest time value must be within to be returned.
        If the temporal distance is greater than the threshold, output is NaN.
    drop_null : bool, optional, default: True
        Whether to drop any null values from the outputs (outside threshold range).
        If False, indices for any null values are NaN.

    Returns
    -------
    outputs : 1d or 2d array
        The extracted values for the requested time points.
    """

    inds = get_inds_by_times(times, timepoints, threshold, drop_null)

    if drop_null:
        outputs = values.take(indices=inds, axis=-1)
    else:
        outputs = np.full([np.atleast_2d(values).shape[0], len(timepoints)], np.nan)
        mask = inds >= 0
        outputs[:, np.where(mask)[0]] = values.take(indices=inds[mask], axis=-1)
        outputs = np.squeeze(outputs)

    return outputs


def get_values_by_time_range(times, values, t_min, t_max):
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

    select = np.logical_and(times >= t_min, times <= t_max)
    out = values.take(indices=np.where(select)[0], axis=-1)

    return times[select], out


def threshold_spikes_by_times(spikes, times, threshold):
    """Threshold spikes by sub-selecting those are temporally close to a set of time values.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    times : 1d array
        Time indices.
    threshold : float
        The threshold that closest time values must be within to be kept.
        For any time indices greater than this threshold, the spike value is dropped.

    Returns
    -------
    spikes : 1d array
        Sub-selected spike times, in seconds.
    """

    mask = np.empty_like(spikes, dtype=bool)
    for ind, spike in enumerate(spikes):
        mask[ind] = np.min(np.abs(times - spike)) < threshold

    return spikes[mask]


def threshold_spikes_by_values(spikes, times, values, data_threshold,
                               time_threshold=None, comp_type='greater'):
    """Threshold spikes by sub-selecting those are exceed a value on another data stream.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    times : 1d array
        Time indices.
    values : 1d array
        Data values, corresponding to the times vector.
    data_threshold : float
        The threshold that closest data values must be within to be kept.
    time_threshold : float, optional
        The threshold that closest time values must be within to be kept.
        For any time indices greater than this threshold, the spike value is dropped.
    comp_type : {'greater', 'less'}
        Which comparison function to use.
        This defines whether selected values must be greater than or less than the data threshold.

    Returns
    -------
    spikes : 1d array
        Sub-selected spike times, in seconds.
    """

    values = get_values_by_times(times, values, spikes, time_threshold, drop_null=False)

    mask = ~np.isnan(values)
    spikes, values = spikes[mask], values[mask]
    spikes = spikes[get_comp_func(comp_type)(values, data_threshold)]

    return spikes
