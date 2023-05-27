"""Utility functions for extracting data segments of interest."""

import numpy as np

from spiketools.utils.checks import check_axis, check_param_type
from spiketools.utils.options import get_comp_func

###################################################################################################
###################################################################################################

def create_mask(data, min_value=None, max_value=None):
    """Create a mask to select data points based on value.

    Parameters
    ----------
    data : 1d array
        Array of data.
    min_value, max_value : float, optional
        Minimum and/or maximum value to extract from the input array.
        The minimum value is inclusive, but the maximum value is exclusive.

    Returns
    -------
    mask : 1d array of bool
        Mask to select data points from the input array.

    Examples
    --------
    Create a mask to select data within a given value range:

    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> create_mask(data, min_value=2.5, max_value=6.5)
    array([False, False,  True,  True,  True,  True, False, False])
    """

    min_value = -np.inf if min_value is None else min_value
    max_value = np.inf if max_value is None else max_value

    # Make the mask inclusive for min / exclusive for max value
    #   If inclusive for both, there can be issues double-selecting spikes
    #   For example, if a spike ==  time_range, and select contiguous segments
    mask = np.logical_and(data >= min_value, data < max_value)

    return mask


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
        Data array, restricted to desired range.

    Examples
    --------
    Get all values greater than a specific value:

    >>> data = np.array([5, 10, 15, 20, 25, 30])
    >>> get_range(data, min_value=10, max_value=None)
    array([10, 15, 20, 25, 30])

    Get all values less than a specific value:

    >>> data = np.array([5, 10, 15, 20, 25, 30])
    >>> get_range(data, min_value=None, max_value=22.5)
    array([ 5, 10, 15, 20])

    Get a specified range from a data array:

    >>> data = np.array([5, 10, 15, 20, 25, 30])
    >>> get_range(data, min_value=10, max_value=22.5)
    array([10, 15, 20])
    """

    mask = create_mask(data, min_value, max_value)

    data = data[mask]

    if reset:
        data = data - reset

    return data


def get_value_range(timestamps, data, min_value=None, max_value=None, reset=None):
    """Extract data and associated timestamps for a desired value range.

    Parameters
    ----------
    timestamps : 1d array
        Timestamps, in seconds.
    data : 1d array
        Data values, corresponding to the timestamps.
    min_value, max_value : float, optional
        Minimum and/or maximum value to extract from the input array.
        The minimum value is inclusive, but the maximum value is exclusive.
    reset : float, optional
        If provided, resets the time values by the given reset value.

    Returns
    -------
    timestamps : 1d array
        Timestamps, in seconds, selected by value.
    data : 1d array
        Array of data, selected by value.

    Examples
    --------
    Extract data and corresponding timestamps based on value range:

    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> timestamps = np.array([0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.2, 1.5])
    >>> get_value_range(timestamps, data, min_value=2.5, max_value=6.5)
    (array([0.4, 0.5, 0.7, 0.9]), array([3, 4, 5, 6]))
    """

    mask = create_mask(data, min_value, max_value)

    timestamps, data = timestamps[mask], data[mask]

    if reset:
        timestamps = timestamps - reset

    return timestamps, data


def get_ind_by_value(values, value, threshold=None):
    """Get the index for a set of values closest to a specified value.

    Parameters
    ----------
    values : 1d array
        Values.
    value : float
        The value to extract the index for.
    threshold : float, optional
        The threshold that the closest value must be within to be returned.
        If the distance is greater than the threshold, output is -1.

    Returns
    -------
    ind : int
        The index value for the requested value, or -1 if out of threshold range.

    Examples
    --------
    Get the index for a specified value:

    >>> values = np.array([5, 10, 15, 20, 25])
    >>> get_ind_by_value(values, 12)
    1
    """

    check_param_type(value, 'value', (int, float, np.int64, np.float64))
    assert not np.isnan(value), "The given `value` is nan - cannot continue."

    ind = np.abs(values - value).argmin()

    if threshold:
        if np.abs(values[ind] - value) > threshold:
            ind = -1

    return ind


def get_ind_by_time(timestamps, timepoint, threshold=None):
    """Get the index for a set of timepoints closest to a specified timepoint.

    Parameters
    ----------
    timestamps : 1d array
        Timestamps, in seconds.
    timepoint : float
        The time value to extract the index for.
    threshold : float, optional
        The threshold that the closest time value must be within to be returned.
        If the temporal distance is greater than the threshold, output is -1.

    Returns
    -------
    ind : int
        The index value for the requested timepoint, or -1 if out of threshold range.

    Examples
    --------
    Get the index for a specified timepoint:

    >>> timestamps = np.array([0.5, 1, 1.5, 2, 2.5, 3 ])
    >>> get_ind_by_time(timestamps, 2.5)
    4
    """

    return get_ind_by_value(timestamps, timepoint, threshold)


def get_inds_by_values(values, select, threshold=None, drop_null=True):
    """Get indices for a set of specified time points.

    Parameters
    ----------
    values : 1d array
        Values to select from.
    select : 1d array
        The values to extract indices for.
    threshold : float, optional
        The threshold that the closest value must be within to be returned.
        If the distance is greater than the threshold, output is NaN.
    drop_null : bool, optional, default: True
        Whether to drop any null indices from the outputs (outside threshold range).
        If False, indices for any null values are -1.

    Returns
    -------
    inds : 1d array
        Indices for all requested values.

    Examples
    --------
    Get the corresponding indices for specified values:

    >>> values = np.array([10, 15, 20, 25, 30])
    >>> select = np.array([11, 21])
    >>> get_inds_by_values(values, select)
    array([0, 2])
    """

    inds = np.zeros(len(select), dtype=int)
    for ind, value in enumerate(select):
        inds[ind] = get_ind_by_value(values, value, threshold=threshold)

    if drop_null:
        inds = inds[inds >= 0]

    return inds


def get_inds_by_times(timestamps, timepoints, threshold=None, drop_null=True):
    """Get indices for a set of specified time points.

    Parameters
    ----------
    timestamps : 1d array
        Timestamps, in seconds.
    timepoints : 1d array
        The time values, in seconds, to extract indices for.
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

    Examples
    --------
    Get the corresponding indices for specified timepoints:

    >>> timestamps = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    >>> timepoints = np.array([1, 2, 3])
    >>> get_inds_by_times(timestamps, timepoints)
    array([1, 3, 5])
    """

    return get_inds_by_values(timestamps, timepoints, threshold, drop_null)


def get_value_by_time(timestamps, values, timepoint, threshold=None, axis=None):
    """Get the value from a data array at a specific time point.

    Parameters
    ----------
    timestamps : 1d array
        Timestamps, in seconds.
    values : 1d or 2d array
        Data values, corresponding to the time values in `timestamps`.
    timepoint : float
        Time value to extract.
    threshold : float, optional
        The threshold that the closest time value must be within to be returned.
        If the temporal distance is greater than the threshold, output is NaN.
    axis : {0, 1}, optional
        The axis argument for the `values` data, if it's a 2d array, as {0: column, 1: row}.
        If not provided, is inferred from the `values` array.

    Returns
    -------
    out : float or 1d array
        The value(s) at the requested time point.

    Examples
    --------
    Get the data values for a specified timepoint:

    >>> timestamps = np.array([0.5, 1, 1.5, 2, 2.5, 3 ])
    >>> values = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    >>> get_value_by_time(timestamps, values, 1.5)
    array([3, 9])
    """

    idx = get_ind_by_time(timestamps, timepoint, threshold=threshold)
    out = values.take(indices=idx, axis=check_axis(axis, values)) if idx >= 0 else np.nan

    return out


def get_values_by_times(timestamps, values, timepoints, threshold=None, drop_null=True, axis=None):
    """Get values from a data array for a set of specified time points.

    Parameters
    ----------
    timestamps : 1d array
        Timestamps, in seconds.
    values : 1d or 2d array
        Data values, corresponding to the time values in `timestamps`.
    timepoints : 1d array
        The time values, in seconds, to extract corresponding values for.
    threshold : float, optional
        The threshold that the closest time value must be within to be returned.
        If the temporal distance is greater than the threshold, output is NaN.
    drop_null : bool, optional, default: True
        Whether to drop any null values from the outputs (outside threshold range).
        If False, outputs for any null values are NaN.
    axis : {0, 1}, optional
        The axis argument for the `values` data, if it's a 2d array, as {0: column, 1: row}.
        If not provided, is inferred from the `values` array.

    Returns
    -------
    outputs : 1d or 2d array
        The extracted values for the requested time points.

    Examples
    --------
    Get the data values for a specified set of timepoints:

    >>> timestamps = np.array([0.5, 1, 1.5, 2, 2.5, 3 ])
    >>> values = np.array([1, 2, 3, 4, 5, 6])
    >>> timepoints = np.array([1, 2, 3])
    >>> get_values_by_times(timestamps, values, timepoints)
    array([2, 4, 6])
    """

    inds = get_inds_by_times(timestamps, timepoints, threshold, drop_null)

    if drop_null:
        outputs = values.take(indices=inds, axis=check_axis(axis, values))
    else:
        outputs = np.full([np.atleast_2d(values).shape[0], len(timepoints)], np.nan)
        mask = inds >= 0
        outputs[:, np.where(mask)[0]] = values.take(indices=inds[mask],
                                                    axis=check_axis(axis, values))
        outputs = np.squeeze(outputs)

    return outputs


def get_values_by_time_range(timestamps, values, t_min, t_max, axis=None):
    """Extract data for a requested time range.

    Parameters
    ----------
    timestamps : 1d array
        Timestamps, in seconds.
    values : ndarray
        Data values, corresponding to the time values in `timestamps`.
    t_min, t_max : float
        Time range to extract.
    axis : {0, 1}, optional
        The axis argument for the `values` data, if it's a 2d array, as {0: column, 1: row}.
        If not provided, is inferred from the `values` array.

    Returns
    -------
    timestamps : 1d array
        Selected timestamp values.
    out : ndarray
        Selected data values.

    Examples
    --------
    Extract data within a specified time range:

    >>> timestamps = np.array([1, 2, 3, 4, 5, 6, 7])
    >>> values = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5])
    >>> get_values_by_time_range(timestamps, values, t_min=2, t_max=6)
    (array([2, 3, 4, 5, 6]), array([1. , 1.5, 2. , 2.5, 3. ]))
    """

    select = np.logical_and(timestamps >= t_min, timestamps <= t_max)
    out = values.take(indices=np.where(select)[0], axis=check_axis(axis, values))

    return timestamps[select], out


def threshold_spikes_by_times(spikes, timestamps, threshold):
    """Threshold spikes by sub-selecting those that are temporally close to a set of time values.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    timestamps : 1d array
        Timestamps, in seconds.
    threshold : float
        The threshold value for the time between the spike and a timestamp for the spike to be kept.
        Any spikes further in time from a timestamp value are dropped.

    Returns
    -------
    spikes : 1d array
        Sub-selected spike times, in seconds.

    Examples
    --------
    Extract spikes based on their proximity to timestamps:

    >>> spikes = np.array([0.76, 1.12, 1.72, 2.05, 2.32, 2.92, 3.11, 3.63, 3.91])
    >>> timestamps = np.array([1.0, 1.25, 1.5, 1.75, 2.0, 3.5, 3.75, 4.0])
    >>> threshold_spikes_by_times(spikes, timestamps, threshold=0.25)
    array([0.76, 1.12, 1.72, 2.05, 3.63, 3.91])
    """

    mask = np.empty_like(spikes, dtype=bool)
    for ind, spike in enumerate(spikes):
        mask[ind] = np.min(np.abs(timestamps - spike)) < threshold

    return spikes[mask]


def threshold_spikes_by_values(spikes, timestamps, values, data_threshold,
                               time_threshold=None, data_comparison='greater'):
    """Threshold spikes by sub-selecting based on thresholding values on another data stream.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    timestamps : 1d array
        Timestamps, in seconds.
    values : 1d array
        Data values, corresponding to the timestamps.
    data_threshold : float
        The threshold for the data, used to select spikes based on data values
    time_threshold : float, optional
        The threshold value for the time between the spike and a timestamp for the spike to be kept.
        Any spikes further in time from a timestamp value are dropped.
    data_comparison : {'greater', 'less'}
        Which comparison function to use for the data threshold.
        This defines whether selected values must be greater than or less than the data threshold.

    Returns
    -------
    spikes : 1d array
        Sub-selected spike times, in seconds.

    Examples
    --------
    Threshold spikes based on a minimum data threshold:

    >>> spikes = np.array([0.1, 0.3, 0.4, 0.5, 1, 1.2, 1.6, 1.8])
    >>> timestamps = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
    >>> values = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> threshold_spikes_by_values(spikes, timestamps, values, 2)
    array([1.6, 1.8])
    """

    values = get_values_by_times(timestamps, values, spikes, time_threshold, drop_null=False)

    mask = ~np.isnan(values)
    spikes, values = spikes[mask], values[mask]
    spikes = spikes[get_comp_func(data_comparison)(values, data_threshold)]

    return spikes


def drop_range(spikes, time_range, check_empty=True):
    """Drop a specified time range from a vector spike times.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    time_range : list of [float, float] or list of list of [float, float]
        Time range(s), in seconds, to drop from spike times.
        Each time range should be defined as [start_add_time, end_add_time].
    check_empty : bool, optional, default: True
        Whether to check if the dropped range is empty of spikes.
        If True, and there are spikes within the drop `time_range`, an error is raised.

    Returns
    -------
    out_spikes : 1d array
        Spike times, in seconds, with the time range removed.

    Examples
    --------
    Drop an empty time range from a set of spike times:

    >>> spikes = np.array([0.24, 0.73, 1.22, 1.65, 10.15, 10.95, 11.52, 11.84])
    >>> time_range = [2, 10]
    >>> drop_range(spikes, time_range)
    array([0.24, 0.73, 1.22, 1.65, 2.15, 2.95, 3.52, 3.84])
    """

    # Check for time range is not empty (if is empty list, do nothing)
    if time_range:

        # Operate on a copy of the input, to not overwrite original array
        spikes = spikes.copy()

        total_len = 0
        for trange in np.array(time_range, ndmin=2):

            if total_len > 0:
                trange = [trange[0] - total_len, trange[1] - total_len]

            if check_empty:
                assert get_range(spikes, *trange).size == 0, \
                    "Extracted range {} is not empty.".format(trange)

            tlen = trange[1] - trange[0]
            spikes = np.hstack([get_range(spikes, max_value=trange[0]),
                                get_range(spikes, min_value=trange[1], reset=tlen)])

            total_len += trange[1] - trange[0]

    return spikes


def _reinstate_range_1d(spikes, time_range):
    """Reinstate a dropped time range into a vector of spike times.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    time_range : list of [float, float]
        Time range, in seconds, to reinstate into spike times, as [start_add_time, end_add_time].

    Returns
    -------
    out_spikes : 1d array
        Spike times, in seconds, with the time range reinstated.
    """

    tlen = time_range[1] - time_range[0]
    out_spikes = np.hstack([get_range(spikes, max_value=time_range[0]),
                            get_range(spikes, min_value=time_range[0]) + tlen])

    return out_spikes


def reinstate_range(spikes, time_range):
    """Reinstate a dropped time range into an array of spike times.

    Parameters
    ----------
    spikes : 1d or 2d array
        An array of spikes times, in seconds.
    time_range : list of [float, float] or list of list of [float, float]
        Time range(s), in seconds, to reinstate into shuffled spike times.
        Each time range should be defined as [start_add_time, end_add_time].

    Returns
    -------
    spikes_out : 1d or 2d array
        An array of spikes times, in seconds, with the time range reinstated.

    Examples
    --------
    Reinstate a time range into a set of spike times:

    >>> spikes = np.array([0.24, 0.73, 1.22, 1.65, 2.15, 2.95, 3.52, 3.84])
    >>> time_range = [2, 10]
    >>> reinstate_range(spikes, time_range)
    array([ 0.24,  0.73,  1.22,  1.65, 10.15, 10.95, 11.52, 11.84])
    """

    assert spikes.ndim < 3, 'The reinstate_range function only supports 1d or 2d arrays.'

    # Operate on a copy of the input, to not overwrite original array
    spikes = spikes.copy()

    # By enforcing 2d (and later squeezing), the loops works for both 1d & 2d arrays
    spikes = np.atleast_2d(spikes)
    for trange in np.array(time_range, ndmin=2):
        for ind, spikes_1d in enumerate(spikes):
            spikes[ind, :] = _reinstate_range_1d(spikes_1d, trange)

    spikes = np.squeeze(spikes)

    return spikes
