"""Utility functions for managing data."""

import numpy as np

###################################################################################################
###################################################################################################

def restrict_range(values, min_time=None, max_time=None):
    """Restrict a vector of time values to a specified range.

    Parameters
    ----------
    values : 1d array
        Array of time values.
    min_time, max_time : float, optional, default: None
        Mininum and/or maximum time to restrict time values to.

    Returns
    -------
    1d array
        Time values, restricted to desired time range.

    Examples
    --------
    Select all values after a specific time point:

    >>> values = np.array([5, 10, 15, 20, 25, 30])
    >>> restrict_range(values, min_time=10, max_time=None)
    array([10, 15, 20, 25, 30])

    Select all values before a specific time point:

    >>> values = np.array([5, 10, 15, 20, 25, 30])
    >>> restrict_range(values, min_time=None, max_time=25)
    array([ 5, 10, 15, 20, 25])

    Restrict a time values to a specific range:

    >>> values = np.array([5, 10, 15, 20, 25, 30])
    >>> restrict_range(values, min_time=10, max_time=20)
    array([10, 15, 20])
    """

    min_time = -np.inf if min_time is None else min_time
    max_time = np.inf if max_time is None else max_time

    return values[(values >= min_time) & (values <= max_time)]


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


def epoch_trials(spikes, events, window):
    """Epoch spiking data into trials.

    Parameters
    ----------
    spikes : 1d array
        Spike times.
    events : 1d array
        The set of event times to extract from the data.
    window : list of [float, float]
        The time window to extract around each event.

    Returns
    -------
    trials : list of 1d array
        Spike

    Notes
    -----
    For each trial, the returned spike times will be relative to each event time, set as zero.
    """

    trials = []
    for event in events:
        trial = restrict_range(spikes, event + window[0], event + window[1])
        trials.append(trial - event)

    return trials
