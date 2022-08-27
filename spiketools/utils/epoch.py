"""Utilities for epoching data."""

import numpy as np

from spiketools.utils.extract import get_range, get_value_by_time, get_values_by_time_range

###################################################################################################
###################################################################################################

def epoch_spikes_by_event(spikes, events, window):
    """Epoch spiking data into trials, based on events of interest.

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
        Spike data per trial.

    Notes
    -----
    For each trial, the returned spike times will be relative to each event time, set as zero.

    Examples
    --------
    Epoch an array of spiking data based on the event window of [0.1, 0.2]:

    >>> spikes = np.array([0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 1, 1.4])
    >>> events = np.array([0.2, 0.8, 1.2])
    >>> window = [0.1, 0.2]
    >>> epoch_spikes_by_event(spikes, events, window)
    [array([0.2]), array([0.2]), array([0.2])]
    """

    trials = [None] * len(events)
    for ind, event in enumerate(events):
        trials[ind] = get_range(spikes, event + window[0], event + window[1]) - event

    return trials


def epoch_spikes_by_range(spikes, starts, stops, reset=False):
    """Epoch spiking data into trials, based on time ranges of interest.

    Parameters
    ----------
    spikes : 1d array
        Spike times.
    starts : list
        The start times for each epoch to extract.
    stops : list
        The stop times of each epoch to extract.
    reset : bool, optional, default: False
        Whether to reset each set of trial timestamps to start at zero.

    Returns
    -------
    trials : list of 1d array
        Spike data per trial.

    Examples
    --------
    Epoch an array of spiking data into trials and reset the starting timestamps of each trial to zero:

    >>> spikes = np.array([0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 1, 1.4])
    >>> starts, stops = [0.05, 0.45, 0.8], [0.42, 0.73, 1.5]
    >>> epoch_spikes_by_range(spikes, starts, stops, reset=True)
    [array([0.05, 0.25, 0.35]), array([0.05, 0.15, 0.25]), array([0.2, 0.6])]
    """

    trials = [None] * len(starts)
    for ind, (start, stop) in enumerate(zip(starts, stops)):
        trial = get_range(spikes, start, stop)
        if reset:
            trial = trial - start
        trials[ind] = trial

    return trials


def epoch_spikes_by_segment(spikes, segments):
    """Epoch spikes by segments.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    segments : list or 1d array of float
        Time values that define the segments.
        Each segment time is defined as the interval between segment[n] and segment[n+1].

    Returns
    -------
    segment_spikes : list of 1d array
        Spike data per segment.

    Examples
    --------
    Epoch spiking data into 4 pre-defined segements:

    >>> spikes = np.array([0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 1, 1.4])
    >>> segments = [0, 0.35, 0.55, 0.8, 1.5]
    >>> epoch_spikes_by_segment(spikes, segments)
    [array([0.1, 0.3]), array([0.4, 0.5]), array([0.6, 0.7]), array([1. , 1.4])]
    """

    segment_spikes = [None] * (len(segments) - 1)
    for ind, (seg_start, seg_end) in enumerate(zip(segments, segments[1:])):
        segment_spikes[ind] = get_range(spikes, seg_start, seg_end)

    return segment_spikes


def epoch_data_by_time(timestamps, values, timepoints, threshold=None):
    """Epoch data into trials, based on individual timepoints of interest.

    Parameters
    ----------
    timestamps : 1d array
        Timestamps.
    values : 1d array
        Data values.
    timepoints : list of float
        The time value to extract per trial.
    threshold : float, optional
        The threshold that the closest time value must be within to be returned.
        If the temporal distance is greater than the threshold, output is NaN.

    Returns
    -------
    trials : list of float
        Selected data points across trial.

    Examples
    --------
    Epoch data values at 3 individual timepoints:

    >>> timestapms = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
    >>> values = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
    >>> timepoints = [0.3, 0.7, 1.3]
    >>> epoch_data_by_time(timestapms, values, timepoints)
    [1.5, 2.5, 4.0]
    """

    trials = [None] * len(timepoints)
    for ind, timepoint in enumerate(timepoints):
        trials[ind] = get_value_by_time(timestamps, values, timepoint, threshold=threshold)

    return trials


def epoch_data_by_event(timestamps, values, events, window):
    """Epoch data into trials, based on events of interest.

    Parameters
    ----------
    timestamps : 1d array
        Timestamps.
    values : 1d array
        Data values.
    events : 1d array
        The set of event times to extract from the data.
    window : list of [float, float]
        The time window to extract around each event.

    Returns
    -------
    trial_times : list of 1d array
        The timestamps, per trial.
    trial_values : list of 1d array
        The values, per trial.

    Examples
    --------
    Epoch data into trials based on the event window of [0.1, 0.2]:

    >>> timestamps = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    >>> values = np.array([1, 1.5, 2, 2.5, 3])
    >>> events = np.array([0.3, 0.6])
    >>> window = [0.1, 0.2]
    >>> epoch_data_by_event(timestamps, values, events, window)
    ([array([0.2]), array([0.1])], [array([2.]), array([2.5])])
    """

    trial_times = [None] * len(events)
    trial_values = [None] * len(events)
    for ind, event in enumerate(events):
        ttimes, tvalues = get_values_by_time_range(\
            timestamps, values, event + window[0], event + window[1])
        trial_times[ind] = ttimes - event
        trial_values[ind] = tvalues

    return trial_times, trial_values


def epoch_data_by_range(timestamps, values, starts, stops, reset=False):
    """Epoch data into trials, based on time ranges of interest.

    Parameters
    ----------
    timestamps : 1d array
        Timestamps.
    values : 1d array
        Data values.
    starts : list of float
        The start times for each epoch to extract.
    stops : list of float
        The stop times of each epoch to extract.
    reset : bool, optional, default: True
        If True, resets the values in each epoch range to the start time of that epoch.

    Returns
    -------
    trial_times : list of 1d array
        The timestamps, per trial.
    trial_values : list of 1d array
        The values, per trial.

    Examples
    --------
    Epoch data values into trials and reset the starting timestamps of each trial to zero:

    >>> timestamps = np.array([0.1, 0.3, 0.4, 0.5, 0.6])
    >>> values = np.array([1, 2, 3, 4, 5])
    >>> starts, stops = [0.2, 0.5], [0.4, 0.6]
    >>> epoch_data_by_range(timestamps, values, starts, stops, reset=True)
    ([array([0.1, 0.2]), array([0. , 0.1])], [array([2, 3]), array([4, 5])])
    """

    trial_times = [None] * len(starts)
    trial_values = [None] * len(starts)
    for ind, (start, stop) in enumerate(zip(starts, stops)):
        ttimes, tvalues = get_values_by_time_range(timestamps, values, start, stop)
        if reset:
            ttimes = ttimes - start
        trial_times[ind] = ttimes
        trial_values[ind] = tvalues

    return trial_times, trial_values


def epoch_data_by_segment(timestamps, values, segments):
    """Epoch data by segments.

    Parameters
    ----------
    timestamps : 1d array
        Timestamps.
    values : 1d array
        Data values.
    segments : list or 1d array of float
        Time values that define the segments.
        Each segment time is defined as the interval between segment[n] and segment[n+1].

    Returns
    -------
    segment_times : list of 1d array
        The timestamps, per segment.
    segment_values : list of 1d array
        The values, per segment.

    Examples
    --------
    Epoch data values into 3 pre-defined segments:

    >>> timestamps = np.array([0.1, 0.4, 0.6, 0.7, 1])
    >>> values = np.array([1, 3, 5, 7, 9])
    >>> segments = [0, 0.35, 0.55, 0.8]
    >>> epoch_data_by_segment(timestamps, values, segments)
    ([array([0.1]), array([0.4]), array([0.6, 0.7])], [array([1]), array([3]), array([5, 7])])
    """

    segment_times = [None] * (len(segments) - 1)
    segment_values = [None] * (len(segments) - 1)
    for ind, (seg_start, seg_end) in enumerate(zip(segments, segments[1:])):
        segment_times[ind], segment_values[ind] = get_values_by_time_range(\
            timestamps, values, seg_start, seg_end)

    return segment_times, segment_values
