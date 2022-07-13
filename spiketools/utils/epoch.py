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
        Spike times.
    segments : list or 1d array of float
        Time values that define the segments.
        Each segment time is defined as the interval between segment[n] and segment[n+1].

    Returns
    -------
    segment_spikes : list of 1d array
        Spike data per segment.
    """

    segment_spikes = [None] * (len(segments) - 1)
    for ind, (seg_start, seg_end) in enumerate(zip(segments, segments[1:])):
        segment_spikes[ind] = get_range(spikes, seg_start, seg_end)

    return segment_spikes


def epoch_data_by_time(timestamps, values, timepoints, threshold=np.inf):
    """Epoch data into trials, based on individual timepoints of interest.

    Parameters
    ----------
    timestamps : 1d array
        Timestamps.
    values : 1d array
        Data values.
    timepoint : list of float
        The time value to extract per trial.
    threshold : float
        The threshold that the closest time value must be within to be returned.
        If the temporal distance is greater than the threshold, output is NaN.

    Returns
    -------
    trials : list of float
        Selected data points across trial.
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
    """

    segment_times = [None] * (len(segments) - 1)
    segment_values = [None] * (len(segments) - 1)
    for ind, (seg_start, seg_end) in enumerate(zip(segments, segments[1:])):
        segment_times[ind], segment_values[ind] = get_values_by_time_range(\
            timestamps, values, seg_start, seg_end)

    return segment_times, segment_values
