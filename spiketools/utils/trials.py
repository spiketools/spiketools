"""Utilities for managing trials and epochs."""

import numpy as np

from spiketools.measures.conversions import convert_times_to_rates
from spiketools.utils.checks import check_time_bins
from spiketools.utils.extract import get_range, get_value_by_time_range

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


def epoch_data_by_event(timestamps, values, events, window):
    """Epoch data with timestamps into trials, based on events of interest.

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
        ttimes, tvalues = get_value_by_time_range(\
            timestamps, values, event + window[0], event + window[1])
        trial_times[ind] = ttimes - event
        trial_values[ind] = tvalues

    return trial_times, trial_values


def epoch_data_by_range(timestamps, values, starts, stops, reset=False):
    """Epoch data with timestamps into trials, based on time ranges of interest.

    Parameters
    ----------
    timestamps : 1d array
        Timestamps.
    values : 1d array
        Data values.
    starts : list
        The start times for each epoch to extract.
    stops : list
        The stop times of each epoch to extract.

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
        ttimes, tvalues = get_value_by_time_range(timestamps, values, start, stop)
        if reset:
            ttimes = ttimes - start
        trial_times[ind] = ttimes
        trial_values[ind] = tvalues

    return trial_times, trial_values


def compute_trial_frs(trial_spikes, bins, trange=None, smooth=None):
    """Compute continuous binned firing rates for a set of epoched spike times.

    Parameters
    ----------
    trial_spikes : list of 1d array
        Spike times per trial.
    bins : float or 1d array
        The binning to apply to the spiking data.
        If float, the length of each bin.
        If array, precomputed bin definitions.
    trange : list of [float, float]
        Time range, in seconds, to create the binned firing rate across.
        Only used if `bins` is a float.
    smooth : float, optional
        If provided, the kernel to use to smooth the continuous firing rate.

    Returns
    -------
    trial_cfrs : 2d array
        Continuous firing rates per trial, with shape [n_trials, n_time_bins].
    """

    bins = check_time_bins(bins, trial_spikes[0], trange=trange)
    trial_cfrs = np.zeros([len(trial_spikes), len(bins) - 1])
    for ind, t_spikes in enumerate(trial_spikes):
        trial_cfrs[ind, :] = convert_times_to_rates(t_spikes, bins, smooth)

    return trial_cfrs
