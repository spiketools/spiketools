"""Functions to compute trial-related measures."""

import numpy as np

from spiketools.utils.extract import get_range
from spiketools.utils.options import get_avg_func
from spiketools.utils.checks import check_time_bins
from spiketools.utils.timestamps import create_bin_times
from spiketools.measures.spikes import compute_firing_rate
from spiketools.measures.conversions import convert_times_to_rates

###################################################################################################
###################################################################################################

def compute_trial_frs(trial_spikes, bins, time_range=None, smooth=None):
    """Compute continuous binned firing rates for a set of epoched spike times.

    Parameters
    ----------
    trial_spikes : list of 1d array
        Spike times per trial, in seconds.
    bins : float or 1d array
        The binning to apply to the spiking data.
        If float, the length of each bin.
        If array, precomputed bin definitions.
    time_range : list of [float, float], optional
        Time range, in seconds, to create the binned firing rate across.
        Only used if `bins` is a float.
    smooth : float, optional
        If provided, the kernel to use to smooth the continuous firing rate.

    Returns
    -------
    bin_times : 1d array
        Timestamps for the computed time bins.
    trial_cfrs : 2d array
        Continuous firing rates per trial, with shape [n_trials, n_time_bins].

    Examples
    --------
    Compute the continuous firing rates across time bins for 3 trials:

    >>> trial_spikes = [np.array([0.005, 0.150, 0.250]),
    ...                 np.array([0.126, 0.275, 0.300, 0.350, 0.600]),
    ...                 np.array([0.250, 0.350, 0.700, 0.900])]
    >>> bins = 0.5
    >>> time_range = [0.0, 1.0]
    >>> bin_times, trial_cfrs = compute_trial_frs(trial_spikes, bins, time_range)
    >>> trial_cfrs
    array([[6., 0.],
           [8., 2.],
           [4., 4.]])
    """

    bins = check_time_bins(bins, time_range)
    bin_times = create_bin_times(bins)

    trial_cfrs = np.zeros([len(trial_spikes), len(bin_times)])
    for ind, t_spikes in enumerate(trial_spikes):
        trial_cfrs[ind, :] = convert_times_to_rates(t_spikes, bins, smooth)

    return bin_times, trial_cfrs


def compute_pre_post_rates(trial_spikes, pre_window, post_window):
    """Compute the firing rates in pre and post event windows.

    Parameters
    ----------
    trial_spikes : list of 1d array
        Spike times per trial.
    pre_window, post_window : list of [float, float]
        The pre and post event time window, in seconds, to compute firing rate across.

    Returns
    -------
    frs_pre, frs_post : 1d array
        Computed pre & post firing rate for each trial.

    Examples
    --------
    Compute the pre & post firing rates for specified time windows across 3 trials:

    >>> trial_spikes = [np.array([-0.150, -0.005, -0.002, 0.250, 0.450, 0.625]),
    ...                 np.array([-0.250, 0.275, 0.290, 0.3, 0.350, 0.6]),
    ...                 np.array([0.550, 0.650, 0.70, 0.9, 0.950])]
    >>> pre_window, post_window = [-0.5, 0.0], [0.5, 0.9]
    >>> compute_pre_post_rates(trial_spikes, pre_window, post_window)
    (array([6., 2., 0.]), array([2.5, 2.5, 7.5]))
    """

    frs_pre = np.array([compute_firing_rate(trial, pre_window) for trial in trial_spikes])
    frs_post = np.array([compute_firing_rate(trial, post_window) for trial in trial_spikes])

    return frs_pre, frs_post


def compute_segment_frs(spikes, segments):
    """Compute firing rate across trial segments.

    Parameters
    ----------
    spikes : 1d array or list of 1d array
        Spike times, in seconds. Can be single array, or list of spike times per trial.
    segments : 2d array
        Time definitions of the segments, per trial, used as time bins.
        Should have shape: [n_trials, n_segments + 1].

    Returns
    -------
    frs : 2d array
        Firing rate per trial, per segment.

    Examples
    --------
    Compute firing rate across segments, with custom segment times per trial:

    >>> spikes = [np.array([0.002, 0.005, 0.120, 0.175, 0.250]),
    ...           np.array([0.275, 0.290, 0.300, 0.350, 0.500]),
    ...           np.array([0.550, 0.650, 0.700, 0.950])]
    >>> segments = np.array([[0.00, 0.15, 0.25],
    ...                      [0.20, 0.40, 0.60],
    ...                      [0.50, 0.70, 0.90]])
    >>> compute_segment_frs(spikes, segments)
    array([[20., 20.],
           [20.,  5.],
           [10.,  5.]])
    """

    if not isinstance(spikes, list):
        spikes = [get_range(spikes, segment[0], segment[-1]) for segment in segments]

    frs = np.zeros([segments.shape[0], segments.shape[1] - 1])
    for ind, (t_spikes, segment) in enumerate(zip(spikes, segments)):
        frs[ind, :] = convert_times_to_rates(t_spikes, segment)

    return frs


def compute_pre_post_averages(frs_pre, frs_post, avg_type='mean'):
    """Compute the average firing rate across pre & post event windows.

    Parameters
    ----------
    frs_pre, frs_post : 1d array
        Firing rates across pre & post event windows.
    avg_type : {'mean', 'median'}
        The type of averaging function to use.

    Returns
    -------
    avg_pre, avg_post : float
        The average firing rates for the pre & post event windows.

    Examples
    --------
    Compute the average pre & post event firing rates:

    >>> frs_pre = np.array([5, 3, 1])
    >>> frs_post = np.array([12, 8, 10])
    >>> compute_pre_post_averages(frs_pre, frs_post, avg_type='mean')
    (3.0, 10.0)
    """

    avg_pre = get_avg_func(avg_type)(frs_pre)
    avg_post = get_avg_func(avg_type)(frs_post)

    return avg_pre, avg_post


def compute_pre_post_diffs(frs_pre, frs_post, average=True, avg_type='mean'):
    """Compute the difference in firing rates between pre & post event windows.

    Parameters
    ----------
    frs_pre, frs_post : 1d array
        Firing rates across pre & post event windows.
    average : bool, optional, default: True
        Whether to average
    avg_type : {'mean', 'median'}
        The type of averaging function to use.

    Returns
    -------
    diffs : float or 1d array
        The difference between firing in pre & post event windows.
        If `average` is True, is a float reflecting the average difference.
        If `average` is False, is an array with trial-by-trial differences.

    Examples
    --------
    Compute the difference between pre & post event firing rates:

    >>> frs_pre = np.array([5, 3, 1])
    >>> frs_post = np.array([12, 8, 10])
    >>> compute_pre_post_diffs(frs_pre, frs_post, average=True, avg_type='mean')
    7.0
    """

    diffs = frs_post - frs_pre

    if average:
        diffs = get_avg_func(avg_type)(diffs)

    return diffs
