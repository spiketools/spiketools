"""Functions to compute measures of spiking activity."""

import numpy as np

from spiketools.utils.extract import get_range
from spiketools.measures.conversions import convert_times_to_counts

###################################################################################################
###################################################################################################

def compute_firing_rate(spikes, start_time=None, stop_time=None):
    """Estimate firing rate from a vector of spike times, in seconds.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    start_time, stop_time : float, optional
        Start and stop time of the range to compute the firing rate over.

    Returns
    -------
    fr : float
        Average firing rate.

    Examples
    --------
    Compute spike rate from spike times:

    >>> spikes = np.array([0.5, 1, 1.5, 2, 2.5, 3])
    >>> compute_firing_rate(spikes)
    2.4
    """

    if start_time or stop_time:
        spikes = get_range(spikes, start_time, stop_time)

    start_time = spikes[0] if start_time is None else start_time
    stop_time = spikes[-1] if stop_time is None else stop_time

    fr = len(spikes) / (stop_time - start_time)

    return fr


def compute_isis(spikes):
    """Compute inter-spike intervals.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.

    Returns
    -------
    isis : 1d array
        Distribution of interspike intervals.

    Examples
    --------
    Compute inter-spike intervals from spike times:

    >>> spikes = np.array([0.5, 0.8, 1.4, 2, 2.2, 2.9])
    >>> compute_isis(spikes)
    array([0.3, 0.6, 0.6, 0.2, 0.7])
    """

    return np.diff(spikes)


def compute_cv(isis):
    """Compute coefficient of variation.

    Parameters
    ----------
    isis : 1d array
        Interspike intervals.

    Returns
    -------
    cv : float
        Coefficient of variation.

    Examples
    --------
    Compute the coefficient of variation from interval-spike intervals:

    >>> isis = [0.3, 0.6, 0.6, 0.2, 0.7]
    >>> compute_cv(isis)
    0.4039733214513607
    """

    return np.std(isis) / np.mean(isis)


def compute_fano_factor(spike_train):
    """Compute the fano factor of a spike train.

    Parameters
    ----------
    spike_train : 1d array
        Spike train.

    Returns
    -------
    fano : float
        Fano factor.

    Examples
    --------
    Compute the fano factor from a spike train:

    >>> spike_train = [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0]
    >>> compute_fano_factor(spike_train)
    0.5
    """

    return np.var(spike_train) / np.mean(spike_train)


def compute_spike_presence(spikes, bins, time_range=None):
    """Compute the spike presence across time bins.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    bins : float or 1d array
        The binning to apply to the spiking data.
        If float, the length of each bin.
        If array, precomputed bin definitions.
    time_range : list of [float, float], optional
        Time range, in seconds, to create the binned firing rate across.
        Only used if `bins` is a float.

    Returns
    -------
    spike_presence : 1d array
        Boolean array indicating spike presence across time bins.
    """

    spike_counts = convert_times_to_counts(spikes, bins, time_range)
    spike_presence = spike_counts != 0

    return spike_presence


def compute_presence_ratio(spikes, bins, time_range=None):
    """Compute the presence ratio for a set of spike times.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    bins : float or 1d array
        The binning to apply to the spiking data.
        If float, the length of each bin.
        If array, precomputed bin definitions.
    time_range : list of [float, float], optional
        Time range, in seconds, to create the binned firing rate across.
        Only used if `bins` is a float.

    Returns
    -------
    presence_ratio : float
        The computed presence ratio.

    Notes
    -----
    The presence ratio reflects the proportion of time bins in which at least 1 spike occurred.
    """

    spike_presence = compute_spike_presence(spikes, bins, time_range)
    presence_ratio = sum(spike_presence) / len(spike_presence)

    return presence_ratio
