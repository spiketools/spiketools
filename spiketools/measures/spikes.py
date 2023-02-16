"""Functions to compute measures of spiking activity."""

import numpy as np

from spiketools.utils.extract import get_range

###################################################################################################
###################################################################################################

def compute_firing_rate(spikes, time_range=None):
    """Estimate firing rate from a vector of spike times, in seconds.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    time_range : list of [float, float], optional
        Time range, in seconds, to calculate the firing rate across.

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

    if time_range:
        spikes = get_range(spikes, time_range[0], time_range[1])

    start_time = spikes[0] if time_range is None else time_range[0]
    stop_time = spikes[-1] if time_range is None else time_range[1]

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
