"""Functions to compute measures of spiking activity."""

import numpy as np

from spiketools.utils.data import restrict_range

###################################################################################################
###################################################################################################

def compute_spike_rate(spikes, start_time=None, stop_time=None):
    """Estimate spike rate from a vector of spike times, in seconds.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    start_time, stop_time : float, optional
        Start and stop time of the range to compute the firing rate over.

    Returns
    -------
    float
        Average firing rate.

    Examples
    --------
    Compute spike rate of 6 spikes

    >>> spikes = [0.5, 1, 1.5, 2, 2.5, 3]
    >>> compute_spike_rate(spikes)
    2.4
    """

    start_time = spikes[0] if not start_time else start_time
    stop_time = spikes[-1] if not stop_time else stop_time

    # If there are spikes before or after requested start / stop time, restrict range
    if np.any(np.array(spikes) < start_time) or np.any(np.array(spikes) > stop_time):
        spikes = restrict_range(spikes, start_time, stop_time)

    return len(spikes) / (stop_time - start_time)


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
    Compute inter-spike intervals of 6 spikes

    >>> spikes = [0.5, 0.8, 1.4, 2, 2.2, 2.9]
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
    Compute the coefficient of variation of 6 interval-spike intervals

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
    Compute the fano factor of a spike train with 6 time points

    >>> spike_train = [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0]
    >>> compute_fano_factor(spike_train)
    0.5
    """

    return np.var(spike_train) / np.mean(spike_train)
