"""Functions to compute measures of interest."""

import numpy as np

###################################################################################################
###################################################################################################

def compute_spike_rate(spikes):
    """Estimate spike rate from a vector of spike times, in seconds.

    Parameters
    ----------
    spikes : 1d array
        Spike times.

    Returns
    -------
    float
        Average firing rate.
    """

    return len(spikes) / (spikes[-1] - spikes[0])


def compute_isis(spikes):
    """Compute inter-spike intervals.

    Parameters
    ----------
    spikes : 1d array
        Spike times.

    Returns
    -------
    isis : 1d array
        Distribution of interspike intervals.
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
    """

    return np.var(spike_train) / np.mean(spike_train)
