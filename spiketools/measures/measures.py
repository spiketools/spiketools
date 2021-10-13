"""Functions to compute measures of spiking activity."""

import numpy as np

###################################################################################################
###################################################################################################

def compute_spike_rate(spikes):
    """Estimate spike rate from a vector of spike times, in seconds.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.

    Returns
    -------
    float
        Average firing rate.
        
    Example
    -------
    Compute spike rate of 6 spikes 
    
    >>> spikes = [0.5, 1, 1.5, 2, 2.5, 3]
    >>> compute_spike_rate(spikes)
    2.4
    """

    return len(spikes) / (spikes[-1] - spikes[0])


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
        
    Example
    -------
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
    
    Example
    -------
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
    
    Example
    -------
    Compute the fano factor of a spike train with 6 time points 
    
    >>> spike_train = [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0]
    >>> compute_fano_factor(spike_train)
    0.5
    """

    return np.var(spike_train) / np.mean(spike_train)
