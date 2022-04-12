"""Utilities for working with simulated spiking data."""

import numpy as np

###################################################################################################
###################################################################################################

def refractory(spike_train, refractory_time, fs=1000):
    """Apply a refractory period to a simulated spike train.

    Parameters
    ----------
    spike_train : 1d array
        Spike train.
    refractory_time : float
        The duration of the refractory period, after a spike, in seconds.
    fs : float, optional, default: 1000
        The sampling rate of the spike train.

    Returns
    -------
    spike_train : 1d array
        Spike train, with refractory period constraint applied.

    Examples
    --------
    Apply a 0.003 seconds refractory period to a binary spike train with 1000 Hz sampling rate.

    >>> spike_train = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0])
    >>> refractory(spike_train, 0.003, 1000)
    array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    """

    ref_len = int(refractory_time * fs)

    for ind in range(spike_train.shape[0]):
        if spike_train[ind]:
            spike_train[ind+1:ind+ref_len] = 0

    return spike_train
