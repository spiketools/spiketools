"""Utilities for working with simulated spiking data."""

import numpy as np

###################################################################################################
###################################################################################################

def refractory(spikes, refractory, fs):
    """Apply a refractory period to a simulated spike train.

    Parameters
    ----------
    spikes : 1d arrayy
        xx
    refractory : float
        xx
    fs : float
        xx

    Returns
    -------
    spikes : 1d array
        Spike train, with refractory period constraint applied.
    """

    ref_len = int(refractory / fs)

    for ind in range(spikes.shape[0]):
        if spikes[ind]:
            spikes[ind+1:ind+ref_len] = 0

    return spikes
