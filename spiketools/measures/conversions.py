"""Functions to convert spiking data."""

import numpy as np

###################################################################################################
###################################################################################################

def create_spike_train(spikes):
    """Convert spike times into a binary spike train.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.

    Returns
    -------
    spike_train : 1d array
        Spike train.
    """

    spike_train = np.zeros(np.ceil(spikes[-1]).astype(int))

    inds = [int(ind) for ind in spikes if ind < spike_train.shape[-1]]

    spike_train[inds] = 1

    return spike_train
