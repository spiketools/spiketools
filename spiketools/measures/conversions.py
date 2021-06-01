"""Functions to convert spiking data."""

import numpy as np

###################################################################################################
###################################################################################################

def create_spike_train(spike_times):
    """Convert spike times into a binary spike train.

    Parameters
    ----------
    spike_times : 1d array
        Spike times, in seconds.

    Returns
    -------
    spike_train : 1d array
        Spike train.
    """

    spike_train = np.zeros(np.ceil(spike_times[-1]).astype(int))
    inds = [int(ind) for ind in spike_times if ind < spike_train.shape[-1]]
    spike_train[inds] = 1

    return spike_train
