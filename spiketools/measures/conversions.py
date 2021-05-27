"""Functions to convert spiking data."""

import numpy as np

###################################################################################################
###################################################################################################

def create_spike_train(spike_times):
    """Convert spike times into a binary spike train."""

    spike_train = np.zeros(np.ceil(spike_times[-1]).astype(int))
    inds = [int(ind) for ind in spike_times if ind < spike_train.shape[-1]]
    spike_train[inds] = 1

    return spike_train
