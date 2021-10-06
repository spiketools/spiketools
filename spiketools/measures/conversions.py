"""Functions to convert spiking data."""

import numpy as np

###################################################################################################
###################################################################################################

def create_spike_train(spikes):
    """Convert spike times into a binary spike train.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in milliseconds.

    Returns
    -------
    spike_train : 1d array
        Spike train.
    """

    spike_train = np.zeros(np.ceil(spikes[-1]).astype(int))

    inds = [int(ind) for ind in spikes if ind < spike_train.shape[-1]]

    spike_train[inds] = 1

    return spike_train


def convert_train_to_times(train):
    """Convert a spike train representation into spike times, in milliseconds.

    Parameters
    ----------
    train : 1d array
        Spike train.

    Returns
    -------
    spikes : 1d array
        Spike times, in milliseconds.
    """

    spikes = np.where(train)[0]

    return spikes


def convert_isis_to_spikes(isis, offset=0, add_offset=True):
    """Convert a sequence of inter-spike intervals to spike times.

    Parameters
    ----------
    isis : 1d array
        xx
    offset : float
        An offset value to add to generated spike times.
    add_offset : bool, optional, default: True
        Whether to prepend the offset value to the beginning of the spike times.

    Returns
    -------
    spikes : 1d array
        Spike times, in milliseconds.
    """

    spikes = np.cumsum(isis, axis=-1)

    if offset:
        spikes = spikes + offset
    if add_offset:
        spikes = np.concatenate((np.array([offset]), spikes))

    return spikes
