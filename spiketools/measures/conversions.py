"""Functions to convert spiking data."""

import numpy as np

###################################################################################################
###################################################################################################

def create_spike_train(spikes):
    """Convert spike times into a binary spike train.
    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds or milliseconds.
    Returns
    -------
    spike_train : 1d array
        Spike train.
        
    Example 
    -------
    
    Convert 6 spike times into a corresponding binary spike train 
    >>> spikes = [250, 500, 750, 1000, 1250, 1500]
    >>> create_spike_train(spikes)
    array([0., 0., 0., ..., 0., 0., 0.])
    
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
        Spike train (assumed sampling rate of 1000 Hz).
    Returns
    -------
    spikes : 1d array
        Spike times, in milliseconds.
    
    Example
    -------
    
    Convert a spike train representation (assumed 1000 Hz sampling rate) into spike times, in milliseconds.
    
    >>> spike_train = [0,0,0,1,0,1,0,0,1,1,1,1,0,1]
    >>> convert_train_to_times(spike_train)
    [4, 6, 9, 10, 11, 12, 14]
    """

    spikes = np.where(train)[0]
    spikes = np.array([x+1 for x in spikes])
    
    return spikes


def convert_isis_to_spikes(isis, offset=0, add_offset=True):
    """Convert a sequence of inter-spike intervals to spike times.
    Parameters
    ----------
    isis : 1d array
        Distribution of interspike intervals, in milliseconds.
    offset : float, optional
        An offset value to add to generated spike times.
    add_offset : bool, optional, default: True
        Whether to prepend the offset value to the beginning of the spike times.
    Returns
    -------
    spikes : 1d array
        Spike times, in milliseconds.
        
    Example
    -------
    
    Convert a sequence of 6 inter-spike intervals (ms) to their corresponding spike times (ms).
    
    >>> isis = [300, 600, 800, 200, 700]
    >>> convert_isis_to_spikes(isis, offset=0, add_offset=True )
    array([   0,  300,  900, 1700, 1900, 2600])
    
    """

    spikes = np.cumsum(isis, axis=-1)

    if offset:
        spikes = spikes + offset
    if add_offset:
        spikes = np.concatenate((np.array([offset]), spikes))

    return spikes
