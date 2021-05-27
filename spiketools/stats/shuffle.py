"""Functions for shuffling data."""

import numpy as np

from spiketools.measures import compute_isis

###################################################################################################
###################################################################################################

def shuffle_isis(spike_times, random_state=None):
    """Use shuffled ISIs to return a set of shuffled spike times.

    Parameters
    ----------
    spike_times : 1d array
        xx
    random_state : int
        xx

    Returns
    -------
    new_spike_times : 1d array
        xx
    """

    rng = np.random.RandomState(random_state)

    isis = compute_isis(spike_times)

    new_spike_times = np.zeros_like(spike_times)
    new_spike_times[1:] = np.cumsum(rng.permutation(isis)) + new_spike_times[0]

    return new_spike_times
