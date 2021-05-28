"""Functions for shuffling data."""

from itertools import chain

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


def varying_bin_circular_shuffle(spike_train, bin_width_range=[50, 2000], random_state=None):
    """Shuffle data with a circular shuffle.

    Parameters
    ----------
    spike_train : 1d array
        xx
    bin_width_range : list of
        xx
    random_state : int
        xx

    Returns
    -------
    spike_time_diff :
        xx

    Notes
    -----
    This approach shuffles data by creating bins of varying length and then
    circularly shuffling within those bins.
    This should disturb the spike to spike structure in a dynamic way while also
    conserving structure uniformly across the distribution of lags.
    """

    rng = np.random.RandomState(random_state)

    right_edges = []
    ind = 0
    while ind < (spike_train.shape[-1] - bin_width_range[0]):
        ind += rng.randint(bin_width_range[0], bin_width_range[1])
        right_edges.append(ind)

    if right_edges[-1] > spike_train.shape[-1]:
        right_edges[-1] = spike_train.shape[-1]
    else:
        right_edges.append(spike_train.shape[-1])

    left_edges = list(right_edges[:-1])
    left_edges.insert(0, 0)

    bins = [cbin for cbin in zip(left_edges, right_edges)]

    # sanity check : the bins should cover the whole length of the spike train
    if np.sum([np.diff(cbin) for cbin in bins]) != spike_train.shape[-1]:
        raise ValueError

    # assign a shuffle amount to each bin
    shuffle_num = rng.uniform(low=bin_width_range[0] / 2,
                              high=bin_width_range[1],
                              size=len(bins)).astype(int)

    # circularly shuffle each bin
    spike_shuff = []
    for ind, cbin in enumerate(bins):
        spike_shuff.append(np.roll(spike_train[cbin[0]:cbin[1]], shuffle_num[ind]))

    # chain them all back together
    spike_shuff = np.array(list(chain.from_iterable(spike_shuff)))

    # get the spike times (ms resolution) back out of it
    spike_time_shuff = np.where(spike_shuff)[0]

    return spike_time_shuff
