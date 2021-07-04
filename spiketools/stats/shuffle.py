"""Functions for shuffling data."""

from itertools import chain

import numpy as np

from spiketools.measures import compute_isis, compute_spike_rate, create_spike_train
from spiketools.stats.generators import poisson_train
from spiketools.stats.permutations import vec_perm

###################################################################################################
###################################################################################################

def shuffle_isis(spike_times, random_state=None):
    """Use shuffled inter-spike intervals to create shuffled spike times.

    Parameters
    ----------
    spikes : 1d array
        Spike times.
    random_state : int
        xx

    Returns
    -------
    shuffled_spikes : Xd array
        xx
    """

    rng = np.random.RandomState(random_state)

    isis = compute_isis(spike_times)

    shuffled_spikes = np.zeros_like(spike_times)
    shuffled_spikes[1:] = np.cumsum(rng.permutation(isis)) + shuffled_spikes[0]

    return shuffled_spikes


def shuffle_bins(spikes, bin_width_range=[50, 2000], random_state=None):
    """Shuffle data with a circular shuffle of the spike train.

    Parameters
    ----------
    spikes : 1d array
        Spike times.
    bin_width_range : list of int
        xx
    random_state : int
        Random state.

    Returns
    -------
    spike_time_diff : Xd array
        xx

    Notes
    -----
    This approach shuffles data by creating bins of varying length and then
    circularly shuffling within those bins.
    This should disturb the spike to spike structure in a dynamic way while also
    conserving structure uniformly across the distribution of lags.

    Originally called: `varying_bin_circular_shuffle`.
    ToDo: switch to taking in spike times, to have consistent API(?)
    """

    rng = np.random.RandomState(random_state)

    spike_train = create_spike_train(spikes)

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
        raise ValueError('Problem with bins covering the whole length.')

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
    shuffled_spikes = np.where(spike_shuff)[0]

    return shuffled_spikes


def shuffle_poisson(spikes, n_shuffles=1000):
    """Shuffle spikes based on a Poisson distribution.

    Parameters
    ----------
    spikes : 1d array
        Spike times.
    n_shuffles : int
        The number of shuffles to create.

    Returns
    -------
    shuffled_spikes : Xd array
        xx

    Notes
    -----
    experimental implementation / has issues matching spike counts.
    """

    length = ((spikes[-1] - spikes[0]) / 1000)
    #fr = len(spikes) / length
    rate = compute_spike_rate(spikes)
    poisson_spikes = [ind for ind in poisson_train(rate, length)] + spikes[0]

    isis = vec_perm(compute_isis(poisson_spikes), n_perms=n_shuffles)
    #ISIs = vec_perm(np.diff(poisson_spikes), n_perms=n_shuffles)

    shuffled_spikes = np.cumsum(isis, axis=1) + spikes[0]

    return shuffled_spikes


def shuffle_circular(spikes, shuffle_min=20000, n_shuffles=1000):
    """Shuffle spikes based on circular....

    Parameters
    ----------
    spikes :
        xx
    shuffle_min :
        xx
    n_shuffles : int
        The number of shuffles to create.

    Returns
    -------
    shuffled_spikes : Xd array
        xx

    Notes
    -----
    experimental implementation / has issues matching spike counts.
    """

    spike_train = create_spike_train(spikes)

    shuffles = np.random.randint(low=shuffle_min,
                                 high=len(spike_train)-shuffle_min,
                                 size=n_shuffles)

    shuffled_spikes = np.zeros([n_shuffles, len(spikes)])

    # THIS FUNCTION IS NOT FINISHED - CURRENTLY RETURNS ZEROS

    return shuffled_spikes
