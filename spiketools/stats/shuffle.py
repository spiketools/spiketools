"""Functions for shuffling data."""

from itertools import chain

import numpy as np

from spiketools.measures import compute_isis, compute_spike_rate
from spiketools.measures.conversions import (create_spike_train, convert_isis_to_spikes,
                                             convert_train_to_times)
from spiketools.stats.generators import poisson_train
from spiketools.stats.permutations import vec_perm

###################################################################################################
###################################################################################################

def shuffle_spikes(spikes, approach='ISI', n_shuffles=1000, random_state=None, **kwargs):
    """Shuffle spike

    Parameters
    ----------
    spikes : 1d array
        Spike times.
    approach : {'ISI', 'BINCIRC', 'POISSON', 'CIRCULAR'}
        Which approach to take for shuffling spike times.
    n_shuffles : int, optional, default: 1000
        The number of shuffles to create.
    random_state : int
        Initialization value for the random state.

    Returns
    -------
    shuffled_spikes : 2d array
        Shuffled spike times.
    """

    if approach == 'ISI':
        shuffled_spikes = shuffle_isis(spikes, n_shuffles=n_shuffles, random_state=random_state)

    elif approach == 'BINCIRC':
        shuffled_spikes = shuffle_bins(spikes, n_shuffles=n_shuffles, random_state=random_state, **kwargs)

    elif approach == 'POISSON':
        shuffled_spikes = shuffle_poisson(spikes, n_shuffles=n_shuffles)

    elif approach == 'CIRCULAR':
        shuffled_spikes = shuffle_circular(spikes, n_shuffles=n_shuffles, **kwargs)

    else:
        raise ValueError('Shuffling approach not understood.')

    return shuffled_spikes


def shuffle_isis(spikes, n_shuffles=1000, random_state=None):
    """Create shuffled spike times using permuted inter-spike intervals.

    Parameters
    ----------
    isis : 1d array
        Inter-spike intervals.
    random_state : int
        Initialization value for the random state.

    Returns
    -------
    shuffled_spikes : 2d array
        Shuffled spike times.
    """

    rng = np.random.RandomState(random_state)

    isis = compute_isis(spikes)

    shuffled_spikes = np.zeros([n_shuffles, spikes.shape[-1]])
    for ind in range(n_shuffles):
        shuffled_spikes[ind, :] = convert_isis_to_spikes(rng.permutation(isis))

    return shuffled_spikes


def shuffle_bins(spikes, bin_width_range=[50, 2000], n_shuffles=1000, random_state=None):
    """Shuffle data with a circular shuffle of the spike train.

    Parameters
    ----------
    spikes : 1d array
        Spike times.
    bin_width_range : list of int
        xx
    random_state : int
        Initialization value for the random state.

    Returns
    -------
    spike_time_diff : 2d array
        Shuffled spike times.

    Notes
    -----
    This approach shuffles spikes by creating bins of varying length and then
    circularly shuffling within those bins.
    This should disturb the spike to spike structure in a dynamic way while also
    conserving structure uniformly across the distribution of lags.
    """

    spike_train = create_spike_train(spikes)

    rng = np.random.RandomState(random_state)

    shuffled_spikes = np.zeros([n_shuffles, spikes.shape[-1]])

    for ind in range(n_shuffles):

        right_edges = []
        edge = 0
        while edge < (spike_train.shape[-1] - bin_width_range[0]):
            edge += rng.randint(bin_width_range[0], bin_width_range[1])
            right_edges.append(edge)

        if right_edges[-1] > spike_train.shape[-1]:
            right_edges[-1] = spike_train.shape[-1]
        else:
            right_edges.append(spike_train.shape[-1])

        left_edges = list(right_edges[:-1])
        left_edges.insert(0, 0)

        bins = [cbin for cbin in zip(left_edges, right_edges)]

        # Error check: the bins should cover the whole length of the spike train
        if np.sum([np.diff(cbin) for cbin in bins]) != spike_train.shape[-1]:
            raise ValueError('Problem with bins covering the whole length.')

        # Assign a shuffle amount to each bin
        shuffle_num = rng.uniform(low=bin_width_range[0] / 2,
                                  high=bin_width_range[1],
                                  size=len(bins)).astype(int)

        # Circularly shuffle each bin
        shuffled_train = []
        for b_ind, cbin in enumerate(bins):
            shuffled_train.append(np.roll(spike_train[cbin[0]:cbin[1]], shuffle_num[ind]))

        # Combine the shuffled bins back together
        shuffled_train_flat = np.concatenate(shuffled_train)

        # Convert back to spike times (with ms resolution) from the shuffled spike train
        shuffled_spikes[ind, :] = convert_train_to_times(shuffled_train_flat)

    return shuffled_spikes


def shuffle_poisson(spikes, n_shuffles=1000, random_state=None):
    """Shuffle spikes based on a Poisson distribution.

    Parameters
    ----------
    spikes : 1d array
        Spike times.
    n_shuffles : int
        The number of shuffles to create.

    Returns
    -------
    shuffled_spikes : 2d array
        Shuffled spike times.

    Notes
    -----
    Experimental implementation / has issues matching spike counts.
    Not fully checked / tested / implemented yet.
    """

    rng = np.random.RandomState(random_state)

    length = ((spikes[-1] - spikes[0]) / 1000)
    rate = compute_spike_rate(spikes)
    poisson_spikes = [ind for ind in poisson_train(rate, length)] + spikes[0]

    # NOTE: might be an issue with vec_perm here
    isis = vec_perm(compute_isis(poisson_spikes), n_perms=n_shuffles)

    shuffled_spikes = np.cumsum(isis, axis=1) + spikes[0]

    return shuffled_spikes


def shuffle_circular(spikes, shuffle_min=20000, n_shuffles=1000, random_state=None):
    """Shuffle spikes based on circularly shifting the spike train.

    Parameters
    ----------
    spikes : 1d array
        Spike times.
    shuffle_min : int
        The minimum amount to rotate date, in terms of units of the spike train.
    n_shuffles : int
        The number of shuffles to create.

    Returns
    -------
    shuffled_spikes : 2d array
        Shuffled spike times.
    """

    spike_train = create_spike_train(spikes)

    rng = np.random.RandomState(random_state)

    shuffles = np.random.randint(low=shuffle_min,
                                 high=len(spike_train)-shuffle_min,
                                 size=n_shuffles)

    shuffled_spikes = np.zeros([n_shuffles, len(spikes)])

    for ind, shuffle in enumerate(shuffles):
        temp_train = np.roll(spike_train, shuffle)
        shuffled_spikes[ind, :] = convert_train_to_times(temp_train)

    return shuffled_spikes
