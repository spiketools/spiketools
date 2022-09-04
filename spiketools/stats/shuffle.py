"""Functions for shuffling data."""

import numpy as np

from spiketools.measures.spikes import compute_isis, compute_firing_rate
from spiketools.measures.conversions import (convert_times_to_train, convert_isis_to_times,
                                             convert_train_to_times)
from spiketools.stats.generators import poisson_generator
from spiketools.stats.permutations import permute_vector
from spiketools.utils.checks import check_param_options

###################################################################################################
###################################################################################################

def shuffle_spikes(spikes, approach='ISI', n_shuffles=1000, **kwargs):
    """Shuffle spikes.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    approach : {'ISI', 'BINCIRC', 'POISSON', 'CIRCULAR'}
        Which approach to take for shuffling spike times.
    n_shuffles : int, optional, default: 1000
        The number of shuffles to create.
    kwargs
        Additional keyword arguments for the shuffle procedure.
        See shuffle sub-functions for details.

    Returns
    -------
    shuffled_spikes : 2d array
        Shuffled spike times.

    Examples
    --------
    Simulate some example spikes for examples:

    >>> from spiketools.sim.times import sim_spiketimes
    >>> spikes = sim_spiketimes(5, 30, 'poisson')

    Create 5 spike time shuffles using the ISI shuffle method:

    >>> shuffled_spikes = shuffle_spikes(spikes, 'ISI', n_shuffles=5)

    Create 5 spike time shuffles using the 'CIRCULAR' shuffle method:

    >>> shuffled_spikes = shuffle_spikes(spikes, 'CIRCULAR', n_shuffles=5, shuffle_min=10000)
    """

    check_param_options(approach, 'approach', ['ISI', 'BINCIRC', 'POISSON', 'CIRCULAR'])

    if approach == 'ISI':
        shuffled_spikes = shuffle_isis(spikes, n_shuffles=n_shuffles)

    elif approach == 'BINCIRC':
        shuffled_spikes = shuffle_bins(spikes, n_shuffles=n_shuffles, **kwargs)

    elif approach == 'POISSON':
        shuffled_spikes = shuffle_poisson(spikes, n_shuffles=n_shuffles)

    elif approach == 'CIRCULAR':
        shuffled_spikes = shuffle_circular(spikes, n_shuffles=n_shuffles, **kwargs)

    return shuffled_spikes


def shuffle_isis(spikes, n_shuffles=1000):
    """Create shuffled spike times using permuted inter-spike intervals.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    n_shuffles : int, optional, default: 1000
        The number of shuffles to create.

    Returns
    -------
    shuffled_spikes : 2d array
        Shuffled spike times.

    Examples
    --------
    Shuffle spike times using the ISI shuffle method:

    >>> from spiketools.sim.times import sim_spiketimes
    >>> spikes = sim_spiketimes(5, 30, 'poisson')
    >>> shuffled_spikes = shuffle_isis(spikes, n_shuffles=5)
    """

    isis = compute_isis(spikes)

    shuffled_spikes = np.zeros([n_shuffles, spikes.shape[-1]])
    for ind in range(n_shuffles):
        shuffled_spikes[ind, :] = convert_isis_to_times(np.random.permutation(isis))

    return shuffled_spikes


def shuffle_bins(spikes, bin_width_range=[.5, 7], n_shuffles=1000):
    """Shuffle data with circular shuffles of randomly sized bins of the spike train.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    bin_width_range : list of [float, float], optional, default : [.5, 7]
        Range of bin widths in seconds from which bin sizes are randomly selected.
    n_shuffles : int, optional, default: 1000
        The number of shuffles to create.

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
    This function can be a little slow when running a lot.
    The main holdup is `np.roll` (unclear if / how to optimize).
    The next biggest hold up is converting the spike train to spike times.
    This shuffling process is very dependent on the `bin_width_range` argument.
    It is recommended that `bin_width_range[1] > 3`, and that the difference
    between the two values of `bin_width_range` is at least 1.

    Examples
    --------
    Shuffle spike times using the circular bin method:

    >>> from spiketools.sim.times import sim_spiketimes
    >>> spikes = sim_spiketimes(5, 30, 'poisson')
    >>> shuffled_spikes = shuffle_bins(spikes, bin_width_range=[3, 4], n_shuffles=5)
    """

    spike_train = convert_times_to_train(spikes)

    shuffled_spikes = np.zeros([n_shuffles, spikes.shape[-1]])

    for ind in range(n_shuffles):

        # Define the bins to use for shuffling
        #  This creates the maximum number of bins, then sub-selects to bins that tile the space
        #  This approach is a little quicker than iterating through and stopping space is tiled
        temp = np.random.randint(bin_width_range[0] * 1000, bin_width_range[1] * 1000,
                                 int(spike_train.shape[-1] / (bin_width_range[0] * 1000)))
        right_edges = np.cumsum(temp)
        right_edges = right_edges[right_edges < spike_train.shape[-1]]
        right_edges[-1] = spike_train.shape[-1]

        # Define the left bin edges
        left_edges = right_edges.copy()[:-1]
        left_edges = np.insert(left_edges, 0, 0)

        # Error check: the bins should cover the whole length of the spike train
        if np.sum([re - le for le, re in zip(left_edges, right_edges)]) != spike_train.shape[-1]:
            raise ValueError('Problem with bins covering the whole length.')

        # Assign a shuffle amount to each bin
        #   QUESTION / CHECK: should the low range here be divided by two?
        shuffle_num = np.random.randint(low=(bin_width_range[0] * 1000) / 2,
                                        high=bin_width_range[1] * 1000,
                                        size=len(left_edges))

        # Circularly shuffle each bin
        shuffled_train = np.zeros(len(spike_train))
        for le, re, shuff in zip(left_edges, right_edges, shuffle_num):
            shuffled_train[le:re] = np.roll(spike_train[le:re], shuff)

        # Convert back to spike times (with ms resolution) from the shuffled spike train
        shuffled_spikes[ind, :] = convert_train_to_times(shuffled_train)

    return shuffled_spikes


def shuffle_poisson(spikes, n_shuffles=1000):
    """Shuffle spikes based on a Poisson distribution.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    n_shuffles : int, optional, default: 1000
        The number of shuffles to create.

    Returns
    -------
    shuffled_spikes : 2d array
        Shuffled spike times.

    Notes
    -----
    This is an experimental implementation, and still has some issues matching spike counts.

    Examples
    --------
    Shuffle spike times using the Poisson method:

    >>> from spiketools.sim.times import sim_spiketimes
    >>> spikes = sim_spiketimes(5, 30, 'poisson')
    >>> shuffled_spikes = shuffle_poisson(spikes, n_shuffles=5)
    """

    rate = compute_firing_rate(spikes)
    length = (spikes[-1] - spikes[0])

    poisson_spikes = list(poisson_generator(rate, length)) + spikes[0]

    isis = permute_vector(compute_isis(poisson_spikes), n_permutations=n_shuffles)

    shuffled_spikes = np.cumsum(isis, axis=1) + spikes[0]

    return shuffled_spikes


def shuffle_circular(spikes, shuffle_min=20000, n_shuffles=1000):
    """Shuffle spikes based on circularly shifting the spike train.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    shuffle_min : int
        The minimum amount to rotate data, in terms of units of the spike train.
    n_shuffles : int, optional, default: 1000
        The number of shuffles to create.

    Returns
    -------
    shuffled_spikes : 2d array
        Shuffled spike times.

    Notes
    -----
    The input shuffle_min should always be less than the maximum time in which a spike occurred.

    Examples
    --------
    Shuffle spike times using the circular method:

    >>> from spiketools.sim.times import sim_spiketimes
    >>> spikes = sim_spiketimes(5, 30, 'poisson')
    >>> shuffled_spikes = shuffle_circular(spikes, shuffle_min=10000, n_shuffles=5)
    """

    spike_train = convert_times_to_train(spikes)

    shuffles = np.random.randint(low=shuffle_min,
                                 high=len(spike_train)-shuffle_min,
                                 size=n_shuffles)

    shuffled_spikes = np.zeros([n_shuffles, len(spikes)])

    for ind, shuffle in enumerate(shuffles):
        temp_train = np.roll(spike_train, shuffle)
        shuffled_spikes[ind, :] = convert_train_to_times(temp_train)

    return shuffled_spikes
