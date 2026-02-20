"""Functions for shuffling data."""

from functools import wraps

import numpy as np

from spiketools.measures.spikes import compute_isis, compute_firing_rate
from spiketools.measures.conversions import (convert_times_to_train, convert_isis_to_times,
                                             convert_train_to_times)
from spiketools.stats.generators import poisson_generator
from spiketools.utils.checks import check_param_options
from spiketools.utils.extract import drop_range, reinstate_range

###################################################################################################
###################################################################################################

def shuffle_spikes(spikes, approach, n_shuffles=1000, start_time=0, **kwargs):
    """Shuffle spikes.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    approach : {'isi', 'circular', 'bincirc'}
        Which approach to take for shuffling spike times.
        See shuffle sub-functions for details.
    n_shuffles : int, optional, default: 1000
        The number of shuffles to create.
    start_time : float, optional
        The start time of the input spikes, used to set the time values of the shuffled outputs.
    kwargs
        Additional keyword arguments for the shuffle procedure.
        See shuffle sub-functions for details.

    Returns
    -------
    shuffled_spikes : 2d array
        Shuffled spike times.

    Notes
    -----
    Spike shuffling also supports dropping time ranges from shuffling.

    This is applied via a decorator which catches the following optional inputs:

    drop_time_range : list of [float, float] or list of list of [float, float]
        Time range(s), in seconds, to drop from spike times before shuffling.
        Each time range should be defined as [start_drop_time, end_drop_time].
    check_empty : bool, optional, default: True
        Whether to check if the dropped range is empty of spikes.
        If True, and there are spikes within the `drop_time_range`, an error is raised.

    Examples
    --------
    Simulate some example spikes for examples:

    >>> from spiketools.sim.times import sim_spiketimes
    >>> spikes = sim_spiketimes(5, 30, 'poisson')

    Create 5 spike time shuffles using the ISI shuffle method:

    >>> shuffled_spikes = shuffle_spikes(spikes, 'isi', n_shuffles=5)

    Create 5 spike time shuffles using a circular shuffle:

    >>> shuffled_spikes = shuffle_spikes(spikes, 'circular', n_shuffles=5, shuffle_min=10000)
    """

    # Use lowered string, for backwards compatibility for options were upper case
    approach = approach.lower()

    check_param_options(approach, 'approach', ['isi', 'circular', 'bincirc'])

    shared_kwargs = {'n_shuffles' : n_shuffles, 'start_time' : start_time}

    if approach == 'isi':
        shuffled_spikes = shuffle_isis(spikes, **shared_kwargs)

    elif approach == 'circular':
        shuffled_spikes = shuffle_circular(spikes, **shared_kwargs, **kwargs)

    elif approach == 'bincirc':
        shuffled_spikes = shuffle_bins(spikes, **shared_kwargs, **kwargs)

    return shuffled_spikes


def drop_shuffle_range(func):
    """Decorator for shuffling functions that allows for dropping a time range for shuffling.

    Notes
    -----
    This function is designed for `shuffle_xx` functions, which takes 1d array `spikes` as
    the first input and return a 2d array `shuffled_spikes` as the sole output.

    If a keyword argument `drop_time_range` is present, this triggers the drop process:

    - The given drop range time is dropped from the given spike times, before shuffling
    - The shuffle function is then run, without the dropped range
    - The drop range time is then reinstated in the shuffled spike times

    If `drop_time_range` is not present in the arguments, this decorator does nothing.
    """

    @wraps(func)
    def decorated(*args, **kwargs):

        spikes = args[0]
        time_range = kwargs.pop('drop_time_range', None)
        check_empty = kwargs.pop('check_empty', True)

        if time_range:
            spikes = drop_range(spikes, time_range, check_empty)

        shuffles = func(spikes, *args[1:], **kwargs)

        if time_range:
            shuffles = reinstate_range(shuffles, time_range)

        return shuffles

    return decorated


@drop_shuffle_range
def shuffle_isis(spikes, n_shuffles=1000, start_time=0):
    """Create shuffled spike times using permuted inter-spike intervals.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    n_shuffles : int, optional, default: 1000
        The number of shuffles to create.
    start_time : float, optional
        The start time of the input spikes, used to set the time values of the shuffled outputs.

    Returns
    -------
    shuffled_spikes : 2d array
        Shuffled spike times.

    Notes
    -----
    Spike shuffling also supports dropping time ranges from shuffling, using optional inputs
    `drop_time_range` and `check_empty` which are caught by a decorator and applied.
    See additional notes on doing this in the `shuffle_spikes` function.

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
        shuffled_spikes[ind, :] = convert_isis_to_times(\
            np.random.permutation(isis), start_time=start_time)

    return shuffled_spikes


@drop_shuffle_range
def shuffle_circular(spikes, shuffle_min=20000, n_shuffles=1000, start_time=0):
    """Shuffle spikes based on circularly shifting the spike train.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    shuffle_min : int
        The minimum amount to rotate data, in terms of units of the spike train.
    n_shuffles : int, optional, default: 1000
        The number of shuffles to create.
    start_time : float, optional
        The start time of the input spikes, used to set the time values of the shuffled outputs.

    Returns
    -------
    shuffled_spikes : 2d array
        Shuffled spike times.

    Notes
    -----
    The input shuffle_min should always be less than the maximum time in which a spike occurred.

    Spike shuffling also supports dropping time ranges from shuffling, using optional inputs
    `drop_time_range` and `check_empty` which are caught by a decorator and applied.
    See additional notes on doing this in the `shuffle_spikes` function.

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
        shuffled_spikes[ind, :] = convert_train_to_times(temp_train) + start_time

    return shuffled_spikes


@drop_shuffle_range
def shuffle_bins(spikes, bin_width_range=[.5, 7], n_shuffles=1000, start_time=0):
    """Shuffle data with circular shuffles of randomly sized bins of the spike train.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    bin_width_range : list of [float, float], optional, default : [.5, 7]
        Range of bin widths in seconds from which bin sizes are randomly selected.
    n_shuffles : int, optional, default: 1000
        The number of shuffles to create.
    start_time : float, optional
        The start time of the input spikes, used to set the time values of the shuffled outputs.

    Returns
    -------
    shuffled_spikes : 2d array
        Shuffled spike times.

    Notes
    -----
    This approach shuffles spikes by creating bins of varying length and then circularly shuffling
    within each bin. This shuffles the time stamps of all spikes within bins, while avoiding
    a simple circular rotation and preserving slow dynamics.

    This shuffling process is very dependent on the `bin_width_range` argument.
    It is recommended that `bin_width_range[1] > 3`, and that the difference
    between the two values of `bin_width_range` is at least 1.

    Note that this approach is slower than simpler approaches, due to spending time on
    the multiple calls to `np.roll` to rotate within each bin, as well as due to converting
    between spike times and spike trains.

    Spike shuffling also supports dropping time ranges from shuffling, using optional inputs
    `drop_time_range` and `check_empty` which are caught by a decorator and applied.
    See additional notes on doing this in the `shuffle_spikes` function.

    Examples
    --------
    Shuffle spike times using the circular bin method:

    >>> from spiketools.sim.times import sim_spiketimes
    >>> spikes = sim_spiketimes(5, 30, 'poisson')
    >>> shuffled_spikes = shuffle_bins(spikes, bin_width_range=[3, 4], n_shuffles=5)
    """

    fs = 1000
    spike_train = convert_times_to_train(spikes, fs=fs)

    shuffled_spikes = np.zeros([n_shuffles, spikes.shape[-1]])

    for ind in range(n_shuffles):

        # Define the bins to use for shuffling
        #  This creates the maximum number of bins, then sub-selects to bins that tile the space
        #  This approach is a little quicker than iterating through and stopping space is tiled
        temp = np.random.randint(bin_width_range[0] * fs, bin_width_range[1] * fs,
                                 int(spike_train.shape[-1] / (bin_width_range[0] * fs)))
        right_edges = np.cumsum(temp)
        right_edges = right_edges[right_edges < spike_train.shape[-1]]
        right_edges[-1] = spike_train.shape[-1]

        # Define the left bin edges
        left_edges = right_edges.copy()[:-1]
        left_edges = np.insert(left_edges, 0, 0)

        # Error check: the bins should cover the whole length of the spike train
        if np.sum([re - le for le, re in zip(left_edges, right_edges)]) != spike_train.shape[-1]:
            raise ValueError('Problem with bins covering the whole length.')

        # QUESTION / CHECK: should the low range here be divided by two?
        # Assign a shuffle amount to each bin
        shuffle_num = np.random.randint(low=(bin_width_range[0] * fs) / 2,
                                        high=bin_width_range[1] * fs,
                                        size=len(left_edges))

        # Circularly shuffle each bin
        shuffled_train = np.zeros(len(spike_train))
        for le, re, shuff in zip(left_edges, right_edges, shuffle_num):
            shuffled_train[le:re] = np.roll(spike_train[le:re], shuff)

        # Convert back to spike times (with ms resolution) from the shuffled spike train
        shuffled_spikes[ind, :] = convert_train_to_times(shuffled_train, start_time=start_time)

    return shuffled_spikes


@drop_shuffle_range
def shuffle_poisson(spikes, n_shuffles=1000, start_time=0):
    """Shuffle spikes based on generating new spike trains from a Poisson distribution.

    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    n_shuffles : int, optional, default: 1000
        The number of shuffles to create.
    start_time : float, optional
        The start time of the input spikes, used to set the time values of the shuffled outputs.

    Returns
    -------
    shuffled_spikes : list of 1d array
        Shuffled spike times.

    Notes
    -----
    This approach creates "shuffles" by simulating new spike trains from a Poisson distribution.

    Note that this approach is therefore not strictly a "shuffle" in the sense that the outputs
    are not literally actually shuffled versions of the input, and are instead new simulated sets
    of spikes sampled based on the statistics of the input.

    In addition, since this approach simulates new spike trains based on an average rate,
    shuffles are not guaranteed to have the same number of spikes as the input, nor are different
    iterations of the shuffles guaranteed to have the same number of spikes as each other.
    This is why the outputs are returned as a list of shuffles, due to inconsistent shape.

    Spike shuffling also supports dropping time ranges from shuffling, using optional inputs
    `drop_time_range` and `check_empty` which are caught by a decorator and applied.
    See additional notes on doing this in the `shuffle_spikes` function.

    Examples
    --------
    Shuffle spike times using the Poisson method:

    >>> from spiketools.sim.times import sim_spiketimes
    >>> spikes = sim_spiketimes(5, 30, 'poisson')
    >>> shuffled_spikes = shuffle_poisson(spikes, n_shuffles=5)
    """

    rate = compute_firing_rate(spikes)
    length = (spikes[-1] - spikes[0])

    shuffled_spikes = [None] * n_shuffles
    for ind in range(n_shuffles):
        shuffled_spikes[ind] = list(poisson_generator(rate, length, start_time))

    return shuffled_spikes
