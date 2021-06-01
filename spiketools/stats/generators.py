"""Generators for drawing data from statistical distributions."""

import numpy as np

###################################################################################################
###################################################################################################

def poisson_train(frequency, duration, start_time=0, seed=None):
    """Generator function for a Homogeneous Poisson train.

    Parameters
    ----------
    frequency : float
        The mean spiking frequency.
    duration : float
        Maximum duration.
    start_time: float
        Timestamp.
    seed : int
        Seed for the random number generator.
        If None, this will be decided by np, which chooses the system time.

    Yields
    ------
    val
        A relative spike time from t=start_time, in seconds (not ms).

    Examples
    --------
    Make a list of spikes at 20 Hz for 3 seconds:

    >>> poisson_generator = poisson_train(20, 3)
    >>> spikes = [spike for spike in poisson_generator]

    Sample spikes continuously from a generator:

    >>> spike_gen = poisson_train(20, duration=np.inf)
    >>> for ind in range(10):
    ...     spike = next(spike_gen)
    """

    cur_time = start_time
    rangen = np.random.mtrand.RandomState()

    if seed is not None:
        rangen.seed(seed)

    isi = 1. / frequency

    while cur_time <= duration:

        cur_time += isi * rangen.exponential()

        if cur_time > duration:
            return

        yield cur_time
