"""Generators for drawing data from statistical distributions."""

import numpy as np

###################################################################################################
###################################################################################################

def poisson_train(frequency, duration, start_time=0):
    """Generator function for a Homogeneous Poisson train.

    Parameters
    ----------
    frequency : float
        The mean spiking frequency.
    duration : float
        Maximum duration.
    start_time: float, optional
        Timestamp of the start time for the generated sequence.

    Yields
    ------
    float
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

    isi = 1. / frequency

    cur_time = start_time
    while cur_time <= duration:

        cur_time += isi * np.random.exponential()

        if cur_time > duration:
            return

        yield cur_time
