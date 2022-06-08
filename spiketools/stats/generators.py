"""Generators for drawing data from statistical distributions."""

import numpy as np

###################################################################################################
###################################################################################################

def poisson_generator(rate, duration, start_time=0):
    """Generator function for a Homogeneous Poisson distribution.

    Parameters
    ----------
    rate : float
        The average rate for the generator.
    duration : float
        Maximum duration. After this time, the generator will return.
    start_time: float, optional
        Timestamp of the start time for the generated sequence.

    Yields
    ------
    float
        A sample from the distribution.
        Sample is a relative value, based on `start_time`, in seconds.

    Examples
    --------
    Create a Poisson generator and sample from it:

    >>> gen = poisson_train(20, duration=np.inf)
    >>> for ind in range(10):
    ...     sample = next(gen)
    """

    isi = 1. / rate
    cur_time = start_time
    while cur_time <= duration:

        cur_time += isi * np.random.exponential()

        if cur_time > duration:
            return

        yield cur_time
