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

    EXAMPLE::

        # Make a list of spikes at 20 Hz for 3 seconds
        spikes = [i for i in poisson_train(20, 3)]

    EXAMPLE::

        # Use dynamically in a program
        # Care needs to be taken with this scenario because the generator will
        # generate spikes until the program or spike_gen object is terminated.
        spike_gen = poisson_train(20, duration=sys.float_info.max)
        spike = spike_gen.next()
        # Process the spike, to other programmatic things
        spike = spike_gen.next() # Get another spike
        # etc.
        # Terminate the program.
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
