"""Spike related utility functions."""

import numpy as np

###################################################################################################
###################################################################################################

def restrict_range(spikes, min_time=None, max_time=None):
    """Restrict a vector of spike times to a specified range.

    Parameters
    ----------
    spikes : 1d array
        Spike times.
    min_time, max_time : float, optional, default: None
        Mininum and/or maximum time to restrict spike times to.

    Returns
    -------
    spikes : 1d array
        Spike times, restricted to desired time range.

    Examples
    --------
    Select all spikes after a specific time point:

    >>> spikes = np.array([50, 100, 150, 200, 250, 300])
    >>> restrict_range(spikes, min_time=100, max_time=None)
    array([100, 150, 200, 250, 300])

    Select all spikes before a specific time point:

    >>> spikes = np.array([50, 100, 150, 200, 250, 300])
    >>> restrict_range(spikes, min_time=None, max_time=250)
    array([ 50, 100, 150, 200, 250])

    Restrict a vector of spike times to a specific range:

    >>> spikes = np.array([50, 100, 150, 200, 250, 300])
    >>> restrict_range(spikes, min_time=100, max_time=200)
    array([100, 150, 200])
    """

    min_time = -np.inf if min_time is None else min_time
    max_time = np.inf if max_time is None else max_time

    return spikes[(spikes >= min_time) & (spikes <= max_time)]
