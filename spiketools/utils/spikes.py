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
    """

    min_time = -np.inf if min_time is None else min_time
    max_time = np.inf if max_time is None else max_time

    return spikes[(spikes >= min_time) & (spikes <= max_time)]
