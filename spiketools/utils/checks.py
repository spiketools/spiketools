"""Spike related checking functions."""

import warnings

import numpy as np

###################################################################################################
###################################################################################################

def check_spike_time_unit(spikes):
    """Check if spikes are in time unit milliseconds.

    Parameters
    ----------
    spikes : 1d array
        Spike times.
    """

    # If there are any two spikes within the same time unit, show warning.
    if len(np.unique((spikes).astype(int))) < len(np.unique(spikes)):
        warnings.warn("There are 2 or more spikes within a same unit of time. " \
                      "Spikes might be in seconds, should be in milliseconds.")

    # If the mean time between spikes is too low, show warning.
    if len(spikes) > 1:
        if np.mean(np.diff(spikes)) < 10:
            warnings.warn("Spikes should be in milliseconds, but appear to be in seconds.")
