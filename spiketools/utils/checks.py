"""Spike related checking functions."""

import warnings

import numpy as np

###################################################################################################
###################################################################################################

def check_spike_time_unit(spikes):
    """Check if spike times seem to be in second or millisecond time units..

    Parameters
    ----------
    spikes : 1d array
        Spike times.
    """

    # If there are any two spikes within the same time unit, show warning
    if len(np.unique((spikes).astype(int))) < len(np.unique(spikes)):
        warnings.warn("Based on number of events in a unit time, spikes appear to be in seconds.")

    # If the mean time between spikes is too low, show warning
    if len(spikes) > 1:
        if np.mean(np.diff(spikes)) < 10:
            warnings.warn("Based on the time between events, spikes appear to be seconds.")
