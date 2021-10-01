"""Spike related checking functions."""

import numpy as np
import warnings

###################################################################################################
###################################################################################################

def check_spikes_in_ms(spikes):
    """Check if input spikes are in ms.

    Parameters
    ----------
    spikes : 1d array
        Spike times.
    """
	
    # If there are any two spikes within the same time unit, show warning.
    if len(np.unique((test_spikes).astype(int)))<len(np.unique(test_spikes)):
        warnings.warn('There are 2 or more spikes within a same unit of time. Spikes might be in seconds, should be in milliseconds.')
    # If the mean time between spikes is too low, show warning.
    if len(spikes)>1:
        if np.mean(np.diff(spikes))<10:
            warnings.warn('Spikes should be in milliseconds, but appear to be in seconds.')