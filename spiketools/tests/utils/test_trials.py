"""Tests for spiketools.utils.trials"""

import numpy as np

from spiketools.utils.trials import *

###################################################################################################
###################################################################################################

def test_epoch_trials():

    spikes = np.array([2.5, 3.5, 4.25, 5.5, 6.1, 8., 9.25, 9.75, 10.5, 12., 14.1, 15.2, 15.9])
    events = np.array([5, 10, 15])
    window = [-1, 1]

    trials = epoch_trials(spikes, events, window)
    assert isinstance(trials, list)
    assert isinstance(trials[0], np.ndarray)
    assert len(trials) == len(events)
    assert np.array_equal(trials[0], np.array([4.25, 5.5]) - events[0])
    assert np.array_equal(trials[1], np.array([9.25, 9.75, 10.5]) - events[1])
    assert np.array_equal(trials[2], np.array([14.1, 15.2, 15.9]) - events[2])
