"""Tests for spiketools.measures.conversions"""

import numpy as np

from spiketools.measures import compute_isis

from spiketools.measures.conversions import *

###################################################################################################
###################################################################################################

def test_convert_times_to_train(tspikes):

    spike_train = convert_times_to_train(tspikes)
    assert isinstance(spike_train, np.ndarray)
    assert spike_train.shape[-1] > tspikes.shape[-1]
    assert sum(spike_train) == tspikes.shape[-1]

def test_convert_train_to_times():

    train = np.zeros(100)
    spike_inds = np.array([12, 24, 36, 45, 76, 79, 90])
    for ind in spike_inds:
        train[ind] = 1
    # Define expected spike times, for a fs of 1000
    expected = (spike_inds / 1000) + 0.001

    # Check spike train for sampling rate of 1000
    spikes = convert_train_to_times(train, fs=1000)
    assert isinstance(spikes, np.ndarray)
    assert spikes.shape[-1] == spike_inds.shape[-1]
    assert np.array_equal(spikes, expected)

    # Check different sampling rate
    spikes = convert_train_to_times(train, fs=500)
    assert isinstance(spikes, np.ndarray)
    assert spikes.shape[-1] == spike_inds.shape[-1]
    assert np.array_equal(spikes, expected * 2)


def test_convert_isis_to_times(tspikes):

    isis = compute_isis(tspikes)

    spikes1 = convert_isis_to_times(isis)
    assert spikes1.shape[-1] == tspikes.shape[-1]

    spikes2 = convert_isis_to_times(isis, offset=2.)
    assert spikes2[0] == 2.

    spikes3 = convert_isis_to_times(isis, add_offset=False)
    assert len(spikes3) == len(isis)

    spikes4 = convert_isis_to_times(isis, offset=tspikes[0])
    assert np.array_equal(spikes4, tspikes)
