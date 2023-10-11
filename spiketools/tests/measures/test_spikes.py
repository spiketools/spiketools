"""Tests for spiketools.measures.spikes"""

import numpy as np

from spiketools.measures.spikes import *

###################################################################################################
###################################################################################################

def test_compute_firing_rate(tspikes):

    rate = compute_firing_rate(tspikes, time_range=[0, 10])

    assert isinstance(rate, float)
    assert np.isclose(rate, 2.0)

def test_compute_isis(tspikes):

    isis = compute_isis(tspikes)
    assert isinstance(isis, np.ndarray)
    assert isis.shape[-1] + 1 == tspikes.shape[-1]
    assert np.allclose(isis, np.array([1.0, 0.5, 0.5, 0.5, 0.2, 0.5, 0.3, 0.2, 0.5, 0.3,
                                       0.7, 0.3, 1.0, 0.5, 0.5, 0.2, 0.5, 0.5, 0.7]))
    assert sum(isis < 0) == 0

def test_compute_cv():

    isis1 = np.array([0.5, 0.5, 0.5])
    isis2 = np.array([0.25, 0.75, 0.25, 0.75])

    cv1 = compute_cv(isis1)
    assert isinstance(cv1, float)
    assert np.isclose(cv1, 0.)

    cv2 = compute_cv(isis2)
    assert isinstance(cv2, float)
    assert cv2 > 0.

def test_compute_fano_factor():

    spike_train1 = np.array([1, 1, 1, 1, 1])
    spike_train2 = np.array([0, 1, 0, 0, 1])

    fano1 = compute_fano_factor(spike_train1)
    assert isinstance(fano1, float)
    assert np.isclose(fano1, 0.)

    fano2 = compute_fano_factor(spike_train2)
    assert isinstance(fano2, float)
    assert fano2 < 1

def test_compute_spike_presence(tspikes):

    tspike_presence = compute_spike_presence(tspikes, 0.5, [0, 10])
    assert isinstance(tspike_presence, np.ndarray)
    assert tspike_presence.dtype == 'bool'

    spikes = np.array([1.1, 1.75, 2.25, 2.9])
    bins = np.array([0, 1, 2, 3, 4, 5])
    spike_presence = compute_spike_presence(spikes, bins)
    assert np.array_equal(spike_presence, np.array([False, True, True, False, False]))

def test_compute_presence_ratio():

    spikes1 = np.array([0.1, 0.3, 0.4, 0.775, 0.825, 0.900])
    presence_ratio1 = compute_presence_ratio(spikes1, 0.250)
    assert presence_ratio1 == 0.75

    spikes2 = np.array([0.05, 0.15])
    presence_ratio2 = compute_presence_ratio(spikes2, np.arange(0, 1.1, 0.1))
    assert presence_ratio2 == 0.2
