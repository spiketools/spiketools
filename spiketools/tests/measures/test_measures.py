"""Tests for spiketools.measures.measures"""

import numpy as np

from spiketools.measures.measures import *

###################################################################################################
###################################################################################################

def test_compute_spike_rate():

    spikes = np.array([0.0, 0.5, 1.5, 2.])

    fr = compute_spike_rate(spikes)

    assert isinstance(fr, float)
    assert np.isclose(fr, 2.0)

def test_compute_isis():

    spikes = np.array([0.0, 0.5, 1.5, 2.])

    isis = compute_isis(spikes)

    assert isinstance(isis, np.ndarray)
    assert np.allclose(isis, np.array([0.5, 1, 0.5]))

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
