"""Tests for spiketools.measures.measures"""

import numpy as np

from spiketools.measures.measures import *

###################################################################################################
###################################################################################################

def test_compute_spike_rate(tspikes_s):

    rate = compute_spike_rate(tspikes_s)

    assert isinstance(rate, float)
    assert np.isclose(rate, 2.0)

def test_compute_isis(tspikes_s):

    isis = compute_isis(tspikes_s)
    assert isinstance(isis, np.ndarray)
    assert isis.shape[-1] + 1 == tspikes_s.shape[-1]
    assert np.allclose(isis, np.array([0.5, 1, 0.5]))
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
