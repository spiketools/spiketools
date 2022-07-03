"""Tests for spiketools.measures.trials"""

import numpy as np

from spiketools.measures.trials import *

###################################################################################################
###################################################################################################

def test_compute_trial_frs(ttrial_spikes):

    trial_spikes = [ttrial_spikes, ttrial_spikes]
    bins = np.arange(-1, 1 + 0.25, 0.25)
    out = compute_trial_frs(trial_spikes, bins)
    assert isinstance(out, np.ndarray)
    assert out.shape == (len(trial_spikes), len(bins) - 1)

def test_compute_pre_post_rates(ttrial_spikes):

    trial_spikes = [ttrial_spikes, ttrial_spikes, ttrial_spikes]
    pre_window = [-0.75, -0.25]
    post_window = [0.25, 0.75]

    frs_pre, frs_post = compute_pre_post_rates(trial_spikes, pre_window, post_window)
    assert len(frs_pre) == len(frs_post) == len(trial_spikes)

def test_compute_pre_post_averages():

    frs1 = np.array([1, 2, 3, 1, 3])
    frs2 = np.array([2, 4, 6, 2, 6])

    avg_pre, avg_post = compute_pre_post_averages(frs1, frs2)
    assert all(isinstance(param, float) for param in [avg_pre, avg_post])
    assert avg_pre == 2.0
    assert avg_post == 4.0
