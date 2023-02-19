"""Tests for spiketools.measures.trials"""

import numpy as np

from spiketools.measures.trials import *

###################################################################################################
###################################################################################################

def test_compute_trial_frs(ttrial_spikes):

    trial_spikes = [ttrial_spikes, ttrial_spikes]
    bins = np.arange(-1, 1 + 0.25, 0.25)
    bin_times, trial_frs = compute_trial_frs(trial_spikes, bins)
    assert isinstance(trial_frs, np.ndarray)
    assert trial_frs.shape == (len(trial_spikes), len(bins) - 1)

def test_compute_pre_post_rates(ttrial_spikes):

    trial_spikes = [ttrial_spikes, ttrial_spikes, ttrial_spikes]
    pre_window = [-0.75, -0.25]
    post_window = [0.25, 0.75]

    frs_pre, frs_post = compute_pre_post_rates(trial_spikes, pre_window, post_window)
    assert len(frs_pre) == len(frs_post) == len(trial_spikes)

def test_compute_segment_frs():

    segments = np.array([[1, 2, 3], [4, 5, 6]])
    spikes = np.array([0.5, 1.5, 2.5, 4.5, 5.5, 6.5 ])
    trial_spikes = np.array([[1.5, 2.5], [4.5, 5.5]])

    frs1 = compute_segment_frs(spikes, segments)
    assert isinstance(frs1, np.ndarray)
    assert frs1.shape == (segments.shape[0], segments.shape[1] - 1)
    assert np.array_equal(frs1, np.array([[1, 1], [1, 1]]))

    frs2 = compute_segment_frs(trial_spikes, segments)
    assert isinstance(frs2, np.ndarray)
    assert frs2.shape == (segments.shape[0], segments.shape[1] - 1)
    assert np.array_equal(frs1, frs2)

def test_compute_pre_post_averages():

    frs1 = np.array([1, 2, 3, 1, 3])
    frs2 = np.array([2, 4, 6, 2, 6])

    avg_pre, avg_post = compute_pre_post_averages(frs1, frs2)
    assert all(isinstance(param, float) for param in [avg_pre, avg_post])
    assert avg_pre == 2.0
    assert avg_post == 4.0

def test_compute_pre_post_diffs():

    frs1 = np.array([1, 2, 3, 1, 3])
    frs2 = np.array([2, 4, 6, 2, 6])

    diffs_avg = compute_pre_post_diffs(frs1, frs2, average=True)
    assert isinstance(diffs_avg, float)
    assert diffs_avg == 2.0

    diffs_all = compute_pre_post_diffs(frs1, frs2, average=False)
    assert isinstance(diffs_all, np.ndarray)
    assert np.array_equal(diffs_all, np.array([1, 2, 3, 1, 3]))
