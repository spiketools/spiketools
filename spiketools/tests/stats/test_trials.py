"""Tests for spiketools.stats.trials"""

import numpy as np

from spiketools.stats.trials import *

###################################################################################################
###################################################################################################

def test_compute_pre_post_ttest():

    frs_pre = np.array([1, 2, 1, 2, 1])
    frs_post = np.array([3, 4, 4, 4, 4])

    t_val, p_val = compute_pre_post_ttest(frs_pre, frs_post)
    assert all(isinstance(param, float) for param in [t_val, p_val])

def test_compare_pre_post_activity(ttrial_spikes):

    trial_spikes = [ttrial_spikes, ttrial_spikes, ttrial_spikes]
    pre_window = [-0.75, -0.25]
    post_window = [0.25, 0.75]

    avg_pre, avg_post, t_val, p_val = \
        compare_pre_post_activity(trial_spikes, pre_window, post_window)
    assert all(isinstance(param, float) for param in [avg_pre, avg_post, t_val, p_val])

def test_compare_trial_frs(tdata2d):

    stats = compare_trial_frs(tdata2d, tdata2d * 2)
    assert isinstance(stats, list)
    assert isinstance(stats[0].statistic, float)
    assert isinstance(stats[0].pvalue, float)
