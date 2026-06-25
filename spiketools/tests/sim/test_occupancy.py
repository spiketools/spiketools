"""Tests for spiketools.sim.occupancy"""

from spiketools.sim.occupancy import (sim_single_trial_occupancy, sim_uniform_occupancy,
                                      sim_occupancy_trials)
import numpy as np

###################################################################################################
###################################################################################################

def test_single_trial_occupancy():

    num_bins = 100
    min_occ = 0.0
    max_occ = 1.0

    trial_occ = sim_single_trial_occupancy(num_bins, min_occ, max_occ)
    assert isinstance(trial_occ, np.ndarray)
    assert trial_occ.shape == (num_bins,)


def test_single_uniform_occupancy():

    num_bins = 100
    scale = 1.0
    uniform_occ = sim_uniform_occupancy(num_bins, scale)
    assert isinstance(uniform_occ, np.ndarray)
    assert uniform_occ.shape == (num_bins,)


def test_sim_occupancy_trials():

    num_trials = 10
    num_bins = 100
    min_occ = 0.0
    max_occ = 1.0
    use_random = False
    uniform_scale = 1.0

    all_trials_occ, avg_occ = sim_occupancy_trials(num_trials, num_bins, min_occ, max_occ,
     use_random, uniform_scale)
    assert isinstance(all_trials_occ, np.ndarray)
    assert isinstance(avg_occ, np.ndarray)
    assert all_trials_occ.shape == (num_trials, num_bins)
    assert avg_occ.shape == (num_bins,)

    use_random = True
    all_trials_occ_random, avg_occ_random = sim_occupancy_trials(num_trials, num_bins, 
                                    min_occ, max_occ, use_random, uniform_scale)
    assert isinstance(all_trials_occ_random, np.ndarray)
    assert isinstance(avg_occ_random, np.ndarray)
    assert all_trials_occ_random.shape == (num_trials, num_bins)
    assert avg_occ_random.shape == (num_bins,)