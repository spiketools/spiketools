"""Tests for spiketools.sim.trials"""

from spiketools.sim.trials import *

###################################################################################################
###################################################################################################

def test_sim_trials():

    n_trials = 2
    time_pre = 1
    time_post = 2

    for method, params in zip(['poisson'], [{'rate_pre' : 5, 'rate_post' : 10}]):

        trials = sim_trials(n_trials, method, time_pre, time_post, **params)

        assert isinstance(trials, list)
        assert len(trials) == n_trials
        for trial in trials:
            assert np.all(trial > -time_pre)
            assert np.all(trial < time_post)

def test_sim_trials_poisson():

    n_trials = 2
    rate_pre = 5
    rate_post = 10
    time_pre = 1
    time_post = 2

    trials = sim_trials_poisson(n_trials, rate_pre, rate_post, time_pre, time_post)
    assert isinstance(trials, list)
    assert len(trials) == n_trials
    for trial in trials:
        assert np.all(trial > -time_pre)
        assert np.all(trial < time_post)

def test_sim_trial_placefield():

    height_mean = 10
    height_std = 2
    width_mean = 10
    width_std = 2
    noise_std = 1
    base_mean = 0
    base_std = 0
    n_trials = 10
    place_loc_mean = 50
    place_loc_std = 10
    n_bins = 100

    trial_placefield = sim_trial_placefield(height_mean, height_std, width_mean, width_std, place_loc_mean, place_loc_std, n_bins, noise_std, base_mean, base_std,
 n_trials, vary_height=True, vary_width=True, vary_place_loc=True, presence_ratio=0.6)

    trial_placefield_no_var = sim_trial_placefield(height_mean, height_std, width_mean, width_std, place_loc_mean, place_loc_std, n_bins, noise_std, base_mean, base_std, n_trials, vary_height=False, vary_width=False, vary_place_loc=False, presence_ratio=None)
    assert isinstance(trial_placefield, np.ndarray)
    assert trial_placefield.shape == (n_trials, n_bins)
    assert np.all(trial_placefield >= 0)
    assert trial_placefield_no_var.shape == (n_trials, n_bins)
    assert np.all(trial_placefield_no_var >= 0)

def test_sim_skew_trial_placefield():

    height_mean = 10
    height_std = 2
    width_mean = 10
    width_std = 2
    noise_std = 1
    base_mean = 0
    base_std = 0
    n_trials = 10
    place_loc_mean = 50
    place_loc_std = 10
    skewness_mean = 0
    skewness_std = 1
    n_bins = 100

    trial_placefield = sim_skew_trial_placefield(height_mean, height_std, width_mean, width_std, place_loc_mean, place_loc_std, skewness_mean, skewness_std, n_bins, noise_std, base_mean, base_std,
 n_trials, vary_height=True, vary_width=True, vary_place_loc=True, vary_skewness=True, presence_ratio=0.6)
     
    trial_placefield_no_var = sim_skew_trial_placefield(height_mean, height_std, width_mean, width_std, place_loc_mean, place_loc_std, skewness_mean, skewness_std, n_bins, noise_std, base_mean, base_std,
 n_trials, vary_height=False, vary_width=False, vary_place_loc=False, vary_skewness=False, presence_ratio=None)

    assert isinstance(trial_placefield, np.ndarray)
    assert trial_placefield.shape == (n_trials, n_bins)
    assert trial_placefield_no_var.shape == (n_trials, n_bins)
    assert np.all(trial_placefield_no_var >= 0)


def test_sim_trial_multi_placefields():

    n_height_mean = [10, 20, 30]
    n_height_std = [2, 2, 2]
    n_width_mean = [10, 20, 30]
    n_width_std = [2, 2, 2]
    n_place_locs_mean = [50, 60, 70]
    n_place_loc_std = [10, 10, 10]
    n_bins = 100
    n_peaks = 3
    base_mean = 0
    base_std = 0
    noise_std = 1
    n_trials = 10

    trial_placefield = sim_trial_multi_placefields(n_height_mean, n_height_std, n_width_mean, n_width_std, n_place_locs_mean, n_place_loc_std, n_bins, n_peaks, base_mean, base_std, noise_std, n_trials, vary_height=True, vary_width=True, vary_place_loc=True, presence_ratio=0.6)
    trial_placefield_no_var = sim_trial_multi_placefields(n_height_mean, n_height_std, n_width_mean, n_width_std, n_place_locs_mean, n_place_loc_std, n_bins, n_peaks, base_mean, base_std, noise_std, n_trials, vary_height=False, vary_width=False, vary_place_loc=False, presence_ratio=None)
    assert isinstance(trial_placefield, np.ndarray)
    assert trial_placefield.shape == (n_trials, n_bins)
    assert trial_placefield_no_var.shape == (n_trials, n_bins)
    assert np.all(trial_placefield >= 0)
    assert np.all(trial_placefield_no_var >= 0)

def test_sim_trial_multi_skew_placefields():

    n_height_mean = [10, 20, 30]
    n_height_std = [2, 2, 2]
    n_width_mean = [10, 20, 30]
    n_width_std = [2, 2, 2]
    n_place_locs_mean = [50, 60, 70]
    n_place_loc_std = [10, 10, 10]
    n_bins = 100
    n_peaks = 3
    n_skewness_mean = [0, 1, 2]
    n_skewness_std = [1, 1, 1]
    base_mean = 0
    base_std = 0
    noise_std = 1
    n_trials = 10

    trial_placefield = sim_trial_multi_skew_placefields(n_height_mean, n_height_std, n_width_mean, n_width_std, n_skewness_mean, n_skewness_std, n_place_locs_mean, n_place_loc_std, n_bins, n_peaks, base_mean, base_std, noise_std, n_trials, vary_height=True, vary_width=True, vary_place_loc=True, vary_skewness=True, presence_ratio=0.6)
    trial_placefield_no_var = sim_trial_multi_skew_placefields(n_height_mean, n_height_std, n_width_mean, n_width_std, n_skewness_mean, n_skewness_std, n_place_locs_mean, n_place_loc_std, n_bins, n_peaks, base_mean, base_std, noise_std, n_trials, vary_height=False, vary_width=False, vary_place_loc=False, vary_skewness=False, presence_ratio=None)
    assert isinstance(trial_placefield, np.ndarray)
    assert trial_placefield.shape == (n_trials, n_bins)
    assert np.all(trial_placefield >= 0)
    assert isinstance(trial_placefield_no_var, np.ndarray)
    assert trial_placefield_no_var.shape == (n_trials, n_bins)
    assert np.all(trial_placefield_no_var >= 0)