"""Tests for spiketools.plts.trials"""

import numpy as np

from spiketools.tests.tutils import plot_test
from spiketools.tests.tsettings import TEST_PLOTS_PATH

from spiketools.plts.trials import *
from spiketools.sim.trials import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_rasters():

    d_one_trial = [-0.5, -0.25, 0.250, 1.0]
    d_multi_trial1 = [[-0.75, -0.30, 0.125, 0.250, 0.750],
                      [-0.50, -0.40, -0.50, 0.10, 0.125, 0.50, 0.80],
                      [-0.85, -0.50, -0.25, 0.10, 0.40, 0.750, 0.950]]
    d_multi_trial2 = [[-0.40, 0.15, 0.50],
                      [-0.50, 0.25, 0.80]]

    plot_rasters(d_one_trial,
                 file_path=TEST_PLOTS_PATH, file_name='tplot_rasters0.png')
    plot_rasters(d_multi_trial1,
                 file_path=TEST_PLOTS_PATH, file_name='tplot_rasters1.png')
    plot_rasters([d_multi_trial1, d_multi_trial2], colors=['blue', 'red'],
                 file_path=TEST_PLOTS_PATH, file_name='tplot_rasters2.png')
    plot_rasters({'c0' : d_multi_trial1, 'c1' : d_multi_trial2},
                 colors={'c0' : 'blue', 'c1' : 'red'},
                 file_path=TEST_PLOTS_PATH, file_name='tplot_rasters3.png')
    plot_rasters(d_multi_trial1, events=[0.5, 0.6, 0.4],
                 file_path=TEST_PLOTS_PATH, file_name='tplot_rasters4.png')
    plot_rasters([d_multi_trial1, d_multi_trial2],
                 events=[[0.25, 0.5], [0.6], [0.4], [0.6], [0.5]],
                 file_path=TEST_PLOTS_PATH, file_name='tplot_rasters5.png')

@plot_test
def test_plot_rate_by_time():

    x_vals = np.array([1, 2, 3, 4, 5])
    y_vals1 = np.array([[2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]])
    y_vals2 = np.array([[3, 3, 3, 3, 3], [4, 4, 4, 4, 4], [5, 5, 5, 5, 5]])

    plot_rate_by_time(x_vals, np.mean(y_vals1, 0),
                      file_path=TEST_PLOTS_PATH, file_name='tplot_time_rates1.png')

    plot_rate_by_time(x_vals, [y_vals1, y_vals2], average='median', shade='sem',
                      labels=['A', 'B'], stats=[0.5, 0.01, 0.5, 0.01, 0.5],
                      file_path=TEST_PLOTS_PATH, file_name='tplot_time_rates2.png')

    plot_rate_by_time(x_vals, {'c0' : y_vals1, 'c1' : y_vals2},
                      average='median', shade='sem',
                      colors = {'c0' : 'red', 'c1' : 'blue'},
                      file_path=TEST_PLOTS_PATH, file_name='tplot_time_rates3.png')


def test_plot_raster_and_rates():

    trial_spikes = [[-0.75, -0.30, 0.125, 0.250, 0.750],
                    [-0.50, -0.40, -0.50, 0.10, 0.125, 0.50, 0.80],
                    [-0.85, -0.50, -0.25, 0.10, 0.40, 0.750, 0.950],
                    [-0.40, 0.15, 0.50],
                    [-0.50, 0.25, 0.80]]
    trial_spikes = [np.array(el) for el in trial_spikes]

    plot_raster_and_rates(trial_spikes, 0.5, [-1, 1],
                          file_path=TEST_PLOTS_PATH, file_name='tplot_raster_rates1.png')

    plot_raster_and_rates(trial_spikes, 0.5, [-1, 1],
                          conditions=['l', 'r', 'l', 'r', 'l'],
                          colors={'l' : 'red', 'r' : 'blue'},
                          file_path=TEST_PLOTS_PATH, file_name='tplot_raster_rates2.png')

def test_create_raster_title():

    title1 = create_raster_title('label1', 1.0, 2.0)
    assert isinstance(title1, str)

    title2 = create_raster_title('label2', 1.0, 2.0, 2.5, 0.5)
    assert isinstance(title2, str)

    assert title1 != title2


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