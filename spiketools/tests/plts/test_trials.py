"""Tests for spiketools.plts.trials"""

import numpy as np

from spiketools.sim import trials
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


def test_plot_trial_placefield():
    
    trial_place_bins = np.array([[0,1,2,1,0,0,0,0,0,0],
                        [0,.5,1,.5,0,0,0,0,0,0],
                        [0,1,2,1,0,0,0,0,0,0]])
    plot_trial_placefield(trial_place_bins, average='mean', shade='sem', add_traces=True, trace_cmap='Greys', file_path=TEST_PLOTS_PATH, file_name='tplot_trial_placefield.png')
