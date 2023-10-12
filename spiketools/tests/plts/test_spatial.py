"""Tests for spiketools.plts.spatial"""

import numpy as np

from spiketools.tests.tutils import plot_test, fig_test
from spiketools.tests.tsettings import TEST_PLOTS_PATH

from spiketools.plts.spatial import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_positions():

    positions = np.array([[1, 2, 3, 2, 3, 3, 2],
                          [6, 7, 5, 6, 7, 6, 5]])
    spike_pos = np.array([[1, 3, 2],
                          [6, 7, 6]])
    x_bins = [1, 2, 3, 4, 5]
    y_bins = [6, 7, 8, 9]

    plot_positions(positions, spike_pos, x_bins=x_bins, y_bins=y_bins,
                   file_path=TEST_PLOTS_PATH, file_name='tplot_positions.png')

    # Test with list of positions input
    positions_lst = [positions, positions + 1.5]
    plot_positions(positions_lst, spike_pos, x_bins=x_bins, y_bins=y_bins,
                   file_path=TEST_PLOTS_PATH, file_name='tplot_positions_lst.png')

    # Test with landmarks
    landmarks = [{'positions' : np.array([[1, 2], [3, 4]]), 'color' : 'orange', 'alpha' : 0.5},
                 {'positions' : np.array([[1, 2], [5, 2]]), 'color' : 'purple', 'alpha' : 0.75}]
    plot_positions(positions, spike_pos, landmarks,
                   file_path=TEST_PLOTS_PATH, file_name='tplot_positions_landmarks.png')

@plot_test
def test_plot_position_1d():

    position = np.array([1.5, 2.5, 3.5])

    # test single array event input
    events = np.array([1, 2, 3])
    plot_position_1d(position, events,
                     file_path=TEST_PLOTS_PATH, file_name='tplot_position_1d_array.png')

    # test multi array event input
    events = [np.array([1, 2, 3]), np.array([0.5, 2.5, 3.5])]
    plot_position_1d(position, events, colors=['red', 'green'], sizes=[1, 0.75],
                     file_path=TEST_PLOTS_PATH, file_name='tplot_position_1d_list.png')

    # test dict event input
    events = [{'positions' : np.array([1, 2, 3]), 'color' : 'purple', 'size' : 1},
              {'positions' : np.array([0.5, 2.5, 3.5]), 'color' : 'green', 'size' : 0.75}]
    plot_position_1d(position, events,
                     file_path=TEST_PLOTS_PATH, file_name='tplot_position_1d_dict.png')

@plot_test
def test_plot_position_by_time():

    ptimes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    positions = np.array([1, 2, 3, 2, 3, 3, 2, 3, 4, 3])
    spike_times = np.array([3, 5, 8])
    spike_pos = np.array([2, 3, 4])
    event_times = np.array([2, 5])
    event_positions = np.array([3, 3])

    plot_position_by_time(ptimes, positions,
                          file_path=TEST_PLOTS_PATH, file_name='tplot_position_by_time.png')

    # Test with spikes
    plot_position_by_time(ptimes, positions, spike_times, spike_pos,
                          file_path=TEST_PLOTS_PATH, file_name='tplot_position_by_time_spikes.png')

    # Test with events
    plot_position_by_time(ptimes, positions, event_times=event_times, event_positions=event_positions,
                          file_path=TEST_PLOTS_PATH, file_name='tplot_position_by_time_events.png')

@plot_test
def test_plot_heatmap():

    data = np.array([[0., 1., 2.], [0., 2., 1.], [0., 3., 2.]])

    plot_heatmap(data, transpose=True, ignore_zero=True,
                 file_path=TEST_PLOTS_PATH, file_name='tplot_heatmap.png')

    plot_heatmap(data, smooth=True, cbar=True,
                 file_path=TEST_PLOTS_PATH, file_name='tplot_heatmap_smooth.png')

    # Check 1d array input
    data1d = np.array([0., 1., 2., 0., 2., 1.])
    plot_heatmap(data1d,
                 file_path=TEST_PLOTS_PATH, file_name='tplot_heatmap_1d.png')

@fig_test
def test_plot_trial_heatmaps():

    temp = np.array([[0., 1., 2.], [0., 2., 1.], [0., 3., 2.]])
    data = np.stack([temp, temp])

    plot_trial_heatmaps(data,
                        file_path=TEST_PLOTS_PATH, file_name='tplot_trial_heatmaps.png')

def test_create_heatmap_title():

    data = np.array([[0., 1., 2.], [0., 2., 1.], [0., 3., 2.]])

    title1 = create_heatmap_title('label', data)
    assert isinstance(title1, str)

    title2 = create_heatmap_title('label2', data, 2.5, 0.5)
    assert isinstance(title2, str)

    assert title1 != title2
