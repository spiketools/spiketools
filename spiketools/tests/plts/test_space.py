"""Tests for spiketools.plts.space"""

import numpy as np

from spiketools.tests.tutils import plot_test
from spiketools.tests.tsettings import TEST_PLOTS_PATH

from spiketools.plts.space import *

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

    plot_positions(positions, spike_pos, x_bins, y_bins,
                   file_path=TEST_PLOTS_PATH, file_name='tplot_positions.png')

    # Test with list of positions input
    positions_lst = [positions, positions + 1.5]
    plot_positions(positions_lst, spike_pos, x_bins, y_bins,
                   file_path=TEST_PLOTS_PATH, file_name='tplot_positions_lst.png')

    # Test with landmarks
    landmarks = [{'positions' : np.array([[1, 2], [3, 4]]), 'color' : 'orange', 'alpha' : 0.5},
                 {'positions' : np.array([[1, 2], [5, 2]]), 'color' : 'purple', 'alpha' : 0.75}]
    plot_positions(positions, spike_pos, landmarks,
                   file_path=TEST_PLOTS_PATH, file_name='tplot_positions_landmarks.png')

@plot_test
def test_plot_heatmap():

    data = np.array([[0., 1., 2.], [0., 2., 1.], [0., 3., 2.]])

    plot_heatmap(data, transpose=True, ignore_zero=True,
                 file_path=TEST_PLOTS_PATH, file_name='tplot_heatmap.png')

    plot_heatmap(data, smooth=True, cbar=True,
                 file_path=TEST_PLOTS_PATH, file_name='tplot_heatmap_smooth.png')
