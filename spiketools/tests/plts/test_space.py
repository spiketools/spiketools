"""Tests for spiketools.plts.space"""

import numpy as np

from spiketools.tests.tutils import plot_test
from spiketools.tests.tsettings import TEST_PLOTS_PATH

from spiketools.plts.space import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_positions():

    positions = np.array([[1, 2, 3, 2, 4, 3, 2],
                          [6, 7, 5, 6, 7, 6, 5]])

    spike_pos = np.array([[2, 4, 2], [6, 7, 6]])
    x_bins = [1, 2, 3, 4, 5]
    y_bins = [6, 7, 8, 9]

    plot_positions(positions, spike_pos, x_bins, y_bins,
                   file_path=TEST_PLOTS_PATH, file_name='tplot_positions.png')

@plot_test
def test_plot_heatmap():

    data = np.array([[0., 1., 2.], [0., 2., 1.], [0., 3., 2.]])

    plot_heatmap(data, transpose=True, ignore_zero=True,
                 file_path=TEST_PLOTS_PATH, file_name='tplot_heatmap.png')

    plot_heatmap(data, smooth=True, cbar=True,
                 file_path=TEST_PLOTS_PATH, file_name='tplot_heatmap_smooth.png')
