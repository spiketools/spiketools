"""Tests for spiketools.plts.trials"""

import numpy as np

from spiketools.tests.tutils import plot_test
from spiketools.tests.tsettings import TEST_PLOTS_PATH

from spiketools.plts.trials import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_rasters():

    data1 = [[-750, -300, 125, 250, 750],
             [-500, -400, -50, 100, 125, 500, 800],
             [-850, -500, -250, 100, 400, 750, 950]]
    data2 = [data1, [[-400, 150, 500], [-500, 250, 800]]]

    plot_rasters(data1,
                 file_path=TEST_PLOTS_PATH, file_name='tplot_rasters1.png')

    plot_rasters(data2, colors=['blue', 'red'],
                 file_path=TEST_PLOTS_PATH, file_name='tplot_rasters1.png')


@plot_test
def test_plot_firing_rates():

    x_vals = np.array([1, 2, 3, 4, 5])
    y_vals1 = np.array([[2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]])
    y_vals2 = np.array([[3, 3, 3, 3, 3], [4, 4, 4, 4, 4], [5, 5, 5, 5, 5]])

    plot_firing_rates(x_vals, np.mean(y_vals1, 0),
                      file_path=TEST_PLOTS_PATH, file_name='tplot_firing_rates1.png')

    plot_firing_rates(x_vals, [y_vals1, y_vals2], average='median', shade='sem',
                      labels=['A', 'B'], stats=[0.5, 0.01, 0.5, 0.01, 0.5],
                      file_path=TEST_PLOTS_PATH, file_name='tplot_firing_rates2.png')
