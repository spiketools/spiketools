"""Tests for spiketools.plts.trials"""

import numpy as np

from spiketools.tests.tutils import plot_test
from spiketools.tests.tsettings import TEST_PLOTS_PATH

from spiketools.plts.trials import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_rasters():

    data0 = [-0.5, -0.25, 0.250, 1.0]
    data1 = [[-0.75, -0.30, 0.125, 0.250, 0.750],
             [-0.50, -0.40, -0.50, 0.10, 0.125, 0.50, 0.80],
             [-0.85, -0.50, -0.25, 0.10, 0.40, 0.750, 0.950]]
    data2 = [data1, [[-0.40, 0.15, 0.50], [-0.50, 0.25, 0.80]]]

    plot_rasters(data0, file_path=TEST_PLOTS_PATH, file_name='tplot_rasters0.png')
    plot_rasters(data1, file_path=TEST_PLOTS_PATH, file_name='tplot_rasters1.png')
    plot_rasters(data2, colors=['blue', 'red'],
                 file_path=TEST_PLOTS_PATH, file_name='tplot_rasters2.png')


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

def test_create_raster_title():

    title1 = create_raster_title('label1', 1.0, 2.0)
    assert isinstance(title1, str)

    title2 = create_raster_title('label2', 1.0, 2.0, 2.5, 0.5)
    assert isinstance(title2, str)

    assert title1 != title2
