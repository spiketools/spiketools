"""Tests for spiketools.plts.data"""

import numpy as np

from spiketools.tests.tutils import plot_test
from spiketools.tests.tsettings import TEST_PLOTS_PATH

from spiketools.plts.data import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_lines(tdata):

    data1 = np.random.random(10)
    data2 = np.random.random(10)

    plot_lines(data1, data2, vline=0.5,
               file_path=TEST_PLOTS_PATH, file_name='tplot_line.png')

@plot_test
def test_plot_hist(tdata):

    plot_hist(tdata,
              file_path=TEST_PLOTS_PATH, file_name='tplot_bar.png')

    plot_hist(tdata, average='median',
              file_path=TEST_PLOTS_PATH, file_name='tplot_bar_opts.png')

@plot_test
def test_plot_bar():

    plot_bar(np.array([1., 2., 3.]),
            file_path=TEST_PLOTS_PATH, file_name='tplot_bar.png')

    plot_bar(np.array([1., 2., 3.]), labels=['A1', 'A2', 'A3'],
            file_path=TEST_PLOTS_PATH, file_name='tplot_bar_labels.png')


@plot_test
def test_plot_polar_hist():

    data = np.array([45, 225, 250])
    plot_polar_hist(data, bin_width=90,
                    file_path=TEST_PLOTS_PATH, file_name='tplot_polar_hist.png')
