"""Tests for spiketools.plts.data"""

import numpy as np

from spiketools.tests.tutils import plot_test
from spiketools.tests.tsettings import TEST_PLOTS_PATH

from spiketools.plts.data import *

###################################################################################################
###################################################################################################

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
