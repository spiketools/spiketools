"""Tests for spiketools.plts.spikes"""

import numpy as np

from spiketools.tests.tutils import plot_test
from spiketools.tests.tsettings import TEST_PLOTS_PATH

from spiketools.plts.spikes import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_waveform():

    data1 = np.array([0, 0, 0, 1, 2, 3, 4, 5, 3, 1, 0, 0])
    data2 = np.array([0, 0, 1, 2, 3, 4, 5, 3, 1, 0, 0, 0])
    data3 = np.array([0, 0, 0, 0, 1, 2, 3, 4, 5, 3, 1, 0])

    plot_waveform(data1,
                  file_path=TEST_PLOTS_PATH, file_name='tplot_waveform1.png')

    plot_waveform(np.array([data1, data2, data3]), average='mean', shade='var', add_traces=True,
                  file_path=TEST_PLOTS_PATH, file_name='tplot_waveform2.png')


@plot_test
def test_plot_isis():

    data = np.array([0.1, 0.25, 0.4, 0.1, 0.05, 0.2, 0.125])

    plot_isis(data,
              file_path=TEST_PLOTS_PATH, file_name='tplot_isis.png')

@plot_test
def test_plot_unit_frs():

    data = np.array([2.5, 0.5, 1.2, 3.4])

    plot_unit_frs(data,
                  file_path=TEST_PLOTS_PATH, file_name='tplot_units_frs.png')
