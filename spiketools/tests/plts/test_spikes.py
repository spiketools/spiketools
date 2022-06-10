"""Tests for spiketools.plts.spikes"""

import numpy as np

from spiketools.tests.tutils import plot_test
from spiketools.tests.tsettings import TEST_PLOTS_PATH

from spiketools.plts.spikes import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_waveform(twaveform):

    plot_waveform(twaveform,
                  file_path=TEST_PLOTS_PATH, file_name='tplot_waveform1d.png')

    plot_waveform(np.array([twaveform, twaveform + 1, twaveform -1]),
                  average='mean', shade='var', add_traces=True,
                  file_path=TEST_PLOTS_PATH, file_name='tplot_waveform2d.png')

@plot_test
def test_plot_waveforms3d(twaveform):

    plot_waveforms3d(np.vstack([twaveform] *  3),
                     file_path=TEST_PLOTS_PATH, file_name='tplot_waveforms3d.png')

@plot_test
def test_plot_waveform_density(twaveform):

    plot_waveform_density(np.vstack([twaveform] *  3),
                          file_path=TEST_PLOTS_PATH, file_name='tplot_waveform_density.png')

@plot_test
def test_plot_isis(tisis):

    plot_isis(tisis,
              file_path=TEST_PLOTS_PATH, file_name='tplot_isis.png')

@plot_test
def test_plot_firing_rates():

    plot_firing_rates(np.array([2.5, 0.5, 1.2, 3.4]),
                      file_path=TEST_PLOTS_PATH, file_name='tplot_firing_rates.png')
