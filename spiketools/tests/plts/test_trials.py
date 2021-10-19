"""Tests for spiketools.plts.trials"""

import numpy as np

from spiketools.tests.tutils import plot_test
from spiketools.tests.tsettings import TEST_PLOTS_PATH

from spiketools.plts.trials import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_trial_rasters():

    data = [[-750, -300, 125, 250, 750],
            [-500, -400, -50, 100, 125, 500, 800],
            [-850, -500, -250, 100, 400, 750, 950]]

    plot_trial_rasters(data,
                       file_path=TEST_PLOTS_PATH, file_name='test_plot_trial_rasters.png')
