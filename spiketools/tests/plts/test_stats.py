"""Tests for spiketools.plts.stats"""

import numpy as np

from spiketools.tests.tutils import plot_test
from spiketools.tests.tsettings import TEST_PLOTS_PATH

from spiketools.plts.stats import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_surrogates(tdata):

    plot_surrogates(tdata, data_value=2, p_value=0.049,
                    file_path=TEST_PLOTS_PATH, file_name='tplot_surrogates.png')
