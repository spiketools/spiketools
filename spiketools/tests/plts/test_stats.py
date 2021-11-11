"""Tests for spiketools.plts.stats"""

import numpy as np

from spiketools.tests.tutils import plot_test
from spiketools.tests.tsettings import TEST_PLOTS_PATH

from spiketools.plts.stats import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_surrogates():

    surrogates = np.array([1, 2, 4, 3, 5, 2, 4, 3, 5, 6, 2, 0, 1, 2, 3, 2])
    data_value = 7
    p_value = 0.049

    plot_surrogates(surrogates, data_value, p_value,
                    file_path=TEST_PLOTS_PATH, file_name='tplot_surrogates.png')
