"""Tests for spiketools.plts.space"""

import numpy as np

from spiketools.tests.tutils import plot_test
from spiketools.tests.tsettings import TEST_PLOTS_PATH

from spiketools.plts.space import *

###################################################################################################
###################################################################################################

@plot_test
def test_plot_space_heat():

    data = np.array([[0., 1., 2.], [0., 2., 1.], [0., 3., 2.]])

    plot_space_heat(data, transpose=True, ignore_zero=True, title='Test Space Plot',
                    file_path=TEST_PLOTS_PATH, file_name='test_plot_space.png')
