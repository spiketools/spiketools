"""Tests for spiketools.utils.spikes"""

import numpy as np

from spiketools.utils.spikes import *

###################################################################################################
###################################################################################################

def test_restrict_range():

    data = np.array([0.5, 1., 1.5, 2., 2.5])

    out1 = restrict_range(data, min_time=1.)
    assert np.array_equal(out1, np.array([1., 1.5, 2., 2.5]))

    out2 = restrict_range(data, max_time=2.)
    assert np.array_equal(out2, np.array([0.5, 1., 1.5, 2.]))

    out3 = restrict_range(data, min_time=1., max_time=2.)
    assert np.array_equal(out3, np.array([1., 1.5, 2.]))
