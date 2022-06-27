"""Tests for spiketools.utils.data"""

import numpy as np

from spiketools.utils.data import *

###################################################################################################
###################################################################################################

def test_get_range():

    data = np.array([0.5, 1., 1.5, 2., 2.5])
    minv, maxv = get_range(data)
    assert isinstance(minv, float)
    assert isinstance(maxv, float)
    assert minv == 0.5
    assert maxv == 2.5


def test_smooth_data():

    # Check 1d case
    data = np.array([0.5, 1., 1.5, 2., 2.5])
    out = smooth_data(data, 0.5)
    assert isinstance(out, np.ndarray)
    assert not np.array_equal(data, out)

    # Check 2d case
    data = np.array([[0.5, 1., 1.5, 2., 2.5], [0.5, 1., 1.5, 2., 2.5]])
    out = smooth_data(data, 0.5)
    assert isinstance(out, np.ndarray)
    assert not np.array_equal(data, out)
